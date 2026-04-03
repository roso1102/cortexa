from __future__ import annotations

import logging
import os
import re
import threading
from datetime import datetime, timezone
from typing import Any, Dict
from urllib.parse import unquote

from flask import Blueprint, jsonify, request, g
from flask_cors import CORS
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from werkzeug.security import check_password_hash

from src.memory import MemoryManager
from src.reflection import ReflectionService

logger = logging.getLogger(__name__)

# Werkzeug scrypt hashes look like: scrypt:32768:8:1$salt$hexdigest
# Some clients/DB paths have stored parameters-only (missing "scrypt:"), which makes
# werkzeug treat "32768" as the algorithm name and raise ValueError.
_SCRYPARAMS_ONLY = re.compile(r"^\d+:\d+:\d+\$.+")

def _coerce_stored_password_hash(stored: str) -> str:
    p = (stored or "").strip()
    if _SCRYPARAMS_ONLY.match(p) and not p.startswith("scrypt:"):
        return f"scrypt:{p}"
    return p


def _password_ok(stored_hash: str, password: str) -> bool:
    if not stored_hash or stored_hash == "!":
        return False
    pwhash = _coerce_stored_password_hash(stored_hash)
    try:
        return bool(check_password_hash(pwhash, password))
    except ValueError as e:
        logger.warning("password_hash verify failed (malformed hash?): %s", e)
        return False


def _maybe_notify_telegram_login(chat_id: int) -> None:
    """Optional: ping Telegram after successful dashboard login (requires TELEGRAM_TOKEN)."""
    tok = (os.getenv("TELEGRAM_TOKEN") or "").strip()
    if not tok:
        return
    msg = (
        "Dashboard login verified. You can manage memories on the web; "
        "anything you save here stays linked to this chat."
    )
    try:
        import requests

        requests.post(
            f"https://api.telegram.org/bot{tok}/sendMessage",
            json={"chat_id": chat_id, "text": msg},
            timeout=6,
        ).raise_for_status()
    except Exception as exc:
        logger.warning("telegram login notify failed chat_id=%s: %s", chat_id, exc)


def create_api_blueprint(
    *,
    memory: MemoryManager,
    reflection: ReflectionService,
    dashboard_secret: str,
    dashboard_users: Dict[str, str],
    dashboard_public_url: str = "",
) -> Blueprint:
    """
    JSON API for the Cortexa dashboard.

    Security model: per-user token auth.
      - POST /api/auth/login issues a signed token for a given chat_id/password.
      - All other /api/* routes require header: X-Dashboard-Token: <token>
      - CORS enabled for browser dashboard consumption.
    """
    bp = Blueprint("api", __name__, url_prefix="/api")
    CORS(bp)  # token auth still required; CORS just enables browser requests

    serializer = URLSafeTimedSerializer(dashboard_secret or "cortexa-dashboard", salt="cortexa-dashboard")
    _tunnel_lock = threading.Lock()
    _tunnel_runs_in_progress: set[int] = set()

    @bp.before_request
    def _auth() -> Any:
        # CORS preflight: browser sends OPTIONS without X-Dashboard-Token; must not 401
        # or the real GET/POST never runs.
        if request.method == "OPTIONS":
            return None

        # Login endpoint is unauthenticated
        if request.path.endswith("/auth/login") or request.path.endswith("/auth/telegram/start-link"):
            return None

        token = request.headers.get("X-Dashboard-Token", "").strip()
        if not dashboard_secret or not token:
            return jsonify({"error": "unauthorized"}), 401
        try:
            data = serializer.loads(token, max_age=7 * 24 * 3600)
        except (BadSignature, SignatureExpired):
            return jsonify({"error": "unauthorized"}), 401

        user_id = data.get("user_id")
        chat_id = data.get("chat_id")
        if user_id is None or chat_id is None:
            return jsonify({"error": "unauthorized"}), 401

        g.user_id = int(user_id)
        g.chat_id = int(chat_id)
        return None

    @bp.post("/auth/login")
    def login() -> Any:
        if not dashboard_secret:
            return jsonify({"error": "dashboard_auth_disabled"}), 503

        body = request.get_json(silent=True) or {}
        raw_chat_id = str(body.get("chat_id") or "").strip()
        password = str(body.get("password") or "").strip()

        if not raw_chat_id or not password:
            return jsonify({"error": "invalid_credentials"}), 401

        try:
            chat_id_int = int(raw_chat_id)
        except ValueError:
            return jsonify({"error": "invalid_credentials"}), 401

        # Prefer Postgres-backed auth (Option B).
        # Backward-compatible fallback: if DATABASE_URL is not configured, use CORTEXA_USERS map.
        try:
            from src.db import get_user_by_chat_id

            user = get_user_by_chat_id(chat_id_int)
            if not user or not _password_ok(user.password_hash, password):
                return jsonify({"error": "invalid_credentials"}), 401
        except RuntimeError:
            expected = dashboard_users.get(raw_chat_id)
            if not expected or expected != password:
                return jsonify({"error": "invalid_credentials"}), 401

        payload = {"user_id": chat_id_int, "chat_id": chat_id_int}
        token = serializer.dumps(payload)

        notify = body.get("notify_telegram")
        if notify in (True, "true", "1", 1):
            _maybe_notify_telegram_login(chat_id_int)

        return jsonify({"token": token})

    @bp.get("/auth/telegram/start-link")
    def telegram_start_link() -> Any:
        """
        Build a dashboard login URL prefilled with Telegram chat_id from /start flow.
        This removes the need for manual Telegram ID entry in the UI.
        """
        raw_chat_id = str(request.args.get("chat_id") or "").strip()
        if not raw_chat_id:
            return jsonify({"error": "missing_chat_id"}), 400
        try:
            int(raw_chat_id)
        except ValueError:
            return jsonify({"error": "invalid_chat_id"}), 400
        base = (dashboard_public_url or "").strip().rstrip("/")
        if not base:
            return jsonify({"error": "dashboard_public_url_missing"}), 400
        return jsonify({"url": f"{base}/login?chat_id={raw_chat_id}&source=telegram"})

    @bp.get("/memories")
    def get_memories() -> Any:  # noqa: C901
        """
        Query params:
          - q: semantic query text (optional)
          - source_type: filter by metadata source_type (optional)
          - tag: filter by metadata tags contains tag (optional; best-effort)
          - page: 1-based page number (default 1)
          - per_page: items per page (default 20, max 50)
        """
        q = (request.args.get("q") or "").strip()
        source_type = (request.args.get("source_type") or "").strip()
        tag = (request.args.get("tag") or "").strip().lower()
        page = max(int(request.args.get("page") or 1), 1)
        per_page = min(max(int(request.args.get("per_page") or 20), 1), 50)

        filter_obj: Dict[str, Any] = {
            "archived": {"$ne": True},
            "user_id": {"$eq": getattr(g, "user_id", 0)},
        }
        if source_type:
            filter_obj["source_type"] = {"$eq": source_type}

        # Pinecone requires a query vector; list results are approximate.
        # We fetch a larger candidate window so pagination + main-only filtering
        # still returns the expected number of items.
        top_k = min(1000, per_page * page * 10)
        query_text = q or "memories notes links pdf reminders"
        items = memory.query_by_filter(query_text=query_text, filter_obj=filter_obj, k=top_k)

        if tag:
            filtered = []
            for m in items:
                tags = [str(t).lower() for t in (m.get("tags") or [])]
                if tag in tags:
                    filtered.append(m)
            items = filtered

        # Dashboard should show only "main" memories (not chunk children).
        # Storage model:
        # - text/reminder full records: source_type="text"/"reminder", is_full=True
        # - text/reminder chunk children: source_type="text_chunk"/"reminder_chunk"
        # - link/pdf chunk children: source_type="link"/"pdf" with chunk_index>0 (old) or
        #   source_type="link_chunk"/"pdf_chunk" (new)
        cleaned: list[Dict[str, Any]] = []
        for m in items:
            st = str(m.get("source_type") or "")
            if st.endswith("_chunk"):
                continue
            if m.get("is_full") is False:
                continue
            if st in {"link", "pdf"}:
                try:
                    if int(m.get("chunk_index") or 0) != 0:
                        continue
                except Exception:
                    continue
            cleaned.append(m)
        items = cleaned

        total = len(items)
        start = (page - 1) * per_page
        end = start + per_page
        page_items = items[start:end]

        # Trim raw_content for list view payload size
        for m in page_items:
            raw = str(m.get("raw_content") or "")
            m["raw_content"] = raw[:800]

        return jsonify(
            {
                "items": page_items,
                "total": total,
                "page": page,
                "per_page": per_page,
            }
        )

    @bp.get("/memories/<memory_id>")
    def get_memory(memory_id: str) -> Any:
        try:
            # Normalize ID: handle once- or twice-encoded timestamp ids
            prev = memory_id
            for _ in range(5):
                decoded = unquote(prev)
                if decoded == prev:
                    break
                prev = decoded
            normalized_id = prev

            md = memory.get_memory_by_id(normalized_id)
            if not md or int(md.get("user_id") or 0) != getattr(g, "user_id", 0):
                return jsonify({"error": "not_found"}), 404
            return jsonify({"item": md})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @bp.get("/tunnels")
    def get_tunnels() -> Any:
        try:
            from src.db import fetch_tunnels_for_user

            tunnels = fetch_tunnels_for_user(user_id=int(getattr(g, "user_id", 0)), limit=30)
        except Exception:
            # Fallback: legacy Pinecone tunnel objects (if Postgres not configured yet)
            tunnels = memory.query_by_filter(
                query_text="tunnels themes clusters",
                filter_obj={"source_type": {"$eq": "tunnel"}},
                k=30,
            )
        return jsonify({"tunnels": tunnels})

    @bp.post("/tunnels/generate")
    def generate_tunnels_now() -> Any:
        """
        Manually trigger tunnel formation for the authenticated user.
        Returns newly generated tunnel metadata.
        """
        user_id = int(getattr(g, "user_id", 0) or 0)
        if not user_id:
            return jsonify({"error": "unauthorized"}), 401

        with _tunnel_lock:
            if user_id in _tunnel_runs_in_progress:
                return jsonify({"error": "generation_in_progress"}), 409
            _tunnel_runs_in_progress.add(user_id)

        try:
            from src.tunnels import form_tunnels

            # Reuse already-initialized clients from ReflectionService.
            groq_client = getattr(reflection, "_groq", None)
            openrouter_key = str(getattr(reflection, "_openrouter_api_key", "") or "")
            if groq_client is None:
                return jsonify({"error": "groq_client_unavailable"}), 503

            tunnels = form_tunnels(
                memory,
                groq_client,
                user_id=user_id,
                openrouter_api_key=openrouter_key,
            )
            return jsonify(
                {
                    "ok": True,
                    "count": len(tunnels),
                    "tunnels": tunnels,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        except Exception as exc:
            logger.exception("Manual tunnel generation failed for user_id=%s", user_id)
            return jsonify({"error": str(exc)}), 500
        finally:
            with _tunnel_lock:
                _tunnel_runs_in_progress.discard(user_id)

    @bp.get("/tunnels/<tunnel_id>/graph")
    def get_tunnel_graph(tunnel_id: str) -> Any:
        user_id = int(getattr(g, "user_id", 0) or 0)
        if not user_id:
            return jsonify({"error": "unauthorized"}), 401
        try:
            from src.db import fetch_tunnel_graph_for_user

            payload = fetch_tunnel_graph_for_user(user_id=user_id, tunnel_id=tunnel_id)
            return jsonify({"tunnel_id": tunnel_id, **payload})
        except Exception as exc:
            logger.exception("Failed to fetch tunnel graph user_id=%s tunnel_id=%s", user_id, tunnel_id)
            return jsonify({"error": str(exc)}), 500

    @bp.get("/profile")
    def get_profile() -> Any:
        try:
            matches = memory.query_by_filter(
                query_text="profile snapshot",
                filter_obj={
                    "source_type": {"$eq": "profile_snapshot"},
                    "user_id": {"$eq": getattr(g, "user_id", 0)},
                },
                k=1,
            )
            snapshot = matches[0].get("raw_content") if matches else ""
            if not snapshot:
                snapshot = reflection.generate_profile_snapshot_for_user(getattr(g, "user_id", 0))
        except Exception as exc:
            return jsonify({"error": str(exc), "snapshot": ""}), 500
        return jsonify({"snapshot": snapshot})

    @bp.get("/summary/today")
    def get_summary_today() -> Any:
        try:
            text = reflection.summarize_today_for_user(getattr(g, "user_id", 0))
        except Exception as exc:
            return jsonify({"error": str(exc), "text": ""}), 500
        return jsonify(
            {
                "text": text,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    return bp

