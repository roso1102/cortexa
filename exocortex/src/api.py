from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict
from urllib.parse import unquote

from flask import Blueprint, jsonify, request
from flask_cors import CORS

from src.memory import MemoryManager
from src.reflection import ReflectionService


def create_api_blueprint(*, memory: MemoryManager, reflection: ReflectionService, dashboard_secret: str) -> Blueprint:
    """
    JSON API for the Cortexa dashboard.

    Security model: single-user token auth.
      - Require header: X-Dashboard-Token: <DASHBOARD_SECRET>
      - CORS enabled for browser dashboard consumption
    """
    bp = Blueprint("api", __name__, url_prefix="/api")
    CORS(bp)  # token auth still required; CORS just enables browser requests

    @bp.before_request
    def _auth() -> Any:
        token = request.headers.get("X-Dashboard-Token", "").strip()
        if not dashboard_secret or token != dashboard_secret:
            return jsonify({"error": "unauthorized"}), 401
        return None

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

        filter_obj: Dict[str, Any] = {"archived": {"$ne": True}}
        if source_type:
            filter_obj["source_type"] = {"$eq": source_type}

        # Pinecone metadata filtering doesn't support "contains" on arrays in a portable way.
        # We'll fetch a larger window and filter tags in Python.
        top_k = min(200, per_page * page)
        query_text = q or "memories notes links pdf reminders"
        items = memory.query_by_filter(query_text=query_text, filter_obj=filter_obj, k=top_k)

        if tag:
            filtered = []
            for m in items:
                tags = [str(t).lower() for t in (m.get("tags") or [])]
                if tag in tags:
                    filtered.append(m)
            items = filtered

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
            if not md:
                return jsonify({"error": "not_found"}), 404
            return jsonify({"item": md})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @bp.get("/tunnels")
    def get_tunnels() -> Any:
        try:
            tunnels = memory.query_by_filter(
                query_text="tunnels themes clusters",
                filter_obj={"source_type": {"$eq": "tunnel"}},
                k=30,
            )
        except Exception as exc:
            return jsonify({"error": str(exc), "tunnels": []}), 500
        return jsonify({"tunnels": tunnels})

    @bp.get("/profile")
    def get_profile() -> Any:
        try:
            matches = memory.query_by_filter(
                query_text="profile snapshot",
                filter_obj={"source_type": {"$eq": "profile_snapshot"}},
                k=1,
            )
            snapshot = matches[0].get("raw_content") if matches else ""
        except Exception as exc:
            return jsonify({"error": str(exc), "snapshot": ""}), 500
        return jsonify({"snapshot": snapshot})

    @bp.get("/summary/today")
    def get_summary_today() -> Any:
        try:
            text = reflection.summarize_today()
        except Exception as exc:
            return jsonify({"error": str(exc), "text": ""}), 500
        return jsonify(
            {
                "text": text,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    return bp

