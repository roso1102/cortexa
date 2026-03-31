from __future__ import annotations

"""
Smoke tests for dashboard API tenant isolation + no chunk leakage.

Assumes the backend Flask server is running locally/accessible:
  API_BASE_URL=http://127.0.0.1:8080

Requires:
  - DASHBOARD_SECRET set in env
  - Postgres DATABASE_URL set so we can pick a memory id

Set:
  - EVAL_USER_A (chat_id/user_id)
  - EVAL_USER_B (different chat_id/user_id)
"""

import json
import os
from typing import Any, Dict, List

import requests
from itsdangerous import URLSafeTimedSerializer

from src.config import load_config
from src.db import get_engine


def _token(serializer: URLSafeTimedSerializer, *, user_id: int, chat_id: int) -> str:
    payload = {"user_id": int(user_id), "chat_id": int(chat_id)}
    return serializer.dumps(payload)


def _pick_memory_id(user_id: int) -> str:
    engine = get_engine()
    with engine.begin() as conn:
        res = conn.execute(
            "SELECT memory_id FROM memories WHERE user_id=:uid AND is_full=true ORDER BY created_at_ts DESC LIMIT 1",
            {"uid": int(user_id)},
        )
        row = res.fetchone()
        if not row:
            raise RuntimeError(f"No memory found in Postgres for user_id={user_id}. Backfill/dual-write first.")
        return str(row[0])


def main() -> None:
    config = load_config()
    if not config.dashboard_secret:
        raise RuntimeError("DASHBOARD_SECRET must be set for API smoke tests.")

    api_base = os.getenv("API_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
    user_a = int(os.getenv("EVAL_USER_A", str(config.owner_chat_id or 0)))
    user_b_raw = os.getenv("EVAL_USER_B", "")
    if user_b_raw:
        user_b = int(user_b_raw)
    else:
        # Pick a different allowed chat id if possible.
        allowed = sorted(list(config.allowed_chat_ids))
        user_b = allowed[0] if allowed else user_a + 1

    if user_b == user_a:
        user_b = user_a + 1

    serializer = URLSafeTimedSerializer(config.dashboard_secret or "cortexa-dashboard", salt="cortexa-dashboard")
    tok_a = _token(serializer, user_id=user_a, chat_id=user_a)
    tok_b = _token(serializer, user_id=user_b, chat_id=user_b)

    memory_id = _pick_memory_id(user_a)

    headers_a = {"X-Dashboard-Token": tok_a}
    headers_b = {"X-Dashboard-Token": tok_b}

    # Detail endpoint must return 404 for other tenant.
    r = requests.get(f"{api_base}/api/memories/{memory_id}", headers=headers_b, timeout=15)
    assert r.status_code in (401, 404), f"Expected tenant denial, got {r.status_code}: {r.text}"

    # Detail endpoint should return 200 for correct tenant.
    r = requests.get(f"{api_base}/api/memories/{memory_id}", headers=headers_a, timeout=15)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    item = r.json().get("item") or {}
    assert str(item.get("user_id") or user_a) == str(user_a), "Returned item has wrong user_id"

    # List endpoint should only return own tenant's main items.
    r = requests.get(f"{api_base}/api/memories?per_page=10&page=1", headers=headers_b, timeout=15)
    assert r.status_code == 200, f"List endpoint failed: {r.status_code} {r.text}"
    payload = r.json() or {}
    for m in payload.get("items") or []:
        st = str(m.get("source_type") or "")
        assert not st.endswith("_chunk"), f"Chunk leakage in list endpoint: {st}"
        if m.get("user_id") is not None:
            assert int(m["user_id"]) == int(user_b), "Tenant leak in list endpoint"

    print("API tenant isolation + no-chunk smoke tests passed.")


if __name__ == "__main__":
    main()

