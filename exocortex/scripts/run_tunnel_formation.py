#!/usr/bin/env python3
"""
Run tunnel formation once (same logic as the weekly scheduler).

Usage (from repo exocortex/ with .env loaded):
  PYTHONPATH=. python scripts/run_tunnel_formation.py
  PYTHONPATH=. python scripts/run_tunnel_formation.py --user-id 123456789

Defaults user id to OWNER_CHAT_ID from env when --user-id is omitted.
"""
from __future__ import annotations

import argparse

from dotenv import load_dotenv
from groq import Groq

from src.config import load_config
from src.memory import MemoryManager
from src.tunnels import form_tunnels


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Manually run tunnel formation for one user.")
    parser.add_argument(
        "--user-id",
        type=int,
        default=None,
        help="Telegram chat_id / user_id (default: OWNER_CHAT_ID env)",
    )
    args = parser.parse_args()

    config = load_config()
    uid = args.user_id or config.owner_chat_id
    if not uid:
        raise SystemExit(
            "No user id: pass --user-id or set OWNER_CHAT_ID in the environment.",
        )

    memory = MemoryManager(config)
    groq = Groq(api_key=config.groq_api_key)
    tunnels = form_tunnels(memory, groq, user_id=int(uid), openrouter_api_key=config.openrouter_api_key)
    print(f"Tunnel formation done: {len(tunnels)} tunnel(s).")
    for t in tunnels:
        print(f"  - {t.get('tunnel_name')!r} (core_token={t.get('core_tag')!r}, members≈{t.get('memory_count')})")


if __name__ == "__main__":
    main()
