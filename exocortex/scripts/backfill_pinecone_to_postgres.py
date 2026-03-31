from __future__ import annotations

"""
Best-effort backfill for Option B cutover.

Reality check:
Pinecone doesn't support a cheap "scan all vectors" operation on free tier.
So this script does a practical staged backfill by querying Pinecone with
broad semantic queries per chat_id and inserting the retrieved *main* memories
and chunk children into Postgres.

Use after your initial dual-write has started, or when you need to populate
Postgres from legacy Pinecone.
"""

import logging
import os
from typing import Any, Dict, List, Set

from src.config import load_config
from src.db import init_db, insert_chunks, insert_memory
from src.memory import MemoryManager


logger = logging.getLogger(__name__)


def _parse_chat_ids_from_env() -> Set[int]:
    raw = os.getenv("ALLOWED_CHAT_IDS", "").strip()
    ids: Set[int] = set()
    if raw:
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                ids.add(int(part))
            except ValueError:
                pass
    owner_raw = os.getenv("OWNER_CHAT_ID", "").strip()
    if owner_raw:
        try:
            ids.add(int(owner_raw))
        except ValueError:
            pass
    return ids


def _pinecone_md_to_memory_row(md: Dict[str, Any]) -> Dict[str, Any]:
    memory_id = str(md.get("id") or md.get("memory_id") or "")
    user_id = md.get("user_id")
    chat_id = md.get("chat_id")

    raw_content_full = str(md.get("raw_content") or md.get("raw_content_full") or "")
    return {
        "memory_id": memory_id,
        "user_id": int(user_id) if user_id is not None else int(chat_id) if chat_id is not None else 0,
        "chat_id": chat_id,
        "title": md.get("title"),
        "raw_content_full": raw_content_full,
        "source_type": str(md.get("source_type") or "text"),
        "source_url": md.get("url") or md.get("source_url"),
        "tags": md.get("tags"),
        "created_at_ts": int(md.get("created_at_ts") or 0),
        "due_at_ts": md.get("due_at_ts"),
        "last_accessed_ts": md.get("last_accessed_ts"),
        "priority_score": float(md.get("priority_score") or 0.5),
        "last_resurfaced_ts": md.get("last_resurfaced_ts"),
        "visibility": md.get("visibility"),
        "parent_id": md.get("parent_id"),
        "tunnel_id": md.get("tunnel_id"),
        "tunnel_name": md.get("tunnel_name"),
        "is_full": bool(md.get("is_full", True)),
    }


def _pinecone_md_to_chunk_row(md: Dict[str, Any]) -> Dict[str, Any]:
    chunk_id = str(md.get("id") or "")
    return {
        "chunk_id": chunk_id,
        "memory_id": str(md.get("parent_id") or md.get("memory_id") or ""),
        "user_id": int(md.get("user_id") or md.get("chat_id") or 0),
        "chat_id": md.get("chat_id"),
        "chunk_index": int(md.get("chunk_index") or 0),
        "chunk_text": str(md.get("raw_content") or md.get("chunk_text") or ""),
        "source_type": str(md.get("source_type") or "text_chunk"),
        "created_at_ts": int(md.get("created_at_ts") or 0),
    }


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    init_db()

    config = load_config()
    memory = MemoryManager(config)

    chat_ids = _parse_chat_ids_from_env()
    if not chat_ids:
        raise RuntimeError("No chat ids found. Set ALLOWED_CHAT_IDS and/or OWNER_CHAT_ID.")

    # How many pinecone candidates to fetch per chat.
    limit_per_chat = int(os.getenv("BACKFILL_LIMIT_PER_CHAT", "2000"))

    inserted_mains = 0
    inserted_chunks = 0

    for chat_id in sorted(chat_ids):
        logger.info("Backfilling chat_id=%s", chat_id)

        # Broad Pinecone query; with metadata filters this will usually return
        # a large fraction of relevant vectors.
        try:
            candidates = memory.query_by_filter_for_chat(
                query_text="memories notes links pdf reminders",
                chat_id=chat_id,
                filter_obj={"archived": {"$ne": True}},
                k=limit_per_chat,
            )
        except Exception as exc:
            logger.exception("Pinecone query failed for chat_id=%s: %s", chat_id, exc)
            continue

        main_rows: List[Dict[str, Any]] = []
        chunk_rows: List[Dict[str, Any]] = []

        for md in candidates:
            if not memory.is_main_memory(md):
                # chunks are also stored in pinecone; backfill them into memory_chunks
                if str(md.get("id") or "") and md.get("parent_id") and not md.get("is_full", True):
                    chunk_rows.append(_pinecone_md_to_chunk_row(md))
                continue

            row = _pinecone_md_to_memory_row(md)
            if row["memory_id"]:
                main_rows.append(row)

        # Insert (best effort). We don't use upsert here; repeated runs may cause duplicates
        # depending on your Postgres constraints. Prefer one-off execution.
        try:
            for r in main_rows:
                insert_memory(r)
            inserted_mains += len(main_rows)
        except Exception:
            logger.exception("Failed inserting main rows for chat_id=%s", chat_id)

        try:
            if chunk_rows:
                insert_chunks(chunk_rows)
                inserted_chunks += len(chunk_rows)
        except Exception:
            logger.exception("Failed inserting chunk rows for chat_id=%s", chat_id)

        logger.info(
            "chat_id=%s done: mains=%d chunks=%d (candidates=%d)",
            chat_id,
            len(main_rows),
            len(chunk_rows),
            len(candidates),
        )

    logger.info("Backfill finished: mains=%d chunks=%d", inserted_mains, inserted_chunks)


if __name__ == "__main__":
    main()

