from __future__ import annotations

"""
Smoke-evaluation harness for Option B retrieval.

Checks:
- no chunk leakage in top-k results
- tenant isolation: returned items match the requested chat_id/user_id

This is not a formal ML evaluation (Recall@k), but it's a reliable operator sanity gate.
"""

import json
import logging
import os
from typing import Any, Dict, List

from src.config import load_config
from src.retrieval import HybridRetriever
from src.memory import MemoryManager


def _load_queries() -> List[str]:
    raw = os.getenv("EVAL_QUERIES", "").strip()
    if not raw:
        return [
            "what did i save about memory",
            "what did i save about allergy",
            "what did i save about FPGA",
            "memex",
            "show me my saved links",
        ]
    try:
        arr = json.loads(raw)
        if isinstance(arr, list):
            return [str(x) for x in arr]
    except Exception:
        pass
    # fallback: newline separated
    return [q.strip() for q in raw.splitlines() if q.strip()]


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    config = load_config()

    chat_id_raw = os.getenv("EVAL_CHAT_ID", "").strip()
    if chat_id_raw:
        chat_id = int(chat_id_raw)
    else:
        chat_id = int(config.owner_chat_id or 0)
    if not chat_id:
        raise RuntimeError("Set EVAL_CHAT_ID or OWNER_CHAT_ID in .env for eval.")

    memory = MemoryManager(config)
    retriever = HybridRetriever(memory)

    queries = _load_queries()
    k = int(os.getenv("EVAL_K", "5"))

    print(f"Eval chat_id={chat_id} k={k} queries={len(queries)}")

    for q in queries:
        print("\n---")
        print(f"Query: {q}")
        items = retriever.recall(query=q, chat_id=chat_id, k=k)
        ids = []
        for m in items:
            ids.append(str(m.get("id") or m.get("memory_id") or ""))
            st = str(m.get("source_type") or "")
            is_full = m.get("is_full")

            # No chunk leakage
            if st.endswith("_chunk"):
                raise AssertionError(f"Chunk leakage for query {q}: source_type={st}")
            if is_full is False:
                raise AssertionError(f"Non-full memory returned for query {q}")

            # Tenant isolation
            md_chat_id = m.get("chat_id")
            if md_chat_id is not None and int(md_chat_id) != int(chat_id):
                raise AssertionError(f"Tenant leak for query {q}: chat_id={md_chat_id}")

        print(f"Top IDs: {ids}")
        if not items:
            print("WARNING: empty results")

    print("\nAll smoke checks passed.")


if __name__ == "__main__":
    main()

