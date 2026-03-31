from __future__ import annotations

"""
Hybrid retrieval (Option B):
- Postgres FTS (lexical candidates from canonical memories)
- Pinecone semantic recall (embedding candidates, already scoped to chat_id)
- Deterministic merge + re-rank

This module is intentionally defensive:
- If Postgres is not configured (DATABASE_URL missing), it gracefully falls back
  to Pinecone semantic recall.
"""

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Tuple

from sqlalchemy import text as sa_text

from src.db import get_engine
from src.memory import MemoryManager
from src.utils import utc_now_ts


logger = logging.getLogger(__name__)


def _min_max_norm(x: List[float]) -> List[float]:
    if not x:
        return []
    lo = min(x)
    hi = max(x)
    if hi - lo < 1e-9:
        return [0.0 for _ in x]
    return [(v - lo) / (hi - lo) for v in x]


@dataclass(frozen=True)
class HybridResult:
    items: List[Dict[str, Any]]


class HybridRetriever:
    def __init__(self, memory: MemoryManager) -> None:
        self._memory = memory

    def _fts_candidates(self, *, user_id: int, query: str, k: int) -> List[Dict[str, Any]]:
        """
        Lexical candidates from Postgres canonical memories.
        Returns items in the same metadata-ish format expected by Telegram/brains router.
        """
        engine = get_engine()

        # Use the same expression as the optional GIN index in schema.sql.
        vector_expr = (
            "to_tsvector('english', coalesce(title, '') || ' ' || coalesce(raw_content_full, ''))"
        )
        tsquery_expr = "plainto_tsquery('english', :q)"

        sql = sa_text(
            f"""
            SELECT
              memory_id,
              title,
              raw_content_full,
              source_type,
              source_url,
              tags,
              created_at_ts,
              last_accessed_ts,
              priority_score,
              last_resurfaced_ts,
              visibility,
              parent_id,
              is_full,
              ts_rank_cd({vector_expr}, {tsquery_expr}) AS fts_score
            FROM memories
            WHERE
              user_id = :user_id
              AND is_full = true
              AND {vector_expr} @@ {tsquery_expr}
            ORDER BY fts_score DESC
            LIMIT :k
            """
        )

        with engine.begin() as conn:
            rows = conn.execute(sql, {"user_id": user_id, "q": query, "k": k}).fetchall()

        items: List[Dict[str, Any]] = []
        for r in rows:
            d = dict(r._mapping)
            items.append(
                {
                    "id": d.get("memory_id"),
                    "raw_content": d.get("raw_content_full"),
                    "title": d.get("title"),
                    "source_type": d.get("source_type"),
                    "url": d.get("source_url"),
                    "tags": d.get("tags") or [],
                    "created_at_ts": d.get("created_at_ts"),
                    "last_accessed_ts": d.get("last_accessed_ts"),
                    "priority_score": d.get("priority_score"),
                    "last_resurfaced_ts": d.get("last_resurfaced_ts"),
                    "visibility": d.get("visibility"),
                    "parent_id": d.get("parent_id"),
                    "is_full": d.get("is_full"),
                    "fts_score": d.get("fts_score") or 0.0,
                }
            )
        return items

    def recall(self, *, query: str, chat_id: int, k: int = 5) -> List[Dict[str, Any]]:
        """
        Hybrid recall for a specific Telegram chat (tenant isolation).
        """
        logger.debug("Hybrid recall start chat_id=%s k=%s query=%r", chat_id, k, query)

        # Pinecone semantic recall (scoped by chat_id filter inside MemoryManager)
        sem_k = max(20, k * 8)
        sem_candidates = self._memory.recall_context_for_chat(query, chat_id=chat_id, k=sem_k)
        sem_main = [m for m in sem_candidates if self._memory.is_main_memory(m)]

        # Lexical candidates from Postgres; if it fails, we still return Pinecone results.
        fts_candidates: List[Dict[str, Any]] = []
        try:
            # Keep more candidates for stable re-rank.
            fts_candidates = self._fts_candidates(user_id=chat_id, query=query, k=max(20, k * 8))
        except Exception:
            # Most commonly: DATABASE_URL not set yet / Postgres not ready.
            fts_candidates = []

        if not sem_main and not fts_candidates:
            return []

        # Build per-source normalized scores.
        sem_scores = [float(m.get("score") or 0.0) for m in sem_main]
        sem_norms = _min_max_norm(sem_scores)
        sem_by_id: Dict[str, Dict[str, Any]] = {}
        for m, sn in zip(sem_main, sem_norms):
            mid = str(m.get("id") or "")
            if not mid:
                continue
            mm = {**m}
            mm["semantic_norm"] = sn
            sem_by_id[mid] = mm

        fts_scores = [float(m.get("fts_score") or 0.0) for m in fts_candidates]
        fts_norms = _min_max_norm(fts_scores)
        fts_by_id: Dict[str, Dict[str, Any]] = {}
        for m, fn in zip(fts_candidates, fts_norms):
            mid = str(m.get("id") or "")
            if not mid:
                continue
            mm = {**m}
            mm["fts_norm"] = fn
            fts_by_id[mid] = mm

        merged_ids = list(set(sem_by_id.keys()) | set(fts_by_id.keys()))

        now_ts = utc_now_ts()
        ranked: List[Tuple[float, Dict[str, Any]]] = []
        for mid in merged_ids:
            sem_m = sem_by_id.get(mid)
            fts_m = fts_by_id.get(mid)
            base = sem_m or fts_m or {"id": mid}

            semantic_norm = float((sem_m or {}).get("semantic_norm") or 0.0)
            fts_norm = float((fts_m or {}).get("fts_norm") or 0.0)

            created_ts = int((base.get("created_at_ts") or 0) or 0)
            age_days = (now_ts - created_ts) / 86400 if created_ts else 999.0
            recency_norm = 1.0 / (1.0 + age_days)  # 1->recent, ~0->old

            priority = float(base.get("priority_score") or 0.5)
            priority_norm = max(0.0, min(1.0, priority))

            # Weighted sum (deterministic).
            final = (0.55 * semantic_norm) + (0.35 * fts_norm) + (0.08 * recency_norm) + (0.02 * priority_norm)
            ranked.append((final, base))

        ranked.sort(key=lambda x: x[0], reverse=True)
        top = [m for _, m in ranked[:k]]

        logger.debug(
            "Hybrid recall done chat_id=%s semantic=%d fts=%d merged=%d top_ids=%s",
            chat_id,
            len(sem_main),
            len(fts_candidates),
            len(merged_ids),
            [str(m.get("id") or "") for m in top],
        )

        # Ensure fields Telegram expects.
        for m in top:
            if "raw_content" not in m:
                m["raw_content"] = m.get("raw_content_full") or ""
            if "id" not in m:
                m["id"] = m.get("memory_id")
            if "source_url" in m and "url" not in m:
                m["url"] = m.get("source_url")

        return top

