from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec

from src.config import AppConfig
from src.utils import utc_now_iso, utc_now_ts

logger = logging.getLogger(__name__)

_EMBED_CACHE_SIZE = 128
_embed_cache: Dict[str, List[float]] = {}
_embed_cache_order: List[str] = []

# To keep Pinecone metadata under the 40KB limit, we must not
# store arbitrarily large raw_content blobs. We keep a generous
# but safe truncated version here; the canonical full text now
# lives in Postgres.
_MAX_RAW_CONTENT_METADATA_CHARS = 20000

# Back-off state: when a 429 quota error is hit, stop embedding for this many
# seconds so we don't burn the remaining daily quota in a tight retry loop.
_QUOTA_BACKOFF_SECS = 120
_quota_backoff_until: float = 0.0


class MemoryManager:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

        # Configure Gemini — embeddings use genai.embed_content(), not GenerativeModel
        genai.configure(api_key=config.google_api_key)

        # Configure Pinecone
        pc = Pinecone(api_key=config.pinecone_api_key)
        self._index_name = config.pinecone_index_name

        # Pinecone v4 returns IndexModel objects; access .name not ["name"]
        existing = [idx.name for idx in pc.list_indexes()]
        if self._index_name not in existing:
            pc.create_index(
                name=self._index_name,
                dimension=3072,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=os.getenv("PINECONE_ENVIRONMENT", "us-east-1"),
                ),
            )

        self._index = pc.Index(self._index_name)

    def _embed(self, text: str) -> List[float]:
        global _quota_backoff_until

        # Return cached vector if available
        if text in _embed_cache:
            logger.debug("embed cache hit (len=%d)", len(text))
            return _embed_cache[text]

        # Honour quota back-off: if we recently hit 429, skip embedding for now
        if time.monotonic() < _quota_backoff_until:
            remaining = int(_quota_backoff_until - time.monotonic())
            raise RuntimeError(
                f"Gemini embedding quota back-off active ({remaining}s remaining). "
                "Skipping embed to avoid burning daily quota."
            )

        try:
            result = genai.embed_content(
                model="models/gemini-embedding-001",
                content=text,
            )
        except Exception as exc:
            exc_str = str(exc)
            if "ResourceExhausted" in exc_str or "429" in exc_str or "quota" in exc_str.lower():
                _quota_backoff_until = time.monotonic() + _QUOTA_BACKOFF_SECS
                logger.warning(
                    "Gemini embedding quota hit (429). Back-off for %ds.", _QUOTA_BACKOFF_SECS
                )
            raise

        embedding: List[float] = result["embedding"]  # type: ignore[assignment]
        if len(embedding) != 3072:
            raise ValueError(f"Unexpected embedding size: {len(embedding)} (expected 3072)")

        # Store in cache; evict oldest entry when full
        if text not in _embed_cache:
            if len(_embed_cache_order) >= _EMBED_CACHE_SIZE:
                oldest = _embed_cache_order.pop(0)
                _embed_cache.pop(oldest, None)
            _embed_cache[text] = embedding
            _embed_cache_order.append(text)

        logger.debug("embed success (len=%d, dim=%d)", len(text), len(embedding))
        return embedding

    def add_memory(self, text: str, metadata: Dict[str, Any]) -> str:
        vector = self._embed(text)
        memory_id = metadata.get("id") or utc_now_iso()

        # Truncate raw_content stored inside Pinecone metadata so we don't
        # hit the 40KB metadata size limit. Canonical full text is stored
        # in Postgres; Pinecone only needs enough for retrieval + snippets.
        raw_for_metadata = text[:_MAX_RAW_CONTENT_METADATA_CHARS]

        md = {
            **metadata,
            "raw_content": raw_for_metadata,
            "created_at": metadata.get("created_at", utc_now_iso()),
            "created_at_ts": metadata.get("created_at_ts", utc_now_ts()),
        }

        self._index.upsert(
            vectors=[
                {
                    "id": memory_id,
                    "values": vector,
                    "metadata": md,
                }
            ]
        )
        logger.debug(
            "pinecone upsert ok id=%s meta_raw_len=%d",
            memory_id,
            len(raw_for_metadata),
        )
        return memory_id

    def query_by_filter_for_chat(
        self,
        query_text: str,
        chat_id: int,
        filter_obj: Dict[str, Any] | None = None,
        k: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Convenience wrapper: apply a chat_id filter on top of the provided filter_obj.
        Ensures multi-user isolation for Telegram-facing queries.
        """
        base_filter: Dict[str, Any] = {"chat_id": {"$eq": chat_id}}
        if filter_obj:
            base_filter.update(filter_obj)
        return self.query_by_filter(query_text=query_text, filter_obj=base_filter, k=k)

    def query_by_filter(self, query_text: str, filter_obj: Dict[str, Any], k: int = 50) -> List[Dict[str, Any]]:
        """
        Pinecone requires a query vector. We embed a generic query_text and apply a metadata filter.
        Useful for listing items (e.g., links saved today).
        """
        vector = self._embed(query_text)
        res = self._index.query(vector=vector, top_k=k, include_metadata=True, filter=filter_obj)
        matches: List[Dict[str, Any]] = []
        for match in res.get("matches", []):
            md = match.get("metadata") or {}
            # Preserve the Pinecone vector id for downstream updates/deletes
            if match.get("id"):
                md["id"] = match.get("id")
            md["score"] = match.get("score")
            matches.append(md)
        return matches

    @staticmethod
    def is_main_memory(md: Dict[str, Any]) -> bool:
        """
        Return True if this metadata represents a user-visible "main" memory.
        Chunk children should not appear in dashboard lists, resurfacing, or tunnels.
        """
        st = str(md.get("source_type") or "")
        if st.endswith("_chunk"):
            return False
        if md.get("is_full") is False:
            return False
        # Backward compatibility: older link/pdf chunks were stored as source_type=link/pdf with chunk_index>0
        if st in {"link", "pdf"}:
            try:
                if int(md.get("chunk_index") or 0) != 0:
                    return False
            except Exception:
                return False
        return True

    def recall_context(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        vector = self._embed(query)
        res = self._index.query(vector=vector, top_k=k, include_metadata=True)
        matches = []
        now_ts = utc_now_ts()
        for match in res.get("matches", []):
            md = match.get("metadata") or {}
            if match.get("id"):
                md["id"] = match.get("id")
            md["score"] = match.get("score")
            matches.append(md)
            # Update last_accessed_ts in the background — best effort, non-blocking
            mem_id = match.get("id")
            if mem_id:
                try:
                    updated = {**md, "last_accessed_ts": now_ts}
                    updated.pop("score", None)
                    raw = updated.get("raw_content", "")
                    if raw:
                        self._index.upsert(
                            vectors=[{"id": mem_id, "values": match["values"] if "values" in match else vector, "metadata": updated}]
                        )
                except Exception:
                    pass  # access tracking is best-effort
        return matches

    def recall_context_for_chat(self, query: str, chat_id: int, k: int = 3) -> List[Dict[str, Any]]:
        """
        Like recall_context, but restricted to a specific chat_id.
        Used by the Telegram bot so different users don't see each other's memories.
        """
        vector = self._embed(query)
        res = self._index.query(
            vector=vector,
            top_k=k,
            include_metadata=True,
            filter={"chat_id": {"$eq": chat_id}, "archived": {"$ne": True}},
        )
        matches: List[Dict[str, Any]] = []
        now_ts = utc_now_ts()
        for match in res.get("matches", []):
            md = match.get("metadata") or {}
            if match.get("id"):
                md["id"] = match.get("id")
            md["score"] = match.get("score")
            matches.append(md)
            mem_id = match.get("id")
            if mem_id:
                try:
                    updated = {**md, "last_accessed_ts": now_ts}
                    updated.pop("score", None)
                    raw = updated.get("raw_content", "")
                    if raw:
                        self._index.upsert(
                            vectors=[
                                {
                                    "id": mem_id,
                                    "values": match["values"] if "values" in match else vector,
                                    "metadata": updated,
                                }
                            ]
                        )
                except Exception:
                    pass
        return matches

    def soft_archive_low_priority(
        self,
        *,
        priority_threshold: float = 0.15,
        inactive_days: int = 90,
        k: int = 200,
    ) -> int:
        """
        Soft-archive low-priority memories that haven't been accessed recently.
        Returns number of memories archived (best-effort).
        """
        cutoff_ts = utc_now_ts() - (inactive_days * 86400)
        matches = self.query_by_filter(
            query_text="archive old low priority memories",
            filter_obj={
                "priority_score": {"$lte": priority_threshold},
                "last_accessed_ts": {"$lte": cutoff_ts},
                "source_type": {"$nin": ["reminder", "diary_entry", "tunnel", "profile_snapshot"]},
                "archived": {"$ne": True},
            },
            k=k,
        )

        archived = 0
        now_ts = utc_now_ts()
        for md in matches:
            mem_id = str(md.get("id") or "").strip()
            if not mem_id:
                continue
            try:
                self.update_memory_metadata(
                    mem_id,
                    {"archived": True, "archived_at_ts": now_ts},
                )
                archived += 1
            except Exception:
                continue
        return archived

    def get_old_memories(self, older_than_ts: int, exclude_source_types: List[str] | None = None, k: int = 100) -> List[Dict[str, Any]]:
        """Return memories created before older_than_ts, optionally excluding certain source types."""
        filter_obj: Dict[str, Any] = {"created_at_ts": {"$lte": older_than_ts}}
        if exclude_source_types:
            filter_obj["source_type"] = {"$nin": exclude_source_types}
        return self.query_by_filter("memory knowledge idea note", filter_obj, k=k)

    def get_old_memories_for_user(
        self,
        *,
        user_id: int,
        older_than_ts: int,
        exclude_source_types: List[str] | None = None,
        k: int = 100,
    ) -> List[Dict[str, Any]]:
        """Per-user variant of get_old_memories for scheduler resurfacing."""
        filter_obj: Dict[str, Any] = {"created_at_ts": {"$lte": older_than_ts}, "archived": {"$ne": True}}
        if exclude_source_types:
            filter_obj["source_type"] = {"$nin": exclude_source_types}
        items = self.query_by_filter_for_chat("memory knowledge idea note", chat_id=user_id, filter_obj=filter_obj, k=k)
        return [m for m in items if self.is_main_memory(m)]

    def update_memory_metadata(self, memory_id: str, updates: Dict[str, Any]) -> None:
        """
        Update specific metadata fields on an existing memory without re-embedding.
        Uses Pinecone's update() which accepts set_metadata for partial updates.
        """
        self._index.update(id=memory_id, set_metadata=updates)

    def fetch_all_memories(self, exclude_source_types: List[str] | None = None, k: int = 200) -> List[Dict[str, Any]]:
        """Fetch a broad sample of memories for clustering / profile generation."""
        filter_obj: Dict[str, Any] = {}
        if exclude_source_types:
            filter_obj["source_type"] = {"$nin": exclude_source_types}
        return self.query_by_filter("knowledge ideas notes memories thoughts", filter_obj, k=k)

    def delete_memory(self, memory_id: str) -> None:
        self._index.delete(ids=[memory_id])

    def get_memory_by_id(self, memory_id: str) -> Dict[str, Any] | None:
        """Fetch a single memory by Pinecone vector id (metadata only)."""
        try:
            res = self._index.fetch(ids=[memory_id])
            vectors = res.get("vectors") or {}
            vec = vectors.get(memory_id) or {}
            md = vec.get("metadata") or None
            if md and isinstance(md, dict):
                md["id"] = memory_id
                return md
        except Exception:
            return None
        return None

    def get_latest_memory_for_chat(self, chat_id: int) -> Dict[str, Any] | None:
        """
        Best-effort helper used for 'delete last'.
        Fetches a slice of this chat's memories and returns the most recent by created_at_ts.
        """
        items = self.query_by_filter_for_chat(
            query_text="latest memory for this chat",
            chat_id=chat_id,
            filter_obj={"archived": {"$ne": True}},
            k=50,
        )
        if not items:
            return None
        items.sort(key=lambda m: int(m.get("created_at_ts") or 0), reverse=True)
        return items[0]
