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

# Simple in-process LRU-style embedding cache.
# Stores up to _EMBED_CACHE_SIZE distinct texts. Avoids re-embedding the same
# repeated query strings (e.g. "reminder" called every 10 s by the scheduler).
_EMBED_CACHE_SIZE = 128
_embed_cache: Dict[str, List[float]] = {}
_embed_cache_order: List[str] = []

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

        return embedding

    def add_memory(self, text: str, metadata: Dict[str, Any]) -> str:
        vector = self._embed(text)
        memory_id = metadata.get("id") or utc_now_iso()

        md = {
            **metadata,
            "raw_content": text,
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
        return memory_id

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
