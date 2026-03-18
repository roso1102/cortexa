from __future__ import annotations

import os
from typing import Any, Dict, List

import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec

from src.config import AppConfig
from src.utils import utc_now_iso, utc_now_ts


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
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text,
        )
        embedding = result["embedding"]  # type: ignore[assignment]
        # Guardrail: Pinecone index dimension must match embedding length
        if len(embedding) != 3072:
            raise ValueError(f"Unexpected embedding size: {len(embedding)} (expected 3072)")
        return embedding  # type: ignore[return-value]

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
