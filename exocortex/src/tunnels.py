"""
src/tunnels.py

Tunnel Formation — weekly semantic grouping of memories.

A "Tunnel" is a named semantic pathway: a cluster of related memories
that the system has identified as a recurring theme in your thinking.

Strategy:
  Rather than raw k-means on embedding vectors (which would require
  fetching all vectors from Pinecone — expensive on free tier), we
  cluster by **tag token overlap**:

  1. Fetch main memories (Postgres first; Pinecone fallback).
  2. From each memory's tags, extract word tokens (e.g. "love poem" → love, poem).
     Memories sharing a token (e.g. "poem") group together even if full tag strings differ.
  3. Tokens that appear on too few memories are skipped; top clusters capped at _MAX_TUNNELS.
  4. For each cluster, sample snippets and ask OpenRouter to name the tunnel.
  5. Persist tunnel + stamp member memories in Postgres / Pinecone metadata.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List

from groq import Groq
from openai import OpenAI

from src.memory import MemoryManager
from src.db import (
    fetch_main_memories_for_user_for_tunnels,
    insert_tunnel_and_members,
    update_memory_tunnel_fields,
)
from src.utils import utc_now_iso, utc_now_ts

logger = logging.getLogger(__name__)

_TUNNEL_NAMER_PROMPT = """\
You are Exocortex. Below are 3-5 short memory snippets that share a common theme.

Return ONLY valid JSON with keys:
- name: short human-readable tunnel name (3-6 words, title-case)
- reason: 1-2 sentence explanation of why these connect

Snippets:
{snippets}
"""

_MIN_MEMORIES_FOR_TUNNEL = 3  # skip clusters with too few memories
_MAX_TUNNELS = 8               # avoid tunnel explosion
_SNIPPET_CHARS = 200           # max chars per snippet sent to LLM

# Drop generic tokens so we do not form giant "the" / "note" tunnels.
_TOKEN_STOP = frozenset(
    {
        "the",
        "and",
        "for",
        "you",
        "are",
        "not",
        "but",
        "with",
        "that",
        "this",
        "from",
        "your",
        "have",
        "has",
        "was",
        "were",
        "can",
        "our",
        "their",
        "what",
        "when",
        "where",
        "which",
        "how",
        "why",
        "who",
        "its",
        "also",
        "just",
        "like",
        "one",
        "all",
        "any",
        "get",
        "got",
        "use",
        "using",
        "used",
        "new",
        "way",
        "may",
        "out",
        "about",
        "into",
        "over",
        "than",
        "then",
        "some",
        "such",
        "here",
        "there",
        "each",
        "both",
        "few",
        "other",
        "own",
        "same",
        "note",
        "notes",
        "idea",
        "ideas",
        "link",
        "links",
        "saved",
        "save",
        "text",
        "memory",
        "memories",
        "article",
        "blog",
        "post",
        "page",
        "http",
        "https",
        "com",
        "org",
        "www",
    }
)


def _cluster_tokens_from_tags(tags: Any) -> set[str]:
    """Word tokens (len>=3) from all tags on a memory; used to merge e.g. 'poem' + 'love poem'."""
    out: set[str] = set()
    if not tags:
        return out
    seq = tags if isinstance(tags, list) else [tags]
    for tag in seq:
        for tok in re.findall(r"[a-z0-9]+", str(tag).lower()):
            if len(tok) >= 3 and tok not in _TOKEN_STOP:
                out.add(tok)
    return out


def form_tunnels(memory: MemoryManager, groq: Groq, *, user_id: int, openrouter_api_key: str) -> List[Dict[str, Any]]:
    """
    Main entry point called by the scheduler weekly.

    Returns a list of tunnel dicts that were created (or updated).
    """
    logger.info("Starting tunnel formation")
    # Prefer canonical Postgres memories (prevents chunk/tunnel leakage and improves accuracy).
    try:
        memories = fetch_main_memories_for_user_for_tunnels(
            user_id=user_id,
            exclude_source_types=["reminder", "diary_entry", "tunnel", "profile_snapshot"],
            limit=400,
        )
        # Canonical fetch returns only main records; no need memory.is_main_memory here.
    except Exception:
        logger.exception("Failed to fetch canonical memories for tunnel formation; falling back to Pinecone.")
        try:
            memories = memory.query_by_filter_for_chat(
                query_text="knowledge ideas notes memories thoughts",
                chat_id=user_id,
                filter_obj={"source_type": {"$nin": ["reminder", "diary_entry", "tunnel", "profile_snapshot"]}},
                k=400,
            )
            memories = [m for m in memories if memory.is_main_memory(m)]
        except Exception:
            logger.exception("Failed to fetch memories for tunnel formation (Pinecone fallback)")
            return []

    if not memories:
        logger.info("No memories available for tunnel formation")
        return []

    # --- Build token → memories (dedupe by memory id per token) ---
    token_to_ids: dict[str, set[str]] = defaultdict(set)
    id_to_memory: Dict[str, Dict[str, Any]] = {}
    untagged: List[Dict[str, Any]] = []

    for m in memories:
        mid = str(m.get("id") or "").strip()
        if not mid:
            continue
        id_to_memory[mid] = m
        tags = m.get("tags") or []
        tokens = _cluster_tokens_from_tags(tags)
        if not tokens:
            untagged.append(m)
            continue
        for tok in tokens:
            token_to_ids[tok].add(mid)

    qualified_tags: Dict[str, List[Dict[str, Any]]] = {}
    for tok, ids in token_to_ids.items():
        if len(ids) < _MIN_MEMORIES_FOR_TUNNEL:
            continue
        qualified_tags[tok] = [id_to_memory[i] for i in ids if i in id_to_memory]

    if not qualified_tags:
        logger.info(
            "Not enough overlapping tag tokens to form tunnels (need >= %d memories sharing one token)",
            _MIN_MEMORIES_FOR_TUNNEL,
        )
        return []

    # --- Sort by cluster size, take top _MAX_TUNNELS ---
    sorted_tags = sorted(qualified_tags.items(), key=lambda x: len(x[1]), reverse=True)[:_MAX_TUNNELS]

    created_tunnels: List[Dict[str, Any]] = []
    now_ts = utc_now_ts()
    now_iso = utc_now_iso()

    for tag, tag_memories in sorted_tags:
        safe_slug = re.sub(r"[^a-z0-9_]+", "_", tag.strip().lower()).strip("_")[:48] or "theme"
        tunnel_id = f"tunnel_{safe_slug}_{now_ts}"

        # Deduplicate memories by raw_content fingerprint
        seen: set = set()
        unique_mems: List[Dict[str, Any]] = []
        for m in tag_memories:
            key = (m.get("raw_content") or "")[:50]
            if key not in seen:
                seen.add(key)
                unique_mems.append(m)

        # Build representative snippets for naming
        snippets_for_llm = []
        for m in unique_mems[:5]:
            raw = str(m.get("raw_content") or "").strip()
            snippets_for_llm.append(raw[:_SNIPPET_CHARS])

        # Ask OpenRouter to name + explain the tunnel (better reasoning)
        tunnel_name, tunnel_reason = _name_tunnel_openrouter(openrouter_api_key, tag, snippets_for_llm)

        # Persist tunnel + membership in Postgres (canonical).
        try:
            insert_tunnel_and_members(
                tunnel_id=tunnel_id,
                user_id=user_id,
                name=tunnel_name,
                reason=tunnel_reason,
                core_tag=tag,
                memory_count=len(unique_mems),
                created_at_ts=now_ts,
                raw="\n".join(f"- {s[:100]}" for s in snippets_for_llm[:3]),
                member_memory_ids=[str(m.get("id") or "") for m in unique_mems if m.get("id")],
            )
        except Exception:
            logger.exception("Failed to persist tunnel in Postgres: %s", tunnel_name)
            # Keep best-effort stamping on existing memory metadata.

        # Stamp constituent memories with tunnel_id/tunnel_name in both stores.
        stamped = 0
        for m in unique_mems:
            mem_id = str(m.get("id") or "").strip()
            if not mem_id:
                continue
            try:
                # Pinecone metadata stamp for downstream profile/UX that still reads Pinecone.
                memory.update_memory_metadata(mem_id, {"tunnel_id": tunnel_id, "tunnel_name": tunnel_name})
                # Canonical stamp for Option B correctness.
                update_memory_tunnel_fields(
                    user_id=user_id,
                    memory_id=mem_id,
                    tunnel_id=tunnel_id,
                    tunnel_name=tunnel_name,
                )
                stamped += 1
            except Exception:
                pass  # best-effort

        tunnel_metadata: Dict[str, Any] = {
            "id": tunnel_id,
            "source_type": "tunnel",
            "tunnel_name": tunnel_name,
            "reason": tunnel_reason,
            "core_tag": tag,
            "memory_count": len(unique_mems),
            "created_at": now_iso,
            "created_at_ts": now_ts,
            "tags": ["tunnel", tag],
            "user_id": user_id,
            "stamped_count": stamped,
        }
        logger.info("Stamped %d/%d memories with tunnel_id=%s", stamped, len(unique_mems), tunnel_id)
        created_tunnels.append(tunnel_metadata)

    return created_tunnels


def _name_tunnel_openrouter(openrouter_api_key: str, fallback_tag: str, snippets: List[str]) -> tuple[str, str]:
    """Ask OpenRouter to produce a tunnel name + reason from snippets."""
    if not snippets:
        return fallback_tag.title(), "Shared theme inferred from tags."

    snippet_block = "\n---\n".join(snippets)
    prompt = _TUNNEL_NAMER_PROMPT.format(snippets=snippet_block)

    client = OpenAI(api_key=openrouter_api_key, base_url="https://openrouter.ai/api/v1")
    try:
        resp = client.chat.completions.create(
            model="minimax/minimax-01",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=220,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        import json as _json

        obj = _json.loads(raw) if raw else {}
        name = str(obj.get("name") or "").strip()
        reason = str(obj.get("reason") or "").strip()
        if not name:
            name = fallback_tag.title()
        if not reason:
            reason = "Shared theme inferred from memory snippets."
        return name, reason
    except Exception:
        logger.exception("OpenRouter tunnel naming failed, using tag as name")
        return fallback_tag.title(), "Shared theme inferred from memory snippets."
