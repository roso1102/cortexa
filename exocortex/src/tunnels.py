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
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from typing import Any, Dict, List

from groq import Groq
from openai import OpenAI

from src.memory import MemoryManager
from src.db import (
    fetch_main_memories_for_user_for_tunnels,
    insert_tunnel_and_members,
    insert_tunnel_edges,
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
_MAX_EDGE_NODES = 12           # cap pairwise comparisons to keep latency bounded
_MAX_EDGES_PER_TUNNEL = 18
_MAX_DEGREE_PER_NODE = 4       # avoid hub-and-spoke explosion in graph UI
_MIN_EDGE_CONFIDENCE = 0.08    # drop weak links

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


def _content_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) >= 4 and t not in _TOKEN_STOP}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return float(len(a & b)) / float(max(1, len(a | b)))


def _hybrid_pair_score(a: Dict[str, Any], b: Dict[str, Any]) -> tuple[float, float, str]:
    """
    Hybrid pair scoring:
      - content similarity proxy (semantic-ish)
      - lexical diversity boost (non-obvious bridges)
      - source type balancing
    Returns: (weight, bridge_score, rationale_seed_terms)
    """
    a_raw = str(a.get("raw_content") or "")
    b_raw = str(b.get("raw_content") or "")
    a_ctok = _content_tokens(a_raw)
    b_ctok = _content_tokens(b_raw)

    a_ttok = _cluster_tokens_from_tags(a.get("tags") or [])
    b_ttok = _cluster_tokens_from_tags(b.get("tags") or [])

    content_sim = _jaccard(a_ctok, b_ctok)
    tag_sim = _jaccard(a_ttok, b_ttok)
    base_weight = (0.7 * content_sim) + (0.3 * tag_sim)

    st_a = str(a.get("source_type") or "").strip().lower()
    st_b = str(b.get("source_type") or "").strip().lower()
    source_bonus = 0.12 if st_a and st_b and st_a != st_b else 0.0
    source_penalty = 0.05 if st_a and st_b and st_a == st_b else 0.0

    # Encourage bridges that are semantically close but lexically not too obvious.
    lexical_overlap = _jaccard(a_ctok, b_ctok)
    lexical_diversity_boost = max(0.0, 0.16 - lexical_overlap)
    bridge_score = max(0.0, base_weight + source_bonus + lexical_diversity_boost - source_penalty)

    bridge_terms = ", ".join(sorted(list((a_ctok & b_ctok) or (a_ttok & b_ttok)))[:3])
    return base_weight, bridge_score, bridge_terms


def _build_tunnel_edges(fallback_tag: str, mems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build semantic-looking memory-to-memory links with rationale.
    This is a first-pass heuristic scaffold; later phases can replace scoring with embeddings.
    """
    if len(mems) < 2:
        return []

    candidates: List[Dict[str, Any]] = []
    working = mems[:_MAX_EDGE_NODES]
    for a, b in combinations(working, 2):
        a_id = str(a.get("id") or "").strip()
        b_id = str(b.get("id") or "").strip()
        if not a_id or not b_id:
            continue

        weight, bridge_score, bridge_terms = _hybrid_pair_score(a, b)
        if bridge_score < _MIN_EDGE_CONFIDENCE:
            continue

        same_source = str(a.get("source_type") or "") == str(b.get("source_type") or "")
        rationale = (
            f"Linked through shared concepts ({bridge_terms or fallback_tag}) with complementary perspectives in '{fallback_tag}'."
            if not same_source
            else f"Linked through overlapping concepts ({bridge_terms or fallback_tag}) within the '{fallback_tag}' theme."
        )
        candidates.append(
            {
                "from_memory_id": a_id,
                "to_memory_id": b_id,
                "weight": round(weight, 4),
                "bridge_score": round(bridge_score, 4),
                "rationale": rationale,
            }
        )

    candidates.sort(key=lambda x: (x.get("bridge_score") or 0.0, x.get("weight") or 0.0), reverse=True)
    selected: List[Dict[str, Any]] = []
    degree: Dict[str, int] = defaultdict(int)
    for e in candidates:
        a_id = str(e.get("from_memory_id") or "")
        b_id = str(e.get("to_memory_id") or "")
        if degree[a_id] >= _MAX_DEGREE_PER_NODE or degree[b_id] >= _MAX_DEGREE_PER_NODE:
            continue
        selected.append(e)
        degree[a_id] += 1
        degree[b_id] += 1
        if len(selected) >= _MAX_EDGES_PER_TUNNEL:
            break
    return selected


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

        # Persist graph edges (memory-to-memory links + rationale).
        tunnel_edges = _build_tunnel_edges(tag, unique_mems)
        if tunnel_edges:
            try:
                insert_tunnel_edges(tunnel_id=tunnel_id, user_id=user_id, edges=tunnel_edges)
            except Exception:
                logger.exception("Failed to persist tunnel edges for tunnel_id=%s", tunnel_id)

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
            "edge_count": len(tunnel_edges),
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
        choices = getattr(resp, "choices", None) or []
        first = choices[0] if choices else None
        message = getattr(first, "message", None) if first else None
        raw = str(getattr(message, "content", "") or "").strip()
        obj = json.loads(raw) if raw else {}
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
