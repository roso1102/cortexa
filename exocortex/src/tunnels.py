"""
src/tunnels.py

Tunnel Formation — weekly semantic grouping of memories.

A "Tunnel" is a named semantic pathway: a cluster of related memories
that the system has identified as a recurring theme in your thinking.

Strategy:
  Rather than raw k-means on embedding vectors (which would require
  fetching all vectors from Pinecone — expensive on free tier), we
  cluster by tag affinity:

  1. Fetch all non-system memories from Pinecone (with metadata).
  2. Build a tag → [memory_ids] inverted index.
  3. Merge closely related tags into groups using a co-occurrence heuristic.
  4. For each group, sample 3 representative snippet and ask Groq to name
     the tunnel (e.g. "FPGA hardware design", "health & nutrition", …).
  5. Store each tunnel as source_type=tunnel in Pinecone.
  6. Stamp each constituent memory with tunnel_id (metadata update).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List

from groq import Groq

from src.memory import MemoryManager
from src.utils import utc_now_iso, utc_now_ts

logger = logging.getLogger(__name__)

_TUNNEL_NAMER_PROMPT = """\
You are Exocortex. Below are 3-5 short memory snippets that share a common theme.
Give this cluster a short, human-readable tunnel name (3-6 words, title-case).
Respond with ONLY the tunnel name and nothing else.

Snippets:
{snippets}
"""

_MIN_MEMORIES_FOR_TUNNEL = 3  # skip tags with too few memories
_MAX_TUNNELS = 8               # avoid tunnel explosion
_SNIPPET_CHARS = 200           # max chars per snippet sent to LLM


def form_tunnels(memory: MemoryManager, groq: Groq) -> List[Dict[str, Any]]:
    """
    Main entry point called by the scheduler weekly.

    Returns a list of tunnel dicts that were created (or updated).
    """
    logger.info("Starting tunnel formation")
    try:
        memories = memory.fetch_all_memories(
            exclude_source_types=["reminder", "diary_entry", "tunnel", "profile_snapshot"],
            k=300,
        )
    except Exception:
        logger.exception("Failed to fetch memories for tunnel formation")
        return []

    if not memories:
        logger.info("No memories available for tunnel formation")
        return []

    # --- Build tag → memories index ---
    tag_index: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    untagged: List[Dict[str, Any]] = []

    for m in memories:
        tags = m.get("tags") or []
        if not tags:
            untagged.append(m)
            continue
        for tag in tags:
            tag_index[str(tag).strip().lower()].append(m)

    # --- Filter tags with enough memories ---
    qualified_tags = {
        tag: mems
        for tag, mems in tag_index.items()
        if len(mems) >= _MIN_MEMORIES_FOR_TUNNEL
    }

    if not qualified_tags:
        logger.info("Not enough tagged memories to form tunnels (need >= %d per tag)", _MIN_MEMORIES_FOR_TUNNEL)
        return []

    # --- Sort by cluster size, take top _MAX_TUNNELS ---
    sorted_tags = sorted(qualified_tags.items(), key=lambda x: len(x[1]), reverse=True)[:_MAX_TUNNELS]

    created_tunnels: List[Dict[str, Any]] = []
    now_ts = utc_now_ts()
    now_iso = utc_now_iso()

    for tag, tag_memories in sorted_tags:
        tunnel_id = f"tunnel_{tag.replace(' ', '_')}_{now_ts}"

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

        # Ask Groq to name the tunnel
        tunnel_name = _name_tunnel(groq, tag, snippets_for_llm)

        # Store tunnel object in Pinecone
        tunnel_text = (
            f"Tunnel: {tunnel_name}\n"
            f"Core tag: {tag}\n"
            f"Memories: {len(unique_mems)}\n\n"
            + "\n".join(f"- {s[:100]}" for s in snippets_for_llm[:3])
        )
        tunnel_metadata: Dict[str, Any] = {
            "source_type": "tunnel",
            "tunnel_name": tunnel_name,
            "core_tag": tag,
            "memory_count": len(unique_mems),
            "created_at": now_iso,
            "created_at_ts": now_ts,
            "tags": ["tunnel", tag],
        }
        try:
            memory.add_memory(tunnel_text, {**tunnel_metadata, "id": tunnel_id})
            logger.info("Created tunnel: %s (tag=%s, memories=%d)", tunnel_name, tag, len(unique_mems))
        except Exception:
            logger.exception("Failed to store tunnel: %s", tunnel_name)
            continue

        # Stamp constituent memories with tunnel_id (best-effort)
        stamped = 0
        for m in unique_mems:
            mem_id = m.get("id") or m.get("created_at", "")
            if not mem_id:
                continue
            try:
                memory.update_memory_metadata(mem_id, {"tunnel_id": tunnel_id, "tunnel_name": tunnel_name})
                stamped += 1
            except Exception:
                pass  # best-effort

        logger.info("Stamped %d/%d memories with tunnel_id=%s", stamped, len(unique_mems), tunnel_id)
        created_tunnels.append({**tunnel_metadata, "id": tunnel_id, "stamped_count": stamped})

    return created_tunnels


def _name_tunnel(groq: Groq, fallback_tag: str, snippets: List[str]) -> str:
    """Ask Groq LLM to produce a human-readable tunnel name from snippets."""
    if not snippets:
        return fallback_tag.title()

    snippet_block = "\n---\n".join(snippets)
    prompt = _TUNNEL_NAMER_PROMPT.format(snippets=snippet_block)

    try:
        resp = groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=30,
            temperature=0.3,
        )
        name = (resp.choices[0].message.content or "").strip().strip('"').strip("'")
        return name if name else fallback_tag.title()
    except Exception:
        logger.exception("Groq tunnel naming failed, using tag as name")
        return fallback_tag.title()
