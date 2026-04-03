"""
src/tunnels.py

Tunnel Formation — weekly semantic grouping of memories.

Primary strategy (embedding-first):
  1. Fetch main memories (Postgres first; Pinecone fallback).
  2. Dedupe, embed a capped subset with Gemini (same pipeline as MemoryManager).
  3. Greedy cosine-similarity clustering; each tunnel gets a semantic cluster.
  4. Fallback: tag-token overlap clustering (legacy) if embeddings are insufficient.

Tunnels are persisted to Postgres with memory-to-memory edges in tunnel_edges.
"""
from __future__ import annotations

import logging
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from typing import Any, Dict, List, Optional

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


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _tunnel_min_memories() -> int:
    """Minimum memories sharing a tag token required to form one tunnel cluster."""
    v = _env_int("TUNNEL_MIN_MEMORIES", 4)
    return max(3, min(v, 100))


def _tunnel_max_memories_per_tunnel() -> int:
    """Cap memories assigned to a single tunnel (newest first) to keep graphs readable."""
    v = _env_int("TUNNEL_MAX_MEMORIES_PER_TUNNEL", 20)
    return max(5, min(v, 400))


def _tunnel_embed_max_memories() -> int:
    v = _env_int("TUNNEL_EMBED_MAX_MEMORIES", 120)
    return max(20, min(v, 400))


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _tunnel_cluster_min_cosine() -> float:
    v = _env_float("TUNNEL_CLUSTER_MIN_COSINE", 0.72)
    return max(0.55, min(0.95, v))


_SEMANTIC_CORE_TAG = "semantic"


_TUNNEL_NAMER_PROMPT = """\
You are Exocortex. Below are short memory snippets that belong to one latent theme in the user's mind.

The connection may be non-obvious: infer a deeper thread (values, questions, projects, metaphors),
not just a repeated keyword.

Return ONLY valid JSON with keys:
- name: short human-readable tunnel name (3-6 words, title-case)
- reason: 1-2 sentences explaining why these specific snippets belong together

Snippets:
{snippets}
"""

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


def _cosine_vec(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(dot / (na * nb))


def _pair_similarity_scores(
    a: Dict[str, Any],
    b: Dict[str, Any],
    vec_by_id: Optional[Dict[str, List[float]]],
) -> tuple[float, float, str]:
    """Returns (weight, bridge_score, rationale_seed_terms). Uses embeddings when both vectors exist."""
    a_id = str(a.get("id") or "").strip()
    b_id = str(b.get("id") or "").strip()
    va = vec_by_id.get(a_id) if vec_by_id and a_id else None
    vb = vec_by_id.get(b_id) if vec_by_id and b_id else None

    a_raw = str(a.get("raw_content") or "")
    b_raw = str(b.get("raw_content") or "")
    a_ctok = _content_tokens(a_raw)
    b_ctok = _content_tokens(b_raw)
    a_ttok = _cluster_tokens_from_tags(a.get("tags") or [])
    b_ttok = _cluster_tokens_from_tags(b.get("tags") or [])
    content_j = _jaccard(a_ctok, b_ctok)
    tag_j = _jaccard(a_ttok, b_ttok)

    if va is not None and vb is not None:
        cos_e = _cosine_vec(va, vb)
        base_weight = 0.78 * cos_e + 0.22 * content_j
    else:
        base_weight = 0.7 * content_j + 0.3 * tag_j

    st_a = str(a.get("source_type") or "").strip().lower()
    st_b = str(b.get("source_type") or "").strip().lower()
    source_bonus = 0.12 if st_a and st_b and st_a != st_b else 0.0
    source_penalty = 0.05 if st_a and st_b and st_a == st_b else 0.0
    lexical_diversity_boost = max(0.0, 0.16 - content_j)
    bridge_score = max(0.0, base_weight + source_bonus + lexical_diversity_boost - source_penalty)
    bridge_terms = ", ".join(sorted(list((a_ctok & b_ctok) or (a_ttok & b_ttok)))[:3])
    return base_weight, bridge_score, bridge_terms


def _dedupe_memories_pool(memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for m in memories:
        key = str(m.get("raw_content") or "")[:50]
        if key in seen:
            continue
        seen.add(key)
        out.append(m)
    out.sort(key=lambda x: int(x.get("created_at_ts") or 0), reverse=True)
    return out


def _embed_memories_map(memory: MemoryManager, mems: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for m in mems:
        mid = str(m.get("id") or "").strip()
        if not mid:
            continue
        raw = str(m.get("raw_content") or "").strip()
        if not raw:
            continue
        try:
            out[mid] = memory.embed_for_tunnel(raw)
        except Exception as exc:
            logger.debug("tunnel embed skip memory_id=%s: %s", mid, exc)
    return out


def _greedy_embedding_clusters(
    sorted_pool_newest_first: List[Dict[str, Any]],
    vec_by_id: Dict[str, List[float]],
    *,
    min_mem: int,
    max_mem: int,
    max_tunnels: int,
    min_cos: float,
) -> List[List[Dict[str, Any]]]:
    """Each memory appears in at most one cluster. Iteration order prefers newer seeds."""
    clusters: List[List[Dict[str, Any]]] = []
    assigned: set[str] = set()
    for seed in sorted_pool_newest_first:
        sid = str(seed.get("id") or "").strip()
        if not sid or sid in assigned:
            continue
        v_seed = vec_by_id.get(sid)
        if not v_seed:
            continue
        scored: List[tuple[float, Dict[str, Any]]] = []
        for other in sorted_pool_newest_first:
            oid = str(other.get("id") or "").strip()
            if not oid or oid in assigned or oid == sid:
                continue
            vo = vec_by_id.get(oid)
            if not vo:
                continue
            sim = _cosine_vec(v_seed, vo)
            if sim >= min_cos:
                scored.append((sim, other))
        scored.sort(key=lambda x: -x[0])
        cluster: List[Dict[str, Any]] = [seed]
        for sim, other in scored:
            if len(cluster) >= max_mem:
                break
            oid = str(other.get("id") or "").strip()
            if oid in assigned:
                continue
            cluster.append(other)
        if len(cluster) < min_mem:
            continue
        for m in cluster:
            mid = str(m.get("id") or "").strip()
            if mid:
                assigned.add(mid)
        clusters.append(cluster)
        if len(clusters) >= max_tunnels:
            break
    return clusters


def _build_tunnel_edges(
    fallback_tag: str,
    mems: List[Dict[str, Any]],
    vec_by_id: Optional[Dict[str, List[float]]] = None,
) -> List[Dict[str, Any]]:
    """Memory-to-memory links with rationale; uses embeddings in scoring when available."""
    if len(mems) < 2:
        return []

    candidates: List[Dict[str, Any]] = []
    working = mems[:_MAX_EDGE_NODES]
    for a, b in combinations(working, 2):
        a_id = str(a.get("id") or "").strip()
        b_id = str(b.get("id") or "").strip()
        if not a_id or not b_id:
            continue

        weight, bridge_score, bridge_terms = _pair_similarity_scores(a, b, vec_by_id)
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


def _create_single_tunnel(
    memory: MemoryManager,
    *,
    user_id: int,
    openrouter_api_key: str,
    tunnel_id: str,
    unique_mems: List[Dict[str, Any]],
    core_tag: str,
    now_ts: int,
    now_iso: str,
    vec_by_id: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, Any]:
    snippets_for_llm = []
    for m in unique_mems[:5]:
        raw = str(m.get("raw_content") or "").strip()
        snippets_for_llm.append(raw[:_SNIPPET_CHARS])

    tunnel_name, tunnel_reason = _name_tunnel_openrouter(openrouter_api_key, core_tag, snippets_for_llm)

    try:
        insert_tunnel_and_members(
            tunnel_id=tunnel_id,
            user_id=user_id,
            name=tunnel_name,
            reason=tunnel_reason,
            core_tag=core_tag,
            memory_count=len(unique_mems),
            created_at_ts=now_ts,
            raw="\n".join(f"- {s[:100]}" for s in snippets_for_llm[:3]),
            member_memory_ids=[str(m.get("id") or "") for m in unique_mems if m.get("id")],
        )
    except Exception:
        logger.exception("Failed to persist tunnel in Postgres: %s", tunnel_name)

    tunnel_edge_rows = _build_tunnel_edges(core_tag, unique_mems, vec_by_id)
    if tunnel_edge_rows:
        try:
            insert_tunnel_edges(tunnel_id=tunnel_id, user_id=user_id, edges=tunnel_edge_rows)
        except Exception:
            logger.exception("Failed to persist tunnel edges for tunnel_id=%s", tunnel_id)

    stamped = 0
    for m in unique_mems:
        mem_id = str(m.get("id") or "").strip()
        if not mem_id:
            continue
        try:
            memory.update_memory_metadata(mem_id, {"tunnel_id": tunnel_id, "tunnel_name": tunnel_name})
            update_memory_tunnel_fields(
                user_id=user_id,
                memory_id=mem_id,
                tunnel_id=tunnel_id,
                tunnel_name=tunnel_name,
            )
            stamped += 1
        except Exception:
            pass

    logger.info("Stamped %d/%d memories with tunnel_id=%s", stamped, len(unique_mems), tunnel_id)
    return {
        "id": tunnel_id,
        "source_type": "tunnel",
        "tunnel_name": tunnel_name,
        "reason": tunnel_reason,
        "core_tag": core_tag,
        "memory_count": len(unique_mems),
        "created_at": now_iso,
        "created_at_ts": now_ts,
        "tags": ["tunnel", core_tag],
        "user_id": user_id,
        "stamped_count": stamped,
        "edge_count": len(tunnel_edge_rows),
    }


def _form_tunnels_tag_fallback(
    memory: MemoryManager,
    _groq: Groq,
    *,
    user_id: int,
    openrouter_api_key: str,
    memories: List[Dict[str, Any]],
    min_mem: int,
    max_mem: int,
    vec_by_id: Optional[Dict[str, List[float]]],
) -> List[Dict[str, Any]]:
    token_to_ids: dict[str, set[str]] = defaultdict(set)
    id_to_memory: Dict[str, Dict[str, Any]] = {}

    for m in memories:
        mid = str(m.get("id") or "").strip()
        if not mid:
            continue
        id_to_memory[mid] = m
        tokens = _cluster_tokens_from_tags(m.get("tags") or [])
        if not tokens:
            continue
        for tok in tokens:
            token_to_ids[tok].add(mid)

    qualified_tags: Dict[str, List[Dict[str, Any]]] = {}
    for tok, ids in token_to_ids.items():
        if len(ids) < min_mem:
            continue
        qualified_tags[tok] = [id_to_memory[i] for i in ids if i in id_to_memory]

    if not qualified_tags:
        logger.info(
            "Tag-token tunnel fallback: need >= %d memories sharing one token; none qualified",
            min_mem,
        )
        return []

    sorted_tags = sorted(qualified_tags.items(), key=lambda x: len(x[1]), reverse=True)[:_MAX_TUNNELS]
    created_tunnels: List[Dict[str, Any]] = []
    now_ts = utc_now_ts()
    now_iso = utc_now_iso()

    for tag, tag_memories in sorted_tags:
        safe_slug = re.sub(r"[^a-z0-9_]+", "_", tag.strip().lower()).strip("_")[:48] or "theme"
        tunnel_id = f"tunnel_{safe_slug}_{now_ts}"

        seen: set[str] = set()
        unique_mems: List[Dict[str, Any]] = []
        for m in tag_memories:
            key = str(m.get("raw_content") or "")[:50]
            if key in seen:
                continue
            seen.add(key)
            unique_mems.append(m)

        unique_mems.sort(key=lambda m: int(m.get("created_at_ts") or 0), reverse=True)
        unique_mems = unique_mems[:max_mem]

        meta = _create_single_tunnel(
            memory,
            user_id=user_id,
            openrouter_api_key=openrouter_api_key,
            tunnel_id=tunnel_id,
            unique_mems=unique_mems,
            core_tag=tag,
            now_ts=now_ts,
            now_iso=now_iso,
            vec_by_id=vec_by_id,
        )
        created_tunnels.append(meta)

    return created_tunnels


def form_tunnels(
    memory: MemoryManager, _groq: Groq, *, user_id: int, openrouter_api_key: str
) -> List[Dict[str, Any]]:
    """
    Main entry point called by the scheduler weekly.

    Returns a list of tunnel dicts that were created (or updated).
    """
    min_mem = _tunnel_min_memories()
    max_mem = _tunnel_max_memories_per_tunnel()
    embed_cap = _tunnel_embed_max_memories()
    min_cos = _tunnel_cluster_min_cosine()
    logger.info(
        "Starting tunnel formation (min_memories=%s max_per_tunnel=%s embed_cap=%s min_cos=%s)",
        min_mem,
        max_mem,
        embed_cap,
        min_cos,
    )

    try:
        memories = fetch_main_memories_for_user_for_tunnels(
            user_id=user_id,
            exclude_source_types=["reminder", "diary_entry", "tunnel", "profile_snapshot"],
            limit=400,
        )
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

    pool = _dedupe_memories_pool(memories)
    to_embed = pool[:embed_cap]
    vec_by_id = _embed_memories_map(memory, to_embed)
    logger.info("Tunnel embed map: %d vectors from %d candidates", len(vec_by_id), len(to_embed))

    now_ts = utc_now_ts()
    now_iso = utc_now_iso()

    if len(vec_by_id) >= min_mem:
        sorted_with_vecs = [m for m in pool if str(m.get("id") or "").strip() in vec_by_id]
        clusters = _greedy_embedding_clusters(
            sorted_with_vecs,
            vec_by_id,
            min_mem=min_mem,
            max_mem=max_mem,
            max_tunnels=_MAX_TUNNELS,
            min_cos=min_cos,
        )
        if clusters:
            logger.info("Embedding-first tunnels: %d cluster(s)", len(clusters))
            created: List[Dict[str, Any]] = []
            for idx, cluster in enumerate(clusters):
                tunnel_id = f"tunnel_semantic_{now_ts}_{idx}"
                meta = _create_single_tunnel(
                    memory,
                    user_id=user_id,
                    openrouter_api_key=openrouter_api_key,
                    tunnel_id=tunnel_id,
                    unique_mems=cluster,
                    core_tag=_SEMANTIC_CORE_TAG,
                    now_ts=now_ts,
                    now_iso=now_iso,
                    vec_by_id=vec_by_id,
                )
                created.append(meta)
            return created

    logger.info("Embedding clusters empty or insufficient vectors; falling back to tag-token clustering")
    return _form_tunnels_tag_fallback(
        memory,
        _groq,
        user_id=user_id,
        openrouter_api_key=openrouter_api_key,
        memories=memories,
        min_mem=min_mem,
        max_mem=max_mem,
        vec_by_id=vec_by_id if vec_by_id else None,
    )


def rebuild_tunnel_edges(
    memory: MemoryManager,
    *,
    user_id: int,
    tunnel_id: str,
    core_tag: str,
) -> tuple[int, str]:
    """
    Replace tunnel_edges for an existing tunnel from current members.
    Returns (edge_count, core_tag_used).
    """
    from src.db import delete_tunnel_edges_for_tunnel, fetch_tunnel_member_memories_for_edges, insert_tunnel_edges

    mems = fetch_tunnel_member_memories_for_edges(user_id=user_id, tunnel_id=tunnel_id)
    if len(mems) < 2:
        delete_tunnel_edges_for_tunnel(user_id=user_id, tunnel_id=tunnel_id)
        return 0, core_tag

    tag = (core_tag or "").strip() or _SEMANTIC_CORE_TAG
    to_embed = mems[: _tunnel_embed_max_memories()]
    vec_by_id = _embed_memories_map(memory, to_embed)
    edges = _build_tunnel_edges(tag, mems, vec_by_id if vec_by_id else None)
    delete_tunnel_edges_for_tunnel(user_id=user_id, tunnel_id=tunnel_id)
    if edges:
        insert_tunnel_edges(tunnel_id=tunnel_id, user_id=user_id, edges=edges)
    return len(edges), tag


def explain_tunnel_edge_openrouter(
    openrouter_api_key: str,
    *,
    memory_a: Dict[str, Any],
    memory_b: Dict[str, Any],
    excerpt_chars: int = 1800,
) -> tuple[str, List[Dict[str, str]], bool]:
    """
    LLM explanation with evidence quotes. Returns (summary, evidence_list, fallback_used).
    """
    id_a = str(memory_a.get("id") or memory_a.get("memory_id") or "")
    id_b = str(memory_b.get("id") or memory_b.get("memory_id") or "")
    text_a = str(memory_a.get("raw_content") or memory_a.get("raw_content_full") or "")
    text_b = str(memory_b.get("raw_content") or memory_b.get("raw_content_full") or "")
    ex_a = text_a[:excerpt_chars]
    ex_b = text_b[:excerpt_chars]

    prompt = f"""You compare two saved memories from the same semantic tunnel.

Return ONLY valid JSON with keys:
- summary: 2-4 sentences explaining why these two items meaningfully connect (causal, thematic, or methodological).
- evidence: array of 1-2 objects, each object MUST have "memory_id" (exactly one of: "{id_a}" or "{id_b}") and "quote" (a VERBATIM substring copied from that memory's excerpt below — copy character-for-character from the excerpt, no invention).

Memory A id={id_a}
EXCERPT_A:
{ex_a}

Memory B id={id_b}
EXCERPT_B:
{ex_b}
"""

    def _normalize_ws(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip())

    def _quotes_valid(
        evidence: List[Dict[str, Any]], full_a: str, full_b: str, eid_a: str, eid_b: str
    ) -> bool:
        if not evidence:
            return False
        norm_a = _normalize_ws(full_a)
        norm_b = _normalize_ws(full_b)
        ok_any = False
        for item in evidence:
            mid = str(item.get("memory_id") or "").strip()
            quote = str(item.get("quote") or "").strip()
            if len(quote) < 8 or mid not in (eid_a, eid_b):
                return False
            nq = _normalize_ws(quote)
            if mid == eid_a and (quote in full_a or (nq and nq in norm_a)):
                ok_any = True
            elif mid == eid_b and (quote in full_b or (nq and nq in norm_b)):
                ok_any = True
            else:
                return False
        return ok_any

    client = OpenAI(api_key=openrouter_api_key, base_url="https://openrouter.ai/api/v1")
    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model="minimax/minimax-01",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.25,
                response_format={"type": "json_object"},
            )
            choices = getattr(resp, "choices", None) or []
            first = choices[0] if choices else None
            message = getattr(first, "message", None) if first else None
            raw = str(getattr(message, "content", "") or "").strip()
            obj = json.loads(raw) if raw else {}
            summary = str(obj.get("summary") or "").strip()
            ev_raw = obj.get("evidence") or []
            evidence_out: List[Dict[str, str]] = []
            if isinstance(ev_raw, list):
                for item in ev_raw:
                    if not isinstance(item, dict):
                        continue
                    evidence_out.append(
                        {
                            "memory_id": str(item.get("memory_id") or "").strip(),
                            "quote": str(item.get("quote") or "").strip(),
                        }
                    )
            if summary and _quotes_valid(evidence_out, text_a, text_b, id_a, id_b):
                return summary, evidence_out, False
            if attempt == 0:
                prompt += "\n\nIMPORTANT: Every quote must appear EXACTLY as a substring in EXCERPT_A or EXCERPT_B. Use shorter quotes if needed."
        except Exception:
            logger.exception("OpenRouter edge explain attempt %s failed", attempt)
    return "", [], True


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
