# Semantic Tunnel Plan

## Goal
Build semantic tunnels that connect non-obvious but meaningful ideas, and explain each connection with clear reasoning.

## Principles
- Prefer memory-to-memory semantic links over token/tag explosions.
- Always provide a human-readable rationale for each displayed link.
- Keep graph readable by default; advanced noisy nodes should be optional.
- Maintain backward compatibility with existing tunnel list and scheduler flows.

## Phase 1: Reliability and API Foundation
- Harden OpenRouter tunnel naming so empty/partial responses do not raise tracebacks.
- Add canonical tunnel graph storage for edges:
  - `tunnel_edges(tunnel_id, from_memory_id, to_memory_id, user_id, weight, rationale, bridge_score)`
- Add API endpoint:
  - `GET /api/tunnels/<tunnel_id>/graph`
  - returns `nodes` (member memories) and `edges` (semantic links with rationale)
- Keep existing `GET /api/tunnels` and `POST /api/tunnels/generate` stable.

## Phase 2: Better Link Quality
- Replace pure tag-token grouping with hybrid scoring:
  - semantic similarity from embeddings
  - lexical diversity boost (prefer non-obvious bridges)
  - source-type balancing to avoid same-type clustering bias
- Add thresholds and caps:
  - max nodes per tunnel
  - max edges per tunnel
  - minimum confidence for edge inclusion

## Phase 3: Explanation Quality
- Generate per-edge rationale using LLM with strict JSON output.
- Add fallback deterministic rationale when LLM output is invalid/missing.
- Store `bridge_score` and `reason_quality` for ranking and debugging.

## Phase 4: UI Graph Contract (Dashboard)
- Default graph: memory nodes + semantic edges only.
- Edge click reveals rationale and evidence snippets.
- Optional toggles:
  - show tags/entities
  - show raw extraction graph
  - show low-confidence links

## Phase 5: Controls and Product Features
- Manual actions:
  - regenerate one tunnel
  - regenerate only weak edges
  - lock/unlock a tunnel
- Quality tools:
  - "Why connected?" panel with cited snippets
  - "Hide this link" feedback loop
- Ops:
  - metrics for edge acceptance, fallback rate, empty-response rate
  - admin debug endpoint for tunnel diagnostics

## Environment variables (tunnel formation)

| Variable | Default | Meaning |
|----------|---------|--------|
| `TUNNEL_MIN_MEMORIES` | `4` | Minimum memories that must share one tag token for a cluster to become a tunnel (clamped 3–100). |
| `TUNNEL_MAX_MEMORIES_PER_TUNNEL` | `20` | After deduplication, at most this many memories are kept per tunnel, newest first (clamped 5–400). |

Set these on Koyeb (backend). Example stricter tunnels: `TUNNEL_MIN_MEMORIES=5` and `TUNNEL_MAX_MEMORIES_PER_TUNNEL=12`.

## Dashboard (required for a sane graph)

The “dozens of random nodes” view happens when the UI builds a graph from **every tag, word, URL, or entity** extracted from a few memories. That is not the tunnel graph.

**Do this instead:**

1. On tunnel detail / semantic map, call **`GET /api/tunnels/<tunnel_id>/graph`** with `X-Dashboard-Token`.
2. Render **only** `nodes` (memories) and **`edges`** (memory-to-memory links). Show `edge.rationale` on click or hover.
3. Do **not** add extra graph nodes from `memory.tags` unless behind an explicit “Show tag nodes” toggle.

Until the dashboard uses this endpoint, the map can stay noisy regardless of backend tuning.

## Immediate Implementation Started
- [x] Plan document created.
- [x] OpenRouter naming hardening.
- [x] Tunnel edge table + persistence.
- [x] `GET /api/tunnels/<id>/graph` endpoint.
- [x] Initial edge generation logic with rationale fallback.
- [x] Phase 2 scoring controls: confidence threshold, source-type balancing, per-node degree cap.
