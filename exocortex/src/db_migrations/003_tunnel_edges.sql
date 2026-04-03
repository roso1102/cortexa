-- Run once for existing deployments to support semantic tunnel graph edges.

CREATE TABLE IF NOT EXISTS public.tunnel_edges (
  tunnel_id text NOT NULL REFERENCES public.tunnels(id) ON DELETE CASCADE,
  from_memory_id text NOT NULL REFERENCES public.memories(memory_id) ON DELETE CASCADE,
  to_memory_id text NOT NULL REFERENCES public.memories(memory_id) ON DELETE CASCADE,
  user_id bigint NOT NULL,
  weight double precision NULL,
  bridge_score double precision NULL,
  rationale text NULL,
  PRIMARY KEY (tunnel_id, from_memory_id, to_memory_id)
);

CREATE INDEX IF NOT EXISTS tunnel_edges_user_idx
  ON public.tunnel_edges (user_id);

CREATE INDEX IF NOT EXISTS tunnel_edges_tunnel_idx
  ON public.tunnel_edges (tunnel_id);
