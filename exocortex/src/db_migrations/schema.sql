-- Cortexa Option B (Supabase Postgres) schema foundation
-- Run this in Supabase SQL editor or via your migration tool.
-- This repository also has a runtime idempotent initializer (src/db.py),
-- but Supabase workflows typically prefer SQL migrations.

-- Users (Telegram identity + dashboard auth)
create table if not exists public.users (
  id bigserial primary key,
  chat_id bigint unique not null,
  username text null,
  password_hash text not null,
  created_at timestamptz not null default now()
);

-- Canonical memories (main records; chunking is a separate table)
create table if not exists public.memories (
  memory_id text primary key,
  -- Tenant key: we store Telegram `chat_id` here (user_id == chat_id),
  -- keeping compatibility with current Pinecone metadata + token payloads.
  user_id bigint not null,
  chat_id bigint null,

  title text null,
  raw_content_full text not null,
  source_type text not null,
  source_url text null,
  text_fingerprint text null,
  url_fingerprint text null,
  tags jsonb null,

  created_at_ts bigint not null,
  due_at_ts bigint null,
  last_accessed_ts bigint null,
  priority_score double precision null,
  last_resurfaced_ts bigint null,
  visibility text null,
  parent_id text null,
  tunnel_id text null,
  tunnel_name text null,
  is_full boolean not null default true
);

create index if not exists memories_user_created_at_ts_idx
  on public.memories (user_id, created_at_ts desc);

create index if not exists memories_user_source_type_idx
  on public.memories (user_id, source_type);

create index if not exists memories_chat_text_fp_idx
  on public.memories (chat_id, text_fingerprint);

create index if not exists memories_chat_url_fp_idx
  on public.memories (chat_id, url_fingerprint);

create index if not exists memories_user_due_at_ts_idx
  on public.memories (user_id, due_at_ts);

-- Basic lexical index (FTS) for hybrid retrieval.
-- (tags json is intentionally not included here; lexical/tag filtering can be done separately.)
create index if not exists memories_fts_title_body_idx
  on public.memories using gin (
    to_tsvector('english', coalesce(title, '') || ' ' || coalesce(raw_content_full, ''))
  );

-- Chunk children (indexed in Pinecone; not meant for user-facing lists)
create table if not exists public.memory_chunks (
  chunk_id text primary key,
  memory_id text not null references public.memories(memory_id) on delete cascade,
  user_id bigint not null,
  chat_id bigint null,

  chunk_index integer not null,
  chunk_text text not null,
  source_type text not null,
  created_at_ts bigint not null
);

create index if not exists memory_chunks_user_memory_idx
  on public.memory_chunks (user_id, memory_id);

-- Reminders
create table if not exists public.reminders (
  id text primary key,
  user_id bigint not null,
  chat_id bigint null,

  text text not null,
  due_at_ts bigint not null,
  timezone text null,
  created_at_ts bigint not null,
  fired boolean not null default false
);

create index if not exists reminders_user_due_idx
  on public.reminders (user_id, due_at_ts);

-- Tunnels (weekly thematic clusters)
create table if not exists public.tunnels (
  id text primary key,
  user_id bigint not null,
  name text not null,
  reason text null,
  core_tag text null,
  memory_count integer null,
  created_at_ts bigint not null,
  raw text not null
);

-- Links: which canonical memories belong to which tunnel
create table if not exists public.tunnel_members (
  tunnel_id text not null references public.tunnels(id) on delete cascade,
  memory_id text not null references public.memories(memory_id) on delete cascade,
  user_id bigint not null,
  primary key (tunnel_id, memory_id)
);

create index if not exists tunnel_members_user_idx
  on public.tunnel_members (user_id);

create index if not exists tunnels_user_created_idx
  on public.tunnels (user_id, created_at_ts desc);

