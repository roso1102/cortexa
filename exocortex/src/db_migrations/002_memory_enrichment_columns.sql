-- Run once if you prefer manual migration over app startup (init_db also applies these).
-- Adds fingerprint + enrichment columns when `memories` was created from an older schema.

ALTER TABLE public.memories ADD COLUMN IF NOT EXISTS text_fingerprint text NULL;
ALTER TABLE public.memories ADD COLUMN IF NOT EXISTS url_fingerprint text NULL;
ALTER TABLE public.memories ADD COLUMN IF NOT EXISTS content_type text NULL;
ALTER TABLE public.memories ADD COLUMN IF NOT EXISTS topics jsonb NULL;

CREATE INDEX IF NOT EXISTS memories_chat_text_fp_idx
  ON public.memories (chat_id, text_fingerprint);

CREATE INDEX IF NOT EXISTS memories_chat_url_fp_idx
  ON public.memories (chat_id, url_fingerprint);

CREATE INDEX IF NOT EXISTS memories_user_content_type_idx
  ON public.memories (user_id, content_type);
