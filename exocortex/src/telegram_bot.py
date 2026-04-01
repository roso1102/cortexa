from __future__ import annotations

import logging
import os
import tempfile
import re
from typing import Any

import fitz  # PyMuPDF
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.config import AppConfig
from src.brains import BrainsRouter
from src.memory import MemoryManager
from src.reflection import ReflectionService
from src.orchestrator import (
    classify_intent,
    route_action,
    INTENT_CHITCHAT,
    INTENT_DELETE,
    INTENT_INGEST_LINK,
    INTENT_QUERY,
    INTENT_REMINDER,
    INTENT_LIST_LINKS,
)
from src.action_schema import (
    ACTION_ANSWER_QUERY,
    ACTION_CLARIFY,
    ACTION_DELETE,
    ACTION_LIST,
    ACTION_SAVE_LINK,
    ACTION_SAVE_TEXT,
    ACTION_SET_REMINDER,
    LIST_LINKS,
    LIST_MEMORIES,
    LIST_POEMS,
)
from datetime import datetime, timedelta, timezone

from src.utils import chunk_text, utc_now_iso, utc_now_ts
from src.reminders import parse_reminder_llm, reminder_to_metadata
from src.link_ingest import extract_urls, validate_url, fetch_and_extract
from src.tagger import tag_text
from src.retrieval import HybridRetriever


logger = logging.getLogger(__name__)


class CortexaBot:
    def __init__(self, config: AppConfig) -> None:
        from groq import Groq  # local import to keep top-level imports clean
        self._config = config
        self._memory = MemoryManager(config)
        self._brains = BrainsRouter(config)
        self._retriever = HybridRetriever(self._memory)
        self._reflection = ReflectionService(config, self._memory)
        self._debug_mode = config.debug_mode
        self._groq_client = Groq(api_key=config.groq_api_key)
        # Ephemeral per-chat listing state (e.g. last poem list for \"show poem 1\")
        # Structure: { chat_id: { \"poems\": [id1, id2, ...] } }
        self._last_lists: dict[int, dict[str, list[str]]] = {}

    def _classify_intent(self, text: str) -> str:
        """
        Wrapper around the LLM-based orchestrator.
        Returns a simple 'query' or 'store' string for backward-compatible call sites
        that still need the old two-class output.
        """
        result = classify_intent(text, self._groq_client)
        if result["intent"] in (INTENT_QUERY, INTENT_LIST_LINKS):
            return "query"
        return "store"

    def _is_links_list_query(self, text: str) -> bool:
        t = (text or "").strip().lower()
        return any(
            phrase in t
            for phrase in (
                "what links did i save",
                "which links did i save",
                "links did i save",
                "links i saved",
                "show links",
                "show me links",
                "show me my links",
                "show me saved links",
                "show me my saved links",
                "list links",
                "my saved links",
                "saved links",
            )
        )

    async def _handle_links_today(self, context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> None:  # type: ignore[type-arg]
        now = datetime.now(timezone.utc)
        start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())

        matches = self._memory.query_by_filter_for_chat(
            query_text="links saved today",
            chat_id=chat_id,
            filter_obj={
                "source_type": {"$eq": "link"},
                "created_at_ts": {"$gte": start_ts, "$lt": end_ts},
            },
            k=50,
        )

        # Deduplicate by URL
        seen: set[str] = set()
        items: list[tuple[str, str]] = []
        for md in matches:
            url = str(md.get("url") or "").strip()
            title = str(md.get("title") or url).strip()
            if not url or url in seen:
                continue
            seen.add(url)
            items.append((title, url))

        if not items:
            await self._send_text(context, chat_id, "No links saved today.")
            if self._debug_mode:
                await self._send_text(context, chat_id, "[debug] links_today count=0")
            return

        lines = ["Links saved today:"]
        for i, (title, url) in enumerate(items[:10], start=1):
            lines.append(f"{i}. {title}\n{url}")
        if len(items) > 10:
            lines.append(f"\n(Showing 10 of {len(items)}.)")

        await self._send_text(context, chat_id, "\n\n".join(lines))
        if self._debug_mode:
            await self._send_text(context, chat_id, f"[debug] links_today count={len(items)}")

    async def _send_text(self, context: ContextTypes.DEFAULT_TYPE, chat_id: int, text: str) -> None:  # type: ignore[type-arg]
        """
        Telegram hard-limits message length. Split long messages safely.
        """
        if text is None:
            return

        text = self._sanitize_llm_text(text)

        max_len = 3500  # conservative vs ~4096 limit (gives room for formatting)
        if len(text) <= max_len:
            await context.bot.send_message(chat_id=chat_id, text=text)
            return

        # Split on paragraph boundaries where possible
        remaining = text
        while remaining:
            if len(remaining) <= max_len:
                await context.bot.send_message(chat_id=chat_id, text=remaining)
                break
            cut = remaining.rfind("\n\n", 0, max_len)
            if cut == -1 or cut < 200:
                cut = max_len
            chunk = remaining[:cut].strip()
            if chunk:
                await context.bot.send_message(chat_id=chat_id, text=chunk)
            remaining = remaining[cut:].lstrip()

    def _sanitize_llm_text(self, text: str) -> str:
        """
        LLMs often emit Markdown (# headings, **bold**, code fences).
        Telegram (in our current setup) shows it as raw text, which looks noisy.
        Convert common Markdown patterns into cleaner plain text.
        """
        t = text.strip()

        # Remove fenced code blocks backticks but keep content
        t = re.sub(r"```[a-zA-Z0-9_-]*\n", "", t)
        t = t.replace("```", "")

        # Headings: '### Title' -> 'Title'
        t = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", t)

        # Bold/italic markers: **text** / *text* -> text
        t = t.replace("**", "")
        t = t.replace("__", "")
        t = t.replace("*", "")

        # Inline code backticks
        t = t.replace("`", "")

        # Clean up excessive blank lines
        t = re.sub(r"\n{3,}", "\n\n", t)

        return t.strip()

    async def handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:  # type: ignore[type-arg]
        if not update.effective_chat:
            return
        text = (
            "🧠 Welcome to cortexa.\n\n"
            "Send me text, links, or PDFs to save them into your semantic memory.\n"
            "Ask me questions in simple English and I'll answer using your past memories."
        )
        await self._send_text(context, update.effective_chat.id, text)

    async def handle_summary_today(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:  # type: ignore[type-arg]
        if not update.effective_chat:
            return
        summary = self._reflection.summarize_today_for_user(update.effective_chat.id)
        await self._send_text(context, update.effective_chat.id, summary)

    async def handle_profile(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:  # type: ignore[type-arg]
        if not update.effective_chat:
            return
        chat_id = update.effective_chat.id
        # Prefer latest stored snapshot; generate if missing
        profile = self._get_latest_profile(chat_id)
        if not profile:
            profile = self._reflection.generate_profile_snapshot_for_user(chat_id)
        await self._send_text(context, chat_id, f"Your profile snapshot:\n\n{profile}")

    async def handle_tunnels(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:  # type: ignore[type-arg]
        if not update.effective_chat:
            return
        chat_id = update.effective_chat.id
        matches = self._memory.query_by_filter(
            query_text="tunnels themes clusters",
            filter_obj={"source_type": {"$eq": "tunnel"}},
            k=30,
        )
        if not matches:
            await self._send_text(context, chat_id, "No tunnels yet. Once you’ve saved more memories, I’ll start forming themes automatically.")
            return

        # Deduplicate by tunnel_id/name
        seen: set[str] = set()
        lines = ["Your tunnels:"]
        for m in matches:
            tid = str(m.get("id") or m.get("tunnel_id") or "").strip()
            name = str(m.get("tunnel_name") or "").strip() or "Untitled Tunnel"
            if tid and tid in seen:
                continue
            if tid:
                seen.add(tid)
            count = m.get("memory_count")
            core_tag = m.get("core_tag")
            extra = []
            if count is not None:
                extra.append(f"{count} memories")
            if core_tag:
                extra.append(f"core: {core_tag}")
            suffix = f" ({', '.join(extra)})" if extra else ""
            lines.append(f"- {name}{suffix}")

        await self._send_text(context, chat_id, "\n".join(lines))

    async def handle_feedback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:  # type: ignore[type-arg]
        """Handle inline button feedback on resurfaced memory items."""
        query = update.callback_query
        if not query or not query.data:
            return
        await query.answer()  # acknowledge immediately so spinner stops

        data = query.data  # format: "fb:action:memory_id"
        parts = data.split(":", 2)
        if len(parts) != 3 or parts[0] != "fb":
            return

        _, action, memory_id = parts
        now_ts = utc_now_ts()

        try:
            if action == "relevant":
                # Boost priority score (cap at 1.0)
                self._memory.update_memory_metadata(memory_id, {
                    "priority_score": min(1.0, 0.7),  # bump to 0.7
                    "last_feedback_ts": now_ts,
                    "feedback": "relevant",
                })
                await query.edit_message_reply_markup(reply_markup=None)
                await query.message.reply_text("Marked as still relevant. I'll keep it on your radar.")  # type: ignore[union-attr]

            elif action == "irrelevant":
                # Reduce priority score (floor at 0.0)
                self._memory.update_memory_metadata(memory_id, {
                    "priority_score": 0.1,
                    "last_feedback_ts": now_ts,
                    "feedback": "irrelevant",
                    "last_resurfaced_ts": now_ts + (30 * 86400),  # suppress for 30 days
                })
                await query.edit_message_reply_markup(reply_markup=None)
                await query.message.reply_text("Got it. I'll stop surfacing that one.")  # type: ignore[union-attr]

            elif action == "snooze":
                snooze_until = now_ts + (7 * 86400)
                self._memory.update_memory_metadata(memory_id, {
                    "last_resurfaced_ts": snooze_until,
                    "last_feedback_ts": now_ts,
                    "feedback": "snoozed",
                })
                await query.edit_message_reply_markup(reply_markup=None)
                await query.message.reply_text("Snoozed for 7 days. I'll bring it back then.")  # type: ignore[union-attr]

        except Exception as exc:
            logger.exception("Feedback handler error: %s", exc)
            await query.message.reply_text("Something went wrong updating that memory.")  # type: ignore[union-attr]

    def _store_confirmation(self, text: str) -> str:
        """Return a warm, human-feeling confirmation line for a stored note."""
        snippet = text.strip()[:60].rstrip(".,;:!?")
        if len(text.strip()) > 60:
            snippet += "..."
        return f'Got it. I\'ve saved "{snippet}" into your memory. You can ask me about it anytime.'

    def _is_allowed(self, chat_id: int) -> bool:
        """Return True if this chat_id is permitted to use the bot."""
        if not self._config.allowed_chat_ids:
            return True  # open/single-user mode
        return chat_id in self._config.allowed_chat_ids

    def _get_latest_profile(self, chat_id: int) -> str | None:
        """Fetch the most recent profile_snapshot from memory to personalise query answers."""
        try:
            matches = self._memory.query_by_filter_for_chat(
                query_text="profile snapshot interests tunnels topics",
                chat_id=chat_id,
                filter_obj={
                    "source_type": {"$eq": "profile_snapshot"},
                    "user_id": {"$eq": chat_id},
                    "archived": {"$ne": True},
                },
                k=1,
            )
            if matches:
                return str(matches[0].get("raw_content") or "")[:600]
        except Exception:
            pass
        return None

    async def _handle_list_poems(self, context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> None:  # type: ignore[type-arg]
        """
        List recently saved poems and remember their ids for follow-up selection like "show poem 1".
        Poems are detected heuristically via tags containing "poem".
        """
        # Pinecone metadata filters do not support "$contains" on arrays,
        # so we fetch a broader slice for this chat and filter by tags client-side.
        candidates = self._memory.query_by_filter_for_chat(
            query_text="poem verse poetry",
            chat_id=chat_id,
            filter_obj={},
            k=50,
        )

        matches = []
        for m in candidates:
            tags = [str(t).lower() for t in (m.get("tags") or [])]
            if "poem" in tags:
                matches.append(m)
            if len(matches) >= 10:
                break

        if not matches:
            await self._send_text(context, chat_id, "I couldn't find any poems you've saved yet.")
            if self._debug_mode:
                await self._send_text(context, chat_id, "[debug] poems_list count=0")  # type: ignore[arg-type]
            return

        poem_ids: list[str] = []
        lines: list[str] = ["You saved these poems:\n"]
        for idx, m in enumerate(matches, start=1):
            mem_id = str(m.get("id") or "").strip()
            if not mem_id:
                continue
            poem_ids.append(mem_id)
            raw = str(m.get("raw_content") or "").strip()
            title = str(m.get("title") or "").strip()
            if not title:
                # Fallback: first non-empty line of content
                first_line = next((ln.strip() for ln in raw.splitlines() if ln.strip()), "")
                title = first_line[:80] or f"Poem {idx}"
            dash_url = self._dashboard_memory_url(mem_id)
            if dash_url:
                lines.append(f"{idx}. {title}\n   Open: {dash_url}")
            else:
                lines.append(f"{idx}. {title}")

        # Remember list for this chat
        self._last_lists.setdefault(chat_id, {})["poems"] = poem_ids

        await self._send_text(context, chat_id, "\n".join(lines).strip())
        if self._debug_mode:
            await self._send_text(context, chat_id, f"[debug] poems_list count={len(poem_ids)}")  # type: ignore[arg-type]

    def _dashboard_memory_url(self, memory_id: str) -> str | None:
        base = (self._config.dashboard_public_url or "").strip()
        if not base:
            return None
        base = base.rstrip("/")
        return f"{base}/memories/{memory_id}"

    def _store_full_and_chunks(self, *, text: str, base_meta: dict[str, Any], chunk_source_type: str) -> str:
        """
        Store a canonical full record (entire text) and optional chunk children for retrieval/understanding.
        Returns the full record's memory_id.
        """
        # Dual-write for Option B:
        # - Pinecone: used by current retrieval
        # - Postgres: canonical store for later hybrid retrieval + dashboard correctness
        try:
            from src.db import insert_chunks, insert_memory
        except Exception:
            insert_chunks = None  # type: ignore[assignment]
            insert_memory = None  # type: ignore[assignment]

        full_id = utc_now_iso()
        full_meta = {
            **base_meta,
            "id": full_id,
            "is_full": True,
            "priority_score": float(base_meta.get("priority_score") or 0.5),
        }
        self._memory.add_memory(text, full_meta)

        if insert_memory is not None:
            row = {
                "memory_id": full_id,
                "user_id": int(base_meta.get("user_id") or base_meta.get("chat_id") or 0),
                "chat_id": base_meta.get("chat_id"),
                "title": base_meta.get("title"),
                "raw_content_full": text,
                "source_type": str(base_meta.get("source_type") or "text"),
                "source_url": base_meta.get("url") or base_meta.get("source_url"),
                "tags": base_meta.get("tags"),
                "created_at_ts": int(base_meta.get("created_at_ts") or utc_now_ts()),
                "due_at_ts": base_meta.get("due_at_ts"),
                "last_accessed_ts": base_meta.get("last_accessed_ts"),
                "priority_score": float(base_meta.get("priority_score") or 0.5),
                "last_resurfaced_ts": base_meta.get("last_resurfaced_ts"),
                "visibility": base_meta.get("visibility"),
                "parent_id": base_meta.get("parent_id"),
                "is_full": True,
            }
            insert_memory(row)

        # Store chunks only for long text
        chunk_rows: list[dict[str, Any]] = []
        if len(text) > 900:
            chunks = chunk_text(text)[:30]
            for idx, chunk in enumerate(chunks):
                child_meta = {
                    **base_meta,
                    "source_type": chunk_source_type,
                    "parent_id": full_id,
                    "chunk_index": idx,
                    "is_full": False,
                    "priority_score": float(base_meta.get("priority_score") or 0.5),
                }
                child_id = self._memory.add_memory(chunk, child_meta)
                if insert_chunks is not None:
                    chunk_rows.append(
                        {
                            "chunk_id": child_id,
                            "memory_id": full_id,
                            "user_id": int(base_meta.get("user_id") or base_meta.get("chat_id") or 0),
                            "chat_id": base_meta.get("chat_id"),
                            "chunk_index": idx,
                            "chunk_text": chunk,
                            "source_type": chunk_source_type,
                            "created_at_ts": int(base_meta.get("created_at_ts") or utc_now_ts()),
                        }
                    )

        if insert_chunks is not None and chunk_rows:
            insert_chunks(chunk_rows)

        return full_id

    def _is_simple_recall_query(self, text: str) -> bool:
        """
        Heuristic: user wants to see what they saved, not analysis.
        We can answer with snippets + links (no LLM call) and optionally dashboard deep-links.
        """
        t = (text or "").strip().lower()
        if any(kw in t for kw in ["analyze", "analysis", "plan", "compare", "summarize", "summarise", "propose", "debug"]):
            return False
        return any(p in t for p in ["what did i save", "what i saved", "show me what i saved", "recall what i saved", "what do i know about"])

    def _rerank_simple_recall(self, contexts: list[dict[str, Any]], user_text: str) -> list[dict[str, Any]]:
        """
        Pinecone semantic recall is sometimes too fuzzy for "what did I save about X".
        For this UX we do a deterministic re-rank using:
        title + raw_content + tags substring hits, combined with the Pinecone prior score.
        """
        stopwords = {
            "what",
            "did",
            "i",
            "save",
            "saved",
            "show",
            "me",
            "my",
            "recall",
            "about",
            "know",
            "tell",
            "anything",
            "everything",
            "the",
            "a",
            "an",
            "and",
            "or",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "is",
            "are",
            "was",
            "were",
            "it",
            "this",
            "that",
            "please",
        }

        tokens = re.findall(r"[a-z0-9]+", (user_text or "").lower())
        tokens = [t for t in tokens if t not in stopwords and len(t) >= 3]
        if not tokens:
            tokens = re.findall(r"[a-z0-9]+", (user_text or "").lower())[:6]

        rescored: list[tuple[float, float, dict[str, Any]]] = []
        max_lex = 0.0
        for m in contexts:
            title_l = str(m.get("title") or "").lower()
            raw_l = str(m.get("raw_content") or "").lower()
            tags_l = " ".join([str(t).lower() for t in (m.get("tags") or [])])
            prior = float(m.get("score") or 0.0)

            lex = 0.0
            for tok in tokens[:10]:
                if tok in title_l:
                    lex += 3.0
                elif tok in tags_l:
                    lex += 2.0
                elif tok in raw_l:
                    lex += 1.0

            max_lex = max(max_lex, lex)
            rescored.append((prior + lex, lex, m))

        # If we have any items with lexical hits, drop pure semantic junk.
        filtered: list[tuple[float, float, dict[str, Any]]] = []
        if max_lex > 0:
            for total, lex, m in rescored:
                if lex > 0:
                    filtered.append((total, lex, m))
        else:
            filtered = rescored

        filtered.sort(key=lambda x: x[0], reverse=True)
        return [m for _, _, m in filtered]

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:  # type: ignore[type-arg]
        if not update.effective_chat or not update.message or not update.message.text:
            return

        text = update.message.text.strip()
        chat_id = update.effective_chat.id

        if not self._is_allowed(chat_id):
            await self._send_text(context, chat_id, "Sorry, this is a private memory system.")
            return

        use_action_router = os.getenv("ACTION_ROUTER", "").strip().lower() in {"1", "true", "yes", "on"}

        # --- Option B: Action router (feature-flagged) ---
        if use_action_router:
            routed = route_action(text, self._groq_client)
            if self._debug_mode:
                await self._send_text(
                    context,
                    chat_id,
                    f"[debug] action={routed.action} confidence={routed.confidence:.2f} reason={routed.reason} args={routed.args}",
                )

            # CLARIFY
            if routed.action == ACTION_CLARIFY:
                q = str(routed.args.get("question") or "").strip() or "Do you want me to save this, or answer it?"
                await self._send_text(context, chat_id, q)
                return

            # LIST (poems/links/memories)
            if routed.action == ACTION_LIST:
                lt = str(routed.args.get("list_type") or "").strip().lower()
                if lt == LIST_POEMS:
                    await self._handle_list_poems(context, chat_id)
                    return
                if lt == LIST_LINKS:
                    await self._handle_links_today(context, chat_id)
                    return
                if lt == LIST_MEMORIES:
                    # Defer to normal recall path by asking a structured query.
                    # Keeps behavior consistent with existing hybrid retrieval.
                    query = "List my recent saved memories."
                    contexts = self._retriever.recall(query=query, chat_id=chat_id, k=6)
                    profile_context = self._get_latest_profile(chat_id)
                    result = self._brains.route_query(query, contexts, profile_context=profile_context, source_refs=[])
                    await self._send_text(context, chat_id, str(result.get("answer") or "").strip() or "I couldn't find any memories yet.")
                    return

            # DELETE
            if routed.action == ACTION_DELETE:
                target = str(routed.args.get("target") or "").strip() or text
                # Reuse existing delete logic by overwriting text and falling through
                text = target

            # SET_REMINDER
            if routed.action == ACTION_SET_REMINDER:
                reminder = parse_reminder_llm(text, self._groq_client)
                if reminder:
                    meta = reminder_to_metadata(reminder, chat_id=chat_id)
                    base_meta = {**meta, "source_type": "reminder", "user_id": chat_id}
                    full_id = self._store_full_and_chunks(
                        text=reminder.text,
                        base_meta=base_meta,
                        chunk_source_type="reminder_chunk",
                    )
                    await self._send_text(
                        context,
                        chat_id,
                        f'Reminder set for {reminder.due_at_iso[:16].replace("T", " ")} UTC. '
                        f'I\'ll remind you: "{reminder.text}".',
                    )
                    if self._debug_mode:
                        await self._send_text(context, chat_id, f"[debug] reminder_saved full_id={full_id} due={reminder.due_at_iso}")
                    return

            # SAVE_LINK
            if routed.action == ACTION_SAVE_LINK:
                urls = routed.args.get("urls")
                if not isinstance(urls, list):
                    urls = extract_urls(text)
                # Reuse existing ingest-link branch via setting urls local var below.
            # SAVE_TEXT
            if routed.action == ACTION_SAVE_TEXT:
                # proceed to ingest-text default at bottom
                pass
            # ANSWER_QUERY
            if routed.action == ACTION_ANSWER_QUERY:
                query = str(routed.args.get("query") or "").strip() or text
                simple = self._is_simple_recall_query(query)
                contexts = self._retriever.recall(query=query, chat_id=chat_id, k=(8 if simple else 5))
                if simple and contexts:
                    contexts = self._rerank_simple_recall(contexts, user_text=query)
                    m = contexts[0]
                    raw = str(m.get("raw_content") or "").strip()
                    mid = str(m.get("id") or "").strip()
                    st = str(m.get("source_type") or "")
                    if st.endswith("_chunk") or m.get("is_full") is False:
                        await self._send_text(context, chat_id, "I couldn't find a clean main memory for that.")
                        return
                    header = f"Here's what I found about \"{query.strip()[:40]}\"".rstrip() + ":\n"
                    lines = [header]
                    if len(raw) <= 420:
                        lines.append(f"- {raw}")
                    else:
                        preview = raw[:220].rstrip(".,;:!?") + "…"
                        link = self._dashboard_memory_url(mid) if mid else None
                        if link:
                            lines.append(f"- {preview}\n  Open: {link}")
                        else:
                            lines.append(f"- {preview}")
                    await self._send_text(context, chat_id, "\n\n".join(lines).strip())
                    return

                profile_context = self._get_latest_profile(chat_id)
                # Let brains filter saved-link refs, or override with [] to suppress.
                result = self._brains.route_query(query, contexts, profile_context=profile_context)
                answer = str(result.get("answer") or "").strip()
                if contexts:
                    top = contexts[0]
                    top_id = str(top.get("id") or "").strip()
                    dash_url = self._dashboard_memory_url(top_id) if top_id else None
                else:
                    dash_url = None

                if len(answer) <= 700 or not dash_url:
                    await self._send_text(context, chat_id, answer)
                else:
                    preview = answer[:520].rstrip(".,;:!?") + "…"
                    keyboard = InlineKeyboardMarkup(
                        [[InlineKeyboardButton("Open in dashboard", url=dash_url)]]
                    )
                    await context.bot.send_message(chat_id=chat_id, text=preview, reply_markup=keyboard)

                if self._debug_mode:
                    await self._send_text(context, chat_id, f"[debug] action=ANSWER_QUERY retrieved={len(contexts)} model={result.get('model')}")
                return

            # If we reach here, fall through to existing delete/link/text branches below.

        # --- Legacy intent classifier path (default / fallback) ---
        intent_result = classify_intent(text, self._groq_client)
        intent = intent_result["intent"]

        if self._debug_mode:
            await self._send_text(
                context,
                chat_id,
                f"[debug] intent={intent} confidence={intent_result.get('confidence', '?'):.2f} "
                f"source={intent_result.get('source', '?')} reason={intent_result.get('summary', '')}",
            )

        t_lower = text.strip().lower()

        # --- POEM LISTING (legacy path) ---
        if any(phrase in t_lower for phrase in ("what poem did i save", "what poems did i save", "list poems", "show my poem", "show my poems", "poems i saved", "my poems")):
            await self._handle_list_poems(context, chat_id)
            return

        # --- POEM SELECTION: \"show poem N\" / \"open poem N\" ---
        if t_lower.startswith("show poem") or t_lower.startswith("open poem"):
            import re as _re

            m = _re.match(r"^(show|open) poem\s+(\d+)", t_lower)
            poem_ids = self._last_lists.get(chat_id, {}).get("poems") or []
            if m and poem_ids:
                idx = int(m.group(2)) - 1
                if 0 <= idx < len(poem_ids):
                    mem_id = poem_ids[idx]
                    md = self._memory.get_memory_by_id(mem_id) or {}
                    raw = str(md.get("raw_content") or "").strip()
                    dash_url = self._dashboard_memory_url(mem_id)
                    if len(raw) <= 900 or not dash_url:
                        await self._send_text(context, chat_id, raw or "(empty poem)")
                    else:
                        preview = raw[:600].rstrip(".,;:!?") + "…"
                        keyboard = InlineKeyboardMarkup(
                            [[InlineKeyboardButton("Open in dashboard", url=dash_url)]]
                        )
                        await context.bot.send_message(chat_id=chat_id, text=preview, reply_markup=keyboard)
                    if self._debug_mode:
                        await self._send_text(
                            context,
                            chat_id,
                            f"[debug] poem_selection index={idx+1} id={mem_id}",
                        )
                    return

        # --- CHITCHAT: greeting / pleasantry — reply warmly, do NOT save ---
        if intent == INTENT_CHITCHAT:
            replies = {
                frozenset({"hi", "hello", "hey", "heyy", "heya"}): "Hey! What's on your mind? You can share a note, ask me something, or drop a link.",
                frozenset({"bye", "goodbye", "good bye", "cya", "see ya", "see you"}): "See you! I'll keep your memories safe until you're back.",
                frozenset({"good night", "goodnight", "gn", "gn!"}): "Good night! Rest well. I'll be here when you need me.",
                frozenset({"good morning", "gm"}): "Good morning! Ready to capture something new today?",
                frozenset({"good afternoon", "good evening"}): "Hello! What are you working on?",
                frozenset({"thanks", "thank you", "ty", "thx"}): "Happy to help! Anything else on your mind?",
                frozenset({"how are you", "how r u", "how are u", "how's it going", "whats up", "what's up", "wassup", "sup"}): "Doing great, ready to help! What would you like to save or recall?",
            }
            t_lower = text.strip().lower().rstrip("!.,?")
            reply = "Got it!"
            for phrase_set, response in replies.items():
                if t_lower in phrase_set:
                    reply = response
                    break
            await self._send_text(context, chat_id, reply)
            return

        # --- LIST_LINKS: structured metadata query, no semantic search ---
        if intent == INTENT_LIST_LINKS or self._is_links_list_query(text):
            await self._handle_links_today(context, chat_id)
            return

        # --- DELETE: CRUD shortcuts ---
        if intent == INTENT_DELETE or text.lower().startswith("delete "):
            t_del = text.strip().lower()

            # delete last → remove most recent memory for this chat
            if t_del == "delete last":
                try:
                    last = self._memory.get_latest_memory_for_chat(chat_id)  # type: ignore[attr-defined]
                except AttributeError:
                    last = None
                if not last:
                    await self._send_text(context, chat_id, "I couldn't find a recent memory to delete.")
                    return
                mem_id = str(last.get("id") or "").strip()
                title = str(last.get("title") or "").strip() or (str(last.get("raw_content") or "").strip()[:40] + "…")
                try:
                    self._memory.delete_memory(mem_id)
                    await self._send_text(context, chat_id, f'Done. Removed your last memory (id: {mem_id}, title: "{title}").')
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Failed to delete last memory: %s", exc)
                    await self._send_text(context, chat_id, "Hmm, I couldn't remove your last memory.")
                return

            # delete poem N → use last poems list state
            if t_del.startswith("delete poem"):
                import re as _re

                m = _re.match(r"^delete poem\s+(\d+)", t_del)
                poem_ids = self._last_lists.get(chat_id, {}).get("poems") or []
                if m and poem_ids:
                    idx = int(m.group(1)) - 1
                    if 0 <= idx < len(poem_ids):
                        mem_id = poem_ids[idx]
                        try:
                            self._memory.delete_memory(mem_id)
                            await self._send_text(context, chat_id, f"Done. Removed poem {idx+1} (id: {mem_id}).")
                        except Exception as exc:  # noqa: BLE001
                            logger.exception("Failed to delete poem memory: %s", exc)
                            await self._send_text(context, chat_id, "Hmm, I couldn't remove that poem.")
                        return

            # delete <id> → explicit id
            if text.lower().startswith("delete "):
                memory_id = text[len("delete "):].strip()
            else:
                memory_id = text.strip()

            if memory_id:
                try:
                    self._memory.delete_memory(memory_id)
                    await self._send_text(context, chat_id, f"Done. Memory removed (id: {memory_id}).")
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Failed to delete memory: %s", exc)
                    await self._send_text(context, chat_id, "I couldn't find that memory. Check the id or use 'delete last'.")
            return

        # --- REMINDER ---
        if intent == INTENT_REMINDER:
            reminder = parse_reminder_llm(text, self._groq_client)
            if reminder:
                meta = reminder_to_metadata(reminder, chat_id=chat_id)
                # Store canonical full reminder + optional chunks
                base_meta = {**meta, "source_type": "reminder", "user_id": chat_id}
                full_id = self._store_full_and_chunks(
                    text=reminder.text,
                    base_meta=base_meta,
                    chunk_source_type="reminder_chunk",
                )
                await self._send_text(
                    context,
                    chat_id,
                    f'Reminder set for {reminder.due_at_iso[:16].replace("T", " ")} UTC. '
                    f'I\'ll remind you: "{reminder.text}".',
                )
                if self._debug_mode:
                    await self._send_text(context, chat_id, f"[debug] intent=REMINDER due={reminder.due_at_iso}")
                    await self._send_text(
                        context,
                        chat_id,
                        f"[debug] reminder_saved full_id={full_id} len={len(reminder.text)}",
                    )
                return
            # If reminder parsing fails, fall through to INGEST_TEXT

        # --- INGEST_LINK ---
        urls = extract_urls(text)
        if intent == INTENT_INGEST_LINK or (urls and intent != INTENT_QUERY):
            if urls:
                for url in urls[:3]:
                    err = validate_url(url)
                    if err:
                        await self._send_text(context, chat_id, f"Can't save that link ({err}):\n{url}")
                        if self._debug_mode:
                            await self._send_text(context, chat_id, f"[debug] link_blocked url={url} reason={err}")
                        continue

                    if self._debug_mode:
                        await self._send_text(context, chat_id, f"[debug] link_fetch_start url={url}")
                    try:
                        extracted = fetch_and_extract(url)
                        if self._debug_mode:
                            await self._send_text(
                                context,
                                chat_id,
                                f"[debug] link_extracted title_len={len(extracted.title)} chars={len(extracted.text)}",
                            )

                        header = f"Title: {extracted.title or extracted.url}\nURL: {extracted.url}\n\n"
                        combined = header + extracted.text.strip()
                        chunks_for_tags = chunk_text(combined)[:1]
                        # Tag from the first chunk (most representative content)
                        link_tags = tag_text(chunks_for_tags[0] if chunks_for_tags else extracted.title, self._groq_client)

                        link_meta: dict[str, Any] = {
                            "source_type": "link",
                            "chat_id": chat_id,
                            "user_id": chat_id,
                            "url": extracted.url,
                            "title": extracted.title or extracted.url,
                            "created_at": utc_now_iso(),
                            "created_at_ts": utc_now_ts(),
                            "tags": link_tags,
                            "priority_score": 0.5,
                        }
                        full_id = self._store_full_and_chunks(
                            text=combined,
                            base_meta=link_meta,
                            chunk_source_type="link_chunk",
                        )

                        title_display = extracted.title or extracted.url
                        await self._send_text(
                            context,
                            chat_id,
                            f'Nice, I\'ve saved "{title_display}" into your memory. '
                            f"You can ask me about it whenever you like.",
                        )
                        if self._debug_mode:
                            await self._send_text(
                                context,
                                chat_id,
                                f"[debug] link_saved full_id={full_id} chars={len(combined)}",
                            )
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("Failed to ingest link: %s", exc)
                        # If extraction fails (403 paywall etc.), still save the URL as a bookmark.
                        try:
                            title_guess = text.splitlines()[0].strip()[:120] if text.strip() else url
                            try:
                                link_tags = tag_text(title_guess, self._groq_client)
                            except Exception:  # noqa: BLE001
                                # Tagging is non-critical; don't fail bookmark saves if Groq is down.
                                link_tags = []
                            combined = (
                                f"Title: {title_guess}\nURL: {url}\n\n"
                                "(Bookmark saved; content fetch failed.)"
                            )
                            link_meta: dict[str, Any] = {
                                "source_type": "link",
                                "chat_id": chat_id,
                                "user_id": chat_id,
                                "url": url,
                                "title": title_guess,
                                "created_at": utc_now_iso(),
                                "created_at_ts": utc_now_ts(),
                                "tags": link_tags,
                                "priority_score": 0.5,
                                "fetch_error": str(exc)[:200],
                            }
                            self._store_full_and_chunks(
                                text=combined,
                                base_meta=link_meta,
                                chunk_source_type="link_chunk",
                            )
                            await self._send_text(
                                context,
                                chat_id,
                                f'I saved the link as a bookmark (the site blocked extraction):\n{url}',
                            )
                        except Exception:
                            await self._send_text(context, chat_id, f"Sorry, I ran into a problem fetching that link.\n{url}")
                        if self._debug_mode:
                            await self._send_text(context, chat_id, f"[debug] link_error type={type(exc).__name__} msg={exc}")
                return

        # --- QUERY ---
        if intent == INTENT_QUERY:
            simple = self._is_simple_recall_query(text)
            contexts = self._retriever.recall(query=text, chat_id=chat_id, k=(8 if simple else 5))
            # If user is asking for a simple recall, avoid LLM and return snippets + dashboard links.
            if simple and contexts:
                contexts = self._rerank_simple_recall(contexts, user_text=text)
                # Show only the single best match to avoid noise.
                m = contexts[0]
                raw = str(m.get("raw_content") or "").strip()
                mid = str(m.get("id") or "").strip()
                st = str(m.get("source_type") or "")
                if st.endswith("_chunk") or m.get("is_full") is False:
                    await self._send_text(context, chat_id, "I couldn't find a clean main memory for that.")
                    return

                header = f"Here's what I found about \"{text.strip()[:40]}\"".rstrip() + ":\n"
                lines = [header]

                if len(raw) <= 420:
                    lines.append(f"- {raw}")
                else:
                    preview = raw[:220].rstrip(".,;:!?") + "…"
                    link = self._dashboard_memory_url(mid) if mid else None
                    if link:
                        lines.append(f"- {preview}\n  Open: {link}")
                    else:
                        lines.append(f"- {preview}")

                await self._send_text(context, chat_id, "\n\n".join(lines).strip())
                return

            profile_context = self._get_latest_profile(chat_id)
            result = self._brains.route_query(text, contexts, profile_context=profile_context)

            answer = str(result.get("answer") or "").strip()
            # If answer is long, send a short preview + dashboard link to the top matching memory.
            if contexts:
                top = contexts[0]
                top_id = str(top.get("id") or "").strip()
                dash_url = self._dashboard_memory_url(top_id) if top_id else None
            else:
                dash_url = None

            if len(answer) <= 700 or not dash_url:
                await self._send_text(context, chat_id, answer)
            else:
                preview = answer[:520].rstrip(".,;:!?") + "…"
                keyboard = InlineKeyboardMarkup(
                    [[InlineKeyboardButton("Open in dashboard", url=dash_url)]]
                )
                await context.bot.send_message(chat_id=chat_id, text=preview, reply_markup=keyboard)

            if self._debug_mode:
                await self._send_text(
                    context,
                    chat_id,
                    f"[debug] intent=QUERY retrieved={len(contexts)} model={result.get('model')}",
                )
            return

        # --- INGEST_TEXT (default) ---
        tags = tag_text(text, self._groq_client)
        metadata: dict[str, Any] = {
            "source_type": "text",
            "chat_id": chat_id,
            "user_id": chat_id,
            "title": text[:80],
            "created_at": utc_now_iso(),
            "created_at_ts": utc_now_ts(),
            "tags": tags,
            "priority_score": 0.5,
        }
        full_id = self._store_full_and_chunks(
            text=text,
            base_meta=metadata,
            chunk_source_type="text_chunk",
        )
        await self._send_text(context, chat_id, self._store_confirmation(text))
        if self._debug_mode:
            await self._send_text(
                context,
                chat_id,
                f"[debug] intent=INGEST_TEXT type=text tags={tags} full_id={full_id} len={len(text)}",
            )

    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:  # type: ignore[type-arg]
        if not update.effective_chat or not update.message or not update.message.document:
            return

        chat_id = update.effective_chat.id

        if not self._is_allowed(chat_id):
            await self._send_text(context, chat_id, "Sorry, this is a private memory system.")
            return

        document = update.message.document

        if not document.file_name or not document.file_name.lower().endswith(".pdf"):
            await self._send_text(context, chat_id, "Currently only PDF documents are supported.")
            return

        if self._debug_mode:
            await self._send_text(
                context,
                chat_id=chat_id,
                text=f"[debug] pdf_received file_name={document.file_name} size={document.file_size} file_id={document.file_id}",
            )

        file = await document.get_file()
        temp_path = os.path.join(tempfile.gettempdir(), f"{document.file_unique_id}.pdf")
        await file.download_to_drive(custom_path=temp_path)

        if self._debug_mode:
            await self._send_text(context, chat_id, f"[debug] pdf_downloaded path={temp_path}")

        try:
            doc = fitz.open(temp_path)
            full_text_parts: list[str] = []
            page_count = 0
            for page in doc:
                full_text_parts.append(page.get_text())
                page_count += 1
            doc.close()

            full_text = "\n".join(full_text_parts)
            if self._debug_mode:
                await self._send_text(
                    context,
                    chat_id=chat_id,
                    text=f"[debug] pdf_extracted pages={page_count} chars={len(full_text)}",
                )

            if not full_text.strip():
                await self._send_text(
                    context,
                    chat_id,
                    "I received the PDF, but extracted 0 text characters.\n"
                    "This usually means it's a scanned/image PDF (no selectable text).",
                )
                return

            chunks = chunk_text(full_text)
            if self._debug_mode:
                await self._send_text(context, chat_id, f"[debug] pdf_chunked chunks={len(chunks)}")

            # Tag from the first chunk (representative of the document)
            pdf_tags = tag_text(chunks[0] if chunks else document.file_name, self._groq_client)

            # Store a canonical full PDF record + optional chunks so the dashboard
            # shows the main item (and hides chunk children).
            chunk_count = min(len(chunks), 30)
            pdf_meta: dict[str, Any] = {
                "source_type": "pdf",
                "chat_id": chat_id,
                "user_id": chat_id,
                "file_name": document.file_name,
                "created_at": utc_now_iso(),
                "created_at_ts": utc_now_ts(),
                "tags": pdf_tags,
                "priority_score": 0.5,
            }
            full_id = self._store_full_and_chunks(
                text=full_text,
                base_meta=pdf_meta,
                chunk_source_type="pdf_chunk",
            )

            await self._send_text(
                context,
                chat_id=chat_id,
                text=f'Got it. I\'ve saved "{document.file_name}" into your memory ({chunk_count} chunks). You can ask me about it anytime.',
            )
            if self._debug_mode:
                await self._send_text(context, chat_id, f"[debug] pdf_saved full_id={full_id} tags={pdf_tags}")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to process PDF: %s", exc)
            await self._send_text(context, chat_id, "Sorry, I couldn't process that PDF.")
            if self._debug_mode:
                await self._send_text(
                    context,
                    chat_id=chat_id,
                    text=f"[debug] pdf_error type={type(exc).__name__} msg={exc}",
                )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                if self._debug_mode:
                    await self._send_text(context, chat_id, "[debug] pdf_temp_deleted=true")


def run_bot(config: AppConfig) -> None:
    from src.scheduler import ReminderScheduler  # local import avoids circular deps

    logging.basicConfig(level=logging.INFO)

    bot = CortexaBot(config)

    app = (
        ApplicationBuilder()
        .token(config.telegram_token)
        .concurrent_updates(True)
        .build()
    )

    app.add_handler(CommandHandler("start", bot.handle_start))
    app.add_handler(CommandHandler("summary_today", bot.handle_summary_today))
    app.add_handler(CommandHandler("profile", bot.handle_profile))
    app.add_handler(CommandHandler("tunnels", bot.handle_tunnels))
    app.add_handler(CallbackQueryHandler(bot.handle_feedback, pattern=r"^fb:"))
    app.add_handler(MessageHandler(filters.Document.ALL, bot.handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_text))

    if config.owner_chat_id:
        scheduler = ReminderScheduler(
            memory=bot._memory,
            reflection=bot._reflection,
            application=app,
            owner_chat_id=config.owner_chat_id,
            groq_client=bot._groq_client,
            daily_digest_time=config.daily_digest_time,
        )
        scheduler.start()

        # Wire up the event loop once run_polling initialises its async context.
        async def _post_init(application: Any) -> None:
            import asyncio as _asyncio
            scheduler.set_event_loop(_asyncio.get_running_loop())

        app.post_init = _post_init  # type: ignore[method-assign]

    app.run_polling()

