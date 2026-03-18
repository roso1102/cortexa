from __future__ import annotations

import logging
import os
import tempfile
import re
from typing import Any

import fitz  # PyMuPDF
from telegram import Update
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
from src.orchestrator import classify_intent, INTENT_QUERY, INTENT_INGEST_TEXT, INTENT_INGEST_LINK, INTENT_REMINDER, INTENT_LIST_LINKS, INTENT_DELETE, INTENT_CHITCHAT
from datetime import datetime, timedelta, timezone

from src.utils import chunk_text, utc_now_iso, utc_now_ts
from src.reminders import parse_reminder_llm, reminder_to_metadata
from src.link_ingest import extract_urls, validate_url, fetch_and_extract
from src.tagger import tag_text


logger = logging.getLogger(__name__)


class CortexaBot:
    def __init__(self, config: AppConfig) -> None:
        from groq import Groq  # local import to keep top-level imports clean
        self._config = config
        self._memory = MemoryManager(config)
        self._brains = BrainsRouter(config)
        self._reflection = ReflectionService(config, self._memory)
        self._debug_mode = config.debug_mode
        self._groq_client = Groq(api_key=config.groq_api_key)

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

        matches = self._memory.query_by_filter(
            query_text="links saved today",
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
        summary = self._reflection.summarize_today()
        await self._send_text(context, update.effective_chat.id, summary)

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

    def _get_latest_profile(self) -> str | None:
        """Fetch the most recent profile_snapshot from memory to personalise query answers."""
        try:
            matches = self._memory.query_by_filter(
                query_text="personal profile interests tunnels topics",
                filter_obj={"source_type": {"$eq": "profile_snapshot"}},
                k=1,
            )
            if matches:
                return str(matches[0].get("raw_content") or "")[:600]
        except Exception:
            pass
        return None

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:  # type: ignore[type-arg]
        if not update.effective_chat or not update.message or not update.message.text:
            return

        text = update.message.text.strip()
        chat_id = update.effective_chat.id

        if not self._is_allowed(chat_id):
            await self._send_text(context, chat_id, "Sorry, this is a private memory system.")
            return

        # --- Classify intent via LLM (with keyword fallback) ---
        intent_result = classify_intent(text, self._groq_client)
        intent = intent_result["intent"]

        if self._debug_mode:
            await self._send_text(
                context,
                chat_id,
                f"[debug] intent={intent} confidence={intent_result.get('confidence', '?'):.2f} "
                f"source={intent_result.get('source', '?')} reason={intent_result.get('summary', '')}",
            )

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

        # --- DELETE: CRUD shortcut ---
        if intent == INTENT_DELETE or text.lower().startswith("delete "):
            memory_id = text[len("delete "):].strip() if text.lower().startswith("delete ") else text.strip()
            if memory_id:
                try:
                    self._memory.delete_memory(memory_id)
                    await self._send_text(context, chat_id, f"Done. Memory removed (id: {memory_id}).")
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Failed to delete memory: %s", exc)
                    await self._send_text(context, chat_id, "Hmm, I couldn't remove that memory. Double-check the id?")
            return

        # --- REMINDER ---
        if intent == INTENT_REMINDER:
            reminder = parse_reminder_llm(text, self._groq_client)
            if reminder:
                meta = reminder_to_metadata(reminder, chat_id=chat_id)
                self._memory.add_memory(reminder.text, meta)
                await self._send_text(
                    context,
                    chat_id,
                    f'Reminder set for {reminder.due_at_iso[:16].replace("T", " ")} UTC. '
                    f'I\'ll remind you: "{reminder.text}".',
                )
                if self._debug_mode:
                    await self._send_text(context, chat_id, f"[debug] intent=REMINDER due={reminder.due_at_iso}")
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
                        chunks = chunk_text(combined)
                        chunks = chunks[:30]
                        # Tag from the first chunk (most representative content)
                        link_tags = tag_text(chunks[0] if chunks else extracted.title, self._groq_client)
                        for idx, chunk in enumerate(chunks):
                            link_meta: dict[str, Any] = {
                                "source_type": "link",
                                "url": extracted.url,
                                "title": extracted.title or extracted.url,
                                "chunk_index": idx,
                                "created_at": utc_now_iso(),
                                "created_at_ts": utc_now_ts(),
                                "tags": link_tags,
                                "priority_score": 0.5,
                            }
                            self._memory.add_memory(chunk, link_meta)

                        title_display = extracted.title or extracted.url
                        await self._send_text(
                            context,
                            chat_id,
                            f'Nice, I\'ve saved "{title_display}" into your memory. '
                            f"You can ask me about it whenever you like.",
                        )
                        if self._debug_mode:
                            await self._send_text(context, chat_id, f"[debug] link_saved chunks={len(chunks)}")
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("Failed to ingest link: %s", exc)
                        await self._send_text(context, chat_id, f"Sorry, I ran into a problem fetching that link.\n{url}")
                        if self._debug_mode:
                            await self._send_text(context, chat_id, f"[debug] link_error type={type(exc).__name__} msg={exc}")
                return

        # --- QUERY ---
        if intent == INTENT_QUERY:
            contexts = self._memory.recall_context(text, k=5)
            profile_context = self._get_latest_profile()
            result = self._brains.route_query(text, contexts, profile_context=profile_context)
            await self._send_text(context, chat_id, result["answer"])
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
            "title": text[:80],
            "created_at": utc_now_iso(),
            "created_at_ts": utc_now_ts(),
            "tags": tags,
            "priority_score": 0.5,
        }
        self._memory.add_memory(text, metadata)
        await self._send_text(context, chat_id, self._store_confirmation(text))
        if self._debug_mode:
            await self._send_text(context, chat_id, f"[debug] intent=INGEST_TEXT type=text tags={tags}")

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

            for idx, chunk in enumerate(chunks):
                pdf_meta: dict[str, Any] = {
                    "source_type": "pdf",
                    "file_name": document.file_name,
                    "chunk_index": idx,
                    "created_at": utc_now_iso(),
                    "created_at_ts": utc_now_ts(),
                    "tags": pdf_tags,
                    "priority_score": 0.5,
                }
                self._memory.add_memory(chunk, pdf_meta)

            await self._send_text(
                context,
                chat_id=chat_id,
                text=f'Got it. I\'ve saved "{document.file_name}" into your memory ({len(chunks)} chunks). You can ask me about it anytime.',
            )
            if self._debug_mode:
                await self._send_text(context, chat_id, f"[debug] pdf_saved tags={pdf_tags}")
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

