from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    from telegram.ext import Application

    from src.memory import MemoryManager
    from src.reflection import ReflectionService

logger = logging.getLogger(__name__)


class ReminderScheduler:
    """
    Background thread that:
    1. Every 10 s — queries Pinecone for due reminders and fires Telegram messages.
    2. Once per day at DAILY_DIGEST_TIME — sends the daily summary to the owner.
    3. Once per week (Sunday) — generates and sends a weekly diary narrative.
    4. Once per day — proactively resurfaces relevant older memories.
    """

    def __init__(
        self,
        memory: "MemoryManager",
        reflection: "ReflectionService",
        application: "Application",  # type: ignore[type-arg]
        owner_chat_id: int,
        daily_digest_time: str = "",  # "HH:MM" UTC, empty = disabled
        resurface_time: str = "08:00",  # "HH:MM" UTC for daily resurfacing
        weekly_diary_time: str = "20:00",  # "HH:MM" UTC on Sundays for weekly diary
        poll_interval: int = 10,
    ) -> None:
        self._memory = memory
        self._reflection = reflection
        self._app = application
        self._owner_chat_id = owner_chat_id
        self._daily_digest_time = daily_digest_time.strip()
        self._resurface_time = resurface_time.strip()
        self._weekly_diary_time = weekly_diary_time.strip()
        self._poll_interval = poll_interval
        self._digest_sent_date: str = ""
        self._resurface_sent_date: str = ""
        self._diary_sent_week: str = ""  # "YYYY-WW" tracks which week diary was sent
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="ReminderScheduler")
        self._thread.start()
        logger.info("ReminderScheduler started (interval=%ds)", self._poll_interval)

    # ------------------------------------------------------------------
    # Private loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while True:
            try:
                self._tick()
            except Exception:
                logger.exception("ReminderScheduler tick failed")
            time.sleep(self._poll_interval)

    def _tick(self) -> None:
        now = datetime.now(timezone.utc)
        now_ts = int(now.timestamp())

        # --- fire due reminders ---
        self._fire_due_reminders(now_ts)

        # --- optional daily digest ---
        if self._daily_digest_time:
            self._maybe_send_daily_digest(now)

        # --- proactive resurfacing (daily) ---
        if self._resurface_time and self._owner_chat_id:
            self._maybe_resurface(now)

        # --- weekly diary (Sundays) ---
        if self._weekly_diary_time and self._owner_chat_id:
            self._maybe_send_weekly_diary(now)

    # ------------------------------------------------------------------
    # Reminder firing
    # ------------------------------------------------------------------

    def _fire_due_reminders(self, now_ts: int) -> None:
        try:
            due = self._memory.query_by_filter(
                query_text="reminder",
                filter_obj={
                    "source_type": {"$eq": "reminder"},
                    "due_at_ts": {"$lte": now_ts},
                    "fired": {"$eq": False},
                },
                k=20,
            )
        except Exception:
            logger.exception("Failed to query due reminders")
            return

        for reminder_md in due:
            self._dispatch_reminder(reminder_md)

    def _dispatch_reminder(self, reminder_md: dict[str, Any]) -> None:
        memory_id = reminder_md.get("id") or reminder_md.get("created_at", "")
        text = reminder_md.get("raw_content") or reminder_md.get("reminder_text") or "Reminder"
        chat_id_raw = reminder_md.get("chat_id")

        # Fall back to owner if no chat_id stored on the reminder
        try:
            chat_id = int(chat_id_raw) if chat_id_raw else self._owner_chat_id
        except (ValueError, TypeError):
            chat_id = self._owner_chat_id

        message = f"Reminder: {text}"

        # Send via Telegram (thread-safe async dispatch)
        self._send_telegram(chat_id, message)

        # Mark as fired in Pinecone (upsert with fired=True)
        if memory_id:
            try:
                updated_md = {**reminder_md, "fired": True}
                # remove score key — not a Pinecone metadata field
                updated_md.pop("score", None)
                raw = updated_md.get("raw_content", text)
                self._memory.add_memory(raw, {**updated_md, "id": memory_id})
                logger.info("Reminder fired and marked: id=%s", memory_id)
            except Exception:
                logger.exception("Failed to mark reminder as fired: id=%s", memory_id)

    # ------------------------------------------------------------------
    # Daily digest
    # ------------------------------------------------------------------

    def _maybe_send_daily_digest(self, now: datetime) -> None:
        today_str = now.strftime("%Y-%m-%d")
        current_hhmm = now.strftime("%H:%M")

        if current_hhmm != self._daily_digest_time:
            return
        if self._digest_sent_date == today_str:
            return  # already sent today

        try:
            summary = self._reflection.summarize_today()
            self._send_telegram(self._owner_chat_id, f"Daily digest:\n\n{summary}")
            self._digest_sent_date = today_str
            logger.info("Daily digest sent for %s", today_str)
        except Exception:
            logger.exception("Failed to send daily digest")

    # ------------------------------------------------------------------
    # Proactive resurfacing
    # ------------------------------------------------------------------

    def _maybe_resurface(self, now: datetime) -> None:
        today_str = now.strftime("%Y-%m-%d")
        current_hhmm = now.strftime("%H:%M")

        if current_hhmm != self._resurface_time:
            return
        if self._resurface_sent_date == today_str:
            return

        try:
            # Find memories older than 7 days (excluding reminders/diary)
            cutoff_ts = int((now - timedelta(days=7)).timestamp())
            old_memories = self._memory.get_old_memories(
                older_than_ts=cutoff_ts,
                exclude_source_types=["reminder", "diary_entry"],
                k=50,
            )

            if not old_memories:
                self._resurface_sent_date = today_str
                return

            # Score by a blend of recency (older = higher urgency) and semantic score
            scored: List[tuple[float, dict]] = []
            for m in old_memories:
                created_ts = int(m.get("created_at_ts") or 0)
                last_accessed_ts = int(m.get("last_accessed_ts") or created_ts)
                score = float(m.get("score") or 0.5)

                # Aging factor: days since last access (normalized, max 90 days)
                days_since = max(0, (now.timestamp() - last_accessed_ts) / 86400)
                aging = min(days_since / 90, 1.0)

                resurface_score = (score * 0.5) + (aging * 0.5)
                scored.append((resurface_score, m))

            scored.sort(key=lambda x: x[0], reverse=True)
            top = [m for _, m in scored[:3]]

            lines = ["Here are a few things from your memory worth revisiting:\n"]
            for i, m in enumerate(top, 1):
                raw = str(m.get("raw_content") or "").strip()
                snippet = raw[:120].rstrip(".,;:!?")
                if len(raw) > 120:
                    snippet += "..."
                source_type = m.get("source_type", "note")
                title = m.get("title") or m.get("file_name") or ""
                label = f'"{title}"' if title and title != snippet else f'"{snippet}"'
                lines.append(f"{i}. [{source_type}] {label}")

            self._send_telegram(self._owner_chat_id, "\n".join(lines))
            self._resurface_sent_date = today_str
            logger.info("Resurfacing sent for %s", today_str)
        except Exception:
            logger.exception("Failed to send resurfacing")

    # ------------------------------------------------------------------
    # Weekly diary
    # ------------------------------------------------------------------

    def _maybe_send_weekly_diary(self, now: datetime) -> None:
        # Only fire on Sundays (weekday 6)
        if now.weekday() != 6:
            return

        current_hhmm = now.strftime("%H:%M")
        if current_hhmm != self._weekly_diary_time:
            return

        week_str = now.strftime("%Y-W%W")
        if self._diary_sent_week == week_str:
            return

        try:
            diary = self._reflection.generate_weekly_diary()
            if diary:
                self._send_telegram(self._owner_chat_id, f"Weekly diary:\n\n{diary}")
                self._diary_sent_week = week_str
                logger.info("Weekly diary sent for week %s", week_str)
        except Exception:
            logger.exception("Failed to send weekly diary")

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Called from the main async context once the event loop is running."""
        self._loop = loop

    # ------------------------------------------------------------------
    # Thread-safe Telegram send
    # ------------------------------------------------------------------

    def _send_telegram(self, chat_id: int, text: str) -> None:
        """
        The scheduler runs in a plain Python thread. The Telegram Application
        owns the asyncio event loop in the main thread (started by run_polling).
        We submit a coroutine to that loop via run_coroutine_threadsafe.
        The loop is injected via set_event_loop() from a post_init hook.
        """
        loop: asyncio.AbstractEventLoop | None = getattr(self, "_loop", None)

        async def _do_send() -> None:
            await self._app.bot.send_message(chat_id=chat_id, text=text)

        if loop and loop.is_running():
            future = asyncio.run_coroutine_threadsafe(_do_send(), loop)
            try:
                future.result(timeout=15)
            except Exception:
                logger.exception("Telegram send failed from scheduler thread")
        else:
            logger.warning("Scheduler: event loop not ready, skipping send to chat_id=%d", chat_id)
