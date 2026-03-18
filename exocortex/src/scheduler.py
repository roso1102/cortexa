from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    from groq import Groq
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
        groq_client: "Groq | None" = None,
        daily_digest_time: str = "",  # "HH:MM" UTC, empty = disabled
        resurface_time: str = "10:35",  # "HH:MM" UTC for daily resurfacing
        weekly_diary_time: str = "18:00",  # "HH:MM" UTC on Sundays for weekly diary
        poll_interval: int = 10,
    ) -> None:
        self._memory = memory
        self._reflection = reflection
        self._app = application
        self._owner_chat_id = owner_chat_id
        self._groq = groq_client
        self._daily_digest_time = daily_digest_time.strip()
        self._resurface_time = resurface_time.strip()
        self._weekly_diary_time = weekly_diary_time.strip()
        self._poll_interval = poll_interval
        self._digest_sent_date: str = ""
        self._resurface_sent_date: str = ""
        self._diary_sent_week: str = ""   # "YYYY-WW"
        self._tunnel_sent_week: str = ""  # "YYYY-WW"
        self._profile_sent_month: str = ""  # "YYYY-MM"
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

        # --- weekly tunnel formation (Sundays, same window as diary) ---
        if self._weekly_diary_time and self._groq:
            self._maybe_form_tunnels(now)

        # --- monthly personal profile (1st of month) ---
        if self._weekly_diary_time and self._owner_chat_id:
            self._maybe_send_profile(now)

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
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        today_str = now.strftime("%Y-%m-%d")
        current_hhmm = now.strftime("%H:%M")

        if current_hhmm != self._resurface_time:
            return
        if self._resurface_sent_date == today_str:
            return

        try:
            cutoff_ts = int((now - timedelta(days=7)).timestamp())  # production: change to days=7
            now_ts = int(now.timestamp())
            old_memories = self._memory.get_old_memories(
                older_than_ts=cutoff_ts,
                exclude_source_types=["reminder", "diary_entry", "tunnel", "profile_snapshot"],
                k=50,
            )

            if not old_memories:
                self._resurface_sent_date = today_str
                return

            # Score: semantic similarity + aging + priority_score boost; exclude recently resurfaced
            scored: List[tuple[float, dict]] = []
            for m in old_memories:
                created_ts = int(m.get("created_at_ts") or 0)
                last_accessed_ts = int(m.get("last_accessed_ts") or created_ts)
                last_resurfaced_ts = int(m.get("last_resurfaced_ts") or 0)
                priority = float(m.get("priority_score") or 0.5)
                score = float(m.get("score") or 0.5)

                # Skip recently resurfaced (within 3 days)
                days_since_resurface = (now_ts - last_resurfaced_ts) / 86400 if last_resurfaced_ts else 999
                if days_since_resurface < 3:
                    continue

                days_since = max(0, (now_ts - last_accessed_ts) / 86400)
                aging = min(days_since / 90, 1.0)

                resurface_score = (score * 0.4) + (aging * 0.4) + (priority * 0.2)
                scored.append((resurface_score, m))

            scored.sort(key=lambda x: x[0], reverse=True)
            top = [m for _, m in scored[:3]]

            if not top:
                self._resurface_sent_date = today_str
                return

            # Send header
            self._send_telegram(self._owner_chat_id, "Here are a few things from your memory worth revisiting:")

            # Send each memory as a separate message with inline feedback buttons
            for m in top:
                raw = str(m.get("raw_content") or "").strip()
                snippet = raw[:160].rstrip(".,;:!?")
                if len(raw) > 160:
                    snippet += "..."
                source_type = m.get("source_type", "note")
                title = m.get("title") or m.get("file_name") or ""
                mem_id = m.get("id") or m.get("created_at", "")

                label = f"[{source_type}] {title}" if title else f"[{source_type}] {snippet}"
                msg = f"{label}\n\n{snippet}"

                keyboard = InlineKeyboardMarkup([
                    [
                        InlineKeyboardButton("Still relevant", callback_data=f"fb:relevant:{mem_id}"),
                        InlineKeyboardButton("Not relevant", callback_data=f"fb:irrelevant:{mem_id}"),
                    ],
                    [
                        InlineKeyboardButton("Snooze 7 days", callback_data=f"fb:snooze:{mem_id}"),
                    ],
                ])
                self._send_telegram(self._owner_chat_id, msg, reply_markup=keyboard)

                # Mark as resurfaced
                try:
                    self._memory.update_memory_metadata(mem_id, {"last_resurfaced_ts": now_ts})
                except Exception:
                    pass

            self._resurface_sent_date = today_str
            logger.info("Resurfacing sent for %s (%d items)", today_str, len(top))
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

    # ------------------------------------------------------------------
    # Weekly tunnel formation
    # ------------------------------------------------------------------

    def _maybe_form_tunnels(self, now: datetime) -> None:
        """Run tunnel formation once per week on Sundays."""
        if now.weekday() != 6:
            return

        current_hhmm = now.strftime("%H:%M")
        # Run tunnels 30 minutes after the weekly diary to avoid API contention
        target_h, target_m = map(int, self._weekly_diary_time.split(":"))
        tunnel_m = (target_m + 30) % 60
        tunnel_h = target_h + (1 if target_m + 30 >= 60 else 0)
        tunnel_time = f"{tunnel_h:02d}:{tunnel_m:02d}"

        if current_hhmm != tunnel_time:
            return

        week_str = now.strftime("%Y-W%W")
        if self._tunnel_sent_week == week_str:
            return

        try:
            from src.tunnels import form_tunnels
            tunnels = form_tunnels(self._memory, self._groq)  # type: ignore[arg-type]
            if tunnels:
                names = ", ".join(t.get("tunnel_name", "?") for t in tunnels[:5])
                self._send_telegram(
                    self._owner_chat_id,
                    f"Weekly tunnels updated. Found {len(tunnels)} theme(s): {names}",
                )
            self._tunnel_sent_week = week_str
            logger.info("Tunnel formation completed for week %s: %d tunnels", week_str, len(tunnels))
        except Exception:
            logger.exception("Tunnel formation failed")

    # ------------------------------------------------------------------
    # Monthly personal profile
    # ------------------------------------------------------------------

    def _maybe_send_profile(self, now: datetime) -> None:
        """Generate and send a personal profile snapshot on the 1st of each month."""
        if now.day != 1:
            return

        current_hhmm = now.strftime("%H:%M")
        # Fire at the same time as the daily digest, or 09:00 UTC if digest disabled
        target_time = self._daily_digest_time or "09:00"
        if current_hhmm != target_time:
            return

        month_str = now.strftime("%Y-%m")
        if self._profile_sent_month == month_str:
            return

        try:
            profile = self._reflection.generate_profile_snapshot()
            if profile:
                self._send_telegram(self._owner_chat_id, f"Monthly profile:\n\n{profile}")
            self._profile_sent_month = month_str
            logger.info("Monthly profile sent for %s", month_str)
        except Exception:
            logger.exception("Failed to send monthly profile")

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Called from the main async context once the event loop is running."""
        self._loop = loop

    # ------------------------------------------------------------------
    # Thread-safe Telegram send
    # ------------------------------------------------------------------

    def _send_telegram(self, chat_id: int, text: str, reply_markup: Any = None) -> None:
        """
        The scheduler runs in a plain Python thread. The Telegram Application
        owns the asyncio event loop in the main thread (started by run_polling).
        We submit a coroutine to that loop via run_coroutine_threadsafe.
        The loop is injected via set_event_loop() from a post_init hook.
        """
        loop: asyncio.AbstractEventLoop | None = getattr(self, "_loop", None)

        async def _do_send() -> None:
            await self._app.bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_markup=reply_markup,
            )

        if loop and loop.is_running():
            future = asyncio.run_coroutine_threadsafe(_do_send(), loop)
            try:
                future.result(timeout=15)
            except Exception:
                logger.exception("Telegram send failed from scheduler thread")
        else:
            logger.warning("Scheduler: event loop not ready, skipping send to chat_id=%d", chat_id)
