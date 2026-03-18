from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from groq import Groq

from src.config import AppConfig
from src.memory import MemoryManager
from src.utils import utc_now_iso, utc_now_ts


class ReflectionService:
    def __init__(self, config: AppConfig, memory: MemoryManager) -> None:
        self._memory = memory
        self._groq = Groq(api_key=config.groq_api_key)

    def summarize_today(self) -> str:
        """
        Fetch today's memories and return a rich reflection:
        - Source-type breakdown (notes / links / PDFs)
        - Upcoming reminders (next 24 h)
        - 3–5 bullet LLM summary
        """
        now = datetime.now(timezone.utc)
        today = now.date()

        contexts: List[Dict[str, Any]] = self._memory.recall_context("today's focus", k=50)

        todays_texts: List[str] = []
        type_counter: Counter = Counter()
        tag_counter: Counter = Counter()

        for m in contexts:
            created_at = m.get("created_at")
            raw = m.get("raw_content")
            source_type = m.get("source_type", "text")
            if not created_at or not raw:
                continue
            try:
                dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except Exception:
                continue
            if dt.date() == today:
                todays_texts.append(raw)
                type_counter[source_type] += 1
                for tag in (m.get("tags") or []):
                    tag_counter[str(tag)] += 1

        # --- Upcoming reminders (next 24 h) ---
        upcoming_reminders = self._get_upcoming_reminders(now)

        if not todays_texts and not upcoming_reminders:
            return "Nothing was captured today and no upcoming reminders."

        # Build header with counts
        header_parts: list[str] = []
        if todays_texts:
            breakdown = ", ".join(f"{count} {stype}(s)" for stype, count in sorted(type_counter.items()))
            header_parts.append(f"Today you captured {len(todays_texts)} item(s): {breakdown}.")

            # Topic distribution from tags
            if tag_counter:
                total_tags = sum(tag_counter.values())
                top_topics = tag_counter.most_common(4)
                topic_parts = []
                for tag, count in top_topics:
                    pct = round(count / total_tags * 100)
                    topic_parts.append(f"{tag.title()} {pct}%")
                header_parts.append("Topics: " + " | ".join(topic_parts))
        if upcoming_reminders:
            reminder_lines = "\n".join(f"  - {r}" for r in upcoming_reminders[:5])
            header_parts.append(f"Upcoming reminders (next 24h):\n{reminder_lines}")

        header = "\n".join(header_parts)

        if not todays_texts:
            return header

        # --- LLM summary ---
        joined = "\n\n---\n\n".join(todays_texts[:20])
        prompt = (
            "You are Exocortex, a personal cognitive assistant writing a daily reflection directly to the user.\n\n"
            "Memories from today:\n"
            f"{joined}\n\n"
            "Write 3–5 concise bullet points summarizing what the user focused on today. "
            "Address the user directly using 'you' and 'your' — never say 'the user' or 'they'. "
            "Be specific, not generic. Example: 'You explored peanut allergy prevention...' not 'The user explored...'"
        )

        try:
            chat = self._groq.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.3,
            )
            llm_summary = chat.choices[0].message.content or "Unable to summarize."
        except Exception:
            llm_summary = "(LLM summary unavailable)"

        return f"{header}\n\n{llm_summary}"

    def generate_weekly_diary(self) -> str:
        """
        Generate a narrative diary entry for the past 7 days.
        Stored as source_type=diary_entry and returned as a string for Telegram delivery.
        """
        now = datetime.now(timezone.utc)
        week_start_ts = int((now - timedelta(days=7)).timestamp())

        # Fetch memories from the past week
        try:
            memories = self._memory.query_by_filter(
                query_text="weekly summary thoughts ideas notes links",
                filter_obj={
                    "created_at_ts": {"$gte": week_start_ts},
                    "source_type": {"$nin": ["reminder", "diary_entry"]},
                },
                k=60,
            )
        except Exception:
            return ""

        if not memories:
            return "Nothing was captured this week."

        # Count by type and gather all tags for topic breakdown
        type_counter: Counter = Counter()
        tag_counter: Counter = Counter()
        texts: List[str] = []
        for m in memories:
            raw = m.get("raw_content", "")
            if raw:
                texts.append(str(raw)[:300])
            type_counter[m.get("source_type", "text")] += 1
            for tag in (m.get("tags") or []):
                tag_counter[str(tag)] += 1

        type_summary = ", ".join(f"{count} {stype}(s)" for stype, count in sorted(type_counter.items()))
        top_tags = [tag for tag, _ in tag_counter.most_common(5)]
        tag_line = ", ".join(top_tags) if top_tags else "mixed topics"

        joined = "\n\n---\n\n".join(texts[:30])
        prompt = (
            "You are Exocortex, writing a personal weekly diary entry for the user.\n\n"
            f"This week ({now.strftime('%B %d, %Y')}) the user captured {len(memories)} item(s): {type_summary}.\n"
            f"Main topics: {tag_line}.\n\n"
            "Memories from this week:\n"
            f"{joined}\n\n"
            "Write a warm, reflective diary entry (3-5 sentences) summarizing the week. "
            "Write directly to the user using 'you' and 'your'. "
            "Mention specific topics, patterns, or interesting things you noticed. "
            "End with a short encouraging line about what to focus on next."
        )

        try:
            chat = self._groq.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=350,
                temperature=0.5,
            )
            diary_text = chat.choices[0].message.content or ""
        except Exception:
            return "(Weekly diary generation failed.)"

        # Store diary entry in Pinecone for future reference
        try:
            diary_metadata: Dict[str, Any] = {
                "source_type": "diary_entry",
                "period": "weekly",
                "week_ending": now.strftime("%Y-%m-%d"),
                "created_at": utc_now_iso(),
                "created_at_ts": utc_now_ts(),
                "tags": ["diary", "weekly_reflection"],
            }
            self._memory.add_memory(diary_text, diary_metadata)
        except Exception:
            pass  # storing diary is best-effort

        return diary_text

    def _get_upcoming_reminders(self, now: datetime) -> List[str]:
        """Return reminder texts due within the next 24 hours."""
        now_ts = int(now.timestamp())
        end_ts = int((now + timedelta(hours=24)).timestamp())
        try:
            matches = self._memory.query_by_filter(
                query_text="reminder",
                filter_obj={
                    "source_type": {"$eq": "reminder"},
                    "due_at_ts": {"$gte": now_ts, "$lte": end_ts},
                    "fired": {"$eq": False},
                },
                k=10,
            )
        except Exception:
            return []

        result: List[str] = []
        for m in matches:
            text = m.get("raw_content") or m.get("reminder_text") or ""
            due = m.get("due_at", "")
            if text:
                label = f"{text} (due: {due})" if due else text
                result.append(label)
        return result
