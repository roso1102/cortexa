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

    def generate_profile_snapshot(self) -> str:
        """
        Monthly personal profile: top tunnels, dominant topics, time-of-day patterns,
        and oldest vs newest interests.
        Stored as source_type=profile_snapshot and returned as a string.
        """
        now = datetime.now(timezone.utc)
        month_start_ts = int((now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)).timestamp())

        try:
            all_memories = self._memory.fetch_all_memories(
                exclude_source_types=["reminder", "diary_entry", "profile_snapshot"],
                k=200,
            )
        except Exception:
            return "(Profile generation failed: could not fetch memories.)"

        if not all_memories:
            return "Not enough memories to build a profile yet."

        tag_counter: Counter = Counter()
        tunnel_counter: Counter = Counter()
        hour_counter: Counter = Counter()
        oldest_ts = float("inf")
        newest_ts = 0.0

        for m in all_memories:
            for tag in (m.get("tags") or []):
                tag_counter[str(tag)] += 1
            tunnel_name = m.get("tunnel_name")
            if tunnel_name:
                tunnel_counter[str(tunnel_name)] += 1
            created_ts = float(m.get("created_at_ts") or 0)
            if created_ts:
                oldest_ts = min(oldest_ts, created_ts)
                newest_ts = max(newest_ts, created_ts)
                hour = datetime.fromtimestamp(created_ts, tz=timezone.utc).hour
                hour_counter[hour] += 1

        # Top topics
        top_tags = tag_counter.most_common(6)
        tag_line = ", ".join(f"{tag} ({cnt})" for tag, cnt in top_tags) if top_tags else "none yet"

        # Top tunnels
        top_tunnels = tunnel_counter.most_common(4)
        tunnel_line = ", ".join(f'"{t}"' for t, _ in top_tunnels) if top_tunnels else "no tunnels formed yet"

        # Time-of-day pattern
        if hour_counter:
            peak_hour = hour_counter.most_common(1)[0][0]
            if 5 <= peak_hour < 12:
                time_pattern = f"morning person (peak ~{peak_hour}:00 UTC)"
            elif 12 <= peak_hour < 17:
                time_pattern = f"afternoon thinker (peak ~{peak_hour}:00 UTC)"
            else:
                time_pattern = f"evening/night thinker (peak ~{peak_hour}:00 UTC)"
        else:
            time_pattern = "unknown"

        oldest_label = (
            datetime.fromtimestamp(oldest_ts, tz=timezone.utc).strftime("%b %Y")
            if oldest_ts != float("inf") else "N/A"
        )
        newest_label = (
            datetime.fromtimestamp(newest_ts, tz=timezone.utc).strftime("%b %Y")
            if newest_ts else "N/A"
        )

        profile_header = (
            f"Memory span: {oldest_label} → {newest_label}\n"
            f"Total memories: {len(all_memories)}\n"
            f"Active tunnels: {tunnel_line}\n"
            f"Top topics: {tag_line}\n"
            f"Capture pattern: {time_pattern}"
        )

        # LLM narrative
        context_snippets = "\n".join(
            f"- {str(m.get('raw_content', ''))[:150]}" for m in all_memories[:20]
        )
        prompt = (
            "You are Exocortex, writing a monthly personal profile for the user.\n\n"
            f"Profile data:\n{profile_header}\n\n"
            f"Sample memories:\n{context_snippets}\n\n"
            "Write a concise 3-4 sentence profile that tells the user who they are intellectually — "
            "what they care about, how they think, and what's evolving in their interests. "
            "Write directly to the user using 'you' and 'your'. Be specific and warm."
        )
        try:
            chat = self._groq.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.4,
            )
            narrative = chat.choices[0].message.content or ""
        except Exception:
            narrative = "(LLM narrative unavailable)"

        full_profile = f"{profile_header}\n\n{narrative}"

        # Store as profile_snapshot
        try:
            self._memory.add_memory(full_profile, {
                "source_type": "profile_snapshot",
                "period": "monthly",
                "month": now.strftime("%Y-%m"),
                "created_at": utc_now_iso(),
                "created_at_ts": utc_now_ts(),
                "tags": ["profile", "monthly_snapshot"],
            })
        except Exception:
            pass

        return full_profile

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
