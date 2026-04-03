from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
import re
from typing import Any, Dict, List

from groq import Groq
from openai import OpenAI

from src.config import AppConfig
from src.memory import MemoryManager
from src.utils import ist_day_range_utc_ts, ist_now, utc_now_iso, utc_now_ts


class ReflectionService:
    def __init__(self, config: AppConfig, memory: MemoryManager) -> None:
        self._memory = memory
        self._groq = Groq(api_key=config.groq_api_key)
        self._openrouter_api_key = (config.openrouter_api_key or "").strip()
        self._openrouter = (
            OpenAI(
                api_key=self._openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
            )
            if self._openrouter_api_key
            else None
        )

    def _clean_summary_text(self, text: str) -> str:
        """
        Dashboard-safe plain text cleanup:
        - strips markdown emphasis/code/header markers
        - keeps simple one-line bullets
        """
        t = (text or "").strip()
        if not t:
            return ""
        t = re.sub(r"[*_`#>]+", "", t)
        parts: List[str] = []
        for ln in t.splitlines():
            s = re.sub(r"\s+", " ", ln).strip()
            if not s:
                continue
            # normalize common list prefixes from model output
            s = re.sub(r"^[-•\d\.\)\s]+", "", s).strip()
            if not s:
                continue
            parts.append(s)
        return "\n".join(parts[:6])

    def _memory_clue(self, m: Dict[str, Any]) -> str:
        title = str(m.get("title") or "").strip()
        raw = str(m.get("raw_content") or "").strip()
        if title:
            return title[:100]
        first = next((ln.strip() for ln in raw.splitlines() if ln.strip()), "")
        if first.lower().startswith("title:"):
            first = first.split(":", 1)[1].strip()
        return re.sub(r"\s+", " ", first)[:100] if first else "a saved memory"

    def _deterministic_topic_lines(self, contexts: List[Dict[str, Any]], tag_counter: Counter) -> str:
        lines: List[str] = []
        top_topics = [str(t).strip() for t, _ in tag_counter.most_common(5) if str(t).strip()]
        used_ids: set[str] = set()
        for topic in top_topics:
            topic_l = topic.lower()
            picked: Dict[str, Any] | None = None
            for m in contexts:
                mid = str(m.get("id") or "")
                if mid and mid in used_ids:
                    continue
                tags = " ".join(str(t).lower() for t in (m.get("tags") or []))
                raw = str(m.get("raw_content") or "").lower()
                title = str(m.get("title") or "").lower()
                if topic_l in tags or topic_l in raw or topic_l in title:
                    picked = m
                    if mid:
                        used_ids.add(mid)
                    break
            if picked is None:
                continue
            clue = self._memory_clue(picked)
            lines.append(f"Around {topic}, you saved {clue}.")
            if len(lines) >= 5:
                break

        if not lines:
            for m in contexts[:4]:
                clue = self._memory_clue(m)
                if clue:
                    lines.append(f"You saved {clue}.")
        return " ".join(lines[:6])

    def _topic_lines_summary(
        self,
        *,
        contexts: List[Dict[str, Any]],
        todays_texts: List[str],
        tag_counter: Counter,
    ) -> str:
        """
        Build a personal, introspective summary with one concise line per topic.
        OpenRouter first, Groq fallback, deterministic fallback last.
        """
        top_topics = [str(t) for t, _ in tag_counter.most_common(5) if str(t).strip()]
        if not top_topics:
            top_topics = ["today's main ideas"]
        # Build compact, per-memory snippets so long single memories don't dominate.
        clue_items: List[str] = []
        for idx, m in enumerate(contexts[:14], start=1):
            clue = self._memory_clue(m)
            if not clue:
                continue
            st = str(m.get("source_type") or "memory")
            clue_items.append(f"{idx}. [{st}] {clue}")
        joined = "\n".join(clue_items) if clue_items else "\n\n---\n\n".join(todays_texts[:10])
        topic_block = ", ".join(top_topics)
        prompt = (
            "You are Exocortex, writing a personal end-of-day diary reflection for the user.\n"
            "Write one short diary-style paragraph (4-7 sentences), human and intuitive.\n"
            "Use second person (you/your), introspective tone, concise and specific.\n"
            "You MUST reference several different saved items from the day (not just one).\n"
            "Use concrete wording from saved content (titles/phrases), avoid vague abstraction.\n"
            "Output plain text only. No markdown emphasis, no headings, no bullets, no numbering.\n\n"
            f"Top topics today: {topic_block}\n\n"
            "Saved items from today:\n"
            f"{joined}\n\n"
            "Write as a diary note addressed to the user."
        )

        # 1) OpenRouter first (better reflective style consistency for this task)
        if self._openrouter is not None:
            try:
                chat = self._openrouter.chat.completions.create(
                    model="minimax/minimax-01",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=420,
                    temperature=0.45,
                )
                content = (chat.choices[0].message.content or "").strip()
                if content:
                    cleaned = self._clean_summary_text(content)
                    if cleaned:
                        return re.sub(r"\s*\n+\s*", " ", cleaned).strip()
            except Exception:
                pass

        # 2) Groq fallback
        try:
            chat = self._groq.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=360,
                temperature=0.35,
            )
            content = (chat.choices[0].message.content or "").strip()
            if content:
                cleaned = self._clean_summary_text(content)
                if cleaned:
                    return re.sub(r"\s*\n+\s*", " ", cleaned).strip()
        except Exception:
            pass

        # 3) Deterministic fallback: concrete and grounded in actual saves.
        return self._deterministic_topic_lines(contexts, tag_counter)

    def summarize_today(self) -> str:
        """
        Fetch today's memories and return a rich reflection:
        - Source-type breakdown (notes / links / PDFs)
        - Upcoming reminders (next 24 h)
        - 3–5 bullet LLM summary
        """
        now_ist = ist_now()
        start_ts, end_ts = ist_day_range_utc_ts(now_ist)
        # Best-effort: owner/global view uses unscoped day-range query.
        contexts: List[Dict[str, Any]] = self._memory.query_by_filter(
            query_text="today memories",
            filter_obj={"created_at_ts": {"$gte": start_ts, "$lt": end_ts}, "archived": {"$ne": True}},
            k=200,
        )
        contexts = [m for m in contexts if self._memory.is_main_memory(m)]

        todays_texts: List[str] = []
        type_counter: Counter = Counter()
        tag_counter: Counter = Counter()

        for m in contexts:
            raw = m.get("raw_content")
            source_type = m.get("source_type", "text")
            if not raw:
                continue
            todays_texts.append(raw)
            type_counter[source_type] += 1
            for tag in (m.get("tags") or []):
                tag_counter[str(tag)] += 1

        # --- Upcoming reminders (next 24 h) ---
        upcoming_reminders = self._get_upcoming_reminders(datetime.now(timezone.utc))

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

        llm_summary = self._topic_lines_summary(contexts=contexts, todays_texts=todays_texts, tag_counter=tag_counter)

        return f"{header}\n\n{llm_summary}"

    def summarize_today_for_user(self, user_id: int) -> str:
        """
        Per-user variant of summarize_today, scoped by user_id.
        Used by the dashboard so each user sees only their own day.
        """
        now_ist = ist_now()
        start_ts, end_ts = ist_day_range_utc_ts(now_ist)
        # Deterministic selection: avoid Pinecone similarity/top_k misses for "created today".
        contexts: List[Dict[str, Any]] = []
        try:
            from src.db import fetch_memories_for_user_created_range

            contexts = fetch_memories_for_user_created_range(
                user_id=user_id,
                start_ts=start_ts,
                end_ts=end_ts,
                limit=400,
            )
            contexts = [m for m in contexts if self._memory.is_main_memory(m)]
        except Exception:
            # Fallback: keep system functional if Postgres isn't configured yet.
            contexts = self._memory.query_by_filter_for_chat(
                query_text="today memories",
                chat_id=user_id,
                filter_obj={"created_at_ts": {"$gte": start_ts, "$lt": end_ts}, "archived": {"$ne": True}},
                k=400,
            )
            contexts = [m for m in contexts if self._memory.is_main_memory(m)]

        todays_texts: List[str] = []
        type_counter: Counter = Counter()
        tag_counter: Counter = Counter()

        for m in contexts:
            raw = m.get("raw_content")
            source_type = m.get("source_type", "text")
            if not raw:
                continue
            todays_texts.append(raw)
            type_counter[source_type] += 1
            for tag in (m.get("tags") or []):
                tag_counter[str(tag)] += 1

        # --- Upcoming reminders (next 24 h) ---
        upcoming_reminders = self._get_upcoming_reminders(datetime.now(timezone.utc), user_id=user_id)

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

        llm_summary = self._topic_lines_summary(contexts=contexts, todays_texts=todays_texts, tag_counter=tag_counter)

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

    def generate_weekly_diary_for_user(self, user_id: int) -> str:
        """
        User-scoped weekly diary (Option B canonical correctness).
        Uses Postgres canonical main memories instead of Pinecone similarity/top_k.
        Stored as source_type=diary_entry (user-scoped) for later reasoning.
        """
        from src.db import fetch_memories_for_user_created_range
        from src.utils import ist_now

        now_ist = ist_now()
        start_ist = now_ist - timedelta(days=7)
        start_ts = int(start_ist.astimezone(timezone.utc).timestamp())
        end_ts = int(now_ist.astimezone(timezone.utc).timestamp())

        # Pull canonical main memories deterministically from Postgres.
        try:
            memories = fetch_memories_for_user_created_range(
                user_id=user_id,
                start_ts=start_ts,
                end_ts=end_ts,
                limit=120,
            )
        except Exception:
            memories = []

        if not memories:
            return "Nothing was captured this week."

        type_counter: Counter = Counter()
        tag_counter: Counter = Counter()
        texts: List[str] = []
        for m in memories:
            raw = m.get("raw_content") or ""
            if raw:
                texts.append(str(raw)[:300])
            type_counter[str(m.get("source_type", "text"))] += 1
            for tag in (m.get("tags") or []):
                tag_counter[str(tag)] += 1

        now_utc = datetime.now(timezone.utc)
        type_summary = ", ".join(f"{count} {stype}(s)" for stype, count in sorted(type_counter.items()))
        top_tags = [tag for tag, _ in tag_counter.most_common(5)]
        tag_line = ", ".join(top_tags) if top_tags else "mixed topics"

        joined = "\n\n---\n\n".join(texts[:30])
        prompt = (
            "You are Exocortex, writing a personal weekly diary entry for the user.\n\n"
            f"This week ({now_utc.strftime('%B %d, %Y')}) you captured {len(memories)} item(s): {type_summary}.\n"
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

        # Store diary entry in Pinecone for continuity (best-effort).
        try:
            diary_metadata: Dict[str, Any] = {
                "source_type": "diary_entry",
                "period": "weekly",
                "week_ending": now_utc.strftime("%Y-%m-%d"),
                "created_at": utc_now_iso(),
                "created_at_ts": utc_now_ts(),
                "tags": ["diary", "weekly_reflection"],
                "chat_id": user_id,
                "user_id": user_id,
                "priority_score": 0.5,
            }
            self._memory.add_memory(diary_text, diary_metadata)
        except Exception:
            pass

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

        # Store as profile_snapshot (owner/global variant)
        try:
            self._memory.add_memory(
                full_profile,
                {
                    "source_type": "profile_snapshot",
                    "period": "monthly",
                    "month": now.strftime("%Y-%m"),
                    "created_at": utc_now_iso(),
                    "created_at_ts": utc_now_ts(),
                    "tags": ["profile", "monthly_snapshot"],
                },
            )
        except Exception:
            pass

        return full_profile

    def generate_profile_snapshot_for_user(self, user_id: int) -> str:
        """
        Per-user profile snapshot used by the dashboard.
        """
        now = datetime.now(timezone.utc)

        try:
            from src.db import fetch_main_memories_for_user_for_profile

            # Canonical selection from Postgres:
            # - deterministic
            # - main records only (no chunk/tunnel leakage)
            all_memories = fetch_main_memories_for_user_for_profile(
                user_id=user_id,
                exclude_source_types=["reminder", "diary_entry", "profile_snapshot", "tunnel"],
                limit=500,
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
            if oldest_ts != float("inf")
            else "N/A"
        )
        newest_label = (
            datetime.fromtimestamp(newest_ts, tz=timezone.utc).strftime("%b %Y")
            if newest_ts
            else "N/A"
        )

        profile_header = (
            f"Memory span: {oldest_label} → {newest_label}\n"
            f"Total memories: {len(all_memories)}\n"
            f"Active tunnels: {tunnel_line}\n"
            f"Top topics: {tag_line}\n"
            f"Capture pattern: {time_pattern}"
        )

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

        # Store as profile_snapshot for this user
        try:
            self._memory.add_memory(
                full_profile,
                {
                    "source_type": "profile_snapshot",
                    "period": "monthly",
                    "month": now.strftime("%Y-%m"),
                    "created_at": utc_now_iso(),
                    "created_at_ts": utc_now_ts(),
                    "tags": ["profile", "monthly_snapshot"],
                    "user_id": user_id,
                },
            )
        except Exception:
            pass

        return full_profile

    def _get_upcoming_reminders(self, now: datetime, user_id: int | None = None) -> List[str]:
        """Return reminder texts due within the next 24 hours."""
        now_ts = int(now.timestamp())
        end_ts = int((now + timedelta(hours=24)).timestamp())
        try:
            if user_id is not None:
                matches = self._memory.query_by_filter_for_chat(
                    query_text="reminder",
                    chat_id=user_id,
                    filter_obj={
                        "source_type": {"$eq": "reminder"},
                        "due_at_ts": {"$gte": now_ts, "$lte": end_ts},
                        "fired": {"$eq": False},
                    },
                    k=10,
                )
            else:
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
