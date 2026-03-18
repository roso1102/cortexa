from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from dateutil import parser as date_parser
from groq import Groq

from src.utils import utc_now_iso, utc_now_ts

logger = logging.getLogger(__name__)

_PARSE_SCHEMA = """\
You are a reminder parser. Given a user message, extract the reminder details and respond ONLY with a JSON object.

The JSON must have exactly these two keys:
- reminder_text: what the user wants to be reminded about
- due_iso: the due date/time as an ISO 8601 UTC string, e.g. 2026-03-20T09:00:00Z

Rules:
- Interpret relative times like "tomorrow", "in 2 hours", "next Monday" relative to today's UTC date/time provided below.
- If no time is given, default to 09:00 UTC on the specified day.
- If no date is specified but a time is, use today.
- Ignore typos and casual language (e.g. "tmrrw" = tomorrow, "remidn" = remind).
- Only return raw JSON. No prose, no markdown.

Today (UTC): {now_iso}
"""


@dataclass
class Reminder:
    id: str
    text: str
    due_at_iso: str
    timezone: str


def parse_reminder_llm(text: str, groq_client: Groq, user_tz: str = "UTC") -> Optional[Reminder]:
    """
    LLM-based reminder parser. Handles natural language like:
    - "remind me tomorrow to check the ASCON paper"
    - "remidn me in 2 hrs to call dad"
    Falls back to the keyword parser on failure.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    prompt = _PARSE_SCHEMA.format(now_iso=now_iso)

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
            max_tokens=100,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = (response.choices[0].message.content or "{}").strip()
        data = json.loads(raw)
        reminder_text = str(data.get("reminder_text") or "").strip()
        due_iso = str(data.get("due_iso") or "").strip()

        if not reminder_text or not due_iso:
            raise ValueError("Incomplete LLM response")

        # Validate the ISO string is parseable
        date_parser.parse(due_iso)

        return Reminder(
            id=utc_now_iso(),
            text=reminder_text,
            due_at_iso=due_iso,
            timezone=user_tz,
        )
    except Exception as exc:
        logger.warning("LLM reminder parse failed (%s), trying keyword fallback.", exc)
        return parse_reminder(text, user_tz)


def parse_reminder(text: str, user_tz: str = "UTC") -> Optional[Reminder]:
    """
    Keyword fallback reminder parser.
    Expects: 'remind me to X on 2026-03-20 18:00'.
    """
    lowered = text.lower()
    if not lowered.startswith("remind me"):
        return None

    if " on " not in lowered:
        return None

    before, _, after = text.partition(" on ")
    try:
        dt = date_parser.parse(after)
    except Exception:
        return None

    return Reminder(
        id=utc_now_iso(),
        text=before.strip(),
        due_at_iso=dt.isoformat(),
        timezone=user_tz,
    )


def reminder_to_metadata(reminder: Reminder, chat_id: Optional[int] = None) -> Dict[str, Any]:
    # Parse due_at_iso to a Unix timestamp
    try:
        due_dt = date_parser.parse(reminder.due_at_iso)
        if due_dt.tzinfo is None:
            due_dt = due_dt.replace(tzinfo=timezone.utc)
        due_at_ts = int(due_dt.timestamp())
    except Exception:
        due_at_ts = utc_now_ts()

    md: Dict[str, Any] = {
        "id": reminder.id,
        "source_type": "reminder",
        "raw_content": reminder.text,
        "due_at": reminder.due_at_iso,
        "due_at_ts": due_at_ts,
        "timezone": reminder.timezone,
        "created_at": utc_now_iso(),
        "created_at_ts": utc_now_ts(),
        "fired": False,
        "tags": ["reminder"],
    }

    if chat_id is not None:
        md["chat_id"] = chat_id

    return md
