from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict

from groq import Groq

logger = logging.getLogger(__name__)

# Canonical intent names
INTENT_INGEST_TEXT = "INGEST_TEXT"
INTENT_INGEST_LINK = "INGEST_LINK"
INTENT_QUERY       = "QUERY"
INTENT_REMINDER    = "REMINDER"
INTENT_LIST_LINKS  = "LIST_LINKS"
INTENT_DELETE      = "DELETE"
INTENT_CHITCHAT    = "CHITCHAT"
INTENT_UNKNOWN     = "UNKNOWN"

_CLASSIFICATION_SCHEMA = """\
You are an intent classifier for a personal cognitive memory system called cortexa.

Given a user message, respond ONLY with a JSON object with these keys:
- intent: one of INGEST_TEXT, INGEST_LINK, QUERY, REMINDER, LIST_LINKS, DELETE, UNKNOWN
- confidence: float 0.0-1.0
- summary: short 1-line reason

Intent definitions:
- INGEST_TEXT: user is sharing a note, idea, thought, task, or fact to save. No URL. No question.
- INGEST_LINK: message contains a URL and user wants to save it.
- QUERY: user is asking for information, asking you to recall something they saved, or asking you to analyze/plan/summarize/explain. Any question about saved content is QUERY.
- REMINDER: user wants to be reminded about something at a specific time or date.
- LIST_LINKS: user is asking to list or show links they saved today or recently.
- DELETE: user wants to delete a saved memory.
- UNKNOWN: message is too ambiguous to classify.

Rules:
- Ignore typos and spelling mistakes. Classify based on meaning, not exact words.
- Any message asking "what did I save", "what do I know about", "tell me about", "recall", "show me what I saved" is QUERY.
- Any message starting with what, how, why, when, where, who is almost always QUERY.
- Any message asking to analyze, plan, compare, summarize, propose, explain is QUERY.
- If there is a URL and no question, prefer INGEST_LINK.
- Only return raw JSON. No prose, no markdown.

Examples:
- "what did i save about allergy" → QUERY
- "wt did i sav bout fpga" → QUERY  
- "analyze my FPGA idea and propose a plan" → QUERY
- "how does ASCON work?" → QUERY
- "I just learned that FPGA can accelerate ML inference" → INGEST_TEXT
- "Key insight: semantic search beats keyword search" → INGEST_TEXT
- "remind me tomorrow to review notes" → REMINDER
- "remidn me tmrrw to check ASCON" → REMINDER
- "call mom this evening" → REMINDER
- "call dad at 6pm" → REMINDER
- "meeting with team tonight at 8" → REMINDER
- "dentist appointment next Monday" → REMINDER
- "https://example.com" → INGEST_LINK
- "what links did i save today" → LIST_LINKS

Key rule: If a message mentions a person AND a time word (this evening, tonight, tomorrow, at Xpm, next week, in X hours/minutes), classify it as REMINDER even without the word "remind".
"""


# Salutations / small-talk phrases — should get a friendly reply, not be saved
_CHITCHAT_PHRASES = frozenset({
    "hi", "hello", "hey", "heyy", "heya",
    "bye", "goodbye", "good bye", "cya", "see ya", "see you",
    "good night", "goodnight", "gn", "gn!",
    "good morning", "gm", "good afternoon", "good evening",
    "how are you", "how r u", "how are u", "how's it going",
    "whats up", "what's up", "wassup", "sup",
    "thanks", "thank you", "ty", "thx",
    "ok", "okay", "k", "kk", "cool", "nice", "great", "perfect",
    "lol", "haha", "hehe", "lmao",
    "yes", "no", "yeah", "nah", "yep", "nope",
})

_QUERY_PREFIXES = (
    "what ", "what's ", "whats ", "wht ", "wat ",
    "how ", "how's ",
    "why ", "when ", "where ", "who ",
    "tell me", "give me",
    "recall ", "remember ", "do you know",
    "analyze ", "analyse ", "plan ", "compare ", "summarize ", "summarise ",
    "explain ", "propose ", "suggest ", "list my ", "find my ",
)

# "show me" only qualifies as a query if it's NOT a list-links request
_LIST_LINK_CUES = (
    "links", "urls", "saved links", "link i saved", "links i saved",
    "links saved today", "link saved today",
)

_QUERY_SUBSTRINGS = (
    "did i save", "did i note", "i saved about", "i know about",
    "what about", "can you tell",
    "step plan", "validation plan", "give me a plan",
)

# Time-of-day / relative-time cues that make a message an implicit reminder.
# "am" / "pm" are matched only when preceded by a digit (e.g. "9am", "10pm")
# to avoid false positives like "i am a llama".
_REMINDER_TIME_CUES = (
    "this evening", "this morning", "this afternoon", "tonight",
    "tomorrow", "next week", "next monday", "next tuesday", "next wednesday",
    "next thursday", "next friday", "next saturday", "next sunday",
    "in 1 hour", "in 2 hours", "in 3 hours", "in an hour",
    "in 1 minute", "in 5 minutes", "in 10 minutes", "in 15 minutes",
    "in 30 minutes", "in 45 minutes",
    " at 6", " at 7", " at 8", " at 9", " at 10", " at 11", " at 12",
)


_AM_PM_RE = re.compile(r"\d\s*[ap]m\b")
_HHMM_RE = re.compile(r"\b([01]?\d|2[0-3]):[0-5]\d\b")
_DATE_CUE_RE = re.compile(
    r"\b(today|tomorrow|tonight|this (morning|afternoon|evening)|next (week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b",
    re.IGNORECASE,
)
_IN_RE = re.compile(r"\bin\s+\d+\s+(minute|minutes|hour|hours|day|days|week|weeks)\b", re.IGNORECASE)
_COMMITMENT_RE = re.compile(
    r"\b(i have to|i need to|i should|i must|dont forget|don't forget|remember to)\b",
    re.IGNORECASE,
)


def _has_time_cue(text: str) -> bool:
    t = (text or "").strip().lower()
    return bool(
        _AM_PM_RE.search(t)
        or _HHMM_RE.search(t)
        or _DATE_CUE_RE.search(t)
        or _IN_RE.search(t)
        or any(cue in t for cue in _REMINDER_TIME_CUES)
    )


def _is_chitchat(text: str) -> bool:
    """Return True if the message is a greeting or social pleasantry."""
    t = text.strip().lower().rstrip("!.,?")
    return t in _CHITCHAT_PHRASES


def _is_list_links_query(text: str) -> bool:
    """Return True if the message is clearly asking to list saved links."""
    t = text.strip().lower()
    return any(cue in t for cue in _LIST_LINK_CUES)


def _is_obvious_query(text: str) -> bool:
    """Fast pre-check: return True if the message is unambiguously a query."""
    t = text.strip().lower()

    # List-links requests should not be short-circuited to QUERY
    if _is_list_links_query(t):
        return False

    if t.endswith("?"):
        return True
    # "show me" without link context is a query
    if t.startswith("show me") and not _is_list_links_query(t):
        return True
    if any(t.startswith(p) for p in _QUERY_PREFIXES):
        return True
    if any(s in t for s in _QUERY_SUBSTRINGS):
        return True
    return False


def _is_implicit_reminder(text: str) -> bool:
    """
    Detect messages like "call mom this evening" or "dentist at 6pm" that
    are implicit reminders (no 'remind me' prefix but contain a time cue).

    "am" / "pm" only count when preceded by a digit (e.g. "9am", "10pm") to
    avoid false positives like "i am a llama".
    """
    raw = text or ""
    t = raw.strip().lower()
    # Never treat long or multi-line messages as implicit reminders.
    # Those are almost always notes/poems/brain-dumps.
    if "\n" in raw:
        return False
    if len(t) > 140:
        return False
    # Explicit "remind me" is already handled by the LLM path
    if "remind me" in t:
        return False
    # If the user is clearly asking to save something, it's ingestion, not a reminder.
    if t.startswith(("save this", "save:", "note this", "note:", "log this", "journal this")):
        return False
    # Check phrase-based cues (evening, tonight, tomorrow, etc.)
    if any(cue in t for cue in _REMINDER_TIME_CUES):
        # Require an action verb to reduce false positives
        if any(v in t for v in ("call ", "meet", "meeting", "pay", "submit", "review", "check", "follow up", "message", "text ", "email", "buy ")):
            return True
        return False
    # Check digit+am/pm patterns (e.g. "9am", "6 pm", "10pm")
    if _AM_PM_RE.search(t):
        if any(v in t for v in ("call ", "meet", "meeting", "pay", "submit", "review", "check", "follow up", "message", "text ", "email", "buy ")):
            return True
        return False
    return False


def classify_intent(text: str, groq_client: Groq) -> Dict[str, Any]:
    """
    Use Groq to classify the user's intent into a structured action.
    Falls back to keyword heuristic if LLM call fails.
    """
    # Fast-path: greetings / pleasantries — reply warmly, don't save
    if _is_chitchat(text):
        return {
            "intent": INTENT_CHITCHAT,
            "confidence": 1.0,
            "summary": "pre-check: greeting or social pleasantry",
            "source": "pre-check",
        }

    # Fast-path: explicit "save/note/log" commands are ingestion, not reminders
    t = (text or "").strip().lower()
    if t.startswith(("save this", "save:", "note this", "note:", "log this", "journal this")) and "http" not in t:
        return {
            "intent": INTENT_INGEST_TEXT,
            "confidence": 0.9,
            "summary": "pre-check: explicit save/note instruction",
            "source": "pre-check",
        }

    # Fast-path: commitment + time cue => reminder (even if long)
    # Example: "i have to read this tomorrow ..." should be REMINDER.
    if _COMMITMENT_RE.search(text or "") and _has_time_cue(text):
        return {
            "intent": INTENT_REMINDER,
            "confidence": 0.9,
            "summary": "pre-check: commitment phrase with time cue",
            "source": "pre-check",
        }

    # Fast-path: long or multiline text defaults to INGEST_TEXT (poems/brain dumps),
    # unless it has explicit reminder structure.
    raw = text or ""
    if ("\n" in raw or len(t) > 200) and ("remind me" not in t) and not _has_time_cue(text):
        return {
            "intent": INTENT_INGEST_TEXT,
            "confidence": 0.85,
            "summary": "pre-check: long/multiline note (no explicit time cue)",
            "source": "pre-check",
        }

    # Fast-path: obvious queries skip the LLM entirely
    if _is_obvious_query(text):
        return {
            "intent": INTENT_QUERY,
            "confidence": 0.95,
            "summary": "pre-check: obvious query phrasing",
            "source": "pre-check",
        }

    # Fast-path: implicit reminders (time cue present, no question)
    if _is_implicit_reminder(text):
        return {
            "intent": INTENT_REMINDER,
            "confidence": 0.85,
            "summary": "pre-check: implicit reminder with time cue",
            "source": "pre-check",
        }

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": _CLASSIFICATION_SCHEMA},
                {"role": "user", "content": text},
            ],
            max_tokens=120,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = (response.choices[0].message.content or "{}").strip()
        result = json.loads(raw)
        intent = result.get("intent", INTENT_UNKNOWN)
        if intent not in {
            INTENT_INGEST_TEXT,
            INTENT_INGEST_LINK,
            INTENT_QUERY,
            INTENT_REMINDER,
            INTENT_LIST_LINKS,
            INTENT_DELETE,
            INTENT_CHITCHAT,
            INTENT_UNKNOWN,
        }:
            intent = INTENT_UNKNOWN

        # Validator: never allow REMINDER without a detected time cue
        if intent == INTENT_REMINDER and not _has_time_cue(text):
            intent = INTENT_INGEST_TEXT
            result["confidence"] = 0.6
            result["summary"] = "validator: reminder requires explicit time cue"

        return {
            "intent": intent,
            "confidence": float(result.get("confidence", 0.5)),
            "summary": str(result.get("summary", "")),
            "source": "llm",
        }
    except Exception as exc:
        logger.warning("LLM intent classification failed (%s), using keyword fallback.", exc)
        return _keyword_fallback(text)


def _keyword_fallback(text: str) -> Dict[str, Any]:
    """
    Rule-based fallback. Used when Groq is unavailable.
    """
    t = (text or "").strip().lower()

    if not t:
        return {"intent": INTENT_INGEST_TEXT, "confidence": 0.5, "source": "fallback"}

    # URL detection
    if "http://" in t or "https://" in t:
        return {"intent": INTENT_INGEST_LINK, "confidence": 0.8, "source": "fallback"}

    # Reminder (explicit and implicit)
    if "remind me" in t or _is_implicit_reminder(t):
        return {"intent": INTENT_REMINDER, "confidence": 0.9, "source": "fallback"}

    # Delete
    if t.startswith("delete ") or t.startswith("remove "):
        return {"intent": INTENT_DELETE, "confidence": 0.9, "source": "fallback"}

    # List links
    link_list_cues = ("what links", "which links", "links saved", "list links", "show links")
    if any(c in t for c in link_list_cues):
        return {"intent": INTENT_LIST_LINKS, "confidence": 0.85, "source": "fallback"}

    # Query cues
    query_cues = (
        "?", "what ", "why ", "how ", "when ", "where ", "who ",
        "analyze", "plan", "compare", "summarize", "summarise",
        "explain", "tell me", "show me", "give me", "debug", "roadmap",
    )
    if any(c in t for c in query_cues):
        return {"intent": INTENT_QUERY, "confidence": 0.7, "source": "fallback"}

    return {"intent": INTENT_INGEST_TEXT, "confidence": 0.6, "source": "fallback"}
