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

# Action router schema (Option B router upgrade)
from src.action_schema import (  # noqa: E402
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
    ROUTER_CLARIFY_THRESHOLD,
    RoutedAction,
    normalize_list_args,
    parse_routed_action,
)

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


_ACTION_ROUTER_SCHEMA = """\
You are an action router for a personal cognitive memory system called cortexa.

Given the user's message, choose the single best next action.
Return ONLY valid JSON with keys:
- action: one of SAVE_TEXT, SAVE_LINK, ANSWER_QUERY, LIST, SET_REMINDER, DELETE, CLARIFY
- confidence: float 0.0-1.0
- reason: short 1-line explanation
- args: object with action-specific fields

Action guidance:
- SAVE_LINK: when message contains a URL and the user is sharing it to save.
  args: { urls: [ ... ] } (1-3 urls)
- SAVE_TEXT: when user is sharing a note/poem/idea to remember.
  args: { text: string }
- ANSWER_QUERY: when user is asking a question (what/how/why/when/where/who, ends with '?', or a request for help/suggestions).
  args: { query: string }
- LIST: when user is asking to list previously saved things.
  args: { list_type: one of poems|links|memories, kind?: string, topic?: string, limit?: number }
- SET_REMINDER: when user wants a reminder at a specific time/date.
  args: { text: string }
- DELETE: when user wants to delete something they saved.
  args: { target: string } (e.g. 'last', an id, or a natural-language reference)
- CLARIFY: when you are not confident whether the user wants to save vs ask.
  args: { question: string }

Rules:
- If there is a URL and no explicit question, prefer SAVE_LINK.
- Messages starting with what/how/why/when/where/who are almost always ANSWER_QUERY.
- For poem listing requests like 'what poem did i save', prefer LIST with list_type='poems'.
- For generic recall prompts like "what limerick/story/news did I save about X", prefer LIST with list_type='memories' and fill kind/topic when possible.
- Only choose SET_REMINDER if the message has an explicit time cue (tomorrow/tonight/at 6pm/in 2 hours/etc).
- If confidence < 0.55, choose CLARIFY and ask a single short question.

Only return raw JSON. No prose, no markdown.
"""


def route_action(text: str, groq_client: Groq) -> RoutedAction:
    """
    Route a user message to a single executable action (tool-calling style).
    This replaces brittle phrase handlers in telegram_bot.py.
    """
    raw_full = text or ""
    header = _header_snippet(raw_full)

    # Fast-path: greetings / pleasantries — CLARIFY/answer in bot layer
    if _is_chitchat(header):
        return RoutedAction(action=ACTION_CLARIFY, confidence=1.0, reason="pre-check: chitchat", args={"question": "Hey — what do you want to do: save something, or ask something?"})

    # Fast-path: URLs present -> save link unless it's clearly a question about the URL
    t = header.strip().lower()
    if ("http://" in t or "https://" in t) and not t.endswith("?"):
        urls = re.findall(r"https?://\S+", header)[:3]
        return RoutedAction(
            action=ACTION_SAVE_LINK,
            confidence=0.85,
            reason="pre-check: url present",
            args={"urls": urls},
        )

    # Fast-path: explicit save commands are save-text
    if t.startswith(("save this", "save:", "note this", "note:", "log this", "journal this")) and "http" not in t:
        return RoutedAction(action=ACTION_SAVE_TEXT, confidence=0.9, reason="pre-check: explicit save", args={"text": raw_full})

    # Fast-path: delete/remove memory commands
    if t.startswith("delete ") or t.startswith("remove "):
        return RoutedAction(
            action=ACTION_DELETE,
            confidence=0.95,
            reason="pre-check: delete/remove command",
            args={"target": raw_full},
        )

    # Fast-path: ambiguous short phrases should clarify instead of being silently saved.
    if _looks_ambiguous_short_phrase(header):
        return RoutedAction(
            action=ACTION_CLARIFY,
            confidence=0.5,
            reason="pre-check: ambiguous short phrase",
            args={"question": "Do you want me to save this, or answer a question about it?"},
        )

    # Fast-path: generic memory listing (with optional kind/topic extraction).
    if _is_list_memory_query(t):
        args = _extract_list_memory_filters(t)
        if args.get("kind") == "poem":
            poem_args: dict[str, str] = {"list_type": LIST_POEMS}
            if args.get("topic"):
                poem_args["topic"] = str(args["topic"])
            args = poem_args
        return RoutedAction(action=ACTION_LIST, confidence=0.95, reason="pre-check: memory listing request", args=normalize_list_args(args))

    # Fast-path: obvious queries
    if _is_obvious_query(header):
        return RoutedAction(action=ACTION_ANSWER_QUERY, confidence=0.9, reason="pre-check: obvious query phrasing", args={"query": raw_full})

    # Fast-path: implicit reminders (short + time cue)
    if _is_implicit_reminder(header) or ("remind me" in t and _has_time_cue(raw_full)):
        return RoutedAction(action=ACTION_SET_REMINDER, confidence=0.8, reason="pre-check: reminder cue", args={"text": raw_full})

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": _ACTION_ROUTER_SCHEMA},
                {"role": "user", "content": header},
            ],
            max_tokens=220,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = (response.choices[0].message.content or "{}").strip()
        parsed = parse_routed_action(raw)
        if not parsed:
            raise ValueError("invalid router json")

        # Normalize/validate actions
        allowed = {
            ACTION_SAVE_TEXT,
            ACTION_SAVE_LINK,
            ACTION_ANSWER_QUERY,
            ACTION_LIST,
            ACTION_SET_REMINDER,
            ACTION_DELETE,
            ACTION_CLARIFY,
        }
        if parsed.action not in allowed:
            return RoutedAction(action=ACTION_CLARIFY, confidence=0.4, reason="validator: unknown action", args={"question": "Do you want me to save this, or answer it?"})

        # Reminder validator: require time cue
        if parsed.action == ACTION_SET_REMINDER and not _has_time_cue(raw_full):
            return RoutedAction(action=ACTION_SAVE_TEXT, confidence=0.6, reason="validator: reminder requires time cue", args={"text": raw_full})

        # LIST validator
        if parsed.action == ACTION_LIST:
            parsed = RoutedAction(
                action=ACTION_LIST,
                confidence=parsed.confidence,
                reason=parsed.reason or "validator: normalized list args",
                args=normalize_list_args(parsed.args),
            )

        # For confidence low, force CLARIFY
        if parsed.confidence < ROUTER_CLARIFY_THRESHOLD and parsed.action != ACTION_CLARIFY:
            return RoutedAction(action=ACTION_CLARIFY, confidence=parsed.confidence, reason="low confidence", args={"question": "Do you want me to save this, or answer it?"})

        return parsed
    except Exception as exc:
        logger.warning("LLM action router failed (%s), falling back to intent classifier.", exc)
        # Fallback to existing intent classifier
        legacy = classify_intent(text, groq_client)
        intent = legacy.get("intent")
        if intent == INTENT_QUERY:
            return RoutedAction(action=ACTION_ANSWER_QUERY, confidence=float(legacy.get("confidence") or 0.6), reason="fallback: legacy intent QUERY", args={"query": raw_full})
        if intent == INTENT_LIST_LINKS:
            return RoutedAction(action=ACTION_LIST, confidence=float(legacy.get("confidence") or 0.6), reason="fallback: legacy intent LIST_LINKS", args={"list_type": LIST_LINKS})
        if intent == INTENT_REMINDER:
            return RoutedAction(action=ACTION_SET_REMINDER, confidence=float(legacy.get("confidence") or 0.6), reason="fallback: legacy intent REMINDER", args={"text": raw_full})
        if intent == INTENT_DELETE:
            return RoutedAction(action=ACTION_DELETE, confidence=float(legacy.get("confidence") or 0.6), reason="fallback: legacy intent DELETE", args={"target": raw_full})
        if intent == INTENT_INGEST_LINK:
            urls = re.findall(r"https?://\S+", raw_full)
            return RoutedAction(action=ACTION_SAVE_LINK, confidence=float(legacy.get("confidence") or 0.6), reason="fallback: legacy intent INGEST_LINK", args={"urls": urls[:3]})
        # default
        return RoutedAction(action=ACTION_SAVE_TEXT, confidence=float(legacy.get("confidence") or 0.6), reason="fallback: legacy intent INGEST_TEXT", args={"text": raw_full})

# Salutations / small-talk phrases — should get a friendly reply, not be saved
_CHITCHAT_PHRASES = frozenset({
    "hi",
    "hello",
    "hey",
    "heyy",
    "heya",
    "hi there",
    "hello there",
    "hey there",
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
_LIST_MEMORY_CUES = (
    "did i save",
    "i saved",
    "show my",
    "list my",
    "what did i save",
    "which did i save",
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


def _header_snippet(text: str, max_chars: int = 200, max_lines: int = 3) -> str:
    """
    Return a short header snippet (first non-empty lines) for intent classification.
    This keeps the classifier focused on how the user phrases the request rather
    than the full body of a long note or article.
    """
    if not text:
        return ""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    header = "\n".join(lines[:max_lines])
    return header[:max_chars]


def _is_chitchat(text: str) -> bool:
    """Return True if the message is a greeting or social pleasantry."""
    t = text.strip().lower().rstrip("!.,?")
    return t in _CHITCHAT_PHRASES


def _is_list_links_query(text: str) -> bool:
    """Return True if the message is clearly asking to list saved links."""
    t = text.strip().lower()
    return any(cue in t for cue in _LIST_LINK_CUES)


def _extract_list_memory_filters(text: str) -> dict[str, str]:
    t = (text or "").strip().lower()
    args: dict[str, str] = {"list_type": LIST_MEMORIES}

    kind_match = re.search(
        r"\b(poem|poems|limerick|limericks|short story|short stories|story|stories|news|article|articles|note|notes|memory|memories)\b",
        t,
    )
    if kind_match:
        kind = kind_match.group(1).strip()
        kind_map = {
            "poems": "poem",
            "limericks": "limerick",
            "short stories": "short story",
            "stories": "story",
            "articles": "article",
            "notes": "note",
            "memories": "memory",
        }
        args["kind"] = kind_map.get(kind, kind)

    about_match = re.search(r"\babout\s+(.+)$", t)
    if about_match:
        topic = about_match.group(1).strip(" .?!,")
        if topic:
            args["topic"] = topic

    return args


def _is_list_memory_query(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    if _is_list_links_query(t):
        return False
    if any(cue in t for cue in _LIST_MEMORY_CUES):
        return True
    return bool(re.search(r"\bwhat\s+(poem|limerick|short story|story|news|article|note|memory)\b", t))


def _is_obvious_query(text: str) -> bool:
    """Fast pre-check: return True if the message is unambiguously a query."""
    t = text.strip().lower()

    # List-links requests should not be short-circuited to QUERY
    if _is_list_links_query(t):
        return False
    if _is_list_memory_query(t):
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


def _looks_ambiguous_short_phrase(text: str) -> bool:
    """
    Short noun-phrases like 'screen recording' are often ambiguous and should clarify.
    """
    t = (text or "").strip().lower()
    if not t or "\n" in t:
        return False
    if "http://" in t or "https://" in t:
        return False
    if t.endswith("?") or _is_obvious_query(t) or _is_list_memory_query(t) or _is_list_links_query(t):
        return False
    if _has_time_cue(t):
        return False
    tokens = re.findall(r"[a-z0-9]+", t)
    if len(tokens) <= 3 and len(t) <= 40:
        if not any(t.startswith(p) for p in ("save", "note", "log", "journal", "delete", "remove", "remind")):
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
    raw_full = text or ""
    header = _header_snippet(raw_full)

    # Minimal fast-path: greetings / pleasantries — reply warmly, don't save
    if _is_chitchat(header):
        return {
            "intent": INTENT_CHITCHAT,
            "confidence": 1.0,
            "summary": "pre-check: greeting or social pleasantry",
            "source": "pre-check",
        }

    # Minimal fast-path: explicit "save/note/log" commands are ingestion, not reminders
    t = header.strip().lower()
    if t.startswith(("save this", "save:", "note this", "note:", "log this", "journal this")) and "http" not in t:
        return {
            "intent": INTENT_INGEST_TEXT,
            "confidence": 0.9,
            "summary": "pre-check: explicit save/note instruction",
            "source": "pre-check",
        }

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": _CLASSIFICATION_SCHEMA},
                {"role": "user", "content": header},
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
        if intent == INTENT_REMINDER and not _has_time_cue(raw_full):
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
