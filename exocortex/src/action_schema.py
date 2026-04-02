from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional


# Canonical action names
ACTION_SAVE_TEXT = "SAVE_TEXT"
ACTION_SAVE_LINK = "SAVE_LINK"
ACTION_ANSWER_QUERY = "ANSWER_QUERY"
ACTION_LIST = "LIST"
ACTION_SET_REMINDER = "SET_REMINDER"
ACTION_DELETE = "DELETE"
ACTION_CLARIFY = "CLARIFY"

ALLOWED_ROUTER_ACTIONS = frozenset(
    {
        ACTION_SAVE_TEXT,
        ACTION_SAVE_LINK,
        ACTION_ANSWER_QUERY,
        ACTION_LIST,
        ACTION_SET_REMINDER,
        ACTION_DELETE,
        ACTION_CLARIFY,
    }
)

LIST_POEMS = "poems"
LIST_LINKS = "links"
LIST_MEMORIES = "memories"
ALLOWED_LIST_TYPES = {LIST_POEMS, LIST_LINKS, LIST_MEMORIES}
ROUTER_CLARIFY_THRESHOLD = 0.55
DEFAULT_LIST_LIMIT = 8
MAX_LIST_LIMIT = 20


@dataclass(frozen=True)
class RoutedAction:
    action: str
    confidence: float
    reason: str
    args: Dict[str, Any]


def parse_routed_action(raw: str) -> Optional[RoutedAction]:
    """
    Parse a JSON string into a RoutedAction.
    Returns None if parsing fails or required keys are missing.
    """
    try:
        obj = json.loads((raw or "").strip() or "{}")
    except Exception:
        return None

    if not isinstance(obj, dict):
        return None

    action = str(obj.get("action") or "").strip()
    if not action:
        return None

    try:
        confidence = float(obj.get("confidence") or 0.0)
    except Exception:
        confidence = 0.0

    reason = str(obj.get("reason") or "").strip()
    args = obj.get("args") if isinstance(obj.get("args"), dict) else {}

    return RoutedAction(action=action, confidence=confidence, reason=reason, args=args)


def should_clarify(*, routed: RoutedAction | None, threshold: float = 0.55) -> bool:
    """
    Decide whether to ask a clarification question.
    """
    if routed is None:
        return True
    if routed.action == ACTION_CLARIFY:
        return True
    return float(routed.confidence or 0.0) < threshold


def normalize_list_args(args: Dict[str, Any] | None) -> Dict[str, Any]:
    """
    Normalize LIST action args into a stable contract:
    - list_type: poems|links|memories (default memories)
    - topic: optional short string
    - kind/content_type: optional short string aliases
    - limit: int in [1, MAX_LIST_LIMIT]
    - time_range: optional short string
    """
    src = args or {}
    out: Dict[str, Any] = {}

    lt = str(src.get("list_type") or "").strip().lower()
    if lt not in ALLOWED_LIST_TYPES:
        lt = LIST_MEMORIES
    out["list_type"] = lt

    topic = str(src.get("topic") or "").strip()
    if topic:
        out["topic"] = topic[:120]

    # Accept both "kind" and "content_type", normalize to "kind".
    kind = str(src.get("kind") or src.get("content_type") or "").strip().lower()
    if kind:
        out["kind"] = kind[:60]

    tr = str(src.get("time_range") or "").strip().lower()
    if tr:
        out["time_range"] = tr[:40]

    raw_limit = src.get("limit")
    try:
        limit = int(raw_limit) if raw_limit is not None else DEFAULT_LIST_LIMIT
    except Exception:
        limit = DEFAULT_LIST_LIMIT
    out["limit"] = max(1, min(limit, MAX_LIST_LIMIT))

    return out

