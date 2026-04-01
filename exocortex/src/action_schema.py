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

LIST_POEMS = "poems"
LIST_LINKS = "links"
LIST_MEMORIES = "memories"


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

