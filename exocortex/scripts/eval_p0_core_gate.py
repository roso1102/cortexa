from __future__ import annotations

import json
import os
from typing import Any

from groq import Groq

from src.config import load_config
from src.orchestrator import route_action


def _load_cases(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise RuntimeError("eval cases must be a JSON array")
    return [x for x in data if isinstance(x, dict)]


def main() -> None:
    config = load_config()
    client = Groq(api_key=config.groq_api_key)

    cases_path = os.getenv("EVAL_CASES_FILE", "scripts/eval_cases_p0.json")
    cases = _load_cases(cases_path)
    failures: list[str] = []

    print(f"Running P0 gate with {len(cases)} cases from {cases_path}")
    for case in cases:
        name = str(case.get("name") or "unnamed")
        text = str(case.get("text") or "")
        if not text:
            failures.append(f"{name}: empty text")
            continue

        routed = route_action(text, client)
        action = routed.action
        args = routed.args or {}
        print(f"- {name}: action={action} confidence={routed.confidence:.2f} args={args}")

        allowed = case.get("expect_action_in")
        if isinstance(allowed, list) and allowed:
            allowed_set = {str(x) for x in allowed}
            if action not in allowed_set:
                failures.append(f"{name}: expected action in {sorted(allowed_set)}, got {action}")

        denied = case.get("expect_action_not_in")
        if isinstance(denied, list) and denied:
            denied_set = {str(x) for x in denied}
            if action in denied_set:
                failures.append(f"{name}: action {action} is forbidden")

        exp_lt = case.get("expect_list_type")
        if exp_lt is not None and action == "LIST":
            got_lt = str(args.get("list_type") or "")
            if got_lt != str(exp_lt):
                failures.append(f"{name}: expected list_type={exp_lt}, got {got_lt}")

    if failures:
        print("\nP0 gate FAILED:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)

    print("\nP0 gate passed.")


if __name__ == "__main__":
    main()
