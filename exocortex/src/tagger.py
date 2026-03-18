from __future__ import annotations

import json
import logging
from typing import List

from groq import Groq

logger = logging.getLogger(__name__)

_TAGGER_PROMPT = """\
You are a topic tagger for a personal knowledge base.

Given a piece of text, assign 2 to 4 short, lowercase topic tags that describe its subject matter.
Return ONLY a JSON object with a single key "tags" whose value is a list of strings.

Rules:
- Tags should be 1-3 words, lowercase, no punctuation.
- Be specific: prefer "hardware security" over "technology".
- Use consistent terms: "machine learning" not "ml" or "AI".
- If the text is too short or ambiguous, use generic tags like "note" or "idea".
- Only return raw JSON. No prose, no markdown.

Examples:
- Text about FPGA and neural networks → ["fpga", "hardware acceleration", "machine learning"]
- Text about peanut allergy symptoms → ["allergy", "health", "peanut allergy"]
- "remind me to call dad" → ["reminder", "personal"]
- "The Rust ownership model prevents data races" → ["rust", "programming", "memory safety"]
"""


def tag_text(text: str, groq_client: Groq) -> List[str]:
    """
    Call Groq to assign 2-4 topic tags to a piece of text.
    Returns an empty list on failure — tagging is best-effort and non-blocking.
    """
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": _TAGGER_PROMPT},
                {"role": "user", "content": text[:800]},  # cap to avoid token waste
            ],
            max_tokens=60,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = (response.choices[0].message.content or "{}").strip()
        data = json.loads(raw)
        tags = data.get("tags", [])
        if isinstance(tags, list):
            return [str(t).lower().strip() for t in tags if t][:4]
        return []
    except Exception as exc:
        logger.debug("Tagging failed (non-fatal): %s", exc)
        return []
