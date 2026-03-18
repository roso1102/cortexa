from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, List

from langchain_text_splitters import RecursiveCharacterTextSplitter


def utc_now_iso() -> str:
    """Return current UTC time as ISO8601 string."""
    return datetime.now(timezone.utc).isoformat()


def utc_now_ts() -> int:
    """Return current UTC time as unix timestamp (seconds)."""
    return int(datetime.now(timezone.utc).timestamp())


_DEFAULT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)


def chunk_text(text: str) -> List[str]:
    """Split text into overlapping chunks suitable for embedding."""
    return _DEFAULT_SPLITTER.split_text(text)


def iter_chunks(texts: Iterable[str]) -> Iterable[str]:
    for t in texts:
        for chunk in chunk_text(t):
            yield chunk

