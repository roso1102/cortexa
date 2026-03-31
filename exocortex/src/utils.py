from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable, List

from langchain_text_splitters import RecursiveCharacterTextSplitter


def utc_now_iso() -> str:
    """Return current UTC time as ISO8601 string."""
    return datetime.now(timezone.utc).isoformat()


def utc_now_ts() -> int:
    """Return current UTC time as unix timestamp (seconds)."""
    return int(datetime.now(timezone.utc).timestamp())


# --- Time helpers (IST) ---
_IST_OFFSET = timedelta(hours=5, minutes=30)


def ist_now() -> datetime:
    """Return current time in IST as an aware datetime."""
    return datetime.now(timezone.utc).astimezone(timezone(_IST_OFFSET))


def ist_day_range_utc_ts(dt_ist: datetime | None = None) -> tuple[int, int]:
    """
    Given an IST-aware datetime (or now), return (start_utc_ts, end_utc_ts)
    for that IST calendar day.
    """
    if dt_ist is None:
        dt_ist = ist_now()
    if dt_ist.tzinfo is None:
        dt_ist = dt_ist.replace(tzinfo=timezone(_IST_OFFSET))

    start_ist = dt_ist.replace(hour=0, minute=0, second=0, microsecond=0)
    end_ist = start_ist + timedelta(days=1)
    start_utc = start_ist.astimezone(timezone.utc)
    end_utc = end_ist.astimezone(timezone.utc)
    return int(start_utc.timestamp()), int(end_utc.timestamp())


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

