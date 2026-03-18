import os
from dataclasses import dataclass, field
from typing import FrozenSet


class ConfigError(RuntimeError):
    pass


def _get_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ConfigError(f"Missing required environment variable: {name}")
    return value


def _parse_chat_ids(raw: str) -> FrozenSet[int]:
    """Parse a comma-separated list of Telegram chat IDs into a frozenset of ints."""
    ids: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            try:
                ids.append(int(part))
            except ValueError:
                pass
    return frozenset(ids)


@dataclass(frozen=True)
class AppConfig:
    telegram_token: str
    pinecone_api_key: str
    pinecone_index_name: str
    google_api_key: str
    groq_api_key: str
    openrouter_api_key: str
    debug_mode: bool
    # Access control: empty set means no restriction (single-user / open mode)
    allowed_chat_ids: FrozenSet[int] = field(default_factory=frozenset)
    # "HH:MM" UTC to auto-send daily digest; empty string disables it
    daily_digest_time: str = ""
    # Owner chat ID for scheduler-initiated messages (reminders, daily digest)
    owner_chat_id: int = 0
    # Token for dashboard API access (X-Dashboard-Token)
    dashboard_secret: str = ""


def load_config() -> AppConfig:
    """
    Load all required configuration from environment variables.
    This function should be called once at startup.
    """
    allowed_raw = os.getenv("ALLOWED_CHAT_IDS", "").strip()
    allowed_ids = _parse_chat_ids(allowed_raw) if allowed_raw else frozenset()

    owner_raw = os.getenv("OWNER_CHAT_ID", "").strip()
    try:
        owner_chat_id = int(owner_raw) if owner_raw else 0
    except ValueError:
        owner_chat_id = 0

    return AppConfig(
        telegram_token=_get_env("TELEGRAM_TOKEN"),
        pinecone_api_key=_get_env("PINECONE_API_KEY"),
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "cortexa"),
        google_api_key=_get_env("GOOGLE_API_KEY"),
        groq_api_key=_get_env("GROQ_API_KEY"),
        openrouter_api_key=_get_env("OPENROUTER_API_KEY"),
        debug_mode=os.getenv("DEBUG_MODE", "").strip().lower() in {"1", "true", "yes", "on"},
        allowed_chat_ids=allowed_ids,
        daily_digest_time=os.getenv("DAILY_DIGEST_TIME", "").strip(),
        owner_chat_id=owner_chat_id,
        dashboard_secret=os.getenv("DASHBOARD_SECRET", "").strip(),
    )
