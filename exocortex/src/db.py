from __future__ import annotations

"""
Postgres schema & thin data-access layer for Cortexa.

This module is the canonical store for:
- users
- memories (main records)
- chunks (child records, also indexed in Pinecone)
- reminders
- tunnels

Supabase exposes Postgres via a standard connection string. We expect:
  DATABASE_URL=postgresql+psycopg2://user:pass@host:port/dbname

For now we use SQLAlchemy Core for a lightweight, explicit schema.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    BigInteger,
    create_engine,
    select,
    text as sa_text,
)
from sqlalchemy.engine import Engine, Result
from sqlalchemy.exc import ProgrammingError, SQLAlchemyError
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import insert as pg_insert


_metadata = MetaData()
_engine: Optional[Engine] = None
_log = logging.getLogger(__name__)


def _get_database_url() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is not set; Postgres is required for canonical storage.")
    return url


def get_engine() -> Engine:
    """Singleton SQLAlchemy engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(_get_database_url(), pool_pre_ping=True, pool_size=5, max_overflow=5)
    return _engine


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

users = Table(
    "users",
    _metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("chat_id", BigInteger, unique=True, nullable=False, index=True),
    Column("username", String(255), nullable=True),
    Column("password_hash", String(255), nullable=False),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)

memories = Table(
    "memories",
    _metadata,
    Column("memory_id", String(128), primary_key=True),  # stable id, often ISO timestamp or UUID
    # Tenant key: we store Telegram `chat_id` here so it stays compatible
    # with Pinecone metadata and token payloads (user_id == chat_id today).
    Column("user_id", BigInteger, nullable=False, index=True),
    Column("chat_id", BigInteger, nullable=True, index=True),
    Column("title", String(512), nullable=True),
    Column("raw_content_full", Text, nullable=False),
    Column("source_type", String(64), nullable=False, index=True),
    Column("source_url", Text, nullable=True),
    Column("text_fingerprint", String(64), nullable=True, index=True),
    Column("url_fingerprint", String(64), nullable=True, index=True),
    Column("content_type", String(64), nullable=True, index=True),
    Column("topics", JSON, nullable=True),
    Column("tags", JSON, nullable=True),
    Column("created_at_ts", BigInteger, nullable=False, index=True),
    Column("due_at_ts", BigInteger, nullable=True, index=True),
    Column("last_accessed_ts", BigInteger, nullable=True, index=True),
    Column("priority_score", Float, nullable=True),
    Column("last_resurfaced_ts", BigInteger, nullable=True, index=True),
    Column("visibility", String(32), nullable=True),
    Column("parent_id", String(128), nullable=True, index=True),
    Column("tunnel_id", String(128), nullable=True, index=True),
    Column("tunnel_name", String(255), nullable=True, index=True),
    Column("is_full", Boolean, nullable=False, server_default=sa_text("true")),
)

chunks = Table(
    "memory_chunks",
    _metadata,
    Column("chunk_id", String(128), primary_key=True),  # matches Pinecone vector id
    Column("memory_id", String(128), ForeignKey("memories.memory_id", ondelete="CASCADE"), nullable=False, index=True),
    Column("user_id", BigInteger, nullable=False, index=True),
    Column("chat_id", BigInteger, nullable=True, index=True),
    Column("chunk_index", Integer, nullable=False),
    Column("chunk_text", Text, nullable=False),
    Column("source_type", String(64), nullable=False, index=True),
    Column("created_at_ts", BigInteger, nullable=False, index=True),
)

reminders = Table(
    "reminders",
    _metadata,
    Column("id", String(128), primary_key=True),
    Column("user_id", BigInteger, nullable=False, index=True),
    Column("chat_id", BigInteger, nullable=True, index=True),
    Column("text", Text, nullable=False),
    Column("due_at_ts", BigInteger, nullable=False, index=True),
    Column("timezone", String(64), nullable=True),
    Column("created_at_ts", BigInteger, nullable=False),
    Column("fired", Boolean, nullable=False, server_default=sa_text("false")),
)

tunnels = Table(
    "tunnels",
    _metadata,
    Column("id", String(128), primary_key=True),
    Column("user_id", BigInteger, nullable=False, index=True),
    Column("name", String(255), nullable=False),
    Column("reason", Text, nullable=True),
    Column("core_tag", String(128), nullable=True, index=True),
    Column("memory_count", Integer, nullable=True),
    Column("created_at_ts", BigInteger, nullable=False, index=True),
    Column("raw", Text, nullable=False),
)

tunnel_members = Table(
    "tunnel_members",
    _metadata,
    Column("tunnel_id", String(128), ForeignKey("tunnels.id", ondelete="CASCADE"), primary_key=True),
    Column("memory_id", String(128), ForeignKey("memories.memory_id", ondelete="CASCADE"), primary_key=True),
    Column("user_id", BigInteger, nullable=False, index=True),
)

tunnel_edges = Table(
    "tunnel_edges",
    _metadata,
    Column("tunnel_id", String(128), ForeignKey("tunnels.id", ondelete="CASCADE"), primary_key=True),
    Column("from_memory_id", String(128), ForeignKey("memories.memory_id", ondelete="CASCADE"), primary_key=True),
    Column("to_memory_id", String(128), ForeignKey("memories.memory_id", ondelete="CASCADE"), primary_key=True),
    Column("user_id", BigInteger, nullable=False, index=True),
    Column("weight", Float, nullable=True),
    Column("bridge_score", Float, nullable=True),
    Column("rationale", Text, nullable=True),
)


# Existing deployments may have created `memories` before enrichment columns existed.
# SQLAlchemy create_all() does not ALTER existing tables, so we add columns idempotently.
_MEMORY_SCHEMA_PATCH_STATEMENTS: tuple[str, ...] = (
    "ALTER TABLE public.memories ADD COLUMN IF NOT EXISTS text_fingerprint text NULL",
    "ALTER TABLE public.memories ADD COLUMN IF NOT EXISTS url_fingerprint text NULL",
    "ALTER TABLE public.memories ADD COLUMN IF NOT EXISTS content_type text NULL",
    "ALTER TABLE public.memories ADD COLUMN IF NOT EXISTS topics jsonb NULL",
    "CREATE INDEX IF NOT EXISTS memories_chat_text_fp_idx ON public.memories (chat_id, text_fingerprint)",
    "CREATE INDEX IF NOT EXISTS memories_chat_url_fp_idx ON public.memories (chat_id, url_fingerprint)",
    "CREATE INDEX IF NOT EXISTS memories_user_content_type_idx ON public.memories (user_id, content_type)",
)


def init_db() -> None:
    """
    Create tables if they do not exist.

    This is idempotent and safe to call at startup.
    """
    try:
        engine = get_engine()
    except RuntimeError as exc:
        _log.warning("Skipping Postgres init (DATABASE_URL missing): %s", exc)
        return

    try:
        _metadata.create_all(engine)
        with engine.begin() as conn:
            for stmt in _MEMORY_SCHEMA_PATCH_STATEMENTS:
                conn.execute(sa_text(stmt))
        _log.info("Postgres schema ensured (tables + memory enrichment columns).")
    except SQLAlchemyError as exc:
        raise RuntimeError(f"Failed to initialize Postgres schema: {exc}") from exc


# ---------------------------------------------------------------------------
# Convenience helpers (minimal for now; can be expanded per feature)
# ---------------------------------------------------------------------------

@dataclass
class UserRow:
    id: int
    chat_id: int
    username: str | None


@dataclass
class UserAuthRow:
    id: int
    chat_id: int
    username: str | None
    password_hash: str


def get_or_create_user(chat_id: int, username: str | None, password_hash: str | None = None) -> UserRow:
    """
    Ensure a user row exists for a given Telegram chat_id.
    For dashboard signups, password_hash should already be hashed.
    """
    engine = get_engine()
    with engine.begin() as conn:
        res: Result = conn.execute(select(users).where(users.c.chat_id == chat_id))
        row = res.fetchone()
        if row:
            return UserRow(id=row.id, chat_id=row.chat_id, username=row.username)

        if password_hash is None:
            # Placeholder hash for system-created users; should be updated via signup.
            password_hash = "!"

        insert_stmt = users.insert().values(chat_id=chat_id, username=username, password_hash=password_hash)
        result = conn.execute(insert_stmt)
        user_id = result.inserted_primary_key[0]
        return UserRow(id=int(user_id), chat_id=chat_id, username=username)


def get_user_by_chat_id(chat_id: int) -> UserAuthRow | None:
    """
    Fetch a user row for auth (password verification).
    """
    engine = get_engine()
    with engine.begin() as conn:
        res: Result = conn.execute(select(users).where(users.c.chat_id == chat_id))
        row = res.fetchone()
        if not row:
            return None
        return UserAuthRow(
            id=int(row.id),
            chat_id=int(row.chat_id),
            username=row.username,
            password_hash=str(row.password_hash),
        )


_MEMORY_INSERT_ENRICHMENT_KEYS = frozenset({"text_fingerprint", "url_fingerprint", "content_type", "topics"})


def insert_memory(row: dict[str, Any]) -> None:
    """
    Insert a canonical memory row.
    Expects keys compatible with the 'memories' table.
    """
    try:
        engine = get_engine()
    except RuntimeError as exc:
        _log.warning("Skipping insert_memory (DATABASE_URL missing): %s", exc)
        return

    def _execute_INSERT(r: dict[str, Any]) -> None:
        with engine.begin() as conn:
            stmt = pg_insert(memories).values(**r).on_conflict_do_nothing(index_elements=["memory_id"])
            conn.execute(stmt)

    try:
        _execute_INSERT(row)
    except ProgrammingError as exc:
        err = str(exc).lower()
        if "undefinedcolumn" in err or "does not exist" in err:
            slim = {k: v for k, v in row.items() if k not in _MEMORY_INSERT_ENRICHMENT_KEYS}
            _log.warning(
                "insert_memory retry without enrichment columns (migrate DB / restart app for auto-patch): %s",
                exc,
            )
            _execute_INSERT(slim)
        else:
            raise
    _log.debug(
        "pg insert_memory ok memory_id=%s user_id=%s source_type=%s",
        row.get("memory_id"),
        row.get("user_id"),
        row.get("source_type"),
    )


def insert_chunks(rows: Iterable[dict[str, Any]]) -> None:
    """Bulk insert chunk rows."""
    try:
        engine = get_engine()
    except RuntimeError as exc:
        _log.warning("Skipping insert_chunks (DATABASE_URL missing): %s", exc)
        return

    engine_rows = list(rows)
    if not engine_rows:
        return
    with engine.begin() as conn:
        stmt = pg_insert(chunks).values(engine_rows).on_conflict_do_nothing(index_elements=["chunk_id"])
        conn.execute(stmt)
    _log.debug("pg insert_chunks ok count=%d", len(engine_rows))


def fetch_memories_for_user_created_range(
    *,
    user_id: int,
    start_ts: int,
    end_ts: int,
    limit: int = 400,
) -> List[Dict[str, Any]]:
    """
    Deterministic time-window selection from Postgres canonical memories.
    Returns ONLY main records (memories table stores main records only in our dual-write).
    """
    engine = get_engine()
    sql = sa_text(
        """
        SELECT
          memory_id,
          chat_id,
          title,
          raw_content_full,
          source_type,
          source_url,
          tags,
          created_at_ts,
          last_accessed_ts,
          priority_score,
          last_resurfaced_ts,
          visibility,
          parent_id,
          is_full,
          due_at_ts
        FROM memories
        WHERE
          user_id = :user_id
          AND created_at_ts >= :start_ts
          AND created_at_ts < :end_ts
          AND is_full = true
        ORDER BY created_at_ts DESC
        LIMIT :limit
        """
    )
    with engine.begin() as conn:
        rows = conn.execute(sql, {"user_id": user_id, "start_ts": start_ts, "end_ts": end_ts, "limit": limit}).fetchall()

    out: List[Dict[str, Any]] = []
    for r in rows:
        d = dict(r._mapping)
        out.append(
            {
                "id": d.get("memory_id"),
                "chat_id": d.get("chat_id"),
                "title": d.get("title"),
                "raw_content": d.get("raw_content_full"),
                "source_type": d.get("source_type"),
                "url": d.get("source_url"),
                "tags": d.get("tags") or [],
                "created_at_ts": d.get("created_at_ts"),
                "last_accessed_ts": d.get("last_accessed_ts"),
                "priority_score": d.get("priority_score"),
                "last_resurfaced_ts": d.get("last_resurfaced_ts"),
                "visibility": d.get("visibility"),
                "parent_id": d.get("parent_id"),
                "is_full": d.get("is_full"),
                "due_at_ts": d.get("due_at_ts"),
            }
        )
    return out


def fetch_upcoming_reminders_for_user(
    *,
    user_id: int,
    now_ts: int,
    window_end_ts: int,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Fetch reminder memories due in [now_ts, window_end_ts).
    """
    engine = get_engine()
    sql = sa_text(
        """
        SELECT
          memory_id,
          raw_content_full,
          created_at_ts,
          due_at_ts
        FROM memories
        WHERE
          user_id = :user_id
          AND source_type = 'reminder'
          AND is_full = true
          AND due_at_ts >= :now_ts
          AND due_at_ts < :window_end_ts
        ORDER BY due_at_ts ASC
        LIMIT :limit
        """
    )
    with engine.begin() as conn:
        rows = conn.execute(sql, {"user_id": user_id, "now_ts": now_ts, "window_end_ts": window_end_ts, "limit": limit}).fetchall()

    out: List[Dict[str, Any]] = []
    for r in rows:
        d = dict(r._mapping)
        out.append(
            {
                "id": d.get("memory_id"),
                "reminder_text": d.get("raw_content_full"),
                "created_at_ts": d.get("created_at_ts"),
                "due_at_ts": d.get("due_at_ts"),
            }
        )
    return out


def fetch_old_main_memories_for_user(
    *,
    user_id: int,
    older_than_ts: int,
    exclude_source_types: List[str] | None = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    Deterministic selection of old main memories for resurfacing.
    """
    engine = get_engine()
    sql = sa_text(
        """
        SELECT
          memory_id,
          raw_content_full,
          source_type,
          tags,
          created_at_ts,
          last_accessed_ts,
          priority_score,
          last_resurfaced_ts
        FROM memories
        WHERE
          user_id = :user_id
          AND is_full = true
          AND created_at_ts <= :older_than_ts
        ORDER BY created_at_ts ASC
        LIMIT :limit
        """
    )
    with engine.begin() as conn:
        rows = conn.execute(sql, {"user_id": user_id, "older_than_ts": older_than_ts, "limit": limit}).fetchall()

    out: List[Dict[str, Any]] = []
    excl = set(exclude_source_types or [])
    for r in rows:
        d = dict(r._mapping)
        st = str(d.get("source_type") or "")
        if st in excl:
            continue
        out.append(
            {
                "id": d.get("memory_id"),
                "raw_content": d.get("raw_content_full"),
                "source_type": st,
                "tags": d.get("tags") or [],
                "created_at_ts": d.get("created_at_ts"),
                "last_accessed_ts": d.get("last_accessed_ts"),
                "priority_score": d.get("priority_score"),
                "last_resurfaced_ts": d.get("last_resurfaced_ts"),
            }
        )
    return out


def fetch_main_memories_for_user_for_profile(
    *,
    user_id: int,
    exclude_source_types: List[str] | None = None,
    limit: int = 500,
) -> List[Dict[str, Any]]:
    """
    Fetch main memories for profile snapshot computation.
    Returns canonical fields needed for topic/tunnel/time analysis.
    """
    engine = get_engine()
    excl = exclude_source_types or []

    if excl:
        sql = sa_text(
            """
            SELECT
              memory_id,
              raw_content_full,
              tags,
              tunnel_name,
              created_at_ts
            FROM memories
            WHERE
              user_id = :user_id
              AND is_full = true
              AND source_type not in :excl
            ORDER BY created_at_ts DESC
            LIMIT :limit
            """
        )
        params: Dict[str, Any] = {"user_id": user_id, "excl": tuple(excl), "limit": limit}
    else:
        sql = sa_text(
            """
            SELECT
              memory_id,
              raw_content_full,
              tags,
              tunnel_name,
              created_at_ts
            FROM memories
            WHERE
              user_id = :user_id
              AND is_full = true
            ORDER BY created_at_ts DESC
            LIMIT :limit
            """
        )
        params = {"user_id": user_id, "limit": limit}

    with engine.begin() as conn:
        rows = conn.execute(sql, params).fetchall()

    out: List[Dict[str, Any]] = []
    for r in rows:
        d = dict(r._mapping)
        out.append(
            {
                "id": d.get("memory_id"),
                "raw_content": d.get("raw_content_full"),
                "tags": d.get("tags") or [],
                "tunnel_name": d.get("tunnel_name"),
                "created_at_ts": d.get("created_at_ts"),
            }
        )
    return out


def fetch_main_memories_for_user_for_tunnels(
    *,
    user_id: int,
    exclude_source_types: List[str] | None = None,
    limit: int = 400,
) -> List[Dict[str, Any]]:
    """
    Fetch main (is_full=true) memories for tunnel candidate generation.
    Uses canonical Postgres storage.
    """
    engine = get_engine()

    params: Dict[str, Any] = {"user_id": user_id, "limit": limit}
    excl = exclude_source_types or []

    if excl:
        sql = sa_text(
            """
            SELECT
              memory_id,
              raw_content_full,
              tags,
              source_type,
              created_at_ts
            FROM memories
            WHERE
              user_id = :user_id
              AND is_full = true
              AND source_type not in :excl
            ORDER BY created_at_ts DESC
            LIMIT :limit
            """
        )
        params["excl"] = tuple(excl)
    else:
        sql = sa_text(
            """
            SELECT
              memory_id,
              raw_content_full,
              tags,
              source_type,
              created_at_ts
            FROM memories
            WHERE
              user_id = :user_id
              AND is_full = true
            ORDER BY created_at_ts DESC
            LIMIT :limit
            """
        )

    with engine.begin() as conn:
        rows = conn.execute(sql, params).fetchall()

    out: List[Dict[str, Any]] = []
    for r in rows:
        d = dict(r._mapping)
        out.append(
            {
                "id": d.get("memory_id"),
                "raw_content": d.get("raw_content_full"),
                "tags": d.get("tags") or [],
                "source_type": d.get("source_type"),
                "created_at_ts": d.get("created_at_ts"),
            }
        )
    return out


def insert_tunnel_and_members(
    *,
    tunnel_id: str,
    user_id: int,
    name: str,
    reason: str,
    core_tag: str,
    memory_count: int,
    created_at_ts: int,
    raw: str,
    member_memory_ids: List[str],
) -> None:
    """
    Persist tunnel + member links into Postgres.
    """
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            tunnels.insert().values(
                id=tunnel_id,
                user_id=user_id,
                name=name,
                reason=reason,
                core_tag=core_tag,
                memory_count=memory_count,
                created_at_ts=created_at_ts,
                raw=raw,
            )
        )
        if member_memory_ids:
            conn.execute(
                tunnel_members.insert(),
                [
                    {"tunnel_id": tunnel_id, "memory_id": mid, "user_id": user_id}
                    for mid in member_memory_ids
                ],
            )


def update_memory_tunnel_fields(*, user_id: int, memory_id: str, tunnel_id: str, tunnel_name: str) -> None:
    """Stamp tunnel_id/tunnel_name onto a canonical memory row."""
    engine = get_engine()
    sql = sa_text(
        """
        UPDATE memories
        SET tunnel_id = :tunnel_id,
            tunnel_name = :tunnel_name
        WHERE user_id = :user_id AND memory_id = :memory_id
        """
    )
    with engine.begin() as conn:
        conn.execute(sql, {"tunnel_id": tunnel_id, "tunnel_name": tunnel_name, "user_id": user_id, "memory_id": memory_id})


def delete_tunnel_edges_for_tunnel(*, user_id: int, tunnel_id: str) -> None:
    engine = get_engine()
    sql = sa_text(
        """
        DELETE FROM tunnel_edges
        WHERE user_id = :user_id AND tunnel_id = :tunnel_id
        """
    )
    with engine.begin() as conn:
        conn.execute(sql, {"user_id": user_id, "tunnel_id": tunnel_id})


def fetch_tunnel_core_tag(*, user_id: int, tunnel_id: str) -> Optional[str]:
    engine = get_engine()
    sql = sa_text(
        """
        SELECT core_tag FROM tunnels
        WHERE user_id = :user_id AND id = :tunnel_id
        LIMIT 1
        """
    )
    with engine.begin() as conn:
        row = conn.execute(sql, {"user_id": user_id, "tunnel_id": tunnel_id}).fetchone()
    if not row:
        return None
    raw = dict(row._mapping).get("core_tag")
    if raw is None:
        return "semantic"
    s = str(raw).strip()
    return s if s else "semantic"


def fetch_tunnel_member_memories_for_edges(*, user_id: int, tunnel_id: str) -> List[Dict[str, Any]]:
    """Member memories with fields needed to rebuild tunnel edges."""
    engine = get_engine()
    sql = sa_text(
        """
        SELECT
          m.memory_id,
          m.raw_content_full,
          m.tags,
          m.source_type,
          m.created_at_ts
        FROM tunnel_members tm
        JOIN memories m ON m.memory_id = tm.memory_id
        WHERE tm.user_id = :user_id
          AND tm.tunnel_id = :tunnel_id
        ORDER BY m.created_at_ts DESC
        """
    )
    with engine.begin() as conn:
        rows = conn.execute(sql, {"user_id": user_id, "tunnel_id": tunnel_id}).fetchall()
    out: List[Dict[str, Any]] = []
    for r in rows:
        d = dict(r._mapping)
        out.append(
            {
                "id": d.get("memory_id"),
                "raw_content": d.get("raw_content_full"),
                "tags": d.get("tags") or [],
                "source_type": d.get("source_type"),
                "created_at_ts": d.get("created_at_ts"),
            }
        )
    return out


def fetch_two_memories_for_user(
    *, user_id: int, memory_id_a: str, memory_id_b: str
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    engine = get_engine()
    sql = sa_text(
        """
        SELECT
          memory_id,
          title,
          raw_content_full,
          source_type,
          tags,
          created_at_ts
        FROM memories
        WHERE user_id = :user_id
          AND memory_id IN (:id_a, :id_b)
          AND is_full = true
        """
    )
    with engine.begin() as conn:
        rows = conn.execute(
            sql, {"user_id": user_id, "id_a": memory_id_a, "id_b": memory_id_b}
        ).fetchall()
    by_id: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        d = dict(r._mapping)
        mid = str(d.get("memory_id") or "")
        by_id[mid] = {
            "id": mid,
            "memory_id": mid,
            "title": d.get("title"),
            "raw_content_full": d.get("raw_content_full"),
            "source_type": d.get("source_type"),
            "tags": d.get("tags") or [],
            "created_at_ts": d.get("created_at_ts"),
        }
    return by_id.get(memory_id_a), by_id.get(memory_id_b)


def verify_both_memories_in_tunnel(
    *, user_id: int, tunnel_id: str, memory_id_a: str, memory_id_b: str
) -> bool:
    engine = get_engine()
    sql = sa_text(
        """
        SELECT COUNT(*) AS c FROM tunnel_members
        WHERE user_id = :user_id
          AND tunnel_id = :tunnel_id
          AND memory_id IN (:id_a, :id_b)
        """
    )
    with engine.begin() as conn:
        row = conn.execute(
            sql, {"user_id": user_id, "tunnel_id": tunnel_id, "id_a": memory_id_a, "id_b": memory_id_b}
        ).fetchone()
    if not row:
        return False
    return int(dict(row._mapping).get("c") or 0) >= 2


def fetch_tunnel_edge_rationale(
    *,
    user_id: int,
    tunnel_id: str,
    from_memory_id: str,
    to_memory_id: str,
) -> Optional[str]:
    engine = get_engine()
    sql = sa_text(
        """
        SELECT rationale FROM tunnel_edges
        WHERE user_id = :user_id
          AND tunnel_id = :tunnel_id
          AND (
            (from_memory_id = :a AND to_memory_id = :b)
            OR (from_memory_id = :b AND to_memory_id = :a)
          )
        LIMIT 1
        """
    )
    with engine.begin() as conn:
        row = conn.execute(
            sql,
            {
                "user_id": user_id,
                "tunnel_id": tunnel_id,
                "a": from_memory_id,
                "b": to_memory_id,
            },
        ).fetchone()
    if not row:
        return None
    r = dict(row._mapping).get("rationale")
    return str(r).strip() if r else None


def insert_tunnel_edges(
    *,
    tunnel_id: str,
    user_id: int,
    edges: List[Dict[str, Any]],
) -> None:
    """
    Persist tunnel graph edges with semantic rationale.
    """
    if not edges:
        return
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            tunnel_edges.insert(),
            [
                {
                    "tunnel_id": tunnel_id,
                    "from_memory_id": str(e.get("from_memory_id") or ""),
                    "to_memory_id": str(e.get("to_memory_id") or ""),
                    "user_id": user_id,
                    "weight": e.get("weight"),
                    "bridge_score": e.get("bridge_score"),
                    "rationale": str(e.get("rationale") or "").strip() or None,
                }
                for e in edges
                if str(e.get("from_memory_id") or "").strip() and str(e.get("to_memory_id") or "").strip()
            ],
        )


def update_memory_last_resurfaced_ts(*, user_id: int, memory_id: str, last_resurfaced_ts: int) -> None:
    """Update last_resurfaced_ts for canonical memory rows."""
    engine = get_engine()
    sql = sa_text(
        """
        UPDATE memories
        SET last_resurfaced_ts = :ts
        WHERE user_id = :user_id AND memory_id = :memory_id
        """
    )
    with engine.begin() as conn:
        conn.execute(sql, {"ts": last_resurfaced_ts, "user_id": user_id, "memory_id": memory_id})


def fetch_tunnels_for_user(
    *,
    user_id: int,
    limit: int = 30,
    min_memory_count: int = 4,
) -> List[Dict[str, Any]]:
    engine = get_engine()
    sql = sa_text(
        """
        SELECT id, name, reason, core_tag, memory_count, created_at_ts, raw
        FROM tunnels
        WHERE user_id = :user_id
          AND COALESCE(memory_count, 0) >= :min_memory_count
        ORDER BY created_at_ts DESC
        LIMIT :limit
        """
    )
    with engine.begin() as conn:
        rows = conn.execute(
            sql,
            {
                "user_id": user_id,
                "limit": limit,
                "min_memory_count": max(0, int(min_memory_count)),
            },
        ).fetchall()
    out: List[Dict[str, Any]] = []
    for r in rows:
        d = dict(r._mapping)
        out.append(
            {
                "id": d.get("id"),
                "tunnel_name": d.get("name"),
                "reason": d.get("reason"),
                "core_tag": d.get("core_tag"),
                "memory_count": d.get("memory_count"),
                "created_at_ts": d.get("created_at_ts"),
                "raw_content": d.get("raw"),
                "source_type": "tunnel",
            }
        )
    return out


def fetch_tunnel_graph_for_user(
    *, user_id: int, tunnel_id: str, min_bridge: Optional[float] = None
) -> Dict[str, Any]:
    """
    Fetch a tunnel graph payload with memory nodes and semantic edges.
    If min_bridge is set, drop edges with bridge_score strictly below it.
    """
    engine = get_engine()
    node_sql = sa_text(
        """
        SELECT
          m.memory_id,
          m.title,
          m.raw_content_full,
          m.tags,
          m.source_type,
          m.created_at_ts
        FROM tunnel_members tm
        JOIN memories m ON m.memory_id = tm.memory_id
        WHERE tm.user_id = :user_id
          AND tm.tunnel_id = :tunnel_id
        ORDER BY m.created_at_ts DESC
        """
    )
    edge_sql = sa_text(
        """
        SELECT
          from_memory_id,
          to_memory_id,
          weight,
          bridge_score,
          rationale
        FROM tunnel_edges
        WHERE user_id = :user_id
          AND tunnel_id = :tunnel_id
        ORDER BY bridge_score DESC NULLS LAST, weight DESC NULLS LAST
        """
    )
    with engine.begin() as conn:
        node_rows = conn.execute(node_sql, {"user_id": user_id, "tunnel_id": tunnel_id}).fetchall()
        try:
            edge_rows = conn.execute(edge_sql, {"user_id": user_id, "tunnel_id": tunnel_id}).fetchall()
        except Exception:
            # Backward compatibility when tunnel_edges migration has not been applied yet.
            edge_rows = []

    nodes: List[Dict[str, Any]] = []
    for r in node_rows:
        d = dict(r._mapping)
        raw = str(d.get("raw_content_full") or "").strip()
        nodes.append(
            {
                "id": d.get("memory_id"),
                "title": d.get("title") or "",
                "snippet": raw[:280],
                "tags": d.get("tags") or [],
                "source_type": d.get("source_type") or "",
                "created_at_ts": d.get("created_at_ts"),
            }
        )

    edges: List[Dict[str, Any]] = []
    for r in edge_rows:
        d = dict(r._mapping)
        edges.append(
            {
                "from_memory_id": d.get("from_memory_id"),
                "to_memory_id": d.get("to_memory_id"),
                "weight": d.get("weight"),
                "bridge_score": d.get("bridge_score"),
                "rationale": d.get("rationale") or "",
            }
        )
    if min_bridge is not None and min_bridge > 0.0:
        edges = [e for e in edges if float(e.get("bridge_score") or 0.0) >= min_bridge]
    return {"nodes": nodes, "edges": edges}


def find_memory_id_by_text_fingerprint(*, chat_id: int, text_fingerprint: str) -> str | None:
    """
    Lookup exact duplicate by normalized text fingerprint.
    Returns memory_id if found, else None.
    """
    if not text_fingerprint:
        return None
    try:
        engine = get_engine()
    except Exception:
        return None
    sql = sa_text(
        """
        SELECT memory_id
        FROM memories
        WHERE chat_id = :chat_id
          AND is_full = true
          AND text_fingerprint = :fp
        ORDER BY created_at_ts DESC
        LIMIT 1
        """
    )
    try:
        with engine.begin() as conn:
            row = conn.execute(sql, {"chat_id": chat_id, "fp": text_fingerprint}).fetchone()
        if row:
            return str(dict(row._mapping).get("memory_id") or "").strip() or None
    except Exception:
        return None
    return None


def find_memory_id_by_url_fingerprint(*, chat_id: int, url_fingerprint: str) -> str | None:
    """
    Lookup exact duplicate link by canonical URL fingerprint.
    Returns memory_id if found, else None.
    """
    if not url_fingerprint:
        return None
    try:
        engine = get_engine()
    except Exception:
        return None
    sql = sa_text(
        """
        SELECT memory_id
        FROM memories
        WHERE chat_id = :chat_id
          AND is_full = true
          AND source_type = 'link'
          AND url_fingerprint = :fp
        ORDER BY created_at_ts DESC
        LIMIT 1
        """
    )
    try:
        with engine.begin() as conn:
            row = conn.execute(sql, {"chat_id": chat_id, "fp": url_fingerprint}).fetchone()
        if row:
            return str(dict(row._mapping).get("memory_id") or "").strip() or None
    except Exception:
        return None
    return None


