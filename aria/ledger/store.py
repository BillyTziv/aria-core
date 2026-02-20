"""SQLite-backed append-only audit event store."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite

from aria.models import AuditEvent, EventType

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path.home() / ".aria" / "ledger.db"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS audit_events (
    event_id    TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    agent_id    TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    input_data  TEXT NOT NULL,
    output_data TEXT NOT NULL,
    confidence  REAL,
    estimated_cost REAL,
    timestamp   TEXT NOT NULL,
    metadata    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_session ON audit_events (session_id);
CREATE INDEX IF NOT EXISTS idx_agent   ON audit_events (agent_id);
CREATE INDEX IF NOT EXISTS idx_type    ON audit_events (event_type);
"""


class LedgerStore:
    """
    Append-only SQLite audit ledger.

    No UPDATE or DELETE statements are ever issued — the ledger is
    immutable by design. The full system state can be reconstructed
    by replaying events in chronological order.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path or os.environ.get("ARIA_LEDGER_PATH", DEFAULT_DB_PATH))
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_CREATE_TABLE)
        await self._db.commit()
        logger.debug("Ledger connected: %s", self.db_path)

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def append(self, event: AuditEvent) -> None:
        if not self._db:
            raise RuntimeError("LedgerStore not connected. Call await ledger.connect() first.")
        await self._db.execute(
            """
            INSERT INTO audit_events
                (event_id, session_id, agent_id, event_type,
                 input_data, output_data, confidence, estimated_cost,
                 timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.session_id,
                event.agent_id,
                event.event_type.value,
                json.dumps(event.input_data),
                json.dumps(event.output_data),
                event.confidence,
                event.estimated_cost,
                event.timestamp.isoformat(),
                json.dumps(event.metadata),
            ),
        )
        await self._db.commit()

    async def query(
        self,
        session_id: str | None = None,
        agent_id: str | None = None,
        event_type: EventType | None = None,
        since: datetime | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        if not self._db:
            raise RuntimeError("LedgerStore not connected.")

        conditions: list[str] = []
        params: list[Any] = []

        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if agent_id:
            conditions.append("agent_id = ?")
            params.append(agent_id)
        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type.value)
        if since:
            conditions.append("timestamp >= ?")
            params.append(since.isoformat())

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        async with self._db.execute(
            f"SELECT * FROM audit_events {where} ORDER BY timestamp ASC LIMIT ?",
            params,
        ) as cursor:
            rows = await cursor.fetchall()

        return [dict(row) for row in rows]

    async def sessions(self) -> list[str]:
        """Return all distinct session IDs in the ledger, newest first."""
        if not self._db:
            raise RuntimeError("LedgerStore not connected.")
        async with self._db.execute(
            "SELECT session_id, MIN(timestamp) AS first_seen "
            "FROM audit_events GROUP BY session_id ORDER BY first_seen DESC"
        ) as cursor:
            rows = await cursor.fetchall()
        return [row[0] for row in rows]
