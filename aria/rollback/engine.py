"""Rollback engine — snapshot-and-restore for reversible agent actions."""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from aria.models import AuditEvent, Checkpoint, EventType

logger = logging.getLogger(__name__)


@runtime_checkable
class Snapshotable(Protocol):
    """
    Interface any tool with side effects must implement to support rollback.

    ARIA calls ``snapshot()`` *before* execution and ``restore(data)``
    during a rollback.
    """

    async def snapshot(self) -> Any:
        """Capture current state. Return any serialisable object."""
        ...

    async def restore(self, data: Any) -> None:
        """Restore state from a previously captured snapshot."""
        ...


class FileSnapshot:
    """Built-in snapshot adapter for file write operations."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._original: bytes | None = None

    async def snapshot(self) -> bytes | None:
        try:
            with open(self.path, "rb") as fh:
                self._original = fh.read()
            return self._original
        except FileNotFoundError:
            self._original = None
            return None

    async def restore(self, data: bytes | None) -> None:
        if data is None:
            try:
                import os
                os.remove(self.path)
                logger.info("Rollback: deleted %s (did not exist before)", self.path)
            except FileNotFoundError:
                pass
        else:
            with open(self.path, "wb") as fh:
                fh.write(data)
            logger.info("Rollback: restored %s (%d bytes)", self.path, len(data))


class RollbackEngine:
    """
    Manages session checkpoints and restores state on demand.

    Usage::

        ckpt_id = await engine.create_checkpoint(session_id, label="before_write")
        # ... agent does stuff ...
        await engine.rollback(session_id, to=ckpt_id)
    """

    def __init__(self, ledger: Any) -> None:
        self.ledger = ledger
        # session_id -> list of (checkpoint, snapshots dict)
        self._checkpoints: dict[str, list[tuple[Checkpoint, dict[str, Any]]]] = {}
        # snapshot registry: snapshot_key -> (adapter, data)
        self._snapshots: dict[str, tuple[Snapshotable, Any]] = {}

    async def register_snapshot(
        self, key: str, adapter: Snapshotable, session_id: str
    ) -> str:
        """Capture a snapshot via adapter and store it under key."""
        data = await adapter.snapshot()
        self._snapshots[key] = (adapter, data)
        logger.debug("Snapshot registered: %s", key)
        return key

    async def create_checkpoint(
        self,
        session_id: str,
        label: str = "",
        snapshot_keys: list[str] | None = None,
    ) -> str:
        checkpoint = Checkpoint(
            session_id=session_id,
            label=label,
            snapshot_refs=snapshot_keys or [],
        )
        if session_id not in self._checkpoints:
            self._checkpoints[session_id] = []
        self._checkpoints[session_id].append((checkpoint, dict(self._snapshots)))

        event = AuditEvent(
            session_id=session_id,
            agent_id="system",
            event_type=EventType.CHECKPOINT_PENDING,
            metadata={"checkpoint_id": checkpoint.checkpoint_id, "label": label},
        )
        await self.ledger.append(event)
        logger.info("Checkpoint created: %s (%s)", checkpoint.checkpoint_id, label)
        return checkpoint.checkpoint_id

    async def rollback(self, session_id: str, to: str) -> bool:
        """
        Restore all snapshots captured at checkpoint ``to`` in reverse order.
        Returns True if the rollback succeeded.
        """
        checkpoints = self._checkpoints.get(session_id, [])
        target: tuple[Checkpoint, dict[str, Any]] | None = None
        for ckpt, snapshots in checkpoints:
            if ckpt.checkpoint_id == to:
                target = (ckpt, snapshots)
                break

        if target is None:
            logger.error("Checkpoint %s not found for session %s", to, session_id)
            return False

        ckpt, snapshots = target
        for key in reversed(ckpt.snapshot_refs):
            if key in snapshots:
                adapter, data = snapshots[key]
                await adapter.restore(data)

        event = AuditEvent(
            session_id=session_id,
            agent_id="system",
            event_type=EventType.ROLLBACK,
            metadata={"checkpoint_id": to},
        )
        await self.ledger.append(event)
        logger.info("Rollback completed to checkpoint %s", to)
        return True

    def list_checkpoints(self, session_id: str) -> list[Checkpoint]:
        return [ckpt for ckpt, _ in self._checkpoints.get(session_id, [])]
