"""Tests for the rollback engine — snapshot, checkpoint, and restore."""

from __future__ import annotations

import pytest

from aria.ledger.store import LedgerStore
from aria.rollback.engine import FileSnapshot, RollbackEngine


@pytest.fixture
async def ledger(tmp_path):
    store = LedgerStore(db_path=tmp_path / "test.db")
    await store.connect()
    yield store
    await store.close()


@pytest.fixture
def engine(ledger):
    return RollbackEngine(ledger=ledger)


class TestRollbackEngine:
    async def test_file_snapshot_and_restore(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("original content")

        snap = FileSnapshot(str(test_file))
        data = await snap.snapshot()

        # Overwrite the file
        test_file.write_text("corrupted content")
        assert test_file.read_text() == "corrupted content"

        # Restore
        await snap.restore(data)
        assert test_file.read_text() == "original content"

    async def test_rollback_restores_file(self, tmp_path, engine):
        test_file = tmp_path / "data.txt"
        test_file.write_text("before change")

        snap = FileSnapshot(str(test_file))
        snap_key = await engine.register_snapshot(
            key="file_snap", adapter=snap, session_id="sess_rb"
        )

        ckpt_id = await engine.create_checkpoint(
            session_id="sess_rb",
            label="safe_point",
            snapshot_keys=[snap_key],
        )

        # Simulate a bad write
        test_file.write_text("after bad write")
        assert test_file.read_text() == "after bad write"

        success = await engine.rollback("sess_rb", to=ckpt_id)
        assert success is True
        assert test_file.read_text() == "before change"

    async def test_rollback_nonexistent_checkpoint_returns_false(self, engine):
        result = await engine.rollback("sess_x", to="nonexistent_ckpt")
        assert result is False

    async def test_multiple_checkpoints(self, tmp_path, engine):
        test_file = tmp_path / "multi.txt"
        test_file.write_text("v1")

        snap = FileSnapshot(str(test_file))
        snap_key = await engine.register_snapshot("s1", snap, "sess_m")
        ckpt1 = await engine.create_checkpoint("sess_m", label="v1", snapshot_keys=[snap_key])

        test_file.write_text("v2")

        snap2 = FileSnapshot(str(test_file))
        snap_key2 = await engine.register_snapshot("s2", snap2, "sess_m")
        ckpt2 = await engine.create_checkpoint("sess_m", label="v2", snapshot_keys=[snap_key2])

        # Roll back to ckpt1 — should restore v1
        await engine.rollback("sess_m", to=ckpt1)
        assert test_file.read_text() == "v1"

    async def test_file_snapshot_nonexistent_file(self, tmp_path):
        """Snapshots of non-existent files restore as deletion."""
        test_file = tmp_path / "new_file.txt"
        # File doesn't exist yet
        snap = FileSnapshot(str(test_file))
        data = await snap.snapshot()
        assert data is None

        # File created by agent
        test_file.write_text("created by agent")

        # Rollback should delete it
        await snap.restore(data)
        assert not test_file.exists()

    async def test_list_checkpoints(self, engine):
        await engine.create_checkpoint("sess_lc", label="first")
        await engine.create_checkpoint("sess_lc", label="second")

        checkpoints = engine.list_checkpoints("sess_lc")
        assert len(checkpoints) == 2
        assert checkpoints[0].label == "first"
        assert checkpoints[1].label == "second"
