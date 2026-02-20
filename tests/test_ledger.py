"""Tests for the audit ledger — append-only writes and query filters."""

from __future__ import annotations

import pytest

from aria.ledger.store import LedgerStore
from aria.models import AuditEvent, EventType


@pytest.fixture
async def ledger(tmp_path):
    store = LedgerStore(db_path=tmp_path / "ledger.db")
    await store.connect()
    yield store
    await store.close()


def make_event(**kwargs) -> AuditEvent:
    defaults = {
        "session_id": "sess_test",
        "agent_id": "agent_test",
        "event_type": EventType.TOOL_CALL,
    }
    defaults.update(kwargs)
    return AuditEvent(**defaults)


class TestLedgerAppend:
    async def test_append_and_retrieve(self, ledger):
        event = make_event(event_type=EventType.AGENT_START)
        await ledger.append(event)

        results = await ledger.query(session_id="sess_test")
        assert len(results) == 1
        assert results[0]["event_id"] == event.event_id

    async def test_multiple_events_ordered_by_time(self, ledger):
        for etype in [EventType.AGENT_START, EventType.THINK, EventType.AGENT_STOP]:
            await ledger.append(make_event(event_type=etype))

        results = await ledger.query(session_id="sess_test")
        assert len(results) == 3
        types = [r["event_type"] for r in results]
        assert types == [
            EventType.AGENT_START.value,
            EventType.THINK.value,
            EventType.AGENT_STOP.value,
        ]

    async def test_filter_by_event_type(self, ledger):
        await ledger.append(make_event(event_type=EventType.TOOL_CALL))
        await ledger.append(make_event(event_type=EventType.SANDBOX_VIOLATION))
        await ledger.append(make_event(event_type=EventType.TOOL_CALL))

        violations = await ledger.query(
            session_id="sess_test", event_type=EventType.SANDBOX_VIOLATION
        )
        assert len(violations) == 1
        assert violations[0]["event_type"] == EventType.SANDBOX_VIOLATION.value

    async def test_filter_by_agent_id(self, ledger):
        await ledger.append(make_event(agent_id="agent_a"))
        await ledger.append(make_event(agent_id="agent_b"))
        await ledger.append(make_event(agent_id="agent_a"))

        results = await ledger.query(session_id="sess_test", agent_id="agent_a")
        assert len(results) == 2
        assert all(r["agent_id"] == "agent_a" for r in results)

    async def test_filter_by_session_id(self, ledger):
        await ledger.append(make_event(session_id="sess_1"))
        await ledger.append(make_event(session_id="sess_2"))
        await ledger.append(make_event(session_id="sess_1"))

        results = await ledger.query(session_id="sess_1")
        assert len(results) == 2

    async def test_limit_respected(self, ledger):
        for _ in range(20):
            await ledger.append(make_event())

        results = await ledger.query(session_id="sess_test", limit=5)
        assert len(results) == 5

    async def test_sessions_list(self, ledger):
        await ledger.append(make_event(session_id="alpha"))
        await ledger.append(make_event(session_id="beta"))
        await ledger.append(make_event(session_id="alpha"))

        sessions = await ledger.sessions()
        assert "alpha" in sessions
        assert "beta" in sessions

    async def test_not_connected_raises(self, tmp_path):
        store = LedgerStore(db_path=tmp_path / "unconnected.db")
        event = make_event()
        with pytest.raises(RuntimeError, match="not connected"):
            await store.append(event)
