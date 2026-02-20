"""Tests for the capability sandbox — policy enforcement."""

from __future__ import annotations

import pytest

from aria.ledger.store import LedgerStore
from aria.models import (
    AuditEvent,
    CapabilityPolicy,
    EventType,
    PolicyAction,
    PolicyRule,
)
from aria.sandbox.executor import ExecutionResult, SandboxExecutor


@pytest.fixture
async def ledger(tmp_path):
    store = LedgerStore(db_path=tmp_path / "test.db")
    await store.connect()
    yield store
    await store.close()


@pytest.fixture
def allow_policy():
    return CapabilityPolicy(
        rules=[PolicyRule(capability="web_search", action=PolicyAction.ALLOW)],
        default_action=PolicyAction.DENY,
    )


@pytest.fixture
async def sandbox(allow_policy, ledger):
    return SandboxExecutor(policy=allow_policy, ledger=ledger)


class TestSandboxPolicyEnforcement:
    async def test_allowed_capability_executes(self, sandbox):
        sandbox.register_manifest("agent_1", ["web_search"])

        async def fake_search(query: str) -> str:
            return f"results for {query}"

        result = await sandbox.execute(
            agent_id="agent_1",
            capability="web_search",
            arguments={"query": "test"},
            tool_fn=fake_search,
            session_id="sess_1",
        )
        assert result.success is True
        assert "results for test" in str(result.value)
        assert result.blocked is False

    async def test_undeclared_capability_is_blocked(self, sandbox):
        sandbox.register_manifest("agent_1", ["web_search"])

        async def evil_fn() -> str:
            return "hacked"

        result = await sandbox.execute(
            agent_id="agent_1",
            capability="delete_database",
            arguments={},
            tool_fn=evil_fn,
            session_id="sess_1",
        )
        assert result.success is False
        assert result.blocked is True

    async def test_denied_by_policy_is_blocked(self, sandbox):
        sandbox.register_manifest("agent_1", ["web_search", "file_write"])

        async def fn() -> str:
            return "ok"

        result = await sandbox.execute(
            agent_id="agent_1",
            capability="file_write",  # allowed in manifest, but denied by policy
            arguments={},
            tool_fn=fn,
            session_id="sess_1",
        )
        assert result.success is False
        assert result.blocked is True

    async def test_violation_is_logged(self, sandbox, ledger):
        sandbox.register_manifest("agent_1", ["web_search"])

        result = await sandbox.execute(
            agent_id="agent_1",
            capability="drop_table",
            arguments={},
            tool_fn=None,
            session_id="sess_log",
        )

        events = await ledger.query(session_id="sess_log", event_type=EventType.SANDBOX_VIOLATION)
        assert len(events) >= 1
        assert events[0]["event_type"] == EventType.SANDBOX_VIOLATION.value

    async def test_missing_tool_fn_is_blocked(self, ledger):
        policy = CapabilityPolicy(default_action=PolicyAction.ALLOW)
        sandbox = SandboxExecutor(policy=policy, ledger=ledger)
        sandbox.register_manifest("agent_2", ["mystery_tool"])

        result = await sandbox.execute(
            agent_id="agent_2",
            capability="mystery_tool",
            arguments={},
            tool_fn=None,
            session_id="sess_2",
        )
        assert result.success is False
        assert result.blocked is True

    async def test_tool_exception_returns_failure_result(self, ledger):
        policy = CapabilityPolicy(default_action=PolicyAction.ALLOW)
        sandbox = SandboxExecutor(policy=policy, ledger=ledger)
        sandbox.register_manifest("agent_3", ["boom"])

        async def boom() -> None:
            raise ValueError("intentional error")

        result = await sandbox.execute(
            agent_id="agent_3",
            capability="boom",
            arguments={},
            tool_fn=boom,
            session_id="sess_3",
        )
        assert result.success is False
        assert "intentional error" in result.reason
