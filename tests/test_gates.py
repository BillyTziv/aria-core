"""Tests for human checkpoint gates — pause, approve, deny."""

from __future__ import annotations

import asyncio

import pytest

from aria.gates.checkpoints import GateDecision, GateEngine
from aria.ledger.store import LedgerStore
from aria.models import EventType, GateCondition, HumanGateRule


@pytest.fixture
async def ledger(tmp_path):
    store = LedgerStore(db_path=tmp_path / "gates.db")
    await store.connect()
    yield store
    await store.close()


@pytest.fixture
def confidence_rule():
    return HumanGateRule(
        condition=GateCondition.CONFIDENCE_BELOW,
        threshold=0.7,
        description="Flag low-confidence actions",
    )


@pytest.fixture
def cost_rule():
    return HumanGateRule(
        condition=GateCondition.COST_ABOVE,
        threshold=10.0,
        description="Flag expensive actions",
    )


class TestGateEvaluation:
    async def test_confidence_rule_triggers_below_threshold(self, ledger, confidence_rule):
        engine = GateEngine(rules=[confidence_rule], ledger=ledger, adapter="programmatic")
        triggered = await engine.evaluate(
            session_id="s1",
            agent_id="a1",
            capability="search",
            confidence=0.5,  # below 0.7
        )
        assert triggered is True

    async def test_confidence_rule_does_not_trigger_above_threshold(self, ledger, confidence_rule):
        engine = GateEngine(rules=[confidence_rule], ledger=ledger, adapter="programmatic")
        triggered = await engine.evaluate(
            session_id="s1",
            agent_id="a1",
            capability="search",
            confidence=0.9,  # above 0.7
        )
        assert triggered is False

    async def test_cost_rule_triggers_above_threshold(self, ledger, cost_rule):
        engine = GateEngine(rules=[cost_rule], ledger=ledger, adapter="programmatic")
        triggered = await engine.evaluate(
            session_id="s1",
            agent_id="a1",
            capability="send_email",
            estimated_cost=50.0,  # above 10.0
        )
        assert triggered is True

    async def test_no_rules_never_triggers(self, ledger):
        engine = GateEngine(rules=[], ledger=ledger, adapter="programmatic")
        triggered = await engine.evaluate(
            session_id="s1",
            agent_id="a1",
            capability="anything",
            confidence=0.1,
            estimated_cost=1000.0,
        )
        assert triggered is False


class TestGatePauseAndResume:
    async def test_approve_resumes_execution(self, ledger):
        engine = GateEngine(rules=[], ledger=ledger, adapter="programmatic", approval_timeout=5.0)
        ckpt_id = await engine.pause("s1", "a1", "send_email", {"to": "test@example.com"})

        async def _approve_soon() -> None:
            await asyncio.sleep(0.05)
            engine.approve(ckpt_id)

        asyncio.create_task(_approve_soon())
        approved = await engine.wait_for_decision(ckpt_id)
        assert approved is True

    async def test_deny_blocks_execution(self, ledger):
        engine = GateEngine(rules=[], ledger=ledger, adapter="programmatic", approval_timeout=5.0)
        ckpt_id = await engine.pause("s1", "a1", "send_email", {"to": "test@example.com"})

        async def _deny_soon() -> None:
            await asyncio.sleep(0.05)
            engine.deny(ckpt_id)

        asyncio.create_task(_deny_soon())
        approved = await engine.wait_for_decision(ckpt_id)
        assert approved is False

    async def test_approved_event_logged(self, ledger):
        engine = GateEngine(rules=[], ledger=ledger, adapter="programmatic", approval_timeout=5.0)
        ckpt_id = await engine.pause("s2", "a2", "delete_record", {})

        async def _approve() -> None:
            await asyncio.sleep(0.05)
            engine.approve(ckpt_id)

        asyncio.create_task(_approve())
        await engine.wait_for_decision(ckpt_id)

        events = await ledger.query(session_id="s2", event_type=EventType.CHECKPOINT_APPROVED)
        assert len(events) == 1

    async def test_pending_checkpoints_list(self, ledger):
        engine = GateEngine(rules=[], ledger=ledger, adapter="programmatic", approval_timeout=5.0)

        ckpt1 = await engine.pause("s3", "a1", "tool_a", {})
        ckpt2 = await engine.pause("s3", "a1", "tool_b", {})

        pending = engine.list_pending()
        assert ckpt1 in pending
        assert ckpt2 in pending

        engine.approve(ckpt1)
        engine.deny(ckpt2)

    async def test_timeout_returns_false(self, ledger):
        engine = GateEngine(rules=[], ledger=ledger, adapter="programmatic", approval_timeout=0.1)
        ckpt_id = await engine.pause("s4", "a1", "slow_tool", {})

        # No one approves — should timeout
        approved = await engine.wait_for_decision(ckpt_id)
        assert approved is False
