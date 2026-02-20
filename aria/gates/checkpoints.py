"""Human checkpoint gates — pause/resume with CLI and webhook adapters."""

from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import Any

from aria.models import AuditEvent, EventType, GateCondition, HumanGateRule

logger = logging.getLogger(__name__)


class GateDecision(str, Enum):
    APPROVED = "approved"
    DENIED = "denied"
    PENDING = "pending"


class PendingCheckpoint:
    def __init__(self, checkpoint_id: str, session_id: str, agent_id: str, capability: str, arguments: dict[str, Any]) -> None:
        self.checkpoint_id = checkpoint_id
        self.session_id = session_id
        self.agent_id = agent_id
        self.capability = capability
        self.arguments = arguments
        self.decision = GateDecision.PENDING
        self._event: asyncio.Event = asyncio.Event()

    def resolve(self, approved: bool) -> None:
        self.decision = GateDecision.APPROVED if approved else GateDecision.DENIED
        self._event.set()

    async def wait(self, timeout: float | None = None) -> bool:
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("Checkpoint %s timed out", self.checkpoint_id)
            return False
        return self.decision == GateDecision.APPROVED


class GateEngine:
    """
    Evaluates HumanGateRule conditions and manages pause/resume flow.

    Adapters available:
    - CLI: interactive prompt in the terminal
    - Webhook: HTTP callback (a POST to resume URL)
    - Programmatic: call approve()/deny() directly in tests
    """

    def __init__(
        self,
        rules: list[HumanGateRule],
        ledger: Any,
        adapter: str = "cli",
        resume_url: str | None = None,
        approval_timeout: float | None = 120.0,
    ) -> None:
        self.rules = rules
        self.ledger = ledger
        self.adapter = adapter
        self.resume_url = resume_url
        self.approval_timeout = approval_timeout
        self._pending: dict[str, PendingCheckpoint] = {}

    async def evaluate(
        self,
        session_id: str,
        agent_id: str,
        capability: str,
        confidence: float | None = None,
        estimated_cost: float | None = None,
    ) -> bool:
        """Return True if any gate rule triggers for this action."""
        for rule in self.rules:
            if rule.applies_to and capability not in rule.applies_to:
                continue
            if rule.condition == GateCondition.CONFIDENCE_BELOW:
                if confidence is not None and rule.threshold is not None and confidence < rule.threshold:
                    logger.info("Gate triggered: confidence %.2f < %.2f for %s", confidence, rule.threshold, capability)
                    return True
            elif rule.condition == GateCondition.COST_ABOVE:
                if estimated_cost is not None and rule.threshold is not None and estimated_cost > rule.threshold:
                    logger.info("Gate triggered: cost %.2f > %.2f for %s", estimated_cost, rule.threshold, capability)
                    return True
        return False

    async def pause(
        self,
        session_id: str,
        agent_id: str,
        capability: str,
        arguments: dict[str, Any],
    ) -> str:
        import uuid
        checkpoint_id = f"gate_{uuid.uuid4().hex[:8]}"
        pending = PendingCheckpoint(
            checkpoint_id=checkpoint_id,
            session_id=session_id,
            agent_id=agent_id,
            capability=capability,
            arguments=arguments,
        )
        self._pending[checkpoint_id] = pending

        event = AuditEvent(
            session_id=session_id,
            agent_id=agent_id,
            event_type=EventType.CHECKPOINT_PENDING,
            input_data={"capability": capability, "arguments": arguments},
            metadata={"checkpoint_id": checkpoint_id, "adapter": self.adapter},
        )
        await self.ledger.append(event)
        logger.info("Execution paused at gate checkpoint %s [%s]", checkpoint_id, capability)

        if self.adapter == "cli":
            self._dispatch_cli_prompt(pending)

        return checkpoint_id

    def _dispatch_cli_prompt(self, pending: PendingCheckpoint) -> None:
        """Fire a background task that prompts the user in the terminal."""
        async def _prompt() -> None:
            print(f"\n[ARIA GATE] Checkpoint: {pending.checkpoint_id}")
            print(f"  Agent    : {pending.agent_id}")
            print(f"  Capability: {pending.capability}")
            print(f"  Arguments: {pending.arguments}")
            response = input("  Approve? [y/N]: ").strip().lower()
            pending.resolve(response == "y")

        asyncio.create_task(_prompt())

    async def wait_for_decision(self, checkpoint_id: str) -> bool:
        pending = self._pending.get(checkpoint_id)
        if not pending:
            logger.error("No pending checkpoint with id %s", checkpoint_id)
            return False
        approved = await pending.wait(timeout=self.approval_timeout)
        event_type = EventType.CHECKPOINT_APPROVED if approved else EventType.CHECKPOINT_DENIED
        event = AuditEvent(
            session_id=pending.session_id,
            agent_id=pending.agent_id,
            event_type=event_type,
            metadata={"checkpoint_id": checkpoint_id},
        )
        await self.ledger.append(event)
        del self._pending[checkpoint_id]
        return approved

    def approve(self, checkpoint_id: str) -> None:
        """Programmatically approve a pending checkpoint (use in tests or webhooks)."""
        if checkpoint_id in self._pending:
            self._pending[checkpoint_id].resolve(approved=True)

    def deny(self, checkpoint_id: str) -> None:
        """Programmatically deny a pending checkpoint."""
        if checkpoint_id in self._pending:
            self._pending[checkpoint_id].resolve(approved=False)

    def list_pending(self) -> list[str]:
        return list(self._pending.keys())
