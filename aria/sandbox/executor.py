"""Capability sandbox — policy enforcement layer for all tool calls."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable

from aria.models import AuditEvent, CapabilityPolicy, EventType, PolicyAction

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    success: bool
    value: Any = None
    blocked: bool = False
    pending_approval: bool = False
    reason: str = ""


class SandboxExecutor:
    """
    Intercepts every tool call and enforces capability policies.

    The policy is evaluated in priority order:
    1. Check the agent's declared ``allowed_capabilities`` in its manifest.
    2. Evaluate the ``CapabilityPolicy`` rules (allow / deny / require_approval).
    3. Check if the tool was marked ``@requires_human_approval``.
    4. Check ``HumanGateRule`` thresholds (confidence, cost).

    Violations are always logged — they never raise silently.
    """

    def __init__(
        self,
        policy: CapabilityPolicy,
        ledger: Any,  # LedgerStore — avoid circular import
        gate_engine: Any | None = None,  # GateEngine
    ) -> None:
        self.policy = policy
        self.ledger = ledger
        self.gate_engine = gate_engine
        # manifest registry: agent_id -> allowed_capabilities
        self._manifests: dict[str, list[str]] = {}

    def register_manifest(self, agent_id: str, allowed_capabilities: list[str]) -> None:
        self._manifests[agent_id] = [c.lower() for c in allowed_capabilities]

    async def execute(
        self,
        agent_id: str,
        capability: str,
        arguments: dict[str, Any],
        tool_fn: Callable | None,
        session_id: str,
        confidence: float | None = None,
        estimated_cost: float | None = None,
    ) -> ExecutionResult:
        cap = capability.lower()

        # 1. Manifest check — capability must be declared
        allowed = self._manifests.get(agent_id, [])
        if allowed and cap not in allowed:
            await self._log_violation(session_id, agent_id, cap, "capability not declared in manifest")
            return ExecutionResult(success=False, blocked=True, reason="capability not declared in manifest")

        # 2. Policy check
        action = self.policy.evaluate(agent_id, cap)
        if action == PolicyAction.DENY:
            await self._log_violation(session_id, agent_id, cap, "denied by policy")
            return ExecutionResult(success=False, blocked=True, reason="denied by policy")

        # 3. Human gate check (explicit decorator or gate engine rules)
        needs_approval = False
        if tool_fn and getattr(tool_fn, "_requires_human_approval", False):
            needs_approval = True
        if not needs_approval and self.gate_engine:
            needs_approval = await self.gate_engine.evaluate(
                session_id=session_id,
                agent_id=agent_id,
                capability=cap,
                confidence=confidence,
                estimated_cost=estimated_cost,
            )

        if action == PolicyAction.REQUIRE_APPROVAL or needs_approval:
            checkpoint_id = await self.gate_engine.pause(
                session_id=session_id,
                agent_id=agent_id,
                capability=cap,
                arguments=arguments,
            ) if self.gate_engine else "__no_gate_engine__"

            if checkpoint_id == "__no_gate_engine__":
                # No gate engine configured; block by default
                return ExecutionResult(success=False, blocked=True, reason="requires human approval but no gate engine configured")

            # Wait for approval signal
            approved = await self.gate_engine.wait_for_decision(checkpoint_id)
            if not approved:
                return ExecutionResult(success=False, blocked=True, reason="human denied the action")

        # 4. Execute
        if tool_fn is None:
            return ExecutionResult(success=False, blocked=True, reason=f"no implementation found for {cap}")

        try:
            if asyncio.iscoroutinefunction(tool_fn):
                result = await tool_fn(**arguments)
            else:
                result = tool_fn(**arguments)
            return ExecutionResult(success=True, value=result)
        except Exception as exc:
            logger.exception("Tool %s raised an exception", cap)
            await self._log_event(session_id, agent_id, EventType.ERROR, {"capability": cap, "error": str(exc)})
            return ExecutionResult(success=False, reason=str(exc))

    async def _log_violation(self, session_id: str, agent_id: str, capability: str, reason: str) -> None:
        logger.warning("[SANDBOX VIOLATION] agent=%s cap=%s reason=%s", agent_id, capability, reason)
        await self._log_event(
            session_id,
            agent_id,
            EventType.SANDBOX_VIOLATION,
            {"capability": capability, "reason": reason},
        )

    async def _log_event(
        self,
        session_id: str,
        agent_id: str,
        event_type: EventType,
        data: dict[str, Any],
    ) -> None:
        event = AuditEvent(
            session_id=session_id,
            agent_id=agent_id,
            event_type=event_type,
            output_data=data,
        )
        await self.ledger.append(event)
