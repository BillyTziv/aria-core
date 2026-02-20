"""ARIA session orchestrator — multi-agent coordination."""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any

from aria.gates.checkpoints import GateEngine
from aria.ledger.store import LedgerStore
from aria.models import CapabilityPolicy, HumanGateRule, PolicyAction, PolicyRule
from aria.rollback.engine import RollbackEngine
from aria.sandbox.executor import SandboxExecutor
from aria.trust.tokens import TrustTokenManager

logger = logging.getLogger(__name__)


class ARIASession:
    """
    Top-level session object — the single entry point for running ARIA agents.

    Usage::

        async with ARIASession() as session:
            agent = MyAgent(manifest, provider, session.policy, session.ledger, session.session_id)
            result = await session.run(agent, task="Research quantum computing")
            print(await session.audit())
    """

    def __init__(
        self,
        session_id: str | None = None,
        ledger_path: Path | str | None = None,
        gate_rules: list[HumanGateRule] | None = None,
        gate_adapter: str = "cli",
        policy: CapabilityPolicy | None = None,
    ) -> None:
        self.session_id = session_id or f"session_{uuid.uuid4().hex[:10]}"
        self.ledger = LedgerStore(db_path=ledger_path)
        self.policy = policy or CapabilityPolicy(default_action=PolicyAction.ALLOW)
        self.gate_engine = GateEngine(
            rules=gate_rules or [],
            ledger=self.ledger,
            adapter=gate_adapter,
        )
        self.rollback_engine = RollbackEngine(ledger=self.ledger)
        self.trust_manager = TrustTokenManager(ledger=self.ledger)
        self.sandbox = SandboxExecutor(
            policy=self.policy,
            ledger=self.ledger,
            gate_engine=self.gate_engine,
        )
        self._connected = False

    async def __aenter__(self) -> ARIASession:
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    async def connect(self) -> None:
        await self.ledger.connect()
        self._connected = True
        logger.info("ARIA session started: %s", self.session_id)

    async def close(self) -> None:
        await self.ledger.close()
        self._connected = False
        logger.info("ARIA session closed: %s", self.session_id)

    async def run(self, agent: Any, task: str) -> str:
        """
        Run a single agent within this session.

        The agent's sandbox executor is patched to use the session's shared
        ledger, gate engine, and policy.
        """
        if not self._connected:
            raise RuntimeError("Session not connected. Use 'async with ARIASession() as session'.")

        # Inject session-scoped sandbox into agent
        agent._sandbox = self.sandbox
        self.sandbox.register_manifest(
            agent_id=agent.manifest.agent_id,
            allowed_capabilities=agent.manifest.allowed_capabilities,
        )
        agent.session_id = self.session_id
        agent.ledger = self.ledger
        return await agent.start(task)

    async def run_pipeline(
        self,
        agents: list[Any],
        tasks: list[str],
        parallel: bool = False,
    ) -> list[str]:
        """
        Run multiple agents, optionally in parallel.

        Set parallel=True only when agents have no data dependencies on each other.
        """
        if not self._connected:
            raise RuntimeError("Session not connected.")

        if parallel:
            results = await asyncio.gather(
                *[self.run(agent, task) for agent, task in zip(agents, tasks)],
                return_exceptions=True,
            )
            return [str(r) for r in results]
        else:
            results = []
            for agent, task in zip(agents, tasks):
                result = await self.run(agent, task)
                results.append(result)
            return results

    async def audit(
        self,
        agent_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return audit events for this session."""
        return await self.ledger.query(
            session_id=self.session_id,
            agent_id=agent_id,
            limit=limit,
        )

    async def rollback(self, to: str) -> bool:
        """Roll back session state to a named checkpoint."""
        return await self.rollback_engine.rollback(self.session_id, to=to)

    def approve(self, checkpoint_id: str) -> None:
        """Approve a pending human gate checkpoint."""
        self.gate_engine.approve(checkpoint_id)

    def deny(self, checkpoint_id: str) -> None:
        """Deny a pending human gate checkpoint."""
        self.gate_engine.deny(checkpoint_id)
