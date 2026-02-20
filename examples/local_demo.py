"""
Local Demo — test all ARIA safety layers with zero API keys.
=============================================================

What this demonstrates:
  1. Basic agent run with sandboxed tools
  2. Sandbox VIOLATION — agent tries to call a tool it wasn't given
  3. Human checkpoint gate — pause, then programmatically approve
  4. Rollback — write a file, roll back, confirm it's restored
  5. Trust delegation — sub-agent blocked from parent-only capabilities
  6. Full audit log for the whole session

Run:
    python examples/local_demo.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from aria.agent.base import AgentBase
from aria.agent.decorators import requires_human_approval, tool
from aria.gates.checkpoints import GateEngine
from aria.models import (
    AgentManifest,
    CapabilityPolicy,
    GateCondition,
    HumanGateRule,
    PolicyAction,
    PolicyRule,
)
from aria.orchestrator import ARIASession
from aria.providers.local import LocalProvider
from aria.rollback.engine import FileSnapshot


# ── Helpers ────────────────────────────────────────────────────────────────

DIVIDER = "-" * 60


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# ── Demo Agents ────────────────────────────────────────────────────────────


class DemoAgent(AgentBase):
    """General-purpose demo agent with several tools."""

    @tool(name="web_search", description="Search the web.")
    async def web_search(self, query: str) -> str:
        return f"[Simulated results for '{query}']"

    @tool(name="summarize", description="Summarize text.")
    async def summarize(self, text: str) -> str:
        return f"[Summary of: {text[:40]}...]"

    @tool(name="send_email", description="Send an email.")
    @requires_human_approval
    async def send_email(self, to: str, subject: str) -> str:
        return f"Email sent to {to} — {subject}"

    async def run_task(self, task: str) -> str:
        result = await self.act("web_search", {"query": task})
        output = str(result.value if hasattr(result, "value") else result)
        await self.observe(output)
        return output


class FileAgent(AgentBase):
    """Agent that writes and reads files — used for rollback demo."""

    def __init__(self, *args, file_path: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._file_path = file_path

    @tool(name="write_file", description="Write content to a file.")
    async def write_file(self, path: str, content: str) -> str:
        Path(path).write_text(content)
        return f"Wrote {len(content)} bytes to {path}"

    @tool(name="read_file", description="Read a file.")
    async def read_file(self, path: str) -> str:
        return Path(path).read_text()

    async def run_task(self, task: str) -> str:
        result = await self.act("write_file", {"path": self._file_path, "content": "BAD DATA — should be rolled back"})
        return str(result.value if hasattr(result, "value") else result)


# ── Demo Steps ─────────────────────────────────────────────────────────────


async def demo_1_basic_run():
    section("1. Basic sandboxed agent run")

    manifest = AgentManifest(name="BasicAgent", allowed_capabilities=["web_search", "summarize"])

    async with ARIASession() as session:
        agent = DemoAgent(
            manifest=manifest,
            provider=LocalProvider(),
            policy=session.policy,
            ledger=session.ledger,
            session_id=session.session_id,
        )
        result = await session.run(agent, task="search for quantum computing news")
        events = await session.audit()

    print(f"  Result : {result}")
    print(f"  Events logged: {len(events)}")
    for ev in events:
        print(f"    [{ev['event_type']:30s}] agent={ev['agent_id']}")
    print("\n  ✓ Agent ran, tools executed, every step logged.")


async def demo_2_sandbox_violation():
    section("2. Sandbox violation — agent calls undeclared tool")

    # Agent manifest only allows web_search — NOT send_email
    manifest = AgentManifest(name="RestrictedAgent", allowed_capabilities=["web_search"])
    policy = CapabilityPolicy(default_action=PolicyAction.ALLOW)

    async with ARIASession(policy=policy) as session:
        agent = DemoAgent(
            manifest=manifest,
            provider=LocalProvider(),
            policy=session.policy,
            ledger=session.ledger,
            session_id=session.session_id,
        )
        # Inject sandbox (normally done automatically by session.run())
        agent._sandbox = session.sandbox
        session.sandbox.register_manifest(
            agent.manifest.agent_id, agent.manifest.allowed_capabilities
        )
        # Directly try to act on a tool not in the manifest
        result = await agent.act("send_email", {"to": "boss@example.com", "subject": "surprise!"})
        from aria.models import EventType
        violations = await session.ledger.query(
            session_id=session.session_id,
            event_type=EventType.SANDBOX_VIOLATION,
        )

    print(f"  Execution blocked : {result.blocked}")
    print(f"  Reason            : {result.reason}")
    print(f"  Violations logged : {len(violations)}")
    print("\n  ✓ Undeclared tool was blocked AND audited — never executed silently.")


async def demo_3_human_gate():
    section("3. Human checkpoint gate — approve programmatically")

    gate_rules = [
        HumanGateRule(
            condition=GateCondition.CONFIDENCE_BELOW,
            threshold=0.5,
            description="Flag low-confidence actions",
        )
    ]
    manifest = AgentManifest(
        name="GatedAgent",
        allowed_capabilities=["web_search", "send_email"],
    )

    async with ARIASession(gate_rules=gate_rules, gate_adapter="programmatic") as session:
        agent = DemoAgent(
            manifest=manifest,
            provider=LocalProvider(confidence=0.3),  # low confidence → triggers gate
            policy=session.policy,
            ledger=session.ledger,
            session_id=session.session_id,
        )

        # Schedule auto-approval 100ms after the gate fires
        async def _auto_approve():
            await asyncio.sleep(0.1)
            pending = session.gate_engine.list_pending()
            for ckpt_id in pending:
                print(f"  [Gate] Auto-approving checkpoint: {ckpt_id}")
                session.approve(ckpt_id)

        asyncio.create_task(_auto_approve())

        # Inject sandbox so gate enforcement is active
        agent._sandbox = session.sandbox
        session.sandbox.register_manifest(
            agent.manifest.agent_id, agent.manifest.allowed_capabilities
        )
        result = await agent.act(
            "send_email",
            {"to": "team@example.com", "subject": "ARIA demo"},
        )

    print(f"  Execution result  : {result.success}")
    print(f"  Value             : {result.value}")
    print("\n  ✓ Execution paused, waited for human signal, then resumed.")


async def demo_4_rollback():
    section("4. Rollback — restore file to state before agent write")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("ORIGINAL CONTENT")
        file_path = f.name

    print(f"  File before: {Path(file_path).read_text()!r}")

    manifest = AgentManifest(
        name="FileAgent",
        allowed_capabilities=["write_file", "read_file"],
    )

    async with ARIASession() as session:
        # Snapshot the file BEFORE the agent writes
        snap = FileSnapshot(file_path)
        snap_key = await session.rollback_engine.register_snapshot(
            key="file_snap",
            adapter=snap,
            session_id=session.session_id,
        )
        ckpt_id = await session.rollback_engine.create_checkpoint(
            session_id=session.session_id,
            label="before_write",
            snapshot_keys=[snap_key],
        )

        agent = FileAgent(
            manifest=manifest,
            provider=LocalProvider(),
            policy=session.policy,
            ledger=session.ledger,
            session_id=session.session_id,
            file_path=file_path,
        )
        await session.run(agent, task="write bad data")
        print(f"  File after write : {Path(file_path).read_text()!r}")

        # Roll back
        success = await session.rollback(to=ckpt_id)
        print(f"  Rollback success : {success}")
        print(f"  File after rollback: {Path(file_path).read_text()!r}")

    print("\n  ✓ File restored to original despite agent write.")


async def demo_5_trust_delegation():
    section("5. Trust delegation — sub-agent cannot exceed parent scope")

    from aria.trust.tokens import TrustTokenManager
    from aria.ledger.store import LedgerStore

    async with ARIASession() as session:
        parent_manifest = AgentManifest(
            agent_id="parent_agent",
            name="ParentAgent",
            allowed_capabilities=["web_search", "summarize"],
        )

        # Issue a token granting only "summarize" to a child
        token = session.trust_manager.issue(
            parent_manifest=parent_manifest,
            grantee_agent_id="child_agent",
            requested_capabilities=["summarize", "web_search", "delete_database"],
        )

        can_summarize = session.trust_manager.verify(token, "summarize")
        can_search    = session.trust_manager.verify(token, "web_search")
        cant_delete   = not session.trust_manager.verify(token, "delete_database")

    print(f"  Token ID          : {token.token_id}")
    print(f"  Granted caps      : {token.scoped_capabilities}")
    print(f"  Can summarize     : {can_summarize}")   # True
    print(f"  Can web_search    : {can_search}")       # True
    print(f"  Blocked delete_db : {cant_delete}")      # True
    print("\n  ✓ Child only received capabilities the parent actually owns.")


async def demo_6_full_audit():
    section("6. Full audit log across a complete session")

    manifest = AgentManifest(name="AuditAgent", allowed_capabilities=["web_search", "summarize"])

    async with ARIASession() as session:
        for _ in range(3):
            agent = DemoAgent(
                manifest=manifest,
                provider=LocalProvider(),
                policy=session.policy,
                ledger=session.ledger,
                session_id=session.session_id,
            )
            await session.run(agent, task="search for AI safety research")

        events = await session.audit(limit=50)
        print(f"  Session ID    : {session.session_id}")
        print(f"  Total events  : {len(events)}")

        from collections import Counter
        counts = Counter(ev["event_type"] for ev in events)
        for etype, count in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"    {etype:30s}: {count}")

    print("\n  ✓ Every think/act/observe logged. Queryable anytime via `aria audit`.")


# ── Main ───────────────────────────────────────────────────────────────────

async def main():
    print("\n██████████████████████████████████████████████████████████")
    print("  ARIA — Local Demo (no API key needed)")
    print("  Agentic Runtime for Intelligent Autonomy")
    print("██████████████████████████████████████████████████████████")

    await demo_1_basic_run()
    await demo_2_sandbox_violation()
    await demo_3_human_gate()
    await demo_4_rollback()
    await demo_5_trust_delegation()
    await demo_6_full_audit()

    print(f"\n{'=' * 60}")
    print("  All demos complete.")
    print(f"  Run `aria audit` to inspect the ledger.")
    print(f"  Run `pytest tests/ -v` to run the full test suite.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
