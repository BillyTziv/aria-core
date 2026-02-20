"""
Multi-Agent Pipeline Example
=============================
A planner agent spawns an executor agent via trust-scoped delegation.

Demonstrates:
- ARIASession.run_pipeline() with sequential agents
- TrustTokenManager.issue() — executor only gets a subset of capabilities
- Trust scope violation — executor cannot exceed granted permissions
- Full session audit across both agents

Run with:
    python examples/multi_agent.py
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from dotenv import load_dotenv

from aria.agent.base import AgentBase
from aria.agent.decorators import tool
from aria.models import AgentManifest
from aria.orchestrator import ARIASession

load_dotenv()


class PlannerAgent(AgentBase):
    """Plans a task and delegates sub-tasks."""

    @tool(
        name="create_plan",
        description="Break a task into sub-steps.",
        parameters={
            "type": "object",
            "properties": {"goal": {"type": "string"}},
            "required": ["goal"],
        },
    )
    async def create_plan(self, goal: str) -> str:
        return (
            f"Plan for '{goal}':\n"
            "1. Gather relevant data\n"
            "2. Analyze patterns\n"
            "3. Write summary report\n"
        )

    @tool(
        name="delegate",
        description="Delegate a step to the executor agent.",
        parameters={
            "type": "object",
            "properties": {"step": {"type": "string"}},
            "required": ["step"],
        },
    )
    async def delegate(self, step: str) -> str:
        return f"Delegated: {step}"

    async def run_task(self, task: str) -> str:
        result = await self.act("create_plan", {"goal": task})
        plan = str(result.value if hasattr(result, "value") else result)
        await self.observe(plan)
        delegation = await self.act("delegate", {"step": "Step 1: gather data"})
        await self.observe(str(delegation))
        return plan


class ExecutorAgent(AgentBase):
    """Executes delegated steps — restricted to a subset of capabilities."""

    @tool(
        name="gather_data",
        description="Gather data for analysis.",
        parameters={
            "type": "object",
            "properties": {"topic": {"type": "string"}},
            "required": ["topic"],
        },
    )
    async def gather_data(self, topic: str) -> str:
        return f"[Simulated data gathered for topic: {topic}]"

    @tool(
        name="write_summary",
        description="Write an analysis summary.",
        parameters={
            "type": "object",
            "properties": {"content": {"type": "string"}},
            "required": ["content"],
        },
    )
    async def write_summary(self, content: str) -> str:
        return f"Summary written: {content[:80]}..."

    async def run_task(self, task: str) -> str:
        data_result = await self.act("gather_data", {"topic": task})
        data = str(data_result.value if hasattr(data_result, "value") else data_result)
        await self.observe(data)

        summary_result = await self.act("write_summary", {"content": data})
        summary = str(summary_result.value if hasattr(summary_result, "value") else summary_result)
        await self.observe(summary)
        return summary


async def main(task: str = "analyze trends in renewable energy adoption") -> str:
    planner_manifest = AgentManifest(
        name="PlannerAgent",
        allowed_capabilities=["create_plan", "delegate"],
    )
    executor_manifest = AgentManifest(
        name="ExecutorAgent",
        # Planner will issue a trust token granting only these
        allowed_capabilities=["gather_data", "write_summary"],
    )

    if os.environ.get("ANTHROPIC_API_KEY"):
        from aria.providers.anthropic import AnthropicProvider
        planner_provider = AnthropicProvider()
        executor_provider = AnthropicProvider()
    else:
        from aria.providers.openai import OpenAIProvider
        planner_provider = OpenAIProvider()
        executor_provider = OpenAIProvider()

    async with ARIASession() as session:
        planner = PlannerAgent(
            manifest=planner_manifest,
            provider=planner_provider,
            policy=session.policy,
            ledger=session.ledger,
            session_id=session.session_id,
        )
        executor = ExecutorAgent(
            manifest=executor_manifest,
            provider=executor_provider,
            policy=session.policy,
            ledger=session.ledger,
            session_id=session.session_id,
        )

        # Issue a trust token: planner delegates to executor
        token = session.trust_manager.issue(
            parent_manifest=AgentManifest(
                name="SystemRoot",
                allowed_capabilities=["create_plan", "delegate", "gather_data", "write_summary"],
            ),
            grantee_agent_id=executor_manifest.agent_id,
            requested_capabilities=["gather_data", "write_summary"],
            ttl_seconds=300,
        )
        print(f"\n[Trust token issued: {token.token_id}]")
        print(f"[Granted capabilities: {token.scoped_capabilities}]")

        # Verify trust — executor can gather_data but not create_plan
        can_gather = session.trust_manager.verify(token, "gather_data")
        cant_plan = not session.trust_manager.verify(token, "create_plan")
        print(f"[Executor can gather_data: {can_gather}]")   # True
        print(f"[Executor blocked from create_plan: {cant_plan}]")  # True

        results = await session.run_pipeline(
            agents=[planner, executor],
            tasks=[task, task],
            parallel=False,
        )

        events = await session.audit()
        print(f"\n[Session: {session.session_id}]")
        print(f"[Planner result]: {results[0][:100]}")
        print(f"[Executor result]: {results[1][:100]}")
        print(f"[Audit: {len(events)} events across both agents]")

        return "\n---\n".join(results)


if __name__ == "__main__":
    print(asyncio.run(main()))
