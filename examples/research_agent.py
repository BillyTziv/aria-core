"""
Research Agent Example
======================
An agent that can search the web (simulated) and summarize results.

Demonstrates:
- AgentBase with @tool declarations
- Human gate triggered before writing the final report
- Audit trail visible via `aria audit`

Run with:
    python examples/research_agent.py
    # or
    aria run examples/research_agent.py --task "summarize quantum computing news"
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from dotenv import load_dotenv

import aria
from aria.agent.base import AgentBase
from aria.agent.decorators import requires_human_approval, tool
from aria.models import AgentManifest, CapabilityPolicy, GateCondition, HumanGateRule, PolicyAction
from aria.orchestrator import ARIASession

load_dotenv()


class ResearchAgent(AgentBase):
    """Agent that searches and summarizes information on a topic."""

    @tool(
        name="web_search",
        description="Search the web for information on a topic.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"],
        },
    )
    async def web_search(self, query: str) -> str:
        # Simulated search — replace with real httpx/serper/brave calls
        return (
            f"[Simulated results for '{query}']\n"
            "1. Quantum computing achieves 1000-qubit milestone (Nature, 2025)\n"
            "2. IBM unveils Heron r2 processor with error correction (IBM, 2025)\n"
            "3. Google Willow solves RCS sampling in 5 minutes (Google DeepMind, 2024)\n"
        )

    @tool(
        name="write_report",
        description="Write the final research report to a file.",
        parameters={
            "type": "object",
            "properties": {
                "filename": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["filename", "content"],
        },
    )
    @requires_human_approval
    async def write_report(self, filename: str, content: str) -> str:
        path = f"/tmp/{filename}"
        with open(path, "w") as f:
            f.write(content)
        return f"Report saved to {path}"

    async def run_task(self, task: str) -> str:
        # Step 1: think about the task
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You are a research assistant. Use the tools provided."},
            {"role": "user", "content": task},
        ]

        response = await self.think(messages)

        # Step 2: if the LLM wants to call a tool, act on it
        tool_calls = response.get("tool_calls", [])
        search_results = ""
        for tc in tool_calls:
            result = await self.act(tc["name"], tc["arguments"])
            search_results += str(result.value if hasattr(result, "value") else result)
            await self.observe(str(result))

        # Step 3: if no tool calls (or after tool calls), synthesize a summary
        if not search_results:
            # Fallback: run the search directly
            result = await self.act("web_search", {"query": task})
            search_results = str(result.value if hasattr(result, "value") else result)
            await self.observe(search_results)

        summary = (
            f"Research Summary for: {task}\n\n"
            f"{search_results}\n\n"
            "Analysis: Based on the search results, significant progress has been made in "
            "quantum computing in 2024-2025, with major milestones in qubit count, "
            "error correction, and computational speed."
        )

        # Step 4: write report (triggers human approval gate)
        write_result = await self.act(
            "write_report",
            {"filename": "research_report.txt", "content": summary},
        )
        await self.observe(str(write_result))

        return summary


async def main(task: str = "summarize the latest quantum computing breakthroughs") -> str:
    gate_rules = [
        HumanGateRule(
            condition=GateCondition.EXPLICIT,
            description="Require approval before writing any file",
        )
    ]

    manifest = AgentManifest(
        name="ResearchAgent",
        allowed_capabilities=["web_search", "write_report"],
    )

    # Choose provider based on available API keys
    if os.environ.get("ANTHROPIC_API_KEY"):
        from aria.providers.anthropic import AnthropicProvider
        provider = AnthropicProvider()
    else:
        from aria.providers.openai import OpenAIProvider
        provider = OpenAIProvider()

    async with ARIASession(gate_rules=gate_rules, gate_adapter="cli") as session:
        agent = ResearchAgent(
            manifest=manifest,
            provider=provider,
            policy=session.policy,
            ledger=session.ledger,
            session_id=session.session_id,
        )
        result = await session.run(agent, task)
        print(f"\n[Session: {session.session_id}]")
        print(f"[Audit: {len(await session.audit())} events logged]")
        return result


if __name__ == "__main__":
    import sys
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "summarize the latest quantum computing breakthroughs"
    print(asyncio.run(main(task)))
