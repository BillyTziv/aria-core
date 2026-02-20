"""
File Agent Example
==================
An agent that reads and writes files safely, with rollback support.

Demonstrates:
- FileSnapshot for reversible file operations
- RollbackEngine checkpointing
- Simulated bad write followed by `aria rollback`

Run with:
    python examples/file_agent.py
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Any

from dotenv import load_dotenv

from aria.agent.base import AgentBase
from aria.agent.decorators import tool
from aria.models import AgentManifest
from aria.orchestrator import ARIASession
from aria.rollback.engine import FileSnapshot

load_dotenv()


class FileAgent(AgentBase):
    """Agent that can read/write files with full rollback support."""

    @tool(
        name="read_file",
        description="Read the contents of a file.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    )
    async def read_file(self, path: str) -> str:
        try:
            with open(path) as f:
                return f.read()
        except FileNotFoundError:
            return f"[File not found: {path}]"

    @tool(
        name="write_file",
        description="Write content to a file.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    )
    async def write_file(self, path: str, content: str) -> str:
        with open(path, "w") as f:
            f.write(content)
        return f"Wrote {len(content)} bytes to {path}"

    async def run_task(self, task: str) -> str:
        return f"FileAgent ready. Task received: {task}"


async def main(task: str = "demonstrate file rollback") -> str:
    manifest = AgentManifest(
        name="FileAgent",
        allowed_capabilities=["read_file", "write_file"],
    )

    if os.environ.get("ANTHROPIC_API_KEY"):
        from aria.providers.anthropic import AnthropicProvider
        provider = AnthropicProvider()
    else:
        from aria.providers.openai import OpenAIProvider
        provider = OpenAIProvider()

    async with ARIASession() as session:
        agent = FileAgent(
            manifest=manifest,
            provider=provider,
            policy=session.policy,
            ledger=session.ledger,
            session_id=session.session_id,
        )
        await session.run(agent, task)

        # Create a temp file to demonstrate rollback
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Original content\n")
            tmpfile = f.name

        print(f"\n[Demo file created: {tmpfile}]")
        print(f"[Contents: {open(tmpfile).read().strip()}]")

        # Register a snapshot BEFORE the write
        snap = FileSnapshot(tmpfile)
        snap_key = await session.rollback_engine.register_snapshot(
            key=f"snap_{tmpfile}", adapter=snap, session_id=session.session_id
        )

        # Create a checkpoint
        ckpt_id = await session.rollback_engine.create_checkpoint(
            session_id=session.session_id,
            label="before_write",
            snapshot_keys=[snap_key],
        )
        print(f"[Checkpoint created: {ckpt_id}]")

        # Simulate the agent doing a "bad" write
        result = await agent.act("write_file", {"path": tmpfile, "content": "CORRUPTED DATA!!!\n"})
        print(f"[After bad write: {open(tmpfile).read().strip()}]")

        # Roll back to the checkpoint
        success = await session.rollback(to=ckpt_id)
        print(f"[Rollback success: {success}]")
        print(f"[After rollback: {open(tmpfile).read().strip()}]")

        # Clean up
        os.unlink(tmpfile)

        events = await session.audit()
        print(f"\n[Session: {session.session_id}]")
        print(f"[Audit: {len(events)} events logged]")

        return f"Rollback demo complete. Checkpoint: {ckpt_id}"


if __name__ == "__main__":
    print(asyncio.run(main()))
