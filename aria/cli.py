"""ARIA CLI — run, audit, replay, rollback."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click

from aria.ledger.store import LedgerStore
from aria.rollback.engine import RollbackEngine


@click.group()
@click.version_option(package_name="aria-core")
def cli() -> None:
    """ARIA — Agentic Runtime for Intelligent Autonomy."""


# ---------------------------------------------------------------------------
# aria run
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("script", type=click.Path(exists=True))
@click.option("--task", "-t", default="", help="Task description to pass to the agent.")
@click.option(
    "--ledger",
    "-l",
    default=None,
    help="Path to the ledger DB file. Defaults to ~/.aria/ledger.db.",
)
def run(script: str, task: str, ledger: str | None) -> None:
    """Run an agent script.

    SCRIPT is the path to a Python file that defines and runs an ARIA agent.
    The script must implement an async main(task: str) -> str function.
    """
    import importlib.util
    import os

    if ledger:
        os.environ["ARIA_LEDGER_PATH"] = ledger

    spec = importlib.util.spec_from_file_location("_aria_script", script)
    if spec is None or spec.loader is None:
        click.echo(f"Cannot load script: {script}", err=True)
        sys.exit(1)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    if not hasattr(module, "main"):
        click.echo(f"{script} must define an async main(task: str) -> str function.", err=True)
        sys.exit(1)

    result = asyncio.run(module.main(task))
    click.echo(result)


# ---------------------------------------------------------------------------
# aria audit
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--session", "-s", default=None, help="Filter by session ID.")
@click.option("--agent", "-a", default=None, help="Filter by agent ID.")
@click.option("--type", "-t", "event_type", default=None, help="Filter by event type.")
@click.option("--limit", "-n", default=100, help="Max number of events to show.")
@click.option(
    "--ledger",
    "-l",
    default=None,
    help="Path to the ledger DB. Defaults to ~/.aria/ledger.db.",
)
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
def audit(
    session: str | None,
    agent: str | None,
    event_type: str | None,
    limit: int,
    ledger: str | None,
    as_json: bool,
) -> None:
    """Show the audit log."""
    from aria.models import EventType

    async def _run() -> None:
        store = LedgerStore(db_path=ledger)
        await store.connect()
        try:
            et = EventType(event_type) if event_type else None
            events = await store.query(
                session_id=session,
                agent_id=agent,
                event_type=et,
                limit=limit,
            )
        finally:
            await store.close()

        if as_json:
            click.echo(json.dumps(events, indent=2, default=str))
            return

        if not events:
            click.echo("No events found.")
            return

        for ev in events:
            ts = ev["timestamp"][:19].replace("T", " ")
            click.echo(
                f"[{ts}] {ev['event_type']:30s}  agent={ev['agent_id']}  "
                f"session={ev['session_id'][:12]}"
            )

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# aria sessions
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--ledger",
    "-l",
    default=None,
    help="Path to the ledger DB. Defaults to ~/.aria/ledger.db.",
)
def sessions(ledger: str | None) -> None:
    """List all session IDs in the ledger."""

    async def _run() -> None:
        store = LedgerStore(db_path=ledger)
        await store.connect()
        try:
            session_ids = await store.sessions()
        finally:
            await store.close()

        if not session_ids:
            click.echo("No sessions found.")
            return

        for sid in session_ids:
            click.echo(sid)

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# aria replay
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("session_id")
@click.option(
    "--ledger",
    "-l",
    default=None,
    help="Path to the ledger DB. Defaults to ~/.aria/ledger.db.",
)
def replay(session_id: str, ledger: str | None) -> None:
    """Replay all audit events for a session (read-only, for debugging)."""

    async def _run() -> None:
        store = LedgerStore(db_path=ledger)
        await store.connect()
        try:
            events = await store.query(session_id=session_id, limit=10000)
        finally:
            await store.close()

        if not events:
            click.echo(f"No events found for session {session_id}.")
            return

        click.echo(f"\n=== Replay: {session_id} ({len(events)} events) ===\n")
        for ev in events:
            ts = ev["timestamp"][:19].replace("T", " ")
            in_d = json.loads(ev["input_data"]) if isinstance(ev["input_data"], str) else ev["input_data"]
            out_d = json.loads(ev["output_data"]) if isinstance(ev["output_data"], str) else ev["output_data"]
            click.echo(f"[{ts}] {ev['event_type']}")
            if in_d:
                click.echo(f"  IN  : {json.dumps(in_d)[:120]}")
            if out_d:
                click.echo(f"  OUT : {json.dumps(out_d)[:120]}")

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# aria rollback
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("session_id")
@click.option("--to", "checkpoint_id", required=True, help="Checkpoint ID to roll back to.")
@click.option(
    "--ledger",
    "-l",
    default=None,
    help="Path to the ledger DB. Defaults to ~/.aria/ledger.db.",
)
def rollback(session_id: str, checkpoint_id: str, ledger: str | None) -> None:
    """Roll back a session to a checkpoint."""

    async def _run() -> None:
        store = LedgerStore(db_path=ledger)
        await store.connect()
        engine = RollbackEngine(ledger=store)
        try:
            success = await engine.rollback(session_id, to=checkpoint_id)
        finally:
            await store.close()

        if success:
            click.echo(f"Rolled back session {session_id} to checkpoint {checkpoint_id}.")
        else:
            click.echo(f"Rollback failed: checkpoint {checkpoint_id} not found in memory.", err=True)
            click.echo("Note: in-memory rollback data is cleared when the process exits.", err=True)
            sys.exit(1)

    asyncio.run(_run())


if __name__ == "__main__":
    cli()
