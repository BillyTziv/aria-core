# ARIA — Agentic Runtime for Intelligent Autonomy

> Safety infrastructure for AI agents: sandboxed capabilities, immutable audit
> ledger, rollback, trust-scoped sub-agent delegation, and human checkpoint gates.

[![CI](https://github.com/YOUR_USERNAME/aria-core/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/aria-core/actions)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## The Problem

AI agents in 2026 take real actions — browsing the web, writing files, calling
APIs, sending emails. But there's no standard safety net. If an agent goes
wrong, there's no easy way to:

- See *exactly* what it did and why
- Stop it mid-task and roll back the damage
- Prevent it from doing things you never intended
- Hand control back to a human at a critical moment

**ARIA is the safety layer you plug underneath any AI agent.**

---

## What You Get

| Feature | What it does |
|---|---|
| **Capability Sandbox** | Agents can only call tools they declared. Violations are blocked and logged — never silently swallowed. |
| **Immutable Audit Ledger** | Every thought, action, and result is written to an append-only SQLite log. |
| **Rollback Engine** | Snapshot state before a risky action. Roll back to any checkpoint with one call. |
| **Trust Delegation** | Sub-agents inherit a *subset* of the parent's permissions — privilege escalation is impossible. |
| **Human Checkpoint Gates** | Define rules that pause execution and wait for your approval before proceeding. |
| **CLI** | `aria audit`, `aria replay`, `aria rollback` — inspect and control any session from the terminal. |

---

## Installation

```bash
pip install aria-core
```

Or from source:

```bash
git clone https://github.com/YOUR_USERNAME/aria-core
cd aria-core
pip install -e ".[dev]"
```

---

## 5-Minute Quickstart

```python
import asyncio
from aria.agent.base import AgentBase
from aria.agent.decorators import tool, requires_human_approval
from aria.models import AgentManifest
from aria.orchestrator import ARIASession
from aria.providers.openai import OpenAIProvider


class MyAgent(AgentBase):

    @tool(name="web_search", description="Search the web")
    async def web_search(self, query: str) -> str:
        return f"Results for: {query}"  # replace with real implementation

    @tool(name="write_report", description="Save a report to disk")
    @requires_human_approval
    async def write_report(self, filename: str, content: str) -> str:
        with open(filename, "w") as f:
            f.write(content)
        return f"Saved to {filename}"

    async def run_task(self, task: str) -> str:
        result = await self.act("web_search", {"query": task})
        return str(result.value)


async def main():
    manifest = AgentManifest(
        name="MyAgent",
        allowed_capabilities=["web_search", "write_report"],
    )

    async with ARIASession() as session:
        agent = MyAgent(
            manifest=manifest,
            provider=OpenAIProvider(),
            policy=session.policy,
            ledger=session.ledger,
            session_id=session.session_id,
        )
        result = await session.run(agent, task="quantum computing news")
        print(result)

asyncio.run(main())
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    ARIASession                       │
│                (orchestrator.py)                     │
├──────────────┬──────────────┬────────────────────────┤
│  AgentBase   │   Sandbox    │   Human Gates           │
│  think/act/  │  Executor    │   GateEngine            │
│  observe     │  (policy     │   (pause/resume)        │
│              │   enforce)   │                         │
├──────────────┴──────────────┴────────────────────────┤
│               Audit Ledger (SQLite)                   │
│               Append-only AuditEvent log              │
├─────────────────────────────────────────────────────┤
│  Rollback Engine      │  Trust Token Manager          │
│  (snapshot/restore)   │  (HMAC-signed delegation)     │
└─────────────────────────────────────────────────────┘
```

---

## CLI Reference

```bash
# Run an agent script
aria run examples/research_agent.py --task "quantum computing news"

# View the audit log
aria audit --session SESSION_ID

# List all sessions
aria sessions

# Replay a session (read-only, for debugging)
aria replay SESSION_ID

# Roll back to a checkpoint
aria rollback SESSION_ID --to CHECKPOINT_ID
```

---

## Examples

| Example | What it demonstrates |
|---|---|
| `examples/research_agent.py` | Web search + human gate before writing a report |
| `examples/file_agent.py` | File read/write + rollback after a simulated bad write |
| `examples/multi_agent.py` | Planner + executor pipeline with trust delegation |

```bash
# Set up your API keys
cp .env.example .env
# Edit .env with your OPENAI_API_KEY or ANTHROPIC_API_KEY

# Run an example
python examples/file_agent.py
python examples/multi_agent.py
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `ARIA_LEDGER_PATH` | `~/.aria/ledger.db` | Path to the audit ledger |
| `ARIA_SESSION_SECRET` | auto-generated | HMAC secret for trust tokens |

---

## Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Make your changes and add tests
4. Run `pytest tests/ -v` and `ruff check aria/ tests/`
5. Open a pull request

---

## License

MIT — see [LICENSE](LICENSE).
