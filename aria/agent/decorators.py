"""Decorators for declaring agent tools and human approval gates."""

from __future__ import annotations

import functools
from typing import Any, Callable


def tool(
    name: str | None = None,
    description: str = "",
    parameters: dict[str, Any] | None = None,
) -> Callable:
    """
    Mark an AgentBase method as a sandboxed tool (capability).

    Usage::

        @tool(name="web_search", description="Search the web for a query")
        async def search(self, query: str) -> str:
            ...

    The capability name (lowercased) must appear in the agent's
    ``AgentManifest.allowed_capabilities`` list, otherwise the
    ``SandboxExecutor`` will block the call.
    """

    def decorator(fn: Callable) -> Callable:
        cap_name = (name or fn.__name__).lower()

        @functools.wraps(fn)
        async def wrapper(self: Any, **kwargs: Any) -> Any:
            return await fn(self, **kwargs)

        wrapper._is_aria_tool = True  # type: ignore[attr-defined]
        wrapper._capability_name = cap_name  # type: ignore[attr-defined]
        wrapper._tool_description = description or fn.__doc__ or ""  # type: ignore[attr-defined]
        wrapper._tool_parameters = parameters or {}  # type: ignore[attr-defined]
        wrapper._requires_human_approval = False  # type: ignore[attr-defined]
        return wrapper

    return decorator


def requires_human_approval(fn: Callable) -> Callable:
    """
    Mark a tool as requiring explicit human approval before execution.

    Apply *after* @tool::

        @tool(name="send_email", description="Send an email")
        @requires_human_approval
        async def send_email(self, to: str, body: str) -> str:
            ...

    The ``GateEngine`` will pause execution and emit a ``CHECKPOINT_PENDING``
    event. Execution resumes only after ``session.approve(checkpoint_id)``
    is called.
    """

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        return await fn(*args, **kwargs)

    # Copy ARIA metadata if present
    for attr in ("_is_aria_tool", "_capability_name", "_tool_description", "_tool_parameters"):
        if hasattr(fn, attr):
            setattr(wrapper, attr, getattr(fn, attr))

    wrapper._requires_human_approval = True  # type: ignore[attr-defined]
    return wrapper
