"""Agent base class — the think → act → observe loop with sandbox enforcement."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from aria.models import AgentManifest, AuditEvent, CapabilityPolicy, EventType

if TYPE_CHECKING:
    from aria.ledger.store import LedgerStore
    from aria.providers.base import LLMProvider
    from aria.sandbox.executor import SandboxExecutor

logger = logging.getLogger(__name__)


class AgentBase:
    """
    Base class for all ARIA agents.

    Subclass this, declare your tools with @tool (from aria.agent.decorators),
    and implement run_task(task: str) -> str.

    The sandbox executor intercepts every tool call, the ledger records every
    step, and the gate engine pauses execution at human checkpoints.
    """

    def __init__(
        self,
        manifest: AgentManifest,
        provider: LLMProvider,
        policy: CapabilityPolicy,
        ledger: LedgerStore,
        session_id: str,
        sandbox: SandboxExecutor | None = None,
    ) -> None:
        self.manifest = manifest
        self.provider = provider
        self.policy = policy
        self.ledger = ledger
        self.session_id = session_id
        self._sandbox: SandboxExecutor | None = sandbox
        self._tools: dict[str, Any] = {}
        self._running = False
        self._register_tools()

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------

    def _register_tools(self) -> None:
        """Discover methods decorated with @tool and register them."""
        for attr_name in dir(self):
            method = getattr(self.__class__, attr_name, None)
            if method and getattr(method, "_is_aria_tool", False):
                self._tools[method._capability_name] = getattr(self, attr_name)

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Return tool schemas in LLM provider format."""
        schemas = []
        for attr_name in dir(self):
            method = getattr(self.__class__, attr_name, None)
            if method and getattr(method, "_is_aria_tool", False):
                schemas.append(
                    {
                        "name": method._capability_name,
                        "description": method._tool_description,
                        "parameters": method._tool_parameters,
                    }
                )
        return schemas

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, task: str) -> str:
        """Entry point called by the orchestrator."""
        self._running = True
        await self._emit(EventType.AGENT_START, input_data={"task": task})
        try:
            result = await self.run_task(task)
        except Exception as exc:
            await self._emit(EventType.ERROR, output_data={"error": str(exc)})
            raise
        finally:
            self._running = False
            await self._emit(EventType.AGENT_STOP, output_data={"result": result if self._running is False else ""})
        return result

    async def run_task(self, task: str) -> str:
        """
        Override in subclasses to implement agent logic.
        Use self.think(), self.act(), self.observe() for structured loops.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Think / Act / Observe
    # ------------------------------------------------------------------

    async def think(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Call the LLM provider and log the reasoning step."""
        await self._emit(EventType.THINK, input_data={"messages": messages})
        response = await self.provider.complete(messages, tools=self.get_tool_schemas())
        await self._emit(
            EventType.THINK,
            output_data={"response": response.model_dump()},
            confidence=response.confidence,
        )
        return response.model_dump()

    async def act(self, capability: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool through the sandbox and log the action."""
        await self._emit(
            EventType.TOOL_CALL,
            input_data={"capability": capability, "arguments": arguments},
        )

        if self._sandbox:
            result = await self._sandbox.execute(
                agent_id=self.manifest.agent_id,
                capability=capability,
                arguments=arguments,
                tool_fn=self._tools.get(capability),
                session_id=self.session_id,
            )
        else:
            tool_fn = self._tools.get(capability)
            if tool_fn is None:
                raise ValueError(f"Unknown tool: {capability}")
            result = await tool_fn(**arguments) if asyncio.iscoroutinefunction(tool_fn) else tool_fn(**arguments)

        await self._emit(
            EventType.TOOL_RESULT,
            input_data={"capability": capability},
            output_data={"result": str(result)},
        )
        return result

    async def observe(self, observation: str) -> None:
        """Log an observation from the environment."""
        await self._emit(EventType.OBSERVE, input_data={"observation": observation})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _emit(
        self,
        event_type: EventType,
        input_data: dict[str, Any] | None = None,
        output_data: dict[str, Any] | None = None,
        confidence: float | None = None,
    ) -> None:
        event = AuditEvent(
            session_id=self.session_id,
            agent_id=self.manifest.agent_id,
            event_type=event_type,
            input_data=input_data or {},
            output_data=output_data or {},
            confidence=confidence,
        )
        await self.ledger.append(event)
        logger.debug("[%s] %s: %s", self.manifest.agent_id, event_type.value, input_data)
