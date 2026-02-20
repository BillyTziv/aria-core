"""LLM Provider interface and response model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]
    call_id: str = ""


class LLMResponse(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] = []
    confidence: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = ""

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class LLMProvider(ABC):
    """Abstract base for all LLM provider adapters."""

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send messages to the LLM and return a structured response."""
        ...
