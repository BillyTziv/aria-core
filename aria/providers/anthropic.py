"""Anthropic provider adapter."""

from __future__ import annotations

import logging
import os
from typing import Any

from aria.providers.base import LLMProvider, LLMResponse, ToolCall

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """
    Adapter for Anthropic's Messages API (Claude 3.x and newer).

    Translates ARIA's generic tool schema to Anthropic tool format and maps
    tool_use content blocks back to ``LLMResponse``.
    """

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: str | None = None,
        max_tokens: int = 4096,
    ) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")

        self.model = model
        self.max_tokens = max_tokens
        self._client = AsyncAnthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        # Anthropic requires system messages to be passed separately
        system: str = ""
        filtered: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "system":
                system = msg.get("content", "")
            else:
                filtered.append(msg)

        anthropic_tools = None
        if tools:
            anthropic_tools = [
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "input_schema": t.get("parameters", {"type": "object", "properties": {}}),
                }
                for t in tools
            ]

        kwargs_final: dict[str, Any] = {"max_tokens": self.max_tokens}
        if system:
            kwargs_final["system"] = system
        if anthropic_tools:
            kwargs_final["tools"] = anthropic_tools
        kwargs_final.update(kwargs)

        response = await self._client.messages.create(
            model=self.model,
            messages=filtered,
            **kwargs_final,
        )

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        name=block.name,
                        arguments=block.input,
                        call_id=block.id,
                    )
                )

        return LLMResponse(
            content="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            confidence=None,  # Anthropic doesn't expose logprobs yet
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason or "",
        )
