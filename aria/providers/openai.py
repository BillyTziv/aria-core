"""OpenAI provider adapter."""

from __future__ import annotations

import logging
import os
from typing import Any

from aria.providers.base import LLMProvider, LLMResponse, ToolCall

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """
    Adapter for OpenAI's chat completions API (GPT-4o and newer).

    Translates ARIA's generic tool schema to OpenAI function-calling format
    and maps tool_call responses back to ``LLMResponse``.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        temperature: float = 0.2,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")

        self.model = model
        self.temperature = temperature
        self._client = AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        openai_tools = None
        if tools:
            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                    },
                }
                for t in tools
            ]

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=openai_tools or [],
            temperature=self.temperature,
            **kwargs,
        )

        choice = response.choices[0]
        message = choice.message

        tool_calls: list[ToolCall] = []
        if message.tool_calls:
            import json
            for tc in message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                        call_id=tc.id,
                    )
                )

        # Approximate confidence from logprobs if available, else None
        confidence: float | None = None
        if hasattr(choice, "logprobs") and choice.logprobs:
            import math
            lp = choice.logprobs.content
            if lp:
                avg_lp = sum(t.logprob for t in lp) / len(lp)
                confidence = round(math.exp(avg_lp), 4)

        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            confidence=confidence,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            stop_reason=choice.finish_reason or "",
        )
