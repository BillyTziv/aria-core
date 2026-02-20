"""
Local (mock) provider — runs without any API key.

Returns deterministic canned responses based on keyword matching.
Useful for local testing, CI, and demos of ARIA's safety layers.
"""

from __future__ import annotations

from typing import Any

from aria.providers.base import LLMProvider, LLMResponse, ToolCall


class LocalProvider(LLMProvider):
    """
    A fully offline LLM provider for testing ARIA without API keys.

    Scans the last user message for keywords and routes to canned tool calls
    or text responses. Extend ``_routes`` to add your own keyword → tool mappings.
    """

    def __init__(self, confidence: float = 0.95) -> None:
        self.confidence = confidence
        self._routes: list[tuple[str, str, dict[str, Any]]] = [
            # (keyword, tool_name, arguments)
            ("search", "web_search", {"query": "local test query"}),
            ("report", "write_report", {"filename": "report.txt", "content": "Local test report content."}),
            ("plan", "create_plan", {"goal": "local test goal"}),
            ("delegate", "delegate", {"step": "Step 1: gather data"}),
            ("gather", "gather_data", {"topic": "local test topic"}),
            ("summary", "write_summary", {"content": "Local test summary content."}),
            ("read", "read_file", {"path": "/tmp/test.txt"}),
            ("write", "write_file", {"path": "/tmp/test.txt", "content": "hello from ARIA"}),
        ]

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        text = (last_user or "").lower()
        available_tool_names = {t["name"] for t in (tools or [])}

        # Try to match a keyword route to an available tool
        for keyword, tool_name, arguments in self._routes:
            if keyword in text and tool_name in available_tool_names:
                return LLMResponse(
                    content=None,
                    tool_calls=[ToolCall(name=tool_name, arguments=arguments)],
                    confidence=self.confidence,
                    stop_reason="tool_use",
                )

        # Default: just respond with text
        return LLMResponse(
            content=f"[LocalProvider] Processed: {last_user[:80]}",
            tool_calls=[],
            confidence=self.confidence,
            stop_reason="end_turn",
        )
