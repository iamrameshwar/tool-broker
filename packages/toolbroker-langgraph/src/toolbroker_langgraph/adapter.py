"""Rendering a selection as LangChain tools.

Emits ``StructuredTool`` objects, which LangGraph's ``ToolNode`` and
LangChain's ``bind_tools`` both accept, so one adapter covers a graph and a
plain agent.

This lives outside the core deliberately: LangChain moves fast, and a breaking
change there should force a release of *this* package, not of ToolBroker itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from toolbroker.adapters.invocation import Invoker, bind_all
from toolbroker.errors import AdapterError
from toolbroker.types import Tool


class LangGraphAdapter:
    """Renders tools as LangChain ``StructuredTool`` instances."""

    id = "langgraph"

    def __init__(
        self,
        *,
        invoker: Invoker | None = None,
        skip_uninvokable: bool = False,
        infer_schema: bool = False,
    ) -> None:
        """Configure rendering.

        Args:
            invoker: Called as ``invoker(tool, kwargs)`` for tools with no local
                callable — typically forwarding to an MCP server.
            skip_uninvokable: Drop tools that cannot be invoked instead of
                raising.
            infer_schema: Let LangChain derive the argument schema from the
                function signature instead of using the tool's JSON Schema. Off
                by default, because a remote tool's signature is ``**kwargs``
                and inferring from it would discard the real schema.
        """
        self._invoker = invoker
        self._skip = skip_uninvokable
        self._infer_schema = infer_schema

    def render(self, tools: Sequence[Tool]) -> list[Any]:
        """Return LangChain tools for ``tools``."""
        try:
            from langchain_core.tools import StructuredTool
        except ImportError as exc:  # pragma: no cover - depends on install
            raise AdapterError(
                "LangGraphAdapter requires langchain-core. "
                "Install with: pip install toolbroker-langgraph"
            ) from exc

        bindings = bind_all(
            tools,
            invoker=self._invoker,
            skip_uninvokable=self._skip,
            adapter="LangGraphAdapter",
        )
        return [
            StructuredTool.from_function(
                func=binding.func,
                name=binding.name,
                description=binding.tool.description,
                args_schema=binding.tool.input_schema or None,
                infer_schema=self._infer_schema,
                metadata={
                    "toolbroker_id": binding.id,
                    "source_id": binding.tool.source_id,
                    "risk": binding.tool.risk.value,
                },
            )
            for binding in bindings
        ]
