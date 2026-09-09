"""Rendering a selection as Claude Agent SDK tools.

The SDK hosts tools as an in-process MCP server, so this adapter emits
``SdkMcpTool`` objects and can wrap them into the server config
``ClaudeAgentOptions`` expects.

Two details this adapter exists to get right:

* **Handlers must return MCP content blocks**, not bare values. Whatever the
  underlying function returns is wrapped, so an existing Python function works
  unchanged.
* **The SDK gates tools by the name ``mcp__<server>__<tool>``.** Building that
  string by hand is easy to get subtly wrong, and the failure — the model sees a
  tool and is then refused permission to call it — reads like a bug in the
  model. :meth:`ClaudeAgentAdapter.allowed_tool_names` derives it from the same
  selection so the two cannot drift.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from toolbroker.adapters.invocation import Invoker, ToolBinding, bind_all
from toolbroker.errors import AdapterError
from toolbroker.types import JSONSchema, Tool

#: Turns whatever a tool returned into an MCP result payload.
ResultFormatter = Callable[[Any], dict[str, Any]]


def default_result_formatter(result: Any) -> dict[str, Any]:
    """Wrap a tool's return value as MCP content.

    A mapping that already looks like an MCP result passes through untouched;
    anything else becomes a single text block. That keeps ordinary Python
    functions usable without rewriting them to know about MCP.
    """
    if isinstance(result, dict) and "content" in result:
        return result
    if result is None:
        return {"content": []}
    return {"content": [{"type": "text", "text": str(result)}]}


def _normalize(schema: JSONSchema) -> JSONSchema:
    """Return a schema the SDK will accept, filling in an empty object."""
    if not schema:
        return {"type": "object", "properties": {}}
    normalized = dict(schema)
    normalized.setdefault("type", "object")
    normalized.setdefault("properties", {})
    return normalized


class ClaudeAgentAdapter:
    """Renders tools as Claude Agent SDK ``SdkMcpTool`` instances."""

    id = "claude-agent"

    def __init__(
        self,
        *,
        invoker: Invoker | None = None,
        skip_uninvokable: bool = False,
        result_formatter: ResultFormatter | None = None,
        use_qualified_names: bool = False,
    ) -> None:
        """Configure rendering.

        Args:
            invoker: Called as ``invoker(tool, kwargs)`` for tools with no local
                callable — typically forwarding to an MCP server.
            skip_uninvokable: Drop tools that cannot be invoked instead of
                raising.
            result_formatter: Converts a return value into an MCP result.
                Override to emit image or resource blocks.
            use_qualified_names: Name tools ``namespace__name``. Turn this on
                when two namespaces expose the same tool name.
        """
        self._invoker = invoker
        self._skip = skip_uninvokable
        self._format = result_formatter or default_result_formatter
        self._qualified = use_qualified_names
        self._by_name: dict[str, Tool] = {}

    @property
    def names(self) -> dict[str, Tool]:
        """Map of emitted tool name to the tool it came from."""
        return dict(self._by_name)

    def resolve_name(self, name: str) -> Tool:
        """Map an emitted tool name back to its tool."""
        tool = self._by_name.get(name)
        if tool is None:
            raise AdapterError(f"unknown tool name {name!r}; known: {sorted(self._by_name)}")
        return tool

    def tool_name(self, tool: Tool) -> str:
        """Return the name this adapter would emit for ``tool``."""
        if self._qualified and tool.namespace != "default":
            return f"{tool.namespace}__{tool.name}"
        return tool.name

    def allowed_tool_names(self, tools: Sequence[Tool], *, server: str = "toolbroker") -> list[str]:
        """Return the ``mcp__<server>__<tool>`` names the SDK gates on.

        Pass the result as ``ClaudeAgentOptions(allowed_tools=...)``. Derived
        from the same selection you rendered, so the permission list cannot
        drift out of sync with the tools the model can see.
        """
        return [f"mcp__{server}__{self.tool_name(tool)}" for tool in tools]

    def render(self, tools: Sequence[Tool]) -> list[Any]:
        """Return ``SdkMcpTool`` objects for ``tools``."""
        sdk_tool = self._sdk_tool()
        bindings = bind_all(
            tools,
            invoker=self._invoker,
            skip_uninvokable=self._skip,
            adapter="ClaudeAgentAdapter",
        )

        self._by_name = {}
        rendered: list[Any] = []
        for binding in bindings:
            name = self.tool_name(binding.tool)
            if name in self._by_name:
                raise AdapterError(
                    f"two tools render to the name {name!r} "
                    f"({self._by_name[name].id} and {binding.id}). "
                    "Pass use_qualified_names=True to disambiguate by namespace."
                )
            self._by_name[name] = binding.tool
            decorator = sdk_tool(
                name,
                binding.tool.description,
                _normalize(binding.tool.input_schema),
            )
            rendered.append(decorator(self._handler(binding)))
        return rendered

    def create_server(
        self,
        tools: Sequence[Tool],
        *,
        name: str = "toolbroker",
        version: str = "1.0.0",
    ) -> Any:
        """Wrap ``tools`` into the in-process MCP server config the SDK expects."""
        try:
            from claude_agent_sdk import create_sdk_mcp_server
        except ImportError as exc:  # pragma: no cover - depends on install
            raise AdapterError(
                "ClaudeAgentAdapter requires the claude-agent-sdk package. "
                "Install with: pip install toolbroker-claude-agent"
            ) from exc
        return create_sdk_mcp_server(name=name, version=version, tools=self.render(tools))

    @staticmethod
    def _sdk_tool() -> Any:
        """Import the SDK's tool decorator, with an actionable error."""
        try:
            from claude_agent_sdk import tool
        except ImportError as exc:  # pragma: no cover - depends on install
            raise AdapterError(
                "ClaudeAgentAdapter requires the claude-agent-sdk package. "
                "Install with: pip install toolbroker-claude-agent"
            ) from exc
        return tool

    def _handler(self, binding: ToolBinding) -> Any:
        """Build the SDK's async handler for one binding."""
        formatter = self._format

        async def handler(arguments: Any) -> dict[str, Any]:
            payload = dict(arguments) if isinstance(arguments, dict) else {}
            return formatter(await binding.acall(payload))

        handler.__name__ = binding.name
        handler.__doc__ = binding.tool.description
        return handler
