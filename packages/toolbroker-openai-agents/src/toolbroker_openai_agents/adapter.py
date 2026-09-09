"""Rendering a selection as OpenAI Agents SDK tools.

The SDK's ``FunctionTool`` takes an async ``on_invoke_tool(context, args_json)``
that receives the model's arguments as a **JSON string**, so this adapter owns
the parse. Malformed JSON gets a clear error naming the tool rather than a
``JSONDecodeError`` from somewhere inside the run loop.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from toolbroker.adapters.invocation import Invoker, ToolBinding, bind_all
from toolbroker.errors import AdapterError
from toolbroker.types import JSONSchema, Tool


def _strict_schema(schema: JSONSchema) -> JSONSchema:
    """Coerce a schema into the shape strict mode demands."""
    strict = dict(schema) if schema else {}
    strict.setdefault("type", "object")
    properties = strict.setdefault("properties", {})
    strict["additionalProperties"] = False
    strict["required"] = sorted(properties)
    return strict


def _normalize(schema: JSONSchema) -> JSONSchema:
    """Return a schema the SDK will accept, filling in an empty object."""
    if not schema:
        return {"type": "object", "properties": {}}
    normalized = dict(schema)
    normalized.setdefault("type", "object")
    normalized.setdefault("properties", {})
    return normalized


class OpenAIAgentsAdapter:
    """Renders tools as OpenAI Agents SDK ``FunctionTool`` instances."""

    id = "openai-agents"

    def __init__(
        self,
        *,
        invoker: Invoker | None = None,
        skip_uninvokable: bool = False,
        strict: bool = False,
        use_qualified_names: bool = False,
    ) -> None:
        """Configure rendering.

        Args:
            invoker: Called as ``invoker(tool, kwargs)`` for tools with no local
                callable — typically forwarding to an MCP server.
            skip_uninvokable: Drop tools that cannot be invoked instead of
                raising.
            strict: Emit strict-mode schemas. Off by default because strict mode
                requires every property to be required with
                ``additionalProperties: false``, which most real MCP and OpenAPI
                schemas do not satisfy.
            use_qualified_names: Name tools ``namespace__name`` instead of
                ``name``. Turn this on when two namespaces expose the same tool
                name; the SDK has no notion of namespaces and the model would
                otherwise see a duplicate.
        """
        self._invoker = invoker
        self._skip = skip_uninvokable
        self._strict = strict
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

    def _name_for(self, tool: Tool) -> str:
        if self._qualified and tool.namespace != "default":
            return f"{tool.namespace}__{tool.name}"
        return tool.name

    def render(self, tools: Sequence[Tool]) -> list[Any]:
        """Return ``FunctionTool`` objects for ``tools``."""
        try:
            from agents import FunctionTool
        except ImportError as exc:  # pragma: no cover - depends on install
            raise AdapterError(
                "OpenAIAgentsAdapter requires the openai-agents package. "
                "Install with: pip install toolbroker-openai-agents"
            ) from exc

        bindings = bind_all(
            tools,
            invoker=self._invoker,
            skip_uninvokable=self._skip,
            adapter="OpenAIAgentsAdapter",
        )

        self._by_name = {}
        rendered: list[Any] = []
        for binding in bindings:
            name = self._name_for(binding.tool)
            if name in self._by_name:
                raise AdapterError(
                    f"two tools render to the name {name!r} "
                    f"({self._by_name[name].id} and {binding.id}). "
                    "Pass use_qualified_names=True to disambiguate by namespace."
                )
            self._by_name[name] = binding.tool
            schema = binding.tool.input_schema
            rendered.append(
                FunctionTool(
                    name=name,
                    description=binding.tool.description,
                    params_json_schema=(
                        _strict_schema(schema) if self._strict else _normalize(schema)
                    ),
                    on_invoke_tool=self._handler(binding, name),
                    strict_json_schema=self._strict,
                )
            )
        return rendered

    @staticmethod
    def _handler(binding: ToolBinding, name: str) -> Any:
        """Build the SDK's async invoke callback for one binding."""

        async def on_invoke_tool(context: Any, arguments: str) -> Any:
            del context
            try:
                parsed = json.loads(arguments) if arguments else {}
            except json.JSONDecodeError as exc:
                # Without the tool name this surfaces as a bare decode error
                # from inside the run loop, which is near-impossible to trace.
                raise AdapterError(
                    f"model sent invalid JSON arguments for tool {name!r}: {exc}"
                ) from exc
            if not isinstance(parsed, dict):
                raise AdapterError(
                    f"tool {name!r} expected a JSON object of arguments, "
                    f"got {type(parsed).__name__}"
                )
            return await binding.acall(parsed)

        return on_invoke_tool
