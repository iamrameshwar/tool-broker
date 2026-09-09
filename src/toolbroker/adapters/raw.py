"""Provider-native JSON schemas, no framework required.

The "no framework" path. If you are writing your own tool-calling loop against
the OpenAI or Anthropic API, these hand you exactly the list those APIs expect.

Both providers cap tool-name characters and neither allows ``/``, so the
``namespace/name`` id is flattened to ``namespace__name``.
:meth:`resolve_name` maps it back, which is what a caller needs when the model
replies with a tool name and they have to find the tool again.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from ..errors import AdapterError
from ..types import JSONSchema, Tool

SEPARATOR = "__"
_INVALID = re.compile(r"[^a-zA-Z0-9_-]")
MAX_NAME_LENGTH = 64


def flatten_name(tool: Tool) -> str:
    """Return a provider-safe name for ``tool``.

    Truncation keeps the tail of the tool name rather than the head: ``name``
    is more distinguishing than ``namespace`` when they collide.
    """
    raw = f"{tool.namespace}{SEPARATOR}{tool.name}" if tool.namespace != "default" else tool.name
    safe = _INVALID.sub("_", raw)
    if len(safe) <= MAX_NAME_LENGTH:
        return safe
    return safe[: MAX_NAME_LENGTH - len(tool.name) - 1] + "_" + tool.name[-MAX_NAME_LENGTH:]


def _normalize_schema(schema: JSONSchema) -> JSONSchema:
    """Return a schema both providers accept, filling in an empty object."""
    if not schema:
        return {"type": "object", "properties": {}}
    normalized = dict(schema)
    normalized.setdefault("type", "object")
    normalized.setdefault("properties", {})
    return normalized


class _BaseRawAdapter:
    """Shared name-resolution logic for the provider adapters."""

    id = "raw"

    def __init__(self) -> None:
        """Start with an empty name map."""
        self._by_flat_name: dict[str, Tool] = {}

    def _register(self, tools: Sequence[Tool]) -> None:
        mapping: dict[str, Tool] = {}
        collisions: dict[str, list[str]] = {}
        for tool in tools:
            flat = flatten_name(tool)
            existing = mapping.get(flat)
            if existing is not None:
                collisions.setdefault(flat, [existing.id]).append(tool.id)
            else:
                mapping[flat] = tool

        if collisions:
            # Silently dropping one of them would leave the model able to see a
            # tool it can never successfully call.
            detail = "; ".join(
                f"{flat!r} <- {', '.join(ids)}" for flat, ids in sorted(collisions.items())
            )
            raise AdapterError(
                f"tool names collide after flattening for the provider API: {detail}. "
                "Rename one of them or use distinct namespaces."
            )
        self._by_flat_name = mapping

    def resolve_name(self, flat_name: str) -> Tool:
        """Map a provider tool name back to the tool it came from."""
        tool = self._by_flat_name.get(flat_name)
        if tool is None:
            raise AdapterError(
                f"unknown tool name {flat_name!r}; known: {sorted(self._by_flat_name)}"
            )
        return tool

    def callable_for(self, flat_name: str) -> Any:
        """Return the Python callable behind a tool name, if the source had one.

        ToolBroker does not invoke it. This exists so the caller's loop can.
        """
        tool = self.resolve_name(flat_name)
        handle = tool.metadata.get("callable")
        if handle is None:
            raise AdapterError(
                f"tool {tool.id!r} has no local callable; it is served by "
                f"{tool.source_id!r} and must be invoked through that source"
            )
        return handle

    @property
    def names(self) -> Mapping[str, Tool]:
        """The current flattened-name to tool map."""
        return dict(self._by_flat_name)


class OpenAIAdapter(_BaseRawAdapter):
    """Renders tools as OpenAI ``tools=[...]`` entries."""

    id = "openai"

    def __init__(self, *, strict: bool = False) -> None:
        """Configure rendering.

        Args:
            strict: Emit OpenAI structured-outputs strict mode. Requires every
                property to be required and ``additionalProperties: false``,
                so it is off by default — most MCP schemas do not satisfy it.
        """
        super().__init__()
        self._strict = strict

    def render(self, tools: Sequence[Tool]) -> list[dict[str, Any]]:
        """Return the OpenAI function-calling representation."""
        self._register(tools)
        rendered: list[dict[str, Any]] = []
        for tool in tools:
            schema = _normalize_schema(tool.input_schema)
            function: dict[str, Any] = {
                "name": flatten_name(tool),
                "description": tool.description,
                "parameters": schema,
            }
            if self._strict:
                schema["additionalProperties"] = False
                schema["required"] = sorted(schema.get("properties", {}))
                function["strict"] = True
            rendered.append({"type": "function", "function": function})
        return rendered


class AnthropicAdapter(_BaseRawAdapter):
    """Renders tools as Anthropic Messages API ``tools=[...]`` entries."""

    id = "anthropic"

    def render(self, tools: Sequence[Tool]) -> list[dict[str, Any]]:
        """Return the Anthropic tool-use representation."""
        self._register(tools)
        return [
            {
                "name": flatten_name(tool),
                "description": tool.description,
                "input_schema": _normalize_schema(tool.input_schema),
            }
            for tool in tools
        ]
