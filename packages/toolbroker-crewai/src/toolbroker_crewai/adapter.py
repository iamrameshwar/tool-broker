"""Rendering a selection as CrewAI tools.

CrewAI's ``BaseTool`` wants ``args_schema`` as a pydantic model class rather
than a JSON Schema dict, so each tool gets a model generated from its schema by
:func:`toolbroker.schema.model_from_schema`.

Sync and async both work. ``_run`` hands back whatever the bound function
returns — including a coroutine, which CrewAI's own ``run`` awaits — and
``_arun`` awaits it directly for callers already inside a loop.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from toolbroker.adapters.invocation import Invoker, ToolBinding, bind_all
from toolbroker.errors import AdapterError
from toolbroker.schema import model_from_schema
from toolbroker.types import Tool


def _class_name(tool: Tool) -> str:
    """Return a model class name for ``tool``'s arguments."""
    parts = [segment for segment in tool.id.replace("/", "_").split("_") if segment]
    return "".join(part[:1].upper() + part[1:] for part in parts) + "Args"


class CrewAIAdapter:
    """Renders tools as CrewAI ``BaseTool`` instances."""

    id = "crewai"

    def __init__(
        self,
        *,
        invoker: Invoker | None = None,
        skip_uninvokable: bool = False,
        use_qualified_names: bool = False,
        result_as_answer: bool = False,
    ) -> None:
        """Configure rendering.

        Args:
            invoker: Called as ``invoker(tool, kwargs)`` for tools with no local
                callable — typically forwarding to an MCP server.
            skip_uninvokable: Drop tools that cannot be invoked instead of
                raising.
            use_qualified_names: Name tools ``namespace__name``. Turn this on
                when two namespaces expose the same tool name; CrewAI has no
                notion of namespaces and the agent would otherwise see a
                duplicate.
            result_as_answer: Set CrewAI's flag so a tool's output is returned
                as the task answer directly, skipping further reasoning.
        """
        self._invoker = invoker
        self._skip = skip_uninvokable
        self._qualified = use_qualified_names
        self._result_as_answer = result_as_answer
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

    def render(self, tools: Sequence[Tool]) -> list[Any]:
        """Return CrewAI ``BaseTool`` objects for ``tools``."""
        base_tool = self._base_tool()
        bindings = bind_all(
            tools,
            invoker=self._invoker,
            skip_uninvokable=self._skip,
            adapter="CrewAIAdapter",
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
            rendered.append(self._build(base_tool, binding, name))
        return rendered

    def _build(self, base_tool: type[Any], binding: ToolBinding, name: str) -> Any:
        """Construct one CrewAI tool for a binding."""
        args_model = model_from_schema(binding.tool.input_schema, name=_class_name(binding.tool))
        # CrewAI requires a description; an empty one makes the tool invisible
        # to the agent's reasoning.
        description = binding.tool.description or f"Call {binding.id}."

        class RenderedTool(base_tool):  # type: ignore[misc]
            """A ToolBroker-selected tool, in CrewAI's shape."""

            def _run(self, *args: Any, **kwargs: Any) -> Any:
                """Invoke the bound function.

                Returns a coroutine untouched when the function is async;
                CrewAI's own ``run`` awaits it.
                """
                del args
                return binding.call(kwargs)

            async def _arun(self, *args: Any, **kwargs: Any) -> Any:
                """Invoke the bound function, awaiting if it is async."""
                del args
                return await binding.acall(kwargs)

        RenderedTool.__name__ = f"{_class_name(binding.tool)[:-4]}Tool"
        return RenderedTool(
            name=name,
            description=description,
            args_schema=args_model,
            result_as_answer=self._result_as_answer,
        )

    @staticmethod
    def _base_tool() -> type[Any]:
        """Import CrewAI's ``BaseTool``, with an actionable error."""
        try:
            from crewai.tools import BaseTool
        except ImportError as exc:  # pragma: no cover - depends on install
            raise AdapterError(
                "CrewAIAdapter requires the crewai package. "
                "Install with: pip install toolbroker-crewai"
            ) from exc
        return BaseTool
