"""Resolving how a selected tool actually gets called.

Every framework adapter faces the same three cases and should answer them the
same way:

* the tool came from a Python function, so call it directly;
* the tool is served by a remote source, so hand it to the caller's ``invoker``;
* neither applies, so fail loudly rather than emit a tool that raises at call
  time — a broken tool the model can see is worse than one it cannot.

This module holds that logic once so adapters do not each reinvent it, and so
the error message a user sees is the same whichever framework they are on.

ToolBroker still invokes nothing itself. :class:`ToolBinding` only *describes*
the call; something else has to make it.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

from ..errors import AdapterError
from ..types import Tool

#: Called as ``invoker(tool, arguments)`` for tools with no local callable —
#: typically forwarding to the MCP server that owns them. May be sync or async.
Invoker: TypeAlias = Callable[[Tool, dict[str, Any]], Any]

CALLABLE_KEY = "callable"


@dataclass(frozen=True, slots=True)
class ToolBinding:
    """A tool paired with the function that runs it."""

    tool: Tool
    func: Callable[..., Any]
    is_remote: bool

    @property
    def name(self) -> str:
        """The tool's bare name."""
        return self.tool.name

    @property
    def id(self) -> str:
        """The tool's ``namespace/name`` id."""
        return self.tool.id

    def call(self, arguments: Mapping[str, Any] | None = None) -> Any:
        """Invoke synchronously.

        Returns the coroutine untouched if the underlying function is async, so
        a sync caller can decide what to do with it rather than having one
        silently swallowed.
        """
        return self.func(**dict(arguments or {}))

    async def acall(self, arguments: Mapping[str, Any] | None = None) -> Any:
        """Invoke, awaiting the result if the function is asynchronous."""
        result = self.func(**dict(arguments or {}))
        if inspect.isawaitable(result):
            return await result
        return result


def local_callable(tool: Tool) -> Callable[..., Any] | None:
    """Return the Python function behind ``tool``, if the source supplied one."""
    handle = tool.metadata.get(CALLABLE_KEY)
    return handle if callable(handle) else None


def bind(tool: Tool, invoker: Invoker | None = None) -> ToolBinding | None:
    """Resolve how ``tool`` should be called, or ``None`` if it cannot be.

    Args:
        tool: The tool to bind.
        invoker: Fallback for tools with no local callable.
    """
    local = local_callable(tool)
    if local is not None:
        return ToolBinding(tool=tool, func=local, is_remote=False)
    if invoker is None:
        return None

    def remote(**kwargs: Any) -> Any:
        return invoker(tool, kwargs)

    # Frameworks that introspect the function for a name or docstring should
    # see the tool's, not this closure's.
    remote.__name__ = tool.name
    remote.__doc__ = tool.description
    return ToolBinding(tool=tool, func=remote, is_remote=True)


def bind_all(
    tools: Sequence[Tool],
    *,
    invoker: Invoker | None = None,
    skip_uninvokable: bool = False,
    adapter: str = "adapter",
) -> list[ToolBinding]:
    """Bind every tool, or explain precisely which one could not be bound.

    Args:
        tools: The tools to bind.
        invoker: Fallback for tools with no local callable.
        skip_uninvokable: Drop unbindable tools instead of raising. Useful when
            a catalogue mixes local and remote tools and the caller only wants
            the local ones.
        adapter: Name used in the error message.

    Raises:
        AdapterError: If a tool cannot be bound and ``skip_uninvokable`` is off.
    """
    bindings: list[ToolBinding] = []
    for tool in tools:
        binding = bind(tool, invoker)
        if binding is not None:
            bindings.append(binding)
            continue
        if skip_uninvokable:
            continue
        raise AdapterError(
            f"tool {tool.id!r} has no local callable and no invoker was supplied; "
            f"it is served by {tool.source_id!r}. Pass invoker=... to {adapter} "
            "so the call can be forwarded, or set skip_uninvokable=True to drop it."
        )
    return bindings
