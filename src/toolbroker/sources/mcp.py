"""MCP servers as a tool source.

Built on the MCP SDK's high-level ``Client``, which accepts stdio parameters,
an HTTP URL, or an in-process server object. That last form is what lets the
test suite exercise real MCP round-trips without spawning a subprocess.

Discovery is exposed both synchronously (:meth:`MCPSource.discover`, which
drives its own event loop) and asynchronously (:meth:`MCPSource.adiscover`), so
it fits either style of host application.

On :meth:`call`: forwarding a *single* tool invocation to the server that owns
it is transport, not orchestration. ToolBroker still never runs an agent loop;
this exists so the MCP proxy can complete a call the client already decided to
make.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from ..errors import SourceError
from ..observability import get_logger
from ..risk import RiskClassifier, classify_by_name
from ..types import CostTier, RiskTier, Tool
from .base import BaseSource

logger = get_logger("sources.mcp")

Transport = Literal["stdio", "http"]

#: Kept as an alias so existing imports from this module keep working.
default_risk_classifier = classify_by_name


@dataclass(slots=True)
class MCPServerSpec:
    """Connection details for one MCP server."""

    name: str
    transport: Transport = "stdio"
    command: str | None = None
    args: Sequence[str] = field(default_factory=tuple)
    env: Mapping[str, str] | None = None
    url: str | None = None
    headers: Mapping[str, str] | None = None
    timeout_seconds: float = 30.0
    # An already-constructed server or transport object, for tests and for
    # hosts that manage their own connections.
    target: Any = None

    def __post_init__(self) -> None:
        """Validate that the fields match the chosen transport."""
        if self.target is not None:
            return
        if self.transport == "stdio" and not self.command:
            raise SourceError(f"MCP server {self.name!r}: stdio transport requires 'command'")
        if self.transport == "http" and not self.url:
            raise SourceError(f"MCP server {self.name!r}: http transport requires 'url'")


def _require_mcp() -> Any:
    """Import the MCP SDK, with an actionable message when it is missing."""
    try:
        import mcp
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise SourceError(
            "MCP support requires the 'mcp' package. Install with: pip install 'toolbroker[mcp]'"
        ) from exc
    if not hasattr(mcp, "Client"):  # pragma: no cover - depends on install extras
        raise SourceError(
            "toolbroker requires mcp>=2 (the SDK version with the high-level Client). "
            "Upgrade with: pip install --upgrade 'mcp>=2'"
        )
    return mcp


class MCPSource(BaseSource):
    """Discovers tools from one MCP server."""

    def __init__(
        self,
        spec: MCPServerSpec,
        *,
        namespace: str | None = None,
        tags: Iterable[str] = (),
        risk_classifier: RiskClassifier | None = None,
        default_cost: CostTier = CostTier.LOW,
    ) -> None:
        """Create a source for ``spec``."""
        super().__init__(f"mcp:{spec.name}")
        self._spec = spec
        self._namespace = namespace or spec.name
        self._tags = frozenset(tags)
        self._classify = risk_classifier or classify_by_name
        self._default_cost = default_cost

    @property
    def spec(self) -> MCPServerSpec:
        """The server this source talks to."""
        return self._spec

    def _target(self) -> Any:
        """Return whatever the SDK's ``Client`` should connect to."""
        mcp = _require_mcp()
        if self._spec.target is not None:
            return self._spec.target
        if self._spec.transport == "http":
            return self._spec.url
        return mcp.StdioServerParameters(
            command=self._spec.command or "",
            args=list(self._spec.args),
            env=dict(self._spec.env) if self._spec.env else None,
        )

    @asynccontextmanager
    async def _client(self) -> AsyncIterator[Any]:
        """Open a connected MCP client."""
        mcp = _require_mcp()
        async with mcp.Client(
            self._target(), read_timeout_seconds=self._spec.timeout_seconds
        ) as client:
            yield client

    async def adiscover(self) -> list[Tool]:
        """Discover tools asynchronously."""
        try:
            async with self._client() as client:
                remote_tools = await asyncio.wait_for(
                    client.list_tools(), timeout=self._spec.timeout_seconds
                )
        except SourceError:
            raise
        except (TimeoutError, asyncio.TimeoutError) as exc:
            raise SourceError(
                f"MCP server {self._spec.name!r} did not respond within "
                f"{self._spec.timeout_seconds}s"
            ) from exc
        except Exception as exc:
            raise SourceError(f"MCP server {self._spec.name!r} discovery failed: {exc}") from exc

        tools = [self._to_tool(remote) for remote in _unwrap_tools(remote_tools)]
        logger.info(
            "discovered tools from MCP server",
            extra={"server": self._spec.name, "count": len(tools)},
        )
        return tools

    async def acall(self, name: str, arguments: Mapping[str, Any]) -> Any:
        """Forward a single tool call to the server and return its result.

        Used only by the MCP proxy, to complete a call the client already made.
        """
        try:
            async with self._client() as client:
                return await asyncio.wait_for(
                    client.call_tool(name, dict(arguments)),
                    timeout=self._spec.timeout_seconds,
                )
        except (TimeoutError, asyncio.TimeoutError) as exc:
            raise SourceError(f"MCP server {self._spec.name!r} timed out calling {name!r}") from exc
        except Exception as exc:
            raise SourceError(
                f"MCP server {self._spec.name!r} failed calling {name!r}: {exc}"
            ) from exc

    def call(self, name: str, arguments: Mapping[str, Any]) -> Any:
        """Synchronous wrapper around :meth:`acall`."""
        return _run_sync(self.acall(name, arguments))

    def _to_tool(self, remote: Any) -> Tool:
        """Convert an MCP tool definition into our model."""
        name = getattr(remote, "name", "") or ""
        description = getattr(remote, "description", "") or ""
        schema = getattr(remote, "inputSchema", None) or getattr(remote, "input_schema", None) or {}
        annotations = getattr(remote, "annotations", None)

        risk = self._classify(name, description)
        # A server that declares its own behaviour is trusted over our guess.
        if annotations is not None:
            if getattr(annotations, "readOnlyHint", None) is True:
                risk = RiskTier.LOW
            elif getattr(annotations, "destructiveHint", None) is True:
                risk = RiskTier.HIGH

        return Tool(
            name=name,
            namespace=self._namespace,
            description=description,
            input_schema=dict(schema),
            tags=self._tags,
            risk=risk,
            cost=self._default_cost,
            source_id=self._id,
            metadata={"mcp_server": self._spec.name, "remote_name": name},
        )

    def _discover(self) -> Iterable[Tool]:
        return cast("list[Tool]", _run_sync(self.adiscover()))


def _unwrap_tools(result: Any) -> Sequence[Any]:
    """Return the tool list from either a bare list or a ``ListToolsResult``."""
    if isinstance(result, Sequence):
        return result
    tools = getattr(result, "tools", None)
    if tools is None:
        raise SourceError(f"unexpected list_tools response: {type(result).__name__}")
    return list(tools)


def _run_sync(coro: Any) -> Any:
    """Run ``coro`` to completion.

    Refuses to run inside an already-running loop rather than deadlocking; the
    caller should await the async variant instead.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Close it explicitly, or Python warns about a coroutine that was never
    # awaited on top of the error the caller actually needs to read.
    coro.close()
    raise SourceError(
        "synchronous MCP discovery cannot run inside an active event loop; "
        "await the async variant (adiscover/acall) instead"
    )
