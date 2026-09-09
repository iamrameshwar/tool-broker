"""ToolBroker as a single MCP server.

The zero-code integration path. Instead of attaching 40 MCP servers to a client
and drowning it in 500 tool definitions, attach this one. It exposes three
tools:

* ``search_tools(query, limit)`` — the retrieval and policy layer, as a tool
* ``describe_tool(tool_id)`` — the full schema for one result
* ``call_tool(tool_id, arguments)`` — forwards the call to the owning server

Any MCP client — Claude Desktop, an IDE, a custom agent — gets tool retrieval
without changing a line of its own code.

The two-step shape (search, then call) is what keeps the context small: the
client holds three tool definitions permanently and pulls in real schemas only
for the handful a query actually surfaced.

``call_tool`` forwarding is transport, not orchestration: the client decided to
make the call, and we complete it. ToolBroker still runs no agent loop.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal

from ..catalog import ToolBroker
from ..errors import AdapterError, ToolBrokerError
from ..observability import get_logger
from ..refresh import PeriodicRefresher
from ..sources.mcp import MCPSource
from ..types import Hit, Selection, Tool

logger = get_logger("proxy")

ServeTransport = Literal["stdio", "http"]


class ToolBrokerProxy:
    """Wraps a catalogue in an MCP server."""

    def __init__(
        self,
        broker: ToolBroker,
        *,
        name: str = "toolbroker",
        default_k: int = 5,
        agent: str | None = None,
        scopes: frozenset[str] = frozenset(),
        allow_calls: bool = True,
        refresh_interval: float | None = None,
    ) -> None:
        """Configure the proxy.

        Args:
            broker: The catalogue to serve.
            name: MCP server name advertised to clients.
            default_k: Tools returned per search when the client does not say.
            agent: Agent identity applied to every search, selecting a policy.
            scopes: Scopes granted to this proxy's callers.
            allow_calls: Expose ``call_tool``. Turn this off to run the proxy as
                a pure discovery service, with execution staying on the client's
                own connections to the underlying servers.
            refresh_interval: Seconds between background re-discoveries. A
                proxy is long-lived and the servers behind it are not: without
                this it serves whatever tools existed at startup, forever.
                ``None`` disables refreshing.
        """
        self._broker = broker
        self._name = name
        self._default_k = default_k
        self._agent = agent
        self._scopes = scopes
        self._allow_calls = allow_calls
        self._sources_by_id = {
            source.id: source for source in broker.sources if isinstance(source, MCPSource)
        }
        self._refresher: PeriodicRefresher | None = None
        if refresh_interval is not None:
            self._refresher = PeriodicRefresher(broker, interval=refresh_interval)

    @property
    def broker(self) -> ToolBroker:
        """The catalogue being served."""
        return self._broker

    @property
    def refresher(self) -> PeriodicRefresher | None:
        """The background refresher, if one was configured."""
        return self._refresher

    # -- the three exposed operations -------------------------------------

    def search_tools(self, query: str, limit: int | None = None) -> dict[str, Any]:
        """Return the tools most relevant to ``query``.

        Args:
            query: What you are trying to do, in natural language.
            limit: Maximum tools to return.

        Returns:
            Matching tools with id, description, and score, plus a count of how
            many were considered.
        """
        selection = self._broker.select(
            query,
            k=limit or self._default_k,
            agent=self._agent,
            scopes=self._scopes,
        )
        return {
            "query": query,
            "considered": selection.considered,
            "returned": len(selection),
            "tools": [
                {
                    "tool_id": hit.id,
                    "description": hit.tool.description,
                    "score": round(hit.score, 4),
                    "risk": hit.tool.risk.value,
                }
                for hit in selection.hits
            ],
            "note": "call describe_tool(tool_id) for the full input schema",
        }

    def describe_tool(self, tool_id: str) -> dict[str, Any]:
        """Return the full definition of one tool.

        Subject to the same policy as search. A description states what a tool
        can do and its schema names every parameter, so handing those to a
        client that policy will not let call it is disclosure, not convenience.

        Args:
            tool_id: The ``namespace/name`` id from a search result.
        """
        tool = self._require_visible(tool_id)
        return {
            "tool_id": tool.id,
            "name": tool.name,
            "namespace": tool.namespace,
            "description": tool.description,
            "input_schema": tool.input_schema,
            "risk": tool.risk.value,
            "cost": tool.cost.value,
            "tags": sorted(tool.tags),
            "required_scopes": sorted(tool.required_scopes),
            "source": tool.source_id,
        }

    async def call_tool(self, tool_id: str, arguments: Mapping[str, Any] | None = None) -> Any:
        """Forward a call to the server that owns ``tool_id``.

        Args:
            tool_id: The ``namespace/name`` id from a search result.
            arguments: Arguments matching the tool's input schema.
        """
        if not self._allow_calls:
            raise ToolBrokerError("this proxy is configured for discovery only; calls are disabled")

        tool = self._require_visible(tool_id)
        payload = dict(arguments or {})

        source = self._sources_by_id.get(tool.source_id)
        if source is not None:
            remote_name = str(tool.metadata.get("remote_name", tool.name))
            return await source.acall(remote_name, payload)

        handle = tool.metadata.get("callable")
        if callable(handle):
            result = handle(**payload)
            if hasattr(result, "__await__"):
                return await result
            return result

        raise AdapterError(
            f"tool {tool.id!r} comes from source {tool.source_id!r}, which cannot be invoked "
            "through the proxy"
        )

    # -- internals --------------------------------------------------------

    def _permitted(self, tool: Tool) -> tuple[bool, str]:
        """Return whether policy allows ``tool``, and why not if it does not.

        Search-time filtering is not authorization: a client can name any tool
        id it likes, including one it learned elsewhere, so the policy has to
        hold on every direct lookup too.
        """
        result = self._broker.policy.evaluate(
            [Hit(tool=tool, score=1.0)], agent=self._agent, scopes=self._scopes
        )
        if result.hits:
            return True, ""
        reasons = "; ".join(exclusion.reason for exclusion in result.exclusions)
        return False, reasons or "not permitted"

    def _require_visible(self, tool_id: str) -> Tool:
        """Look up a tool the caller is allowed to see, or refuse.

        A denied tool and a nonexistent one produce the *same* client-facing
        error, and suggestions are drawn only from tools the caller may already
        see. Otherwise the error message is an enumeration oracle: guess a
        substring, and the "did you mean" hint confirms the existence and exact
        id of tools policy was meant to hide.

        The real reason is logged server-side, so an operator debugging a
        legitimate agent still gets a straight answer.
        """
        tool = self._broker.get(tool_id)
        if tool is not None:
            permitted, reason = self._permitted(tool)
            if permitted:
                return tool
            logger.warning(
                "policy denied a direct tool lookup",
                extra={"tool_id": tool.id, "agent": self._agent, "reason": reason},
            )
        else:
            logger.debug("lookup for an unknown tool", extra={"tool_id": tool_id})

        visible = [
            candidate.id
            for candidate in self._broker.tools()
            if (candidate.name == tool_id or tool_id in candidate.id)
            and self._permitted(candidate)[0]
        ][:5]
        hint = f" Did you mean: {', '.join(visible)}?" if visible else ""
        raise ToolBrokerError(f"unknown tool {tool_id!r}.{hint}")

    def build_server(self) -> Any:
        """Construct the MCP server object exposing this proxy."""
        try:
            from mcp.server.mcpserver import MCPServer
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ToolBrokerError(
                "MCP proxy mode requires mcp>=2. Install with: pip install 'toolbroker[mcp]'"
            ) from exc

        server = MCPServer(
            name=self._name,
            instructions=(
                "This server fronts a large tool catalogue. Call search_tools with a "
                "description of what you are trying to do, then describe_tool for the "
                "schema of a result, then call_tool to run it. Do not guess tool ids."
            ),
        )

        server.tool(name="search_tools")(self.search_tools)
        server.tool(name="describe_tool")(self.describe_tool)
        if self._allow_calls:
            server.tool(name="call_tool")(self.call_tool)
        return server

    def run(
        self,
        *,
        transport: ServeTransport = "stdio",
        host: str = "127.0.0.1",
        port: int = 8000,
    ) -> None:
        """Build and run the MCP server. Blocks until interrupted."""
        server = self.build_server()
        logger.info(
            "starting MCP proxy",
            extra={
                "tools": len(self._broker),
                "transport": transport,
                "refreshing": self._refresher is not None,
            },
        )
        if self._refresher is not None:
            self._refresher.start()
        try:
            if transport == "http":
                server.settings.host = host
                server.settings.port = port
                server.run(transport="streamable-http")
            else:
                server.run(transport="stdio")
        finally:
            if self._refresher is not None:
                self._refresher.stop()


def serve(
    broker: ToolBroker,
    *,
    transport: ServeTransport = "stdio",
    host: str = "127.0.0.1",
    port: int = 8000,
    default_k: int = 5,
    refresh_interval: float | None = None,
    **kwargs: Any,
) -> None:
    """Serve ``broker`` over MCP. Convenience wrapper around :class:`ToolBrokerProxy`."""
    ToolBrokerProxy(broker, default_k=default_k, refresh_interval=refresh_interval, **kwargs).run(
        transport=transport, host=host, port=port
    )


def selection_to_payload(selection: Selection) -> str:
    """Render a selection as JSON, for clients that want the raw trace."""
    return json.dumps(selection.model_dump(mode="json"), default=str, indent=2)
