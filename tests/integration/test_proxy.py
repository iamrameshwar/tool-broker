"""The MCP proxy: search, describe, and forwarded calls."""

from __future__ import annotations

import time

import pytest

from toolbroker import DenyTools, PolicyEngine, RiskTier, Tool, ToolBroker
from toolbroker.errors import ToolBrokerError
from toolbroker.index.embedders.hashing import HashingEmbedder
from toolbroker.proxy import ToolBrokerProxy


def search_orders(customer_email: str) -> str:
    """Find recent orders placed by a customer."""
    return f"orders for {customer_email}"


def delete_customer(customer_id: str) -> str:
    """Permanently erase a customer record."""
    return f"deleted {customer_id}"


@pytest.fixture
def proxy():
    broker = ToolBroker(embedder=HashingEmbedder(dim=128), cache_embeddings=False)
    broker.add_functions([search_orders, delete_customer])
    broker.index()
    return ToolBrokerProxy(broker, default_k=2)


async def test_search_returns_ranked_tools(proxy):
    result = proxy.search_tools("what did this customer order")
    assert result["tools"][0]["tool_id"] == "python/search_orders"
    assert result["considered"] == 2


async def test_search_respects_the_limit(proxy):
    assert len(proxy.search_tools("customer", limit=1)["tools"]) == 1


async def test_describe_returns_the_full_schema(proxy):
    described = proxy.describe_tool("python/search_orders")
    assert described["input_schema"]["properties"]["customer_email"]["type"] == "string"
    assert described["risk"] == RiskTier.LOW.value


async def test_unknown_tool_suggests_alternatives(proxy):
    with pytest.raises(ToolBrokerError, match="Did you mean"):
        proxy.describe_tool("search_orders")


async def test_call_forwards_to_a_local_callable(proxy):
    assert await proxy.call_tool("python/search_orders", {"customer_email": "a@b.com"}) == (
        "orders for a@b.com"
    )


async def test_call_is_re_authorized_against_policy(proxy):
    # Search-time filtering is not authorization: a client can name any tool id
    # it likes, so the policy has to hold at call time too.
    proxy.broker.set_policy(PolicyEngine([DenyTools(["python/delete_*"])]))
    with pytest.raises(ToolBrokerError):
        await proxy.call_tool("python/delete_customer", {"customer_id": "1"})


async def test_denied_and_missing_tools_are_indistinguishable(proxy):
    # Distinct errors would be an enumeration oracle: a client could confirm a
    # tool exists by the shape of the refusal.
    proxy.broker.set_policy(PolicyEngine([DenyTools(["python/delete_*"])]))
    denied = missing = ""
    try:
        await proxy.call_tool("python/delete_customer", {"customer_id": "1"})
    except ToolBrokerError as exc:
        denied = str(exc).replace("python/delete_customer", "X")
    try:
        await proxy.call_tool("python/does_not_exist", {})
    except ToolBrokerError as exc:
        missing = str(exc).replace("python/does_not_exist", "X")
    assert denied == missing


async def test_describe_is_gated_by_policy(proxy):
    # A description states what a tool does and its schema names every
    # parameter; handing those to a client that cannot call it is disclosure.
    proxy.broker.set_policy(PolicyEngine([DenyTools(["python/delete_*"])]))
    with pytest.raises(ToolBrokerError):
        proxy.describe_tool("python/delete_customer")


async def test_suggestions_never_name_a_denied_tool(proxy):
    proxy.broker.set_policy(PolicyEngine([DenyTools(["python/delete_*"])]))
    try:
        proxy.describe_tool("delete_customer")
    except ToolBrokerError as exc:
        assert "python/delete_customer" not in str(exc)


async def test_suggestions_still_help_for_visible_tools(proxy):
    try:
        proxy.describe_tool("search_orders")
    except ToolBrokerError as exc:
        assert "python/search_orders" in str(exc)


async def test_discovery_only_mode_refuses_calls():
    broker = ToolBroker(embedder=HashingEmbedder(dim=128), cache_embeddings=False)
    broker.add_functions([search_orders])
    broker.index()
    proxy = ToolBrokerProxy(broker, allow_calls=False)
    with pytest.raises(ToolBrokerError, match="discovery only"):
        await proxy.call_tool("python/search_orders", {"customer_email": "a"})


async def test_build_server_exposes_the_expected_tools(proxy):
    pytest.importorskip("mcp.server.mcpserver")
    server = proxy.build_server()
    names = {tool.name for tool in await server.list_tools()}
    assert names == {"search_tools", "describe_tool", "call_tool"}


async def test_discovery_only_server_omits_call_tool():
    pytest.importorskip("mcp.server.mcpserver")
    broker = ToolBroker(embedder=HashingEmbedder(dim=128), cache_embeddings=False)
    broker.add_functions([search_orders])
    broker.index()
    server = ToolBrokerProxy(broker, allow_calls=False).build_server()
    assert "call_tool" not in {tool.name for tool in await server.list_tools()}


async def test_proxy_has_no_refresher_by_default(proxy):
    assert proxy.refresher is None


async def test_proxy_can_refresh_in_the_background():
    # A proxy is long-lived and the servers behind it are not; without this it
    # serves whatever tools existed at startup, forever.
    from toolbroker.sources.base import BaseSource

    class Mutable(BaseSource):
        def __init__(self):
            super().__init__("mut")
            self.tools = [Tool(name="alpha", namespace="mut", description="Does alpha.")]

        def _discover(self):
            return list(self.tools)

    source = Mutable()
    broker = ToolBroker(embedder=HashingEmbedder(dim=128), cache_embeddings=False)
    broker.add_source(source)
    broker.refresh()

    proxy = ToolBrokerProxy(broker, refresh_interval=0.02)
    assert proxy.refresher is not None
    proxy.refresher.start()
    try:
        source.tools.append(Tool(name="beta", namespace="mut", description="Does beta."))
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and broker.get("mut/beta") is None:
            time.sleep(0.01)
        assert broker.get("mut/beta") is not None
        assert "mut/beta" in [t["tool_id"] for t in proxy.search_tools("does beta")["tools"]]
    finally:
        proxy.refresher.stop()
