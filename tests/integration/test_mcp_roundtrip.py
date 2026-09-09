"""Real MCP round-trips against an in-process server.

The SDK's high-level Client accepts a server object directly, so these are
genuine protocol exchanges — serialisation, schema shapes, annotations — without
spawning a subprocess or opening a socket.
"""

from __future__ import annotations

import pytest

from toolbroker import RiskTier, ToolBroker
from toolbroker.errors import SourceError
from toolbroker.index.embedders.hashing import HashingEmbedder
from toolbroker.sources.mcp import MCPServerSpec, MCPSource, default_risk_classifier

mcp_server = pytest.importorskip("mcp.server.mcpserver")
# asyncio_mode = auto handles the async tests; the mark is only for selection.
pytestmark = pytest.mark.integration


@pytest.fixture
def server():
    server = mcp_server.MCPServer(name="fixture")

    @server.tool()
    def search_orders(customer_email: str, limit: int = 10) -> list[str]:
        """Find recent orders placed by a customer."""
        return [f"order for {customer_email}"][:limit]

    @server.tool()
    def delete_customer(customer_id: str) -> str:
        """Permanently erase a customer record."""
        return f"deleted {customer_id}"

    return server


@pytest.fixture
def source(server):
    return MCPSource(MCPServerSpec(name="fixture", target=server))


async def test_discovery_returns_the_servers_tools(source):
    tools = await source.adiscover()
    assert {tool.name for tool in tools} == {"search_orders", "delete_customer"}


async def test_discovered_tools_are_namespaced_by_server(source):
    tools = await source.adiscover()
    assert {tool.id for tool in tools} == {
        "fixture/search_orders",
        "fixture/delete_customer",
    }


async def test_input_schema_survives_the_round_trip(source):
    tools = {tool.name: tool for tool in await source.adiscover()}
    schema = tools["search_orders"].input_schema
    assert schema["properties"]["customer_email"]["type"] == "string"
    assert "customer_email" in schema["required"]


async def test_descriptions_survive_the_round_trip(source):
    tools = {tool.name: tool for tool in await source.adiscover()}
    assert "Find recent orders" in tools["search_orders"].description


async def test_risk_is_inferred_for_destructive_names(source):
    tools = {tool.name: tool for tool in await source.adiscover()}
    assert tools["delete_customer"].risk is RiskTier.HIGH
    assert tools["search_orders"].risk is RiskTier.LOW


async def test_source_id_identifies_the_server(source):
    tools = await source.adiscover()
    assert all(tool.source_id == "mcp:fixture" for tool in tools)


async def test_call_forwards_to_the_server(source):
    result = await source.acall("search_orders", {"customer_email": "a@b.com", "limit": 1})
    assert "a@b.com" in str(result)


async def test_unreachable_server_raises_source_error():
    spec = MCPServerSpec(name="dead", command="/nonexistent/binary")
    with pytest.raises(SourceError, match="dead"):
        await MCPSource(spec).adiscover()


async def test_catalogue_indexes_mcp_tools(server):
    broker = ToolBroker(embedder=HashingEmbedder(dim=128), cache_embeddings=False)
    source = MCPSource(MCPServerSpec(name="fixture", target=server))
    broker.index(await source.adiscover())

    selection = broker.select("find what a customer ordered recently", k=1)
    assert selection.tool_ids == ("fixture/search_orders",)


async def test_policy_applies_to_mcp_tools(server):
    from toolbroker import MaxRisk, PolicyEngine

    broker = ToolBroker(embedder=HashingEmbedder(dim=128), cache_embeddings=False)
    source = MCPSource(MCPServerSpec(name="fixture", target=server))
    broker.index(await source.adiscover())
    broker.set_policy(PolicyEngine([MaxRisk(RiskTier.LOW)]))

    selection = broker.select("erase the customer permanently", k=5)
    assert "fixture/delete_customer" not in selection.tool_ids


def test_sync_discovery_refuses_inside_a_running_loop(source):
    import asyncio

    async def inner():
        with pytest.raises(SourceError, match="active event loop"):
            source.discover()
        return True

    assert asyncio.run(inner())


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("delete_thing", RiskTier.HIGH),
        ("purge_cache", RiskTier.HIGH),
        ("create_issue", RiskTier.MEDIUM),
        ("send_email", RiskTier.MEDIUM),
        ("list_issues", RiskTier.LOW),
        ("get_user", RiskTier.LOW),
    ],
)
def test_default_risk_classifier(name, expected):
    assert default_risk_classifier(name, "") is expected
