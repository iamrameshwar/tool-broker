from __future__ import annotations

import pytest

from toolbroker import (
    DROP,
    DenyTools,
    Event,
    Filters,
    MaxRisk,
    MaxTools,
    PolicyEngine,
    RiskTier,
    Tool,
    ToolBroker,
)
from toolbroker.errors import ConfigurationError, NotIndexedError
from toolbroker.index.embedders.hashing import HashingEmbedder
from toolbroker.store.memory import InMemoryStore


def test_zero_config_construction_works():
    broker = ToolBroker(embedder=HashingEmbedder(dim=64))
    assert len(broker) == 0


def test_select_before_indexing_raises():
    broker = ToolBroker(embedder=HashingEmbedder(dim=64))
    with pytest.raises(NotIndexedError):
        broker.select("anything")


def test_mismatched_store_and_embedder_are_rejected():
    with pytest.raises(ConfigurationError, match="dimension"):
        ToolBroker(embedder=HashingEmbedder(dim=64), store=InMemoryStore(128))


def test_index_then_select(broker):
    selection = broker.select("how many units are in stock", k=2)
    assert len(selection) == 2
    assert "test/check_inventory" in selection.tool_ids


def test_selection_reports_what_was_considered(broker):
    assert broker.select("stock", k=1).considered == 5


def test_selection_records_timings(broker):
    timings = broker.select("stock", k=1).timings_ms
    assert "retrieval" in timings
    assert "policy" in timings


def test_reindex_replaces_by_default(embedder, sample_tools):
    broker = ToolBroker(embedder=embedder, cache_embeddings=False)
    broker.index(sample_tools)
    broker.index(sample_tools[:2])
    assert len(broker) == 2


def test_add_tools_appends(embedder, sample_tools):
    broker = ToolBroker(embedder=embedder, cache_embeddings=False)
    broker.index(sample_tools[:2])
    broker.add_tools([Tool(name="extra", namespace="test")])
    assert len(broker) == 3


def test_remove_tools(broker):
    assert broker.remove_tools(["test/send_email"]) == 1
    assert broker.get("test/send_email") is None


def test_duplicate_ids_are_reported(embedder):
    broker = ToolBroker(embedder=embedder, cache_embeddings=False)
    report = broker.index([Tool(name="a"), Tool(name="a")])
    assert report.indexed == 1
    assert report.duplicates == ("default/a",)


def test_policy_removes_tools_and_says_why(broker):
    broker.set_policy(PolicyEngine([DenyTools(["test/delete_*"])]))
    selection = broker.select("erase this customer permanently", k=5)
    assert "test/delete_customer" not in selection.tool_ids
    assert any(
        exclusion.tool_id == "test/delete_customer" and "deny pattern" in exclusion.reason
        for exclusion in selection.exclusions
    )


def test_policy_denials_do_not_shrink_the_result(broker):
    # A caller who asks for 3 tools should get 3 even when one candidate is
    # denied, because retrieval overfetches to leave policy room to work.
    broker.set_policy(PolicyEngine([MaxRisk(RiskTier.LOW)]))
    selection = broker.select("customer", k=2)
    assert len(selection) == 2
    assert all(tool.risk is RiskTier.LOW for tool in selection.tools)


def test_max_tools_caps_below_requested_k(broker):
    broker.set_policy(PolicyEngine(selection_rules=[MaxTools(limit=2)]))
    assert len(broker.select("customer", k=5)) == 2


def test_scopes_unlock_gated_tools(broker):
    from toolbroker import RequireScopes

    broker.set_policy(PolicyEngine([RequireScopes()]))
    without = broker.select("refund the customer their money", k=5)
    assert "test/issue_refund" not in without.tool_ids

    granted = broker.select("refund the customer their money", k=5, scopes={"payments:write"})
    assert "test/issue_refund" in granted.tool_ids


def test_namespace_filter_shorthand(broker):
    assert broker.select("customer", k=5, namespaces=["nothing"]).tool_ids == ()


def test_tag_filter_shorthand(broker):
    selection = broker.select("customer", k=5, tags=["warehouse"])
    assert selection.tool_ids == ("test/check_inventory",)


def test_explicit_filters_are_honoured(broker):
    selection = broker.select("customer", k=5, filters=Filters(max_risk=RiskTier.LOW))
    assert all(tool.risk is RiskTier.LOW for tool in selection.tools)


def test_openai_rendering(broker):
    tools = broker.openai_tools("how many units are in stock", k=1)
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "test__check_inventory"


def test_anthropic_rendering(broker):
    tools = broker.anthropic_tools("how many units are in stock", k=1)
    assert "input_schema" in tools[0]


def test_select_for_returns_both_halves(broker):
    rendered, selection = broker.select_for("openai", "how many units are in stock", k=1)
    assert len(rendered) == 1
    assert len(selection) == 1


def test_add_functions_registers_a_source():
    def greet(name: str) -> str:
        """Say hello to someone."""
        return name

    broker = ToolBroker(embedder=HashingEmbedder(dim=64), cache_embeddings=False)
    broker.add_functions([greet])
    broker.index()
    assert broker.get("python/greet") is not None


def test_hooks_can_rewrite_the_query(embedder, sample_tools):
    broker = ToolBroker(embedder=embedder, cache_embeddings=False)
    broker.hooks.register(Event.TRANSFORM_QUERY, lambda query: "inventory stock units")
    broker.index(sample_tools)
    assert broker.select("something else entirely", k=1).tool_ids == ("test/check_inventory",)


def test_hooks_can_drop_tools_at_index_time(embedder, sample_tools):
    broker = ToolBroker(embedder=embedder, cache_embeddings=False)
    broker.hooks.register(
        Event.TRANSFORM_TOOL,
        lambda tool: DROP if tool.risk is RiskTier.CRITICAL else tool,
    )
    broker.index(sample_tools)
    assert broker.get("test/delete_customer") is None


def test_hook_failures_do_not_break_selection(broker):
    def broken(value, **kwargs):
        raise RuntimeError("hook is buggy")

    broker.hooks.register(Event.TRANSFORM_QUERY, broken)
    assert len(broker.select("customer", k=1)) == 1


def test_stats_summarise_the_catalogue(broker):
    stats = broker.stats()
    assert stats["tools"] == 5
    assert stats["namespaces"] == {"test": 5}
    assert stats["risk_tiers"]["critical"] == 1


def test_truncated_hits_are_recorded_as_exclusions(broker):
    selection = broker.select("customer", k=1)
    assert any(exclusion.stage.value == "retrieval" for exclusion in selection.exclusions)
