from __future__ import annotations

import pytest
from pydantic import ValidationError

from toolbroker import Filters, Hit, RiskTier, Selection, Stage, Tool
from toolbroker.types import CostTier, Decision, Exclusion, RuleFiring


def test_tool_id_combines_namespace_and_name():
    assert Tool(name="search", namespace="github").id == "github/search"


def test_tool_rejects_separator_in_name():
    with pytest.raises(ValidationError):
        Tool(name="bad/name")


def test_tool_is_immutable():
    tool = Tool(name="search")
    with pytest.raises(ValidationError):
        tool.name = "other"


def test_risk_tiers_are_ordered():
    assert RiskTier.LOW < RiskTier.MEDIUM < RiskTier.HIGH < RiskTier.CRITICAL
    assert RiskTier.CRITICAL > RiskTier.LOW
    assert RiskTier.LOW <= RiskTier.LOW


def test_filters_empty_matches_everything():
    assert Filters().is_empty()
    assert Filters().matches(Tool(name="anything"))


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        (Filters(namespaces=frozenset({"other"})), False),
        (Filters(namespaces=frozenset({"test"})), True),
        (Filters(tags_any=frozenset({"read"})), True),
        (Filters(tags_any=frozenset({"nope"})), False),
        (Filters(tags_all=frozenset({"read", "commerce"})), True),
        (Filters(tags_all=frozenset({"read", "missing"})), False),
        (Filters(exclude_tags=frozenset({"read"})), False),
        (Filters(max_risk=RiskTier.LOW), True),
        (Filters(exclude_tool_ids=frozenset({"test/search"})), False),
    ],
)
def test_filter_matching(filters, expected):
    tool = Tool(
        name="search",
        namespace="test",
        tags=frozenset({"read", "commerce"}),
        risk=RiskTier.LOW,
    )
    assert filters.matches(tool) is expected


def test_filter_max_risk_excludes_higher_tiers():
    tool = Tool(name="drop", risk=RiskTier.CRITICAL)
    assert not Filters(max_risk=RiskTier.HIGH).matches(tool)


def test_selection_exposes_tools_and_ids():
    hits = (Hit(tool=Tool(name="a"), score=0.9), Hit(tool=Tool(name="b"), score=0.4))
    selection = Selection(query="q", hits=hits, requested_k=2, considered=10)
    assert len(selection) == 2
    assert selection.tool_ids == ("default/a", "default/b")
    assert [tool.name for tool in selection] == ["a", "b"]


def test_explain_reports_scores_rules_and_exclusions():
    selection = Selection(
        query="refund the customer",
        hits=(Hit(tool=Tool(name="refund"), score=0.82, components={"vector": 0.82}),),
        agent="support",
        requested_k=1,
        considered=500,
        firings=(
            RuleFiring(
                rule="max_risk",
                decision=Decision.DENY,
                reason="risk critical exceeds ceiling high",
                tool_id="default/wipe",
            ),
        ),
        exclusions=(Exclusion(tool_id="default/wipe", stage=Stage.POLICY, reason="too risky"),),
        timings_ms={"retrieval": 1.5},
    )
    text = selection.explain()
    assert "refund the customer" in text
    assert "agent: support" in text
    assert "considered 500 tools" in text
    assert "score=0.8200" in text
    assert "default/wipe (policy): too risky" in text
    assert "max_risk -> deny" in text
    assert "retrieval=1.50ms" in text


def test_explain_flags_dry_run():
    selection = Selection(query="q", hits=(), dry_run=True)
    assert "DRY RUN" in selection.explain()


def test_cost_tier_values():
    assert CostTier.FREE.value == "free"


def test_with_metadata_returns_a_copy():
    tool = Tool(name="a", metadata={"x": 1})
    updated = tool.with_metadata(y=2)
    assert updated.metadata == {"x": 1, "y": 2}
    assert tool.metadata == {"x": 1}
