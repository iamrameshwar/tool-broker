from __future__ import annotations

import pytest

from toolbroker import (
    AgentPolicy,
    AllowTools,
    DenyTags,
    DenyTools,
    Hit,
    MaxCost,
    MaxRisk,
    MaxTools,
    MinScore,
    PolicyEngine,
    PredicateRule,
    RequireScopes,
    RequireTags,
    RiskTier,
    Tool,
)
from toolbroker.types import CostTier, Decision


def hit(name: str, score: float = 1.0, **kwargs) -> Hit:
    return Hit(tool=Tool(name=name, namespace="test", **kwargs), score=score)


def test_default_policy_allows_everything():
    hits = [hit("a"), hit("b")]
    assert len(PolicyEngine().evaluate(hits).hits) == 2


def test_deny_pattern_blocks_matching_tools():
    result = PolicyEngine([DenyTools(["test/delete_*"])]).evaluate(
        [hit("delete_user"), hit("read_user")]
    )
    assert [h.id for h in result.hits] == ["test/read_user"]


def test_deny_records_a_reason():
    result = PolicyEngine([DenyTools(["test/delete_*"])]).evaluate([hit("delete_user")])
    assert result.exclusions[0].tool_id == "test/delete_user"
    assert "deny pattern" in result.exclusions[0].reason


def test_namespace_wide_deny():
    result = PolicyEngine([DenyTools(["admin/*"])]).evaluate(
        [Hit(tool=Tool(name="x", namespace="admin"), score=1.0), hit("y")]
    )
    assert [h.id for h in result.hits] == ["test/y"]


def test_allowlist_denies_unlisted_tools():
    result = PolicyEngine([AllowTools(["test/read_*"])]).evaluate(
        [hit("read_user"), hit("write_user")]
    )
    assert [h.id for h in result.hits] == ["test/read_user"]


def test_deny_overrides_allow_by_default():
    engine = PolicyEngine([AllowTools(["test/*"]), DenyTools(["test/secret"])])
    result = engine.evaluate([hit("secret")])
    assert result.hits == ()


def test_first_match_stops_at_the_first_opinion():
    engine = PolicyEngine(
        [AllowTools(["test/secret"]), DenyTools(["test/secret"])], combining="first_match"
    )
    assert len(engine.evaluate([hit("secret")]).hits) == 1


def test_max_risk_blocks_higher_tiers():
    engine = PolicyEngine([MaxRisk(RiskTier.MEDIUM)])
    result = engine.evaluate([hit("safe", risk=RiskTier.LOW), hit("scary", risk=RiskTier.HIGH)])
    assert [h.id for h in result.hits] == ["test/safe"]


def test_max_cost_blocks_expensive_tools():
    engine = PolicyEngine([MaxCost(CostTier.LOW)])
    result = engine.evaluate([hit("cheap", cost=CostTier.FREE), hit("pricey", cost=CostTier.HIGH)])
    assert [h.id for h in result.hits] == ["test/cheap"]


def test_scopes_gate_tools_that_require_them():
    engine = PolicyEngine([RequireScopes()])
    candidates = [hit("refund", required_scopes=frozenset({"payments:write"})), hit("read")]

    without = engine.evaluate(candidates)
    assert [h.id for h in without.hits] == ["test/read"]

    with_scope = engine.evaluate(candidates, scopes=frozenset({"payments:write"}))
    assert len(with_scope.hits) == 2


def test_missing_scope_is_named_in_the_reason():
    engine = PolicyEngine([RequireScopes()])
    result = engine.evaluate([hit("refund", required_scopes=frozenset({"payments:write"}))])
    assert "payments:write" in result.exclusions[0].reason


def test_deny_tags_and_require_tags():
    denied = PolicyEngine([DenyTags(frozenset({"destructive"}))]).evaluate(
        [hit("wipe", tags=frozenset({"destructive"})), hit("read", tags=frozenset({"safe"}))]
    )
    assert [h.id for h in denied.hits] == ["test/read"]

    required = PolicyEngine([RequireTags(frozenset({"approved"}))]).evaluate(
        [hit("ok", tags=frozenset({"approved"})), hit("unvetted")]
    )
    assert [h.id for h in required.hits] == ["test/ok"]


def test_max_tools_caps_the_selection():
    engine = PolicyEngine(selection_rules=[MaxTools(limit=2)])
    result = engine.evaluate([hit(f"t{i}", score=1.0 - i / 10) for i in range(5)])
    assert len(result.hits) == 2
    assert len(result.exclusions) == 3
    assert "max_tools limit of 2" in result.exclusions[0].reason


def test_max_tools_keeps_pinned_tools():
    engine = PolicyEngine(selection_rules=[MaxTools(limit=2, keep_pinned=frozenset({"test/t4"}))])
    result = engine.evaluate([hit(f"t{i}", score=1.0 - i / 10) for i in range(5)])
    assert "test/t4" in [h.id for h in result.hits]
    assert len(result.hits) == 2


def test_max_tools_preserves_score_order():
    engine = PolicyEngine(selection_rules=[MaxTools(limit=2, keep_pinned=frozenset({"test/t4"}))])
    result = engine.evaluate([hit(f"t{i}", score=1.0 - i / 10) for i in range(5)])
    scores = [h.score for h in result.hits]
    assert scores == sorted(scores, reverse=True)


def test_min_score_drops_weak_matches():
    engine = PolicyEngine(selection_rules=[MinScore(0.5)])
    result = engine.evaluate([hit("strong", score=0.9), hit("weak", score=0.1)])
    assert [h.id for h in result.hits] == ["test/strong"]


def test_predicate_rule_can_deny():
    engine = PolicyEngine(
        [PredicateRule(predicate=lambda tool, ctx: tool.name != "blocked", reason="blocked")]
    )
    result = engine.evaluate([hit("blocked"), hit("fine")])
    assert [h.id for h in result.hits] == ["test/fine"]


def test_predicate_rule_sees_the_agent():
    seen: list[str | None] = []

    def predicate(tool, context):
        seen.append(context.agent)
        return True

    PolicyEngine([PredicateRule(predicate=predicate)]).evaluate([hit("a")], agent="support")
    assert seen == ["support"]


def test_dry_run_records_but_does_not_enforce():
    engine = PolicyEngine([DenyTools(["test/*"])], [MaxTools(limit=1)], dry_run=True)
    result = engine.evaluate([hit("a"), hit("b")])
    assert len(result.hits) == 2
    assert len(result.exclusions) >= 2
    assert engine.dry_run


def test_default_deny_blocks_tools_no_rule_mentions():
    engine = PolicyEngine([], default=Decision.DENY)
    assert engine.evaluate([hit("a")]).hits == ()


def test_firings_carry_rule_names_and_tool_ids():
    result = PolicyEngine([DenyTools(["test/x"], name="no_x")]).evaluate([hit("x")])
    assert result.firings[0].rule == "no_x"
    assert result.firings[0].tool_id == "test/x"
    assert result.firings[0].decision is Decision.DENY


def test_with_rules_returns_a_new_engine():
    base = PolicyEngine([DenyTools(["test/a"])])
    extended = base.with_rules([DenyTools(["test/b"])])
    assert len(base.rules) == 1
    assert len(extended.rules) == 2


def test_agent_policy_routes_by_agent():
    policy = AgentPolicy(
        {
            "support": PolicyEngine([MaxRisk(RiskTier.LOW)]),
            "ops": PolicyEngine([MaxRisk(RiskTier.CRITICAL)]),
        }
    )
    candidates = [hit("restart", risk=RiskTier.HIGH)]
    assert policy.evaluate(candidates, agent="support").hits == ()
    assert len(policy.evaluate(candidates, agent="ops").hits) == 1


def test_unknown_agent_gets_the_default_policy():
    policy = AgentPolicy({"support": PolicyEngine([MaxRisk(RiskTier.LOW)])})
    assert len(policy.evaluate([hit("x", risk=RiskTier.CRITICAL)], agent="unknown").hits) == 1


@pytest.mark.parametrize("pattern", ["*/delete_*", "test/*", "test/delete_user"])
def test_glob_patterns_match_full_ids(pattern):
    result = PolicyEngine([DenyTools([pattern])]).evaluate([hit("delete_user")])
    assert result.hits == ()
