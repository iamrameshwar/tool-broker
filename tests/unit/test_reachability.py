"""Nobody can review a glob. They can review what it grants."""

from __future__ import annotations

from toolbroker import (
    AgentPolicy,
    AllowTools,
    DenyTools,
    MaxRisk,
    MaxTools,
    MinScore,
    PolicyEngine,
    RiskTier,
    Tool,
)
from toolbroker.reachability import diff_policies, reach_report, reachable_tools

TOOLS = [
    Tool(name="list_orders", namespace="billing", description="List orders.", risk=RiskTier.LOW),
    Tool(name="issue_refund", namespace="billing", description="Refund.", risk=RiskTier.MEDIUM),
    Tool(name="delete_user", namespace="identity", description="Delete.", risk=RiskTier.HIGH),
    Tool(name="delete_order", namespace="billing", description="Delete.", risk=RiskTier.HIGH),
]


def _policy(*rules, selection=(), agent="support") -> AgentPolicy:
    return AgentPolicy({agent: PolicyEngine(rules, selection)})


# --- reachability ----------------------------------------------------------


def test_no_rules_means_everything_is_reachable():
    reach = reachable_tools(PolicyEngine(), TOOLS)
    assert reach.total == len(TOOLS)
    assert reach.denied == ()


def test_a_deny_glob_removes_matching_tools():
    reach = reachable_tools(PolicyEngine([DenyTools(["*/delete_*"])]), TOOLS)
    assert reach.reachable == ("billing/issue_refund", "billing/list_orders")
    assert reach.denied == ("billing/delete_order", "identity/delete_user")


def test_an_allow_list_restricts_to_it():
    reach = reachable_tools(PolicyEngine([AllowTools(["billing/*"])]), TOOLS)
    assert "identity/delete_user" in reach.denied


def test_a_risk_cap_removes_high_risk_tools():
    reach = reachable_tools(PolicyEngine([MaxRisk(RiskTier.LOW)]), TOOLS)
    assert reach.reachable == ("billing/list_orders",)


def test_reach_is_broken_down_by_risk():
    reach = reachable_tools(PolicyEngine(), TOOLS)
    assert reach.by_risk == {"low": 1, "medium": 1, "high": 2}


def test_max_tools_is_reported_but_does_not_reduce_reach():
    """A cap bounds one response; it does not make a tool unreachable."""
    reach = reachable_tools(PolicyEngine([], [MaxTools(1)]), TOOLS)
    assert reach.total == len(TOOLS)
    assert reach.max_tools == 1


def test_a_score_floor_is_reported_as_undecidable():
    """Silently ignoring it would overstate what the agent can do."""
    reach = reachable_tools(PolicyEngine([], [MinScore(0.5)]), TOOLS)
    assert reach.undecidable == ("min_score",)


def test_reach_report_covers_every_agent_and_the_default():
    policy = AgentPolicy(
        {
            "support": PolicyEngine([DenyTools(["*/delete_*"])]),
            "ops": PolicyEngine(),
        }
    )
    report = reach_report(policy, TOOLS)
    assert set(report) == {"support", "ops", "(default)"}
    assert report["support"].total == 2
    assert report["ops"].total == 4


# --- the diff --------------------------------------------------------------


def test_an_unchanged_policy_reports_no_change():
    before = _policy(DenyTools(["*/delete_*"]))
    after = _policy(DenyTools(["*/delete_*"]))
    diff = diff_policies(before, after, TOOLS)
    assert not diff.changed
    assert "No change" in diff.summary()


def test_a_loosened_glob_shows_exactly_what_it_grants():
    """The review question: `*/delete_*` → `*/delete_user` grants what?"""
    before = _policy(DenyTools(["*/delete_*"]))
    after = _policy(DenyTools(["*/delete_user"]))
    diff = diff_policies(before, after, TOOLS)

    delta = next(d for d in diff.deltas if d.agent == "support")
    assert delta.gained == ("billing/delete_order",)
    assert delta.lost == ()


def test_a_tightened_glob_shows_what_it_removes():
    before = _policy(DenyTools(["*/delete_user"]))
    after = _policy(DenyTools(["*/delete_*"]))
    delta = next(d for d in diff_policies(before, after, TOOLS).deltas if d.agent == "support")
    assert delta.lost == ("billing/delete_order",)
    assert delta.gained == ()


def test_granting_a_high_risk_tool_is_flagged():
    """The gate worth having in CI."""
    before = _policy(MaxRisk(RiskTier.LOW))
    after = _policy(MaxRisk(RiskTier.HIGH))
    diff = diff_policies(before, after, TOOLS)
    assert diff.grants_high_risk
    assert "HIGH risk" in diff.summary()


def test_granting_only_low_risk_tools_is_not_flagged():
    before = _policy(MaxRisk(RiskTier.LOW))
    after = _policy(MaxRisk(RiskTier.MEDIUM))
    diff = diff_policies(before, after, TOOLS)
    assert diff.changed
    assert not diff.grants_high_risk


def test_a_new_agent_shows_everything_it_gains():
    before = AgentPolicy({})
    after = AgentPolicy({"newbot": PolicyEngine([AllowTools(["billing/*"])])})
    delta = next(d for d in diff_policies(before, after, TOOLS).deltas if d.agent == "newbot")
    assert len(delta.gained) == 3
    assert delta.lost == ()


def test_a_removed_agent_shows_everything_it_loses():
    before = AgentPolicy({"oldbot": PolicyEngine()})
    after = AgentPolicy({})
    delta = next(d for d in diff_policies(before, after, TOOLS).deltas if d.agent == "oldbot")
    assert len(delta.lost) == len(TOOLS)


def test_a_changed_cap_is_reported():
    before = _policy(selection=[MaxTools(4)])
    after = _policy(selection=[MaxTools(2)])
    diff = diff_policies(before, after, TOOLS)
    delta = next(d for d in diff.deltas if d.agent == "support")
    assert (delta.max_tools_before, delta.max_tools_after) == (4, 2)
    assert delta.changed
    assert "max_tools: 4 → 2" in diff.summary()


def test_undecidable_rules_are_surfaced_on_the_diff():
    before = _policy()
    after = _policy(selection=[MinScore(0.5)])
    diff = diff_policies(before, after, TOOLS)
    assert diff.undecidable == ("min_score",)
    assert "needs a live query" in diff.summary()


def test_the_summary_names_the_agent_and_the_tools():
    before = _policy(DenyTools(["*/delete_*"]))
    after = _policy(DenyTools(["*/delete_user"]))
    summary = diff_policies(before, after, TOOLS).summary()
    assert "support" in summary
    assert "+ billing/delete_order" in summary


def test_the_diff_records_the_catalogue_it_judged_against():
    """A reach number is meaningless without knowing what it was measured over."""
    diff = diff_policies(_policy(), _policy(), TOOLS)
    assert diff.catalogue_size == len(TOOLS)


def test_an_empty_catalogue_is_not_an_error():
    diff = diff_policies(_policy(), _policy(DenyTools(["*"])), [])
    assert not diff.changed
