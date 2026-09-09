"""What a policy actually grants, without running a query.

A config file exists so agent permissions can be reviewed like any other
change. But nobody can review this:

.. code-block:: diff

    - deny: ["*/delete_*"]
    + deny: ["*/delete_user"]

by reading it. The reviewer needs to know what it *grants*, against the
catalogue they actually have. That is a static question — policy is
deterministic Python over a known set of tools — so it can be answered in CI,
before the change ships, rather than discovered in production.

    before = ToolBrokerConfig.from_file("main.yaml").policy.build()
    after = ToolBrokerConfig.from_file("pr.yaml").policy.build()
    print(diff_policies(before, after, broker.tools()).summary())

What this can and cannot decide
-------------------------------

Per-tool rules — allow and deny globs, risk tiers, tags, required scopes — are
decidable from the tool alone, so reachability is exact for them.

Selection rules are not. ``MaxTools`` caps how many tools reach the model *per
request*; it does not make any particular tool unreachable, so it is reported
separately rather than folded into the count. ``MinScore`` depends on a score
that only exists once there is a query, so it cannot be evaluated here at all
and is named in ``undecidable``. Reporting that honestly matters more than
producing a confident number: a reachability report that quietly ignored a
score floor would overstate what an agent can do.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from .policy.engine import AgentPolicy, PolicyEngine
from .types import Hit, RiskTier, Tool

#: Score handed to rules during static analysis. Any value is a fiction; this
#: one is neutral, and rules that actually depend on it are reported as
#: undecidable rather than judged against it.
_NEUTRAL_SCORE = 1.0

#: Selection rules whose effect cannot be determined without a live query.
_SCORE_DEPENDENT = frozenset({"min_score"})


class AgentReach(BaseModel):
    """Which tools one agent can reach under one policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent: str
    reachable: tuple[str, ...] = ()
    denied: tuple[str, ...] = ()
    #: Reachable tools at each risk tier, for the line a reviewer reads first.
    by_risk: dict[str, int] = {}
    #: Cap on tools per request, if a ``MaxTools`` rule is configured. Not a
    #: reachability limit: it bounds one response, not the whole catalogue.
    max_tools: int | None = None
    #: Rules that could not be evaluated statically, by name.
    undecidable: tuple[str, ...] = ()

    @property
    def total(self) -> int:
        """How many tools this agent can reach."""
        return len(self.reachable)


class AgentDelta(BaseModel):
    """How one agent's reach changed between two policies."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent: str
    gained: tuple[str, ...] = ()
    lost: tuple[str, ...] = ()
    #: Risk tier of each gained tool, so an escalation is visible at a glance.
    gained_risk: dict[str, str] = {}
    max_tools_before: int | None = None
    max_tools_after: int | None = None

    @property
    def changed(self) -> bool:
        """Whether anything moved for this agent."""
        return bool(self.gained or self.lost) or self.max_tools_before != self.max_tools_after

    @property
    def grants_high_risk(self) -> bool:
        """Whether this change hands the agent a high-risk tool it lacked."""
        return any(tier == RiskTier.HIGH.value for tier in self.gained_risk.values())


class PolicyDiff(BaseModel):
    """The reviewable answer to "what does this policy change do"."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    deltas: tuple[AgentDelta, ...] = ()
    catalogue_size: int = 0
    undecidable: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        """Whether any agent's reach moved."""
        return any(delta.changed for delta in self.deltas)

    @property
    def grants_high_risk(self) -> bool:
        """Whether any agent gained a high-risk tool. The CI gate worth having."""
        return any(delta.grants_high_risk for delta in self.deltas)

    def summary(self) -> str:
        """Render as a review comment."""
        if not self.changed:
            # Still report undecidable rules here. Adding a score floor moves no
            # tool out of reach yet can empty a response, so "no change" alone
            # would be a misleading thing to put in front of a reviewer.
            lines = [f"No change in reach for any agent, against {self.catalogue_size} tools."]
            if self.undecidable:
                lines.append(
                    "Not evaluated statically (needs a live query): "
                    + ", ".join(sorted(set(self.undecidable)))
                )
            return "\n".join(lines)

        lines = [f"Policy change against {self.catalogue_size} tools:", ""]
        for delta in self.deltas:
            if not delta.changed:
                continue
            lines.append(f"agent `{delta.agent}`")
            for tool_id in delta.gained:
                tier = delta.gained_risk.get(tool_id, "")
                marker = "  ⚠" if tier == RiskTier.HIGH.value else "   "
                lines.append(f"{marker} + {tool_id}" + (f"  (risk: {tier})" if tier else ""))
            for tool_id in delta.lost:
                lines.append(f"    - {tool_id}")
            if delta.max_tools_before != delta.max_tools_after:
                lines.append(f"    max_tools: {delta.max_tools_before} → {delta.max_tools_after}")
            lines.append("")

        if self.grants_high_risk:
            lines.append("⚠ This change grants at least one HIGH risk tool.")
        if self.undecidable:
            lines.append(
                "Not evaluated statically (needs a live query): "
                + ", ".join(sorted(set(self.undecidable)))
            )
        return "\n".join(lines).rstrip()


def _cap(engine: PolicyEngine) -> int | None:
    """Return the ``MaxTools`` limit configured on ``engine``, if any."""
    for rule in engine.selection_rules:
        limit = getattr(rule, "limit", None)
        if isinstance(limit, int):
            return limit
    return None


def _undecidable(engine: PolicyEngine) -> tuple[str, ...]:
    """Names of selection rules that cannot be evaluated without a query."""
    return tuple(rule.name for rule in engine.selection_rules if rule.name in _SCORE_DEPENDENT)


def reachable_tools(
    engine: PolicyEngine,
    tools: Sequence[Tool],
    *,
    agent: str | None = None,
    scopes: frozenset[str] = frozenset(),
) -> AgentReach:
    """Return which of ``tools`` survive ``engine``'s per-tool rules.

    Selection rules are deliberately not applied: they bound one response, not
    the catalogue. ``max_tools`` and ``undecidable`` report them instead.
    """
    # Only the per-tool rules, so a MaxTools cap does not masquerade as tools
    # being unreachable, and dry-run is off so we measure the policy as written
    # rather than as currently enforced.
    static = PolicyEngine(
        engine.rules,
        (),
        default=engine.default,
        combining=engine.combining,
    )
    hits = [Hit(tool=tool, score=_NEUTRAL_SCORE) for tool in tools]
    result = static.evaluate(hits, agent=agent, scopes=scopes)

    allowed = {hit.id for hit in result.hits}
    reachable = tuple(sorted(tool.id for tool in tools if tool.id in allowed))
    by_risk: dict[str, int] = {}
    for tool in tools:
        if tool.id in allowed:
            by_risk[tool.risk.value] = by_risk.get(tool.risk.value, 0) + 1

    return AgentReach(
        agent=agent or "(default)",
        reachable=reachable,
        denied=tuple(sorted(tool.id for tool in tools if tool.id not in allowed)),
        by_risk=by_risk,
        max_tools=_cap(engine),
        undecidable=_undecidable(engine),
    )


def reach_report(
    policy: AgentPolicy,
    tools: Sequence[Tool],
    *,
    scopes: frozenset[str] = frozenset(),
) -> dict[str, AgentReach]:
    """Return the reach of every registered agent, plus the default."""
    reaches = {
        agent: reachable_tools(policy.for_agent(agent), tools, agent=agent, scopes=scopes)
        for agent in policy.agents
    }
    reaches["(default)"] = reachable_tools(policy.for_agent(None), tools, scopes=scopes)
    return reaches


def diff_policies(
    before: AgentPolicy,
    after: AgentPolicy,
    tools: Sequence[Tool],
    *,
    scopes: frozenset[str] = frozenset(),
) -> PolicyDiff:
    """Return what changes between two policies, against one catalogue."""
    left = reach_report(before, tools, scopes=scopes)
    right = reach_report(after, tools, scopes=scopes)
    risk_of = {tool.id: tool.risk.value for tool in tools}

    deltas: list[AgentDelta] = []
    undecidable: list[str] = []
    for agent in sorted(set(left) | set(right)):
        # An agent that exists on only one side reaches nothing on the other,
        # which is exactly what adding or removing an agent should look like.
        old = left.get(agent)
        new = right.get(agent)
        old_reach = set(old.reachable) if old else set()
        new_reach = set(new.reachable) if new else set()
        gained = tuple(sorted(new_reach - old_reach))
        lost = tuple(sorted(old_reach - new_reach))
        undecidable.extend(new.undecidable if new else ())

        deltas.append(
            AgentDelta(
                agent=agent,
                gained=gained,
                lost=lost,
                gained_risk={tool_id: risk_of.get(tool_id, "") for tool_id in gained},
                max_tools_before=old.max_tools if old else None,
                max_tools_after=new.max_tools if new else None,
            )
        )

    return PolicyDiff(
        deltas=tuple(deltas),
        catalogue_size=len(tools),
        undecidable=tuple(sorted(set(undecidable))),
    )
