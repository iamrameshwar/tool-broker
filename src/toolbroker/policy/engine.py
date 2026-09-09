"""Evaluating rules against a candidate set.

Combining logic defaults to **deny-overrides**: every rule gets a say, and a
single deny wins. That is the model security reviewers already know from IAM,
and it fails closed — a misordered rule list cannot accidentally grant access.

``first_match`` is available for teams who want ordered, escape-early
semantics, but it is opt-in because getting the order wrong is silent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from ..observability import get_logger, span
from ..types import Decision, Exclusion, Hit, PolicyResult, RuleFiring, Stage
from .rules import Rule, RuleContext, SelectionRule

logger = get_logger("policy")

Combining = Literal["deny_overrides", "first_match"]


class PolicyEngine:
    """Applies per-tool rules then selection rules, recording every decision."""

    def __init__(
        self,
        rules: Sequence[Rule] = (),
        selection_rules: Sequence[SelectionRule] = (),
        *,
        default: Decision = Decision.ALLOW,
        combining: Combining = "deny_overrides",
        dry_run: bool = False,
    ) -> None:
        """Configure the engine.

        Args:
            rules: Per-tool rules.
            selection_rules: Whole-selection rules, applied after per-tool ones.
            default: What happens to a tool no rule has an opinion about.
            combining: ``"deny_overrides"`` or ``"first_match"``.
            dry_run: Record decisions without enforcing them. This is how a team
                rolls out a policy: run it against real traffic, read the
                firings, and only then turn it on.
        """
        self._rules = list(rules)
        self._selection_rules = list(selection_rules)
        self._default = default
        self._combining = combining
        self._dry_run = dry_run

    @property
    def dry_run(self) -> bool:
        """Whether decisions are recorded but not enforced."""
        return self._dry_run

    @property
    def rules(self) -> Sequence[Rule]:
        """The per-tool rules."""
        return tuple(self._rules)

    @property
    def selection_rules(self) -> Sequence[SelectionRule]:
        """The whole-selection rules."""
        return tuple(self._selection_rules)

    @property
    def default(self) -> Decision:
        """What happens to a tool no rule has an opinion about."""
        return self._default

    @property
    def combining(self) -> Combining:
        """How conflicting rule decisions are resolved."""
        return self._combining

    def with_rules(
        self,
        rules: Sequence[Rule] = (),
        selection_rules: Sequence[SelectionRule] = (),
    ) -> PolicyEngine:
        """Return a copy with additional rules appended."""
        return PolicyEngine(
            [*self._rules, *rules],
            [*self._selection_rules, *selection_rules],
            default=self._default,
            combining=self._combining,
            dry_run=self._dry_run,
        )

    def evaluate(
        self,
        hits: Sequence[Hit],
        *,
        agent: str | None = None,
        scopes: frozenset[str] = frozenset(),
        query: str = "",
        tenant: str | None = None,
    ) -> PolicyResult:
        """Apply every rule to ``hits``."""
        with span("toolbroker.policy", agent=agent, candidates=len(hits)):
            context = RuleContext(agent=agent, scopes=scopes, query=query, tenant=tenant)
            firings: list[RuleFiring] = []
            exclusions: list[Exclusion] = []
            allowed: list[Hit] = []

            for hit in hits:
                decision, hit_firings = self._judge(hit, context)
                firings.extend(hit_firings)
                if decision is Decision.ALLOW:
                    allowed.append(hit)
                    continue
                exclusions.append(
                    Exclusion(
                        tool_id=hit.id,
                        stage=Stage.POLICY,
                        reason=self._denial_reason(hit_firings),
                    )
                )
                if self._dry_run:
                    allowed.append(hit)

            for rule in self._selection_rules:
                kept, rule_exclusions, rule_firings = rule.apply(allowed, context)
                firings.extend(rule_firings)
                exclusions.extend(rule_exclusions)
                if not self._dry_run:
                    allowed = kept

            if exclusions:
                logger.debug(
                    "policy excluded tools",
                    extra={
                        "agent": agent,
                        "excluded": [exclusion.tool_id for exclusion in exclusions],
                        "dry_run": self._dry_run,
                    },
                )

            return PolicyResult(
                hits=tuple(allowed),
                firings=tuple(firings),
                exclusions=tuple(exclusions),
            )

    def _judge(self, hit: Hit, context: RuleContext) -> tuple[Decision, list[RuleFiring]]:
        """Return the combined decision for one hit and the firings behind it."""
        firings: list[RuleFiring] = []
        verdict: Decision | None = None

        # Mandatory rules first, and always. Under `first_match` an earlier
        # allow-rule would otherwise short-circuit a later isolation boundary,
        # which makes a security guarantee depend on list order.
        ordered = sorted(self._rules, key=lambda rule: not rule.mandatory)
        short_circuit = self._combining == "first_match"

        for rule in ordered:
            outcome = rule.check(hit.tool, context)
            if outcome is None:
                continue
            decision, reason = outcome
            firings.append(
                RuleFiring(rule=rule.name, decision=decision, reason=reason, tool_id=hit.id)
            )
            if rule.mandatory and decision is Decision.DENY:
                # A mandatory deny is final under either combining mode.
                return Decision.DENY, firings
            if short_circuit and not rule.mandatory:
                return decision, firings
            if decision is Decision.DENY:
                verdict = Decision.DENY
            elif verdict is None:
                verdict = Decision.ALLOW

        return (verdict if verdict is not None else self._default), firings

    @staticmethod
    def _denial_reason(firings: Sequence[RuleFiring]) -> str:
        """Summarise why a tool was denied."""
        denials = [firing for firing in firings if firing.decision is Decision.DENY]
        if not denials:
            return "denied by default policy"
        return "; ".join(f"{firing.rule}: {firing.reason}" for firing in denials)

    def __repr__(self) -> str:
        """Show rule counts and mode."""
        return (
            f"PolicyEngine(rules={len(self._rules)}, "
            f"selection_rules={len(self._selection_rules)}, "
            f"default={self._default.value}, dry_run={self._dry_run})"
        )


class AgentPolicy:
    """Routes each agent to its own :class:`PolicyEngine`.

    One process usually hosts several agents with different privileges — a
    read-only support bot and an ops agent that can restart services. This
    keeps that difference in one place instead of scattered across call sites.
    """

    def __init__(
        self,
        policies: Mapping[str, PolicyEngine] | None = None,
        *,
        default: PolicyEngine | None = None,
        strict: bool = False,
    ) -> None:
        """Create the router.

        Args:
            policies: Per-agent engines, keyed by agent name.
            default: Applied when no agent is named. Defaults to
                allow-everything, so calling ``select()`` without an agent
                behaves as if there were no policy rather than silently
                returning nothing.
            strict: Deny everything for an agent that is *named but not
                registered*. Recommended in production.

        An agent name that is not registered is almost always a typo, and
        routing it to a permissive default hands it the entire catalogue. Even
        with ``strict=False`` that case is logged as a warning, because the
        failure is otherwise completely silent — no exception, no empty result,
        just an agent quietly holding more privilege than anyone intended.
        """
        self._policies = dict(policies or {})
        self._default = default if default is not None else PolicyEngine()
        self._strict = strict
        self._deny_all = PolicyEngine([], default=Decision.DENY)
        self._warned: set[str] = set()

    def register(self, agent: str, policy: PolicyEngine) -> None:
        """Attach ``policy`` to ``agent``."""
        self._policies[agent] = policy

    @property
    def strict(self) -> bool:
        """Whether an unregistered agent name is denied everything."""
        return self._strict

    @property
    def agents(self) -> tuple[str, ...]:
        """Registered agent names."""
        return tuple(sorted(self._policies))

    def for_agent(self, agent: str | None) -> PolicyEngine:
        """Return the engine governing ``agent``.

        ``None`` means "no particular agent" and gets the default. A *named*
        agent that is not registered is treated as the mistake it almost always
        is: denied under ``strict``, and warned about either way.
        """
        if agent is None:
            return self._default
        engine = self._policies.get(agent)
        if engine is not None:
            return engine

        if agent not in self._warned:
            self._warned.add(agent)
            logger.warning(
                "no policy registered for this agent; %s",
                "denying everything (strict)"
                if self._strict
                else "falling back to the default engine, which may grant more than intended",
                extra={"agent": agent, "registered": self.agents},
            )
        return self._deny_all if self._strict else self._default

    def evaluate(
        self,
        hits: Sequence[Hit],
        *,
        agent: str | None = None,
        scopes: frozenset[str] = frozenset(),
        query: str = "",
        tenant: str | None = None,
    ) -> PolicyResult:
        """Dispatch to the right engine and evaluate."""
        return self.for_agent(agent).evaluate(
            hits, agent=agent, scopes=scopes, query=query, tenant=tenant
        )

    def __repr__(self) -> str:
        """Show the configured agents."""
        return f"AgentPolicy(agents={list(self.agents)}, strict={self._strict})"
