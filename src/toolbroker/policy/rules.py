"""Policy rules.

Two kinds:

* :class:`Rule` judges one tool at a time — "may this agent see this tool?"
* :class:`SelectionRule` judges the set — "how many tools, in total?"

Every rule states a *reason* alongside its decision. A policy that can only say
"denied" is not auditable, and an agent developer who cannot see why their tool
vanished will work around the policy rather than with it.
"""

from __future__ import annotations

import fnmatch
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from ..types import CostTier, Decision, Exclusion, Hit, RiskTier, RuleFiring, Stage, Tool

_COST_ORDER = {CostTier.FREE: 0, CostTier.LOW: 1, CostTier.MEDIUM: 2, CostTier.HIGH: 3}


@dataclass(frozen=True, slots=True)
class RuleContext:
    """Everything a rule may consult beyond the tool itself."""

    agent: str | None = None
    scopes: frozenset[str] = frozenset()
    query: str = ""
    #: Which tenant this request belongs to. ``None`` means the caller did not
    #: declare one, which :class:`TenantIsolation` treats as "may see only
    #: untenanted tools" rather than as "may see everything".
    tenant: str | None = None


Verdict = tuple[Decision, str]


class Rule(ABC):
    """Judges a single tool."""

    name: str = "rule"

    #: Whether this rule must be consulted whatever the combining mode.
    #:
    #: ``first_match`` returns on the first rule with an opinion, so an
    #: ``allow`` glob earlier in the list can grant a tool before a later rule
    #: ever runs. That is acceptable for a preference and unacceptable for an
    #: isolation boundary: a security guarantee that depends on rule order is
    #: not a guarantee. A mandatory rule is always evaluated, and its ``DENY``
    #: always wins.
    mandatory: bool = False

    @abstractmethod
    def check(self, tool: Tool, context: RuleContext) -> Verdict | None:
        """Return a verdict, or ``None`` to abstain.

        Abstaining matters: a rule about billing tools should have no opinion
        about search tools, and saying so explicitly keeps combining logic
        simple.
        """

    def __repr__(self) -> str:
        """Show the rule name."""
        return f"{type(self).__name__}(name={self.name!r})"


class SelectionRule(ABC):
    """Judges the selection as a whole."""

    name: str = "selection_rule"

    @abstractmethod
    def apply(
        self, hits: Sequence[Hit], context: RuleContext
    ) -> tuple[list[Hit], list[Exclusion], list[RuleFiring]]:
        """Return kept hits, plus the exclusions and firings that produced them."""


def _matches_any(
    tool_id: str, patterns: Iterable[str], *, case_sensitive: bool = False
) -> str | None:
    """Return the first glob in ``patterns`` matching ``tool_id``.

    Case-insensitive by default. Case is not semantically meaningful in a tool
    name, but it is a silent bypass: an operator writing ``*/delete_*`` means
    "block deletion", and a server exposing ``DeleteUser`` — which plenty of
    APIs do — would otherwise sail straight through a rule they believed was
    protecting them, with nothing anywhere to indicate it.
    """
    if case_sensitive:
        return next((p for p in patterns if fnmatch.fnmatchcase(tool_id, p)), None)
    lowered = tool_id.lower()
    return next((p for p in patterns if fnmatch.fnmatchcase(lowered, p.lower())), None)


@dataclass(frozen=True)
class DenyTools(Rule):
    """Denies tools whose id matches any glob pattern.

    Patterns match the full ``namespace/name`` id, so ``"admin/*"`` blocks a
    namespace and ``"*/delete_*"`` blocks a verb across all of them.
    """

    patterns: tuple[str, ...]
    name: str = "deny_tools"
    reason: str = "matched a deny pattern"
    case_sensitive: bool = False

    def __init__(
        self,
        patterns: Iterable[str],
        *,
        name: str = "deny_tools",
        reason: str = "matched a deny pattern",
        case_sensitive: bool = False,
    ) -> None:
        """Create the rule from an iterable of glob patterns.

        Args:
            patterns: Globs matched against the full ``namespace/name`` id.
            name: Rule name, shown in firings.
            reason: Explanation attached to a denial.
            case_sensitive: Match case exactly. Off by default; see
                :func:`_matches_any` for why.
        """
        object.__setattr__(self, "patterns", tuple(patterns))
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "case_sensitive", case_sensitive)

    def check(self, tool: Tool, context: RuleContext) -> Verdict | None:
        """Deny if the tool id matches."""
        del context
        matched = _matches_any(tool.id, self.patterns, case_sensitive=self.case_sensitive)
        if matched is not None:
            return Decision.DENY, f"{self.reason}: {matched!r}"
        return None


@dataclass(frozen=True)
class AllowTools(Rule):
    """Allows tools matching a glob, and denies everything else.

    Use with an allowlist mindset: anything not named is denied by this rule,
    which is the safe direction for a rule that exists to constrain.
    """

    patterns: tuple[str, ...]
    name: str = "allow_tools"
    case_sensitive: bool = False

    def __init__(
        self,
        patterns: Iterable[str],
        *,
        name: str = "allow_tools",
        case_sensitive: bool = False,
    ) -> None:
        """Create the rule from an iterable of glob patterns.

        Note the asymmetry with :class:`DenyTools`: case-insensitive matching
        makes a *deny* list stricter and an *allow* list broader. It is still
        the default here, because an allowlist that silently fails to match the
        tool the operator meant to permit is its own kind of surprise. Pass
        ``case_sensitive=True`` when the distinction matters.
        """
        object.__setattr__(self, "patterns", tuple(patterns))
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "case_sensitive", case_sensitive)

    def check(self, tool: Tool, context: RuleContext) -> Verdict | None:
        """Allow on match, deny otherwise."""
        del context
        matched = _matches_any(tool.id, self.patterns, case_sensitive=self.case_sensitive)
        if matched is not None:
            return Decision.ALLOW, f"matched allow pattern {matched!r}"
        return Decision.DENY, "not in the allowlist"


@dataclass(frozen=True)
class RequireScopes(Rule):
    """Denies tools whose ``required_scopes`` the caller does not hold."""

    name: str = "require_scopes"

    def check(self, tool: Tool, context: RuleContext) -> Verdict | None:
        """Deny when scopes are missing; abstain when the tool needs none."""
        if not tool.required_scopes:
            return None
        missing = tool.required_scopes - context.scopes
        if missing:
            return Decision.DENY, f"missing scopes: {', '.join(sorted(missing))}"
        return Decision.ALLOW, "all required scopes held"


@dataclass(frozen=True)
class TenantIsolation(Rule):
    """Keeps one tenant's tools away from another's.

    Scopes cannot do this job. :class:`RequireScopes` *abstains* for a tool
    that declares none, which is the right behaviour for a capability check and
    exactly wrong for an isolation boundary: a tool nobody remembered to tag
    would be visible to every tenant. This rule decides on every tool, and
    decides ``DENY`` whenever it is not certain the caller should see it.

    A tool's tenant is read from ``tool.metadata[tenant_key]``. Tools without
    one are *shared* — visible to everybody — only when ``shared_tools`` is
    true. Turn it off and an untagged tool is invisible until somebody tags it,
    which is the safer default for a deployment where forgetting to tag is the
    likely mistake.
    """

    tenant_key: str = "tenant"
    shared_tools: bool = True
    name: str = "tenant_isolation"
    # A tenant boundary must hold regardless of where it sits in the list.
    mandatory: bool = True

    def check(self, tool: Tool, context: RuleContext) -> Verdict | None:
        """Allow only tools this request's tenant is entitled to see."""
        owner = tool.metadata.get(self.tenant_key)
        owner = str(owner) if owner is not None else None

        if owner is None:
            if self.shared_tools:
                return Decision.ALLOW, "shared tool, no tenant"
            return Decision.DENY, "tool has no tenant and shared tools are disabled"

        if context.tenant is None:
            # No tenant declared, so nothing tenant-owned can be justified.
            # Failing open here would leak every tenant to an unauthenticated
            # caller, which is the whole class of bug this rule exists for.
            return Decision.DENY, "request declared no tenant"

        if owner == context.tenant:
            return Decision.ALLOW, f"tenant {owner}"
        return Decision.DENY, "belongs to another tenant"


@dataclass(frozen=True)
class MaxRisk(Rule):
    """Denies tools above a risk tier."""

    ceiling: RiskTier = RiskTier.MEDIUM
    name: str = "max_risk"

    def check(self, tool: Tool, context: RuleContext) -> Verdict | None:
        """Deny when the tool's risk exceeds the ceiling."""
        del context
        if tool.risk > self.ceiling:
            return Decision.DENY, f"risk {tool.risk.value} exceeds ceiling {self.ceiling.value}"
        return None


@dataclass(frozen=True)
class MaxCost(Rule):
    """Denies tools above a cost tier."""

    ceiling: CostTier = CostTier.HIGH
    name: str = "max_cost"

    def check(self, tool: Tool, context: RuleContext) -> Verdict | None:
        """Deny when the tool's cost exceeds the ceiling."""
        del context
        if _COST_ORDER[tool.cost] > _COST_ORDER[self.ceiling]:
            return Decision.DENY, f"cost {tool.cost.value} exceeds ceiling {self.ceiling.value}"
        return None


@dataclass(frozen=True)
class DenyTags(Rule):
    """Denies tools carrying any of the given tags."""

    tags: frozenset[str] = frozenset()
    name: str = "deny_tags"

    def check(self, tool: Tool, context: RuleContext) -> Verdict | None:
        """Deny on tag intersection."""
        del context
        overlap = tool.tags & self.tags
        if overlap:
            return Decision.DENY, f"carries denied tag(s): {', '.join(sorted(overlap))}"
        return None


@dataclass(frozen=True)
class RequireTags(Rule):
    """Denies tools missing any of the required tags."""

    tags: frozenset[str] = frozenset()
    name: str = "require_tags"

    def check(self, tool: Tool, context: RuleContext) -> Verdict | None:
        """Deny when required tags are absent."""
        del context
        missing = self.tags - tool.tags
        if missing:
            return Decision.DENY, f"missing required tag(s): {', '.join(sorted(missing))}"
        return None


@dataclass(frozen=True)
class PredicateRule(Rule):
    """Wraps an arbitrary callable as a rule.

    The escape hatch for logic we cannot anticipate — tenant checks, time
    windows, feature flags — without requiring a plugin package.
    """

    predicate: Callable[[Tool, RuleContext], bool]
    name: str = "predicate"
    reason: str = "predicate returned False"
    on_true: Decision = Decision.ALLOW

    def check(self, tool: Tool, context: RuleContext) -> Verdict | None:
        """Apply the predicate and translate it into a verdict."""
        result = self.predicate(tool, context)
        if result:
            if self.on_true is Decision.ALLOW:
                return None
            return Decision.DENY, self.reason
        if self.on_true is Decision.ALLOW:
            return Decision.DENY, self.reason
        return None


@dataclass(frozen=True)
class MaxTools(SelectionRule):
    """Caps how many tools reach the model.

    The cap is the whole point of the library. Everything above it is dropped
    lowest-score-first, and each drop is recorded so a user can see that the
    limit, not the ranking, is what removed their tool.
    """

    limit: int = 8
    name: str = "max_tools"
    keep_pinned: frozenset[str] = field(default_factory=frozenset)

    def apply(
        self, hits: Sequence[Hit], context: RuleContext
    ) -> tuple[list[Hit], list[Exclusion], list[RuleFiring]]:
        """Truncate to the limit, keeping pinned tools regardless of rank."""
        del context
        if self.limit < 0 or len(hits) <= self.limit:
            return list(hits), [], []

        pinned = [hit for hit in hits if hit.id in self.keep_pinned]
        rest = [hit for hit in hits if hit.id not in self.keep_pinned]
        room = max(0, self.limit - len(pinned))
        kept = pinned + rest[:room]
        kept_ids = {hit.id for hit in kept}
        ordered = [hit for hit in hits if hit.id in kept_ids]

        exclusions = [
            Exclusion(
                tool_id=hit.id,
                stage=Stage.POLICY,
                reason=f"beyond max_tools limit of {self.limit}",
            )
            for hit in hits
            if hit.id not in kept_ids
        ]
        firing = RuleFiring(
            rule=self.name,
            decision=Decision.DENY,
            reason=f"capped selection at {self.limit} tools ({len(exclusions)} dropped)",
        )
        return ordered, exclusions, [firing]


@dataclass(frozen=True)
class MinScore(SelectionRule):
    """Drops hits below a score floor.

    Returning a weak match is worse than returning nothing: the model will use
    whatever it is given, so a floor is how you say "no tool fits this query".
    """

    threshold: float = 0.0
    name: str = "min_score"

    def apply(
        self, hits: Sequence[Hit], context: RuleContext
    ) -> tuple[list[Hit], list[Exclusion], list[RuleFiring]]:
        """Filter by score threshold."""
        del context
        kept = [hit for hit in hits if hit.score >= self.threshold]
        exclusions = [
            Exclusion(
                tool_id=hit.id,
                stage=Stage.POLICY,
                reason=f"score {hit.score:.4f} below threshold {self.threshold:.4f}",
            )
            for hit in hits
            if hit.score < self.threshold
        ]
        firings = (
            [
                RuleFiring(
                    rule=self.name,
                    decision=Decision.DENY,
                    reason=f"dropped {len(exclusions)} hits below {self.threshold:.4f}",
                )
            ]
            if exclusions
            else []
        )
        return kept, exclusions, firings
