"""Domain models.

These types are the contract between every component. They are deliberately
plain data: a source produces :class:`Tool`, a retriever produces :class:`Hit`,
a policy produces :class:`PolicyResult`, and the caller receives a
:class:`Selection` that explains how the final list was reached.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

JSONSchema = dict[str, Any]


class _Frozen(BaseModel):
    """Immutable, extra-rejecting base for value objects."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class RiskTier(str, Enum):
    """How much damage a tool can do if the model calls it wrongly.

    Ordered: policies can express "nothing above ``MEDIUM``" without enumerating
    every tier.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def level(self) -> int:
        """Position in the ordering, lowest risk first."""
        return _RISK_ORDER[self]

    def __lt__(self, other: object) -> bool:
        """Compare by ordinal rather than by string value."""
        if not isinstance(other, RiskTier):
            return NotImplemented
        return self.level < other.level

    def __le__(self, other: object) -> bool:
        """Compare by ordinal rather than by string value."""
        if not isinstance(other, RiskTier):
            return NotImplemented
        return self.level <= other.level

    def __gt__(self, other: object) -> bool:
        """Compare by ordinal rather than by string value."""
        if not isinstance(other, RiskTier):
            return NotImplemented
        return self.level > other.level

    def __ge__(self, other: object) -> bool:
        """Compare by ordinal rather than by string value."""
        if not isinstance(other, RiskTier):
            return NotImplemented
        return self.level >= other.level


_RISK_ORDER: dict[RiskTier, int] = {
    RiskTier.LOW: 0,
    RiskTier.MEDIUM: 1,
    RiskTier.HIGH: 2,
    RiskTier.CRITICAL: 3,
}


class CostTier(str, Enum):
    """Rough expense of calling a tool, for budget-style policies."""

    FREE = "free"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Tool(_Frozen):
    """A single callable exposed to an agent.

    ToolBroker never invokes a tool. It carries enough metadata to rank, filter,
    and render one, plus a ``handle`` the caller's framework uses to actually
    call it.
    """

    name: str = Field(min_length=1)
    namespace: str = Field(default="default", min_length=1)
    description: str = ""
    input_schema: JSONSchema = Field(default_factory=dict)
    tags: frozenset[str] = frozenset()
    examples: tuple[str, ...] = ()
    risk: RiskTier = RiskTier.LOW
    cost: CostTier = CostTier.FREE
    required_scopes: frozenset[str] = frozenset()
    source_id: str = "inline"
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("name", "namespace")
    @classmethod
    def _clean_identifier(cls, value: str) -> str:
        """Normalise a name or namespace, rejecting what cannot be made safe.

        Surrounding whitespace is stripped rather than rejected. It is never
        intentional, and a tool called ``"admin "`` would otherwise slip past an
        ``admin/*`` deny rule while looking identical in every log and error
        message an operator might check.

        Control characters are rejected outright: they cannot be displayed, so
        two visually identical tool ids could carry different policy outcomes.
        """
        cleaned = value.strip()
        if "/" in cleaned:
            raise ValueError("must not contain '/'")
        if not cleaned:
            raise ValueError("must not be empty or whitespace")
        if any(character.isprintable() is False for character in cleaned):
            raise ValueError("must not contain control characters")
        return cleaned

    @property
    def id(self) -> str:
        """Globally unique identifier, ``namespace/name``."""
        return f"{self.namespace}/{self.name}"

    def with_metadata(self, **updates: Any) -> Tool:
        """Return a copy with additional metadata entries."""
        merged = {**self.metadata, **updates}
        return self.model_copy(update={"metadata": merged})


class ToolRecord(_Frozen):
    """A tool paired with the text and vector used to find it."""

    tool: Tool
    text: str
    vector: tuple[float, ...]

    @property
    def id(self) -> str:
        """Identifier of the underlying tool."""
        return self.tool.id


class Filters(_Frozen):
    """Pre-retrieval narrowing applied inside the store.

    Applying these before scoring, rather than after, is what keeps ``k``
    meaningful: asking for 5 tools in the ``billing`` namespace returns 5, not
    whatever survives filtering the global top 5.
    """

    namespaces: frozenset[str] | None = None
    tags_any: frozenset[str] | None = None
    tags_all: frozenset[str] | None = None
    exclude_tags: frozenset[str] = frozenset()
    max_risk: RiskTier | None = None
    tool_ids: frozenset[str] | None = None
    exclude_tool_ids: frozenset[str] = frozenset()

    def matches(self, tool: Tool) -> bool:
        """Return whether ``tool`` survives every constraint."""
        if self.namespaces is not None and tool.namespace not in self.namespaces:
            return False
        if self.tool_ids is not None and tool.id not in self.tool_ids:
            return False
        if tool.id in self.exclude_tool_ids:
            return False
        if self.tags_any is not None and not (tool.tags & self.tags_any):
            return False
        if self.tags_all is not None and not self.tags_all <= tool.tags:
            return False
        if tool.tags & self.exclude_tags:
            return False
        return not (self.max_risk is not None and tool.risk > self.max_risk)

    def is_empty(self) -> bool:
        """Return whether this filter set constrains nothing."""
        return self == Filters()


class Hit(_Frozen):
    """A retrieved tool with its score and the components that produced it."""

    tool: Tool
    score: float
    components: Mapping[str, float] = Field(default_factory=dict)

    @property
    def id(self) -> str:
        """Identifier of the underlying tool."""
        return self.tool.id


class Stage(str, Enum):
    """Pipeline stage a decision was made in, for explanations."""

    DISCOVERY = "discovery"
    INDEXING = "indexing"
    FILTERING = "filtering"
    RETRIEVAL = "retrieval"
    RERANKING = "reranking"
    POLICY = "policy"
    ADAPTATION = "adaptation"


class Decision(str, Enum):
    """Outcome of a policy rule."""

    ALLOW = "allow"
    DENY = "deny"


class RuleFiring(_Frozen):
    """A record that one policy rule matched one tool (or the selection)."""

    rule: str
    decision: Decision
    reason: str
    tool_id: str | None = None


class Exclusion(_Frozen):
    """A tool that was dropped, and why."""

    tool_id: str
    stage: Stage
    reason: str


class PolicyResult(_Frozen):
    """What a policy decided about a candidate set."""

    hits: tuple[Hit, ...]
    firings: tuple[RuleFiring, ...] = ()
    exclusions: tuple[Exclusion, ...] = ()


class Selection(_Frozen):
    """The answer to "which tools go in the context, and why".

    Never a bare list. Every score, rule, and exclusion that shaped the result
    travels with it, so a surprising selection can be explained without
    re-running anything.
    """

    query: str
    hits: tuple[Hit, ...]
    agent: str | None = None
    requested_k: int = 0
    considered: int = 0
    firings: tuple[RuleFiring, ...] = ()
    exclusions: tuple[Exclusion, ...] = ()
    timings_ms: Mapping[str, float] = Field(default_factory=dict)
    dry_run: bool = False

    @property
    def tools(self) -> tuple[Tool, ...]:
        """The selected tools, highest scoring first."""
        return tuple(hit.tool for hit in self.hits)

    @property
    def tool_ids(self) -> tuple[str, ...]:
        """Identifiers of the selected tools."""
        return tuple(hit.id for hit in self.hits)

    def __len__(self) -> int:
        """Number of selected tools."""
        return len(self.hits)

    def __iter__(self) -> Iterable[Tool]:  # type: ignore[override]
        """Iterate the selected tools."""
        return iter(self.tools)

    def explain(self) -> str:
        """Render a human-readable trace of how this selection was made."""
        lines = [f'query: "{self.query}"']
        if self.agent:
            lines.append(f"agent: {self.agent}")
        lines.append(
            f"considered {self.considered} tools, "
            f"requested {self.requested_k}, selected {len(self.hits)}"
        )
        if self.dry_run:
            lines.append("DRY RUN: policy decisions recorded but not enforced")
        lines.append("")
        lines.append("selected:")
        for rank, hit in enumerate(self.hits, start=1):
            parts = ", ".join(f"{k}={v:.4f}" for k, v in sorted(hit.components.items()))
            detail = f"  [{parts}]" if parts else ""
            lines.append(f"  {rank}. {hit.id}  score={hit.score:.4f}{detail}")
        if self.exclusions:
            lines.append("")
            lines.append("excluded:")
            for exclusion in self.exclusions:
                lines.append(
                    f"  - {exclusion.tool_id} ({exclusion.stage.value}): {exclusion.reason}"
                )
        if self.firings:
            lines.append("")
            lines.append("rules fired:")
            for firing in self.firings:
                target = firing.tool_id or "*"
                lines.append(
                    f"  - {firing.rule} -> {firing.decision.value} on {target}: {firing.reason}"
                )
        if self.timings_ms:
            lines.append("")
            timings = ", ".join(f"{k}={v:.2f}ms" for k, v in self.timings_ms.items())
            lines.append(f"timings: {timings}")
        return "\n".join(lines)
