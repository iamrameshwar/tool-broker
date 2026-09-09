"""Declarative configuration.

The same catalogue you build in Python can be described in YAML and checked
into a repo, which is what a platform team needs when agent permissions have to
be reviewed like any other config.

    embedder:
      name: fastembed
      options: {model_name: BAAI/bge-small-en-v1.5}
    store:
      name: pgvector
      options: {dsn: "${TOOLBROKER_PG_DSN}"}   # from the environment, not the file
    sources:
      - {type: mcp, name: github, command: npx, args: ["-y", "@x/github-mcp"]}
    policy:
      default_k: 5
      agents:
        support:
          max_tools: 4
          max_risk: low
          deny: ["*/delete_*"]

Everything resolves through the plugin registry, so a config file can name a
third-party store the core has never heard of, by entry point or by dotted path.

``${VAR}`` and ``${VAR:-default}`` are substituted from the environment before
validation, so the file can be committed while the DSNs and API keys in it are
not. See :mod:`toolbroker.interpolate`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .aliases import DEFAULT_MAX_ALIASES, DEFAULT_MIN_COUNT, AliasLearner
from .catalog import ToolBroker
from .drift import DriftGuard
from .errors import ConfigurationError
from .hooks import Event, HookManager
from .index.enrich import EnrichmentConfig
from .interpolate import interpolate
from .policy.engine import AgentPolicy, PolicyEngine
from .policy.rules import (
    AllowTools,
    DenyTags,
    DenyTools,
    MaxCost,
    MaxRisk,
    MaxTools,
    MinScore,
    RequireScopes,
    RequireTags,
    Rule,
    SelectionRule,
    TenantIsolation,
)
from .protocols import Embedder, Source, Store
from .registry import (
    GROUP_EMBEDDERS,
    GROUP_HOOKS,
    GROUP_RERANKERS,
    GROUP_RETRIEVERS,
    GROUP_SOURCES,
    GROUP_STORES,
    GROUP_TRACERS,
    construct,
    create,
    resolve,
)
from .retrieve.resilient import OnError, ResilientRetriever
from .savings import SavingsTally
from .types import CostTier, Decision, RiskTier
from .usage import UsageTracker

#: Retrievers whose construction needs collaborators a config cannot express.
#: Everything else resolves through the plugin registry.
_BUILT_IN_RETRIEVERS = frozenset({"semantic", "keyword", "hybrid"})


class _Section(BaseModel):
    """Strict base so a typo in a config file is an error, not a silent default."""

    model_config = ConfigDict(extra="forbid")


class ComponentConfig(_Section):
    """A plugin name plus its constructor options."""

    name: str
    options: dict[str, Any] = Field(default_factory=dict)


class SourceConfig(_Section):
    """One tool source."""

    type: str
    name: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)

    # MCP shorthands, so the common case does not need a nested options block.
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    url: str | None = None
    env: dict[str, str] | None = None

    def build(self) -> Source:
        """Instantiate the source."""
        options = dict(self.options)
        if self.type == "mcp":
            from .sources.mcp import MCPServerSpec, MCPSource

            if not self.name:
                raise ConfigurationError("mcp sources require a 'name'")
            spec = MCPServerSpec(
                name=self.name,
                transport="http" if self.url else "stdio",
                command=self.command,
                args=tuple(self.args),
                url=self.url,
                env=self.env,
            )
            return MCPSource(spec, **options)
        if self.name and "source_id" not in options:
            options["source_id"] = self.name
        source: Source = create(GROUP_SOURCES, self.type, **options)
        return source


class AgentPolicyConfig(_Section):
    """Rules governing one agent."""

    max_tools: int | None = None
    max_risk: RiskTier | None = None
    max_cost: CostTier | None = None
    min_score: float | None = None
    allow: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)
    deny_tags: list[str] = Field(default_factory=list)
    require_tags: list[str] = Field(default_factory=list)
    require_scopes: bool = True
    #: Isolate tenants by ``tool.metadata[tenant_key]``. Unlike scopes this
    #: decides on every tool, so an untagged one cannot leak by abstention.
    tenant_isolation: bool = False
    tenant_key: str = "tenant"
    #: Whether tools with no tenant are visible to everyone. Off is the safer
    #: setting where forgetting to tag is the likely mistake.
    shared_tools: bool = True
    pinned: list[str] = Field(default_factory=list)
    default: Literal["allow", "deny"] = "allow"
    combining: Literal["deny_overrides", "first_match"] = "deny_overrides"
    dry_run: bool = False

    def build(self) -> PolicyEngine:
        """Instantiate the engine described by this section."""
        rules: list[Rule] = []
        if self.allow:
            rules.append(AllowTools(self.allow))
        if self.deny:
            rules.append(DenyTools(self.deny))
        if self.deny_tags:
            rules.append(DenyTags(frozenset(self.deny_tags)))
        if self.require_tags:
            rules.append(RequireTags(frozenset(self.require_tags)))
        if self.max_risk is not None:
            rules.append(MaxRisk(self.max_risk))
        if self.max_cost is not None:
            rules.append(MaxCost(self.max_cost))
        if self.require_scopes:
            rules.append(RequireScopes())
        if self.tenant_isolation:
            rules.append(
                TenantIsolation(tenant_key=self.tenant_key, shared_tools=self.shared_tools)
            )

        selection_rules: list[SelectionRule] = []
        if self.min_score is not None:
            selection_rules.append(MinScore(self.min_score))
        if self.max_tools is not None:
            selection_rules.append(
                MaxTools(limit=self.max_tools, keep_pinned=frozenset(self.pinned))
            )

        return PolicyEngine(
            rules,
            selection_rules,
            default=Decision.ALLOW if self.default == "allow" else Decision.DENY,
            combining=self.combining,
            dry_run=self.dry_run,
        )


class PolicyConfig(_Section):
    """Default policy plus per-agent overrides."""

    default_k: int = 5
    default: AgentPolicyConfig = Field(default_factory=AgentPolicyConfig)
    agents: dict[str, AgentPolicyConfig] = Field(default_factory=dict)
    #: Deny everything for an agent that is named but not registered here.
    #: Recommended in production: an unregistered name is nearly always a typo,
    #: and the permissive fallback hands it the whole catalogue.
    strict_agents: bool = False

    def build(self) -> AgentPolicy:
        """Instantiate the agent-routing policy."""
        return AgentPolicy(
            {name: section.build() for name, section in self.agents.items()},
            default=self.default.build(),
            strict=self.strict_agents,
        )


class RetrievalConfig(_Section):
    """Retriever selection and pipeline shape.

    ``mode`` is not a closed set. ``semantic``, ``keyword`` and ``hybrid`` are
    built in, and anything else is resolved through the plugin registry — an
    installed entry point in ``toolbroker.retrievers``, a name registered at
    runtime, or a dotted path like ``my_pkg.retrieval:GraphRetriever``. The
    store and embedder are handed to the constructor if it names them.
    """

    mode: str = "semantic"
    #: Constructor arguments for a plugin retriever. Ignored by the built-ins,
    #: which are configured by the fields below.
    options: dict[str, Any] = Field(default_factory=dict)
    weights: list[float] | None = None
    fusion: Literal["rrf", "weighted"] = "rrf"
    overfetch: float = 4.0
    min_score: float = 0.0
    #: What to do when the retriever raises — usually the vector store being
    #: unreachable. ``fail`` re-raises, ``empty`` returns nothing deliberately,
    #: ``fallback`` tries ``fallback`` instead.
    on_error: OnError = "fail"
    #: The degraded-path retriever. ``keyword`` is the useful choice: lexical
    #: retrieval needs no external service, so it survives the outage that took
    #: the primary down.
    fallback: ComponentConfig | None = None
    #: Rerankers applied in order after retrieval. Each is a plugin name plus
    #: its options, resolved the same way as ``mode``.
    rerankers: list[ComponentConfig] = Field(default_factory=list)


class DriftConfig(_Section):
    """Watching for tools that change underneath you."""

    enabled: bool = False
    #: Keep serving the last approved version when a description or schema
    #: changes, instead of accepting the new one.
    quarantine: bool = False
    #: Approve a tool's content the first time it is seen. Off means the
    #: approved set is managed deliberately, via ``approved`` or ``path``.
    trust_on_first_use: bool = True
    #: Where approvals live. Without it, quarantine re-fires on every restart.
    path: Path | None = None
    #: Digests pinned in the config itself, for tools worth being explicit about.
    approved: dict[str, str] = Field(default_factory=dict)

    def build(self) -> DriftGuard | None:
        """Construct the guard, or ``None`` when disabled."""
        if not self.enabled:
            return None
        return DriftGuard(
            quarantine=self.quarantine,
            trust_on_first_use=self.trust_on_first_use,
            approved=self.approved,
            path=self.path,
        )


class AliasConfig(_Section):
    """Learning how users actually ask for a tool."""

    enabled: bool = False
    #: Where learned phrases persist between restarts.
    path: Path | None = None
    #: Times a phrase must recur before it enters the index. The bound that
    #: stops one stray call rewriting retrieval.
    min_count: float = DEFAULT_MIN_COUNT
    #: Most phrases any single tool may contribute.
    max_aliases: int = DEFAULT_MAX_ALIASES
    half_life_days: float = 30.0

    def build(self) -> AliasLearner | None:
        """Construct the learner, restoring persisted phrases if configured."""
        if not self.enabled:
            return None
        settings: dict[str, Any] = {
            "half_life": self.half_life_days * 86400.0,
            "min_count": self.min_count,
            "max_aliases": self.max_aliases,
        }
        if self.path:
            return AliasLearner.load(self.path, **settings)
        return AliasLearner(**settings)


class SavingsConfig(_Section):
    """Accounting for the context retrieval avoided sending."""

    enabled: bool = False
    #: Input-token price for your model, so the report carries a cost figure.
    price_per_million: float | None = None

    def build(self) -> SavingsTally | None:
        """Construct the tally, or ``None`` when disabled."""
        if not self.enabled:
            return None
        return SavingsTally(price_per_million=self.price_per_million)


class HooksConfig(_Section):
    """Handlers attached to pipeline events, by dotted path.

    The point of doing this from config rather than only in Python is that the
    interesting handlers belong to the deployment, not the application: send
    every selection to an audit log, normalise queries through an in-house
    service, drop tools an external authoriser rejects. A platform team should
    be able to add those without a code change in every service that embeds
    the broker.

    .. code-block:: yaml

        hooks:
          strict: false
          after_selection:
            - {name: my_pkg.audit:record}
          transform_query:
            - {name: my_pkg.nlp:Normaliser, options: {lang: en}}

    Without ``options`` the resolved object is used as the handler. With them
    it is *called* with those options and the result is the handler, so a
    parameterised handler is a factory rather than a closure someone has to
    write.
    """

    #: Re-raise handler exceptions instead of logging and continuing. Off by
    #: default: a broken audit hook should not take selection down with it.
    strict: bool = False
    handlers: dict[str, list[ComponentConfig]] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _accept_events_at_the_top_level(cls, data: Any) -> Any:
        """Let event names sit directly under ``hooks:``.

        ``hooks: {after_selection: [...]}`` reads far better than
        ``hooks: {handlers: {after_selection: [...]}}``, and the nested form
        stays valid for anyone who prefers it.
        """
        if not isinstance(data, Mapping):
            return data
        known = {"strict", "handlers"}
        inline = {key: value for key, value in data.items() if key not in known}
        if not inline:
            return data
        merged = {key: value for key, value in data.items() if key in known}
        handlers = dict(merged.get("handlers") or {})
        handlers.update(inline)
        merged["handlers"] = handlers
        return merged

    def build(self) -> HookManager:
        """Construct a hook manager with every configured handler attached."""
        manager = HookManager(strict=self.strict)
        for event_name, entries in self.handlers.items():
            try:
                event = Event(event_name)
            except ValueError as exc:
                known = ", ".join(member.value for member in Event)
                raise ConfigurationError(
                    f"unknown hook event {event_name!r}. Known events: {known}"
                ) from exc
            for entry in entries:
                handler = resolve(GROUP_HOOKS, entry.name)
                if entry.options:
                    handler = handler(**entry.options)
                if not callable(handler):
                    raise ConfigurationError(
                        f"hook {entry.name!r} for {event_name} is not callable"
                    )
                manager.register(event, handler)
        return manager


class ObservabilityConfig(_Section):
    """Where traces go.

    Logging is deliberately absent: ToolBroker logs through the stdlib under the
    ``toolbroker`` logger and never configures the root logger, so any provider
    that speaks ``logging`` — Datadog, Sentry, structlog, a plain file handler —
    is wired up by the application, not by this file. Tracing has no equivalent
    standard, which is why it needs a plugin point.
    """

    #: A tracer plugin. ``otel`` and ``null`` are built in; anything else is
    #: resolved through the ``toolbroker.tracers`` group or a dotted path.
    #: Omitted, OpenTelemetry is used when installed and nothing otherwise.
    tracer: ComponentConfig | None = None


class UsageConfig(_Section):
    """Usage-based boosting.

    Off unless ``enabled``. When on, you still have to call
    :meth:`~toolbroker.catalog.ToolBroker.record_use` — ToolBroker never executes a
    tool, so nothing else can tell it which one was called.
    """

    enabled: bool = False
    #: Maximum fraction of a hit's own score the boost may add. Bounded so a
    #: popular-but-wrong tool cannot displace an unused-but-right one.
    weight: float = Field(default=0.1, ge=0.0, le=1.0)
    #: Seconds after which a recorded use counts for half as much.
    half_life_days: float = Field(default=30.0, gt=0.0)
    #: Where to persist counts. Without it, usage resets on every restart.
    path: str | None = None


class ToolBrokerConfig(_Section):
    """A complete catalogue definition."""

    embedder: ComponentConfig | None = None
    store: ComponentConfig | None = None
    enrichment: EnrichmentConfig = Field(default_factory=EnrichmentConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    usage: UsageConfig = Field(default_factory=UsageConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    hooks: HooksConfig = Field(default_factory=HooksConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)
    aliases: AliasConfig = Field(default_factory=AliasConfig)
    savings: SavingsConfig = Field(default_factory=SavingsConfig)
    sources: list[SourceConfig] = Field(default_factory=list)
    cache_embeddings: bool = True

    @classmethod
    def from_file(cls, path: str | Path) -> ToolBrokerConfig:
        """Load a YAML or JSON config file."""
        file_path = Path(path)
        if not file_path.exists():
            raise ConfigurationError(f"config file not found: {file_path}")
        text = file_path.read_text(encoding="utf-8")
        if file_path.suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover - depends on extras
                raise ConfigurationError(
                    "YAML config requires pyyaml. Install with: pip install 'toolbroker[yaml]'"
                ) from exc
            payload = yaml.safe_load(text) or {}
        else:
            import json

            payload = json.loads(text)
        if not isinstance(payload, Mapping):
            raise ConfigurationError(f"{file_path} must contain a mapping at the top level")
        # Secrets belong in the environment, not in a file that gets committed.
        return cls.model_validate(interpolate(payload))

    def build(self) -> ToolBroker:
        """Construct a :class:`~toolbroker.catalog.ToolBroker` from this config."""
        self._apply_observability()
        hooks = self.hooks.build()
        aliases = self.aliases.build()
        if aliases is not None:
            # Aliases change what gets embedded, so they attach to the index
            # text rather than to retrieval, and take effect on the next index.
            hooks.register(Event.TRANSFORM_INDEX_TEXT, aliases.enrich)
        embedder = self._build_embedder()
        store = self._build_store(embedder.dim)
        sources = [source.build() for source in self.sources]

        broker = ToolBroker(
            embedder=embedder,
            store=store,
            enrichment=self.enrichment,
            sources=sources,
            default_k=self.policy.default_k,
            cache_embeddings=self.cache_embeddings,
            policy=self.policy.build(),
            hooks=hooks,
            drift=self.drift.build(),
            savings=self.savings.build(),
            usage=self._build_usage(),
            usage_weight=self.usage.weight,
        )
        retriever = self._build_retriever(broker)
        if retriever is not None:
            broker._pipeline = retriever
        return broker

    def _apply_observability(self) -> None:
        """Install the configured tracer, if one was named.

        Process-wide rather than per-broker: spans are opened by free functions
        deep in the retrieval path, and threading a tracer through every one of
        them would put an observability concern in every signature.
        """
        if self.observability.tracer is None:
            return
        from .observability import set_tracer

        set_tracer(
            construct(
                GROUP_TRACERS, self.observability.tracer.name, self.observability.tracer.options
            )
        )

    def _build_usage(self) -> UsageTracker | None:
        """Construct the usage tracker, restoring persisted counts if configured."""
        if not self.usage.enabled:
            return None
        half_life = self.usage.half_life_days * 86400.0
        if self.usage.path:
            return UsageTracker.load(self.usage.path, half_life=half_life)
        return UsageTracker(half_life=half_life)

    def _build_embedder(self) -> Embedder:
        if self.embedder is None:
            from .index.embedders import default_embedder

            return default_embedder()
        embedder: Embedder = create(GROUP_EMBEDDERS, self.embedder.name, **self.embedder.options)
        return embedder

    def _build_store(self, dim: int) -> Store:
        if self.store is None:
            from .store.memory import InMemoryStore

            return InMemoryStore(dim)
        options = dict(self.store.options)
        options.setdefault("dim", dim)
        store: Store = create(GROUP_STORES, self.store.name, **options)
        return store

    def _build_retriever(self, broker: ToolBroker) -> Any:
        from .retrieve.hybrid import HybridRetriever
        from .retrieve.keyword import KeywordRetriever
        from .retrieve.pipeline import RetrievalPipeline
        from .retrieve.semantic import SemanticRetriever

        settings = self.retrieval
        if settings.mode in _BUILT_IN_RETRIEVERS:
            semantic = SemanticRetriever(
                broker.store, broker.embedder, hooks=broker.hooks, min_score=settings.min_score
            )
            if settings.mode == "semantic":
                base: Any = semantic
            elif settings.mode == "keyword":
                base = KeywordRetriever(broker.store)
            else:
                base = HybridRetriever(
                    [semantic, KeywordRetriever(broker.store)],
                    weights=settings.weights,
                    mode=settings.fusion,
                    names=["semantic", "keyword"],
                )
        else:
            # A retriever the core has never heard of. It is handed the store,
            # the embedder and the hook manager if its constructor names them.
            base = construct(
                GROUP_RETRIEVERS,
                settings.mode,
                settings.options,
                store=broker.store,
                embedder=broker.embedder,
                hooks=broker.hooks,
            )

        # Usage boosting is deliberately absent here: it is owned by ToolBroker,
        # because the tracker has to outlive any pipeline that gets swapped in.
        # `set_retriever` re-attaches it around whatever this returns.
        rerankers = [
            construct(
                GROUP_RERANKERS,
                reranker.name,
                reranker.options,
                store=broker.store,
                embedder=broker.embedder,
                hooks=broker.hooks,
            )
            for reranker in settings.rerankers
        ]
        if settings.on_error != "fail":
            base = ResilientRetriever(
                base,
                fallback=self._build_fallback(broker),
                on_error=settings.on_error,
            )
        return RetrievalPipeline(base, rerankers, overfetch=settings.overfetch, hooks=broker.hooks)

    def _build_fallback(self, broker: ToolBroker) -> Any:
        """Construct the degraded-path retriever, if one was named."""
        fallback = self.retrieval.fallback
        if fallback is None:
            return None
        if fallback.name == "keyword":
            from .retrieve.keyword import KeywordRetriever

            return KeywordRetriever(broker.store, **fallback.options)
        return construct(
            GROUP_RETRIEVERS,
            fallback.name,
            fallback.options,
            store=broker.store,
            embedder=broker.embedder,
            hooks=broker.hooks,
        )


def load(path: str | Path) -> ToolBroker:
    """Build a catalogue from a config file."""
    return ToolBrokerConfig.from_file(path).build()


def build_policy(sections: Mapping[str, Mapping[str, Any]]) -> AgentPolicy:
    """Build an :class:`AgentPolicy` from plain dictionaries."""
    return PolicyConfig.model_validate({"agents": sections}).build()


def describe_rules(engine: PolicyEngine) -> Sequence[str]:
    """Return one human-readable line per rule, for the CLI and docs."""
    return [
        *(f"tool rule: {rule.name}" for rule in engine.rules),
        *(f"selection rule: {rule.name}" for rule in engine.selection_rules),
    ]
