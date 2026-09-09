"""The public entry point.

``ToolBroker`` wires the six components together and exposes one method that
matters: :meth:`ToolBroker.select`. Everything it depends on is injectable, and
every default is chosen so that constructing it with no arguments works offline.

    from toolbroker import ToolBroker

    broker = ToolBroker()
    broker.add_functions([search_orders, issue_refund, check_inventory])
    broker.index()

    selection = broker.select("customer wants their money back", k=3)
    print(selection.tool_ids)
    print(selection.explain())

This object never calls a tool and never runs a loop.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from .drift import DriftGuard
from .errors import ConfigurationError, NotIndexedError
from .hooks import Event, HookManager
from .index.embedders import CachedEmbedder, default_embedder
from .index.enrich import EnrichmentConfig
from .index.indexer import Indexer, IndexReport, RefreshReport
from .observability import get_logger, span
from .policy.engine import PolicyEngine
from .protocols import Adapter, Embedder, Policy, Retriever, Source, Store
from .retrieve.pipeline import RetrievalPipeline
from .retrieve.rerank import UsageBooster
from .retrieve.semantic import SemanticRetriever
from .risk import RiskClassifier
from .savings import SavingsTally
from .sources.python_fn import PythonFunctionSource
from .store.memory import InMemoryStore
from .types import Filters, Hit, Selection, Stage, Tool
from .usage import UsageTracker

logger = get_logger("catalog")

DEFAULT_K = 5


class ToolBroker:
    """A searchable, policy-governed catalogue of tools."""

    def __init__(
        self,
        *,
        embedder: Embedder | None = None,
        store: Store | None = None,
        retriever: Retriever | None = None,
        policy: Policy | None = None,
        enrichment: EnrichmentConfig | None = None,
        hooks: HookManager | None = None,
        drift: DriftGuard | None = None,
        savings: SavingsTally | None = None,
        sources: Sequence[Source] = (),
        default_k: int = DEFAULT_K,
        cache_embeddings: bool = True,
        usage: UsageTracker | None = None,
        usage_weight: float = 0.1,
    ) -> None:
        """Build a catalogue.

        Args:
            embedder: Defaults to fastembed if installed, else the offline
                hashing embedder.
            store: Defaults to an in-memory store sized to the embedder.
            retriever: Defaults to semantic retrieval over the store.
            policy: Defaults to allow-everything, so the library is useful
                before anyone writes a rule.
            enrichment: Controls what text gets embedded per tool.
            hooks: Hook manager for interception points.
            drift: Watches for tools changing underneath you between
                refreshes, and optionally holds back unapproved changes.
            savings: Accumulates how much context retrieval avoided sending.
                Every selection is recorded automatically when supplied.
            sources: Sources to register immediately.
            default_k: How many tools :meth:`select` returns when not told.
            cache_embeddings: Wrap the embedder in an in-process cache.
            usage: Enables usage boosting: tools that actually get called rank
                slightly higher. You report calls with :meth:`record_use`,
                because ToolBroker never executes anything and so cannot observe
                them itself. ``None`` disables it entirely.
            usage_weight: Maximum fraction of a hit's own score the boost may
                add. Bounded on purpose — see
                :class:`~toolbroker.retrieve.rerank.UsageBooster`.

        Raises:
            ConfigurationError: If a supplied store and embedder disagree on
                dimensionality.
        """
        base = embedder or default_embedder()
        self._embedder: Embedder = (
            CachedEmbedder(base)
            if cache_embeddings and not isinstance(base, CachedEmbedder)
            else base
        )
        # `is None` rather than `or`: a store defines __len__, so an empty one
        # is falsy and `store or ...` would silently discard it.
        self._store: Store = store if store is not None else InMemoryStore(self._embedder.dim)
        if self._store.dim != self._embedder.dim:
            raise ConfigurationError(
                f"store dimension {self._store.dim} does not match embedder "
                f"dimension {self._embedder.dim}; they must be built together"
            )

        self._hooks = hooks if hooks is not None else HookManager()
        self._enrichment = enrichment if enrichment is not None else EnrichmentConfig()
        self._drift = drift
        self._savings = savings
        self._indexer = Indexer(
            self._embedder,
            self._store,
            enrichment=self._enrichment,
            hooks=self._hooks,
            drift=drift,
        )
        base_retriever = (
            retriever
            if retriever is not None
            else SemanticRetriever(self._store, self._embedder, hooks=self._hooks)
        )
        self._usage = usage
        self._booster = UsageBooster(usage, weight=usage_weight) if usage is not None else None
        rerankers = [self._booster] if self._booster is not None else []
        if isinstance(base_retriever, RetrievalPipeline):
            self._pipeline = (
                base_retriever.with_reranker(self._booster)
                if self._booster is not None
                else base_retriever
            )
        else:
            self._pipeline = RetrievalPipeline(base_retriever, rerankers, hooks=self._hooks)
        self._policy: Policy = policy if policy is not None else PolicyEngine()
        self._sources: list[Source] = list(sources)
        self._default_k = default_k
        self._indexed = False
        self._last_report: IndexReport | None = None
        self._last_refresh: RefreshReport | None = None
        self._succeeded_sources: list[str] = []

    # -- introspection ----------------------------------------------------

    @property
    def store(self) -> Store:
        """The vector store."""
        return self._store

    @property
    def embedder(self) -> Embedder:
        """The embedder in use."""
        return self._embedder

    @property
    def drift(self) -> DriftGuard | None:
        """The drift guard, if one was attached."""
        return self._drift

    @property
    def savings(self) -> SavingsTally | None:
        """The savings tally, if one was attached."""
        return self._savings

    @property
    def hooks(self) -> HookManager:
        """The hook manager. Register handlers on this."""
        return self._hooks

    @property
    def policy(self) -> Policy:
        """The policy in force."""
        return self._policy

    @property
    def pipeline(self) -> RetrievalPipeline:
        """The retrieval pipeline."""
        return self._pipeline

    @property
    def sources(self) -> Sequence[Source]:
        """Registered sources."""
        return tuple(self._sources)

    @property
    def usage(self) -> UsageTracker | None:
        """The usage tracker, if usage boosting is enabled."""
        return self._usage

    @property
    def last_report(self) -> IndexReport | None:
        """The result of the most recent :meth:`index` call."""
        return self._last_report

    def __len__(self) -> int:
        """Number of indexed tools."""
        return len(self._store)

    def __repr__(self) -> str:
        """Show size and wiring."""
        return (
            f"ToolBroker(tools={len(self)}, embedder={self._embedder.id!r}, "
            f"sources={[source.id for source in self._sources]})"
        )

    # -- configuration ----------------------------------------------------

    def set_policy(self, policy: Policy) -> ToolBroker:
        """Replace the policy. Returns self for chaining."""
        self._policy = policy
        return self

    def set_retriever(self, retriever: Retriever) -> ToolBroker:
        """Replace the retrieval stage, keeping usage boosting attached.

        Assigning a pipeline directly would silently drop the usage booster,
        and the symptom — boosting quietly stops working — is invisible.
        """
        if isinstance(retriever, RetrievalPipeline):
            pipeline = retriever
        else:
            pipeline = RetrievalPipeline(retriever, hooks=self._hooks)
        if self._booster is not None and self._booster not in pipeline.rerankers:
            pipeline = pipeline.with_reranker(self._booster)
        self._pipeline = pipeline
        return self

    def add_source(self, source: Source) -> ToolBroker:
        """Register a tool source. Returns self for chaining."""
        self._sources.append(source)
        return self

    def add_functions(
        self,
        functions: Sequence[Callable[..., Any]],
        *,
        namespace: str = "python",
        source_id: str = "python",
        risk_classifier: RiskClassifier | None = None,
    ) -> ToolBroker:
        """Register plain Python callables as tools.

        Args:
            functions: Callables to expose.
            namespace: Namespace the tools land in.
            source_id: Identifier stamped onto every tool.
            risk_classifier: Infer risk from each function's name. Off by
                default, since an explicit ``risk=`` beats a guess for code you
                own; pass :func:`toolbroker.risk.classify_by_name` to opt in.
        """
        return self.add_source(
            PythonFunctionSource(
                functions,
                namespace=namespace,
                source_id=source_id,
                risk_classifier=risk_classifier,
            )
        )

    def add_mcp_server(
        self,
        name: str,
        *,
        command: str | None = None,
        args: Sequence[str] = (),
        url: str | None = None,
        **kwargs: Any,
    ) -> ToolBroker:
        """Register an MCP server, over stdio (``command``) or HTTP (``url``)."""
        from .sources.mcp import MCPServerSpec, MCPSource

        spec = MCPServerSpec(
            name=name,
            transport="http" if url else "stdio",
            command=command,
            args=tuple(args),
            url=url,
            **{key: value for key, value in kwargs.items() if key in _SPEC_FIELDS},
        )
        source_kwargs = {key: value for key, value in kwargs.items() if key not in _SPEC_FIELDS}
        return self.add_source(MCPSource(spec, **source_kwargs))

    # -- indexing ---------------------------------------------------------

    def discover(self) -> list[Tool]:
        """Collect tools from every registered source without indexing them.

        Raises whatever a source raises. :meth:`refresh` uses the tolerant
        variant instead, because a live catalogue should survive one server
        being briefly unreachable.
        """
        tools, failures = self._discover_all(tolerate_failures=False)
        del failures
        return tools

    def _discover_all(self, *, tolerate_failures: bool) -> tuple[list[Tool], list[tuple[str, str]]]:
        """Discover from every source, optionally isolating per-source failures."""
        with span("toolbroker.discover", sources=len(self._sources)):
            self._hooks.emit(Event.BEFORE_DISCOVERY, sources=self._sources)
            tools: list[Tool] = []
            failures: list[tuple[str, str]] = []
            succeeded: list[str] = []

            for source in self._sources:
                try:
                    tools.extend(source.discover())
                except Exception as exc:
                    if not tolerate_failures:
                        raise
                    failures.append((source.id, str(exc)))
                    logger.warning(
                        "source failed during discovery; its tools are left in place",
                        extra={"source_id": source.id, "error": str(exc)},
                    )
                    continue
                succeeded.append(source.id)

            self._succeeded_sources = succeeded
            tools = self._hooks.transform(Event.AFTER_DISCOVERY, tools)
            logger.info(
                "discovery complete",
                extra={
                    "tools": len(tools),
                    "sources": len(self._sources),
                    "failed": len(failures),
                },
            )
            return tools, failures

    def index(self, tools: Sequence[Tool] | None = None, *, replace: bool = True) -> IndexReport:
        """Index ``tools``, or everything the registered sources expose.

        Args:
            tools: Index these instead of running discovery.
            replace: Clear the store first. On by default so a re-index after a
                server drops a tool actually removes it, rather than leaving a
                stale entry the model can still select.
        """
        collected = list(tools) if tools is not None else self.discover()
        report = self._indexer.index(collected, replace=replace)
        self._indexed = True
        self._last_report = report
        return report

    def add_tools(self, tools: Sequence[Tool]) -> IndexReport:
        """Index additional tools without clearing what is already there."""
        report = self._indexer.index(tools, replace=False)
        self._indexed = True
        self._last_report = report
        return report

    def remove_tools(self, tool_ids: Sequence[str]) -> int:
        """Remove tools from the index by id."""
        return self._store.delete(tool_ids)

    def sync_tools(
        self,
        tools: Sequence[Tool],
        *,
        prune: bool = True,
        known_sources: Sequence[str] | None = None,
        failed_sources: Sequence[tuple[str, str]] = (),
    ) -> RefreshReport:
        """Apply an already-discovered catalogue incrementally.

        :meth:`refresh` discovers and then applies. This is the second half on
        its own, for callers who run discovery themselves — the async path does
        it concurrently, and anyone assembling a catalogue by hand needs the
        same seam.

        Args:
            tools: The catalogue as it is now.
            prune: Remove tools that are genuinely gone.
            known_sources: Only prune within these sources. Pass the ones that
                actually answered, or an unreachable server's tools are deleted
                on its behalf.
            failed_sources: Recorded on the report so a caller can alert.
        """
        report = self._indexer.sync(
            tools,
            prune=prune,
            known_sources=known_sources,
            failed_sources=failed_sources,
        )
        self._indexed = True
        self._last_refresh = report
        return report

    def refresh(self, *, prune: bool = True) -> RefreshReport:
        """Re-discover and bring the index in line, touching only what changed.

        This is what a long-running process calls when its MCP servers may have
        gained or lost tools. Unlike :meth:`index`, it embeds only tools whose
        indexed text actually changed, so a server adding one tool costs one
        embedding call rather than re-embedding the catalogue.

        A source that fails is isolated: its tools stay in the index and the
        failure is reported. Deleting them would mean a brief outage silently
        strips capabilities from every agent, with no error anywhere the caller
        would look.

        Args:
            prune: Remove tools that are genuinely gone. Only ever prunes within
                sources that answered successfully.

        Returns:
            A :class:`~toolbroker.index.indexer.RefreshReport`.
        """
        tools, failures = self._discover_all(tolerate_failures=True)
        report = self._indexer.sync(
            tools,
            prune=prune,
            known_sources=self._succeeded_sources,
            failed_sources=failures,
        )
        self._indexed = True
        self._last_refresh = report
        return report

    @property
    def last_refresh(self) -> RefreshReport | None:
        """The result of the most recent :meth:`refresh` call."""
        return self._last_refresh

    # -- selection --------------------------------------------------------

    def select(
        self,
        query: str,
        *,
        k: int | None = None,
        agent: str | None = None,
        scopes: Iterable[str] = (),
        tenant: str | None = None,
        filters: Filters | None = None,
        namespaces: Iterable[str] | None = None,
        tags: Iterable[str] | None = None,
    ) -> Selection:
        """Return the tools that should go into the model's context.

        Args:
            query: What the agent is trying to do. A user turn works well.
            k: How many tools to return. Defaults to the catalogue's setting.
            agent: Agent identity, used to pick the governing policy.
            scopes: Scopes the caller holds, for scope-gated tools.
            tenant: Which tenant this request belongs to. Read by
                :class:`~toolbroker.policy.rules.TenantIsolation`, which denies
                every tenant-owned tool when this is not supplied.
            filters: Pre-retrieval narrowing.
            namespaces: Shorthand for a namespace filter.
            tags: Shorthand for a tags-any filter.

        Returns:
            A :class:`~toolbroker.types.Selection` carrying the tools *and* the
            scores, rule firings, and exclusions behind them.

        Raises:
            NotIndexedError: If nothing has been indexed yet.
        """
        if not self._indexed and len(self._store) == 0:
            raise NotIndexedError(
                "no tools indexed; call index() after registering at least one source"
            )

        limit = k if k is not None else self._default_k
        effective = self._merge_filters(filters, namespaces, tags)
        timings: dict[str, float] = {}

        with span("toolbroker.select", agent=agent, k=limit):
            started = time.perf_counter()
            # Overfetch relative to k: policy will remove some candidates, and
            # a caller who asked for 5 tools should still get 5 after a couple
            # are denied.
            candidates = self._pipeline.retrieve(query, self._fetch_size(limit), effective)
            timings["retrieval"] = (time.perf_counter() - started) * 1000

            started = time.perf_counter()
            self._hooks.emit(Event.BEFORE_POLICY, hits=candidates, agent=agent)
            result = self._policy.evaluate(
                candidates,
                agent=agent,
                scopes=frozenset(scopes),
                query=query,
                tenant=tenant,
            )
            self._hooks.emit(Event.AFTER_POLICY, result=result, agent=agent)
            timings["policy"] = (time.perf_counter() - started) * 1000

            selected = list(result.hits)[:limit]
            exclusions = list(result.exclusions)
            kept = {hit.id for hit in selected}
            exclusions.extend(self._truncation_exclusions(result.hits, kept, limit))

            selection = Selection(
                query=query,
                hits=tuple(selected),
                agent=agent,
                requested_k=limit,
                considered=len(self._store),
                firings=result.firings,
                exclusions=tuple(exclusions),
                timings_ms=timings,
                dry_run=getattr(self._policy, "dry_run", False),
            )
            final: Selection = self._hooks.transform(Event.AFTER_SELECTION, selection)
            if self._savings is not None:
                # Recorded here rather than by the caller: select() is the only
                # place that knows both what was sent and what could have been,
                # and asking every call site to remember guarantees some forget.
                self._savings.record(final, self.tools())
            logger.debug(
                "selection complete",
                extra={
                    "agent": agent,
                    "query_length": len(query),
                    "selected": len(final),
                    "considered": final.considered,
                },
            )
            return final

    def _fetch_size(self, k: int) -> int:
        """How many candidates to retrieve so policy has room to deny some."""
        return min(max(k * 4, k + 10), max(len(self._store), k))

    @staticmethod
    def _truncation_exclusions(hits: Sequence[Hit], kept: set[str], limit: int) -> list[Any]:
        """Record hits dropped purely because ``k`` was reached."""
        from .types import Exclusion

        return [
            Exclusion(
                tool_id=hit.id,
                stage=Stage.RETRIEVAL,
                reason=f"ranked below the top {limit}",
            )
            for hit in hits
            if hit.id not in kept
        ]

    @staticmethod
    def _merge_filters(
        filters: Filters | None,
        namespaces: Iterable[str] | None,
        tags: Iterable[str] | None,
    ) -> Filters | None:
        """Fold the shorthand arguments into an explicit filter object."""
        if namespaces is None and tags is None:
            return filters
        updates: dict[str, Any] = {}
        if namespaces is not None:
            updates["namespaces"] = frozenset(namespaces)
        if tags is not None:
            updates["tags_any"] = frozenset(tags)
        base = filters or Filters()
        return base.model_copy(update=updates)

    # -- feedback ---------------------------------------------------------

    def record_use(self, tool_id: str, count: float = 1.0) -> None:
        """Report that ``tool_id`` was actually called.

        ToolBroker never executes a tool, so it cannot observe this. Call it from
        wherever your framework runs the loop:

            selection = broker.select(query, k=5)
            # ... the model picks and calls one ...
            broker.record_use("billing/issue_refund")

        A no-op when usage boosting is not enabled, so instrumenting call sites
        is safe before deciding whether to turn it on.
        """
        if self._usage is not None:
            self._usage.record(tool_id, count)

    def record_uses(self, tool_ids: Iterable[str]) -> None:
        """Report one use of each id."""
        for tool_id in tool_ids:
            self.record_use(tool_id)

    # -- rendering --------------------------------------------------------

    def render(self, adapter: str | Adapter, selection: Selection) -> Any:
        """Render a selection into a framework's native tool shape."""
        resolved = self._resolve_adapter(adapter)
        return resolved.render(selection.tools)

    def select_for(
        self, adapter: str | Adapter, query: str, **kwargs: Any
    ) -> tuple[Any, Selection]:
        """Select and render in one call.

        Returns both the rendered tools and the selection, because the
        explanation is the part you want when the model picks the wrong one.
        """
        selection = self.select(query, **kwargs)
        return self.render(adapter, selection), selection

    def openai_tools(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Convenience: OpenAI-shaped tools for ``query``."""
        rendered, _ = self.select_for("openai", query, **kwargs)
        return list(rendered)

    def anthropic_tools(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Convenience: Anthropic-shaped tools for ``query``."""
        rendered, _ = self.select_for("anthropic", query, **kwargs)
        return list(rendered)

    @staticmethod
    def _resolve_adapter(adapter: str | Adapter) -> Adapter:
        """Turn a name into an adapter instance."""
        if isinstance(adapter, str):
            from .adapters.resolve import get_adapter

            return get_adapter(adapter)
        return adapter

    # -- diagnostics ------------------------------------------------------

    def tools(self) -> list[Tool]:
        """Every indexed tool."""
        return [record.tool for record in self._store.all_records()]

    def get(self, tool_id: str) -> Tool | None:
        """Look up one indexed tool by id."""
        record = self._store.get(tool_id)
        return record.tool if record else None

    def stats(self) -> Mapping[str, Any]:
        """Counts useful for a health endpoint or the CLI."""
        tools = self.tools()
        namespaces: dict[str, int] = {}
        risks: dict[str, int] = {}
        for tool in tools:
            namespaces[tool.namespace] = namespaces.get(tool.namespace, 0) + 1
            risks[tool.risk.value] = risks.get(tool.risk.value, 0) + 1
        return {
            "tools": len(tools),
            "sources": len(self._sources),
            "embedder": self._embedder.id,
            "dimension": self._embedder.dim,
            "namespaces": dict(sorted(namespaces.items())),
            "risk_tiers": dict(sorted(risks.items())),
            "indexed": self._indexed,
        }


_SPEC_FIELDS = frozenset({"env", "headers", "timeout_seconds", "transport"})
