# Changelog

Notable changes to ToolBroker. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and from 1.0 the project
follows [semantic versioning](docs/stability.md) strictly.

## [Unreleased]

Pre-1.0. The public API is not frozen; see [API stability](docs/stability.md).

### Added

- Retrieval and policy core: six swappable protocols, in-memory store, semantic /
  BM25 / hybrid retrieval, 10 policy rule types, per-agent routing, dry-run mode.
- Sources: Python functions, static JSON/JSONL, MCP (stdio and HTTP), OpenAPI 3.x.
- Stores: in-memory, Qdrant, Chroma, PostgreSQL/pgvector.
- Embedders: offline hashing, fastembed, Ollama, OpenAI and Azure OpenAI.
- Adapters: OpenAI and Anthropic JSON, LangGraph, OpenAI Agents SDK, Claude Agent SDK,
  CrewAI.
- MCP proxy mode — serve a whole catalogue as one MCP server.
- `toolbroker calibrate` — derive a `MinScore` floor from the catalogue, no labels needed.
- `toolbroker doctor` / `diagnose_catalogue()` — find tools retrieval can never surface:
  shadowed (a tool's own description does not return it), thin or empty descriptions, and
  the catalogue's closest pairs ranked. Ships no default twin margin, because the gap
  between byte-identical descriptions is 0.02–0.06 under bge-small and 0.07–0.20 under the
  hashing embedder, and no catalogue-internal statistic distinguishes a catalogue full of
  duplicates from one with none. Pass `--samples` to add the tools your real traffic never
  reaches.
- `broker.refresh()` and `PeriodicRefresher` — incremental re-indexing that embeds only
  what changed, and never deletes tools belonging to a source that failed.
- Usage boosting: `UsageTracker` with exponential decay and durable counts, bounded so a
  popular tool cannot displace a clearly more relevant one.
- `toolbroker.testing` — executable conformance suites for stores, adapters, embedders,
  retrievers, rerankers and tracers. Every extension point with a registry group now has
  one, and each suite is verified against deliberate violations rather than only against
  implementations that already pass.
- Benchmark suite: 40 hand-authored tools, 290 hand-written labelled queries across eight
  categories, Wilson confidence intervals, per-category scoring, and harnesses for
  end-to-end accuracy, enrichment ablation, score-floor sweeps, and usage signal quality.
- `bench/ceiling.py` — measures recall@N against fetch depth, so the headroom available
  to a reranker is a number rather than a hope, and reports the share of queries whose
  tool never enters the pool at any depth.

- Every Protocol now has a registry group, so anything can be named from a config file by
  an installed entry point *or* a plain dotted path — no publishing, no registration, no
  fork. Added `toolbroker.rerankers` and `toolbroker.tracers`.
- **Async entry points** (`toolbroker.aio`). `aselect`, `aselect_for`, `adiscover`,
  `aindex`, `arefresh`. Agent frameworks are overwhelmingly async and the hot path was not.
  `aselect` runs the same `select` off the event loop rather than reimplementing it, so an
  async caller cannot drift into different rules from a sync one; discovery fans out across
  sources concurrently, bounded, preferring a source's native `adiscover`.
- **Conversation context** (`toolbroker.conversation`). Selecting on the last user message
  alone is the known weakness of this approach — "check stock there" names nothing. Folding
  one turn of history into the query is worth **+0.13 to +0.23** recall@5 on a new
  multi-turn benchmark, and the gain grows with the catalogue.
- `ToolBroker.sync_tools()` — apply an already-discovered catalogue incrementally, for
  callers running discovery themselves.
- **Tenant isolation** (`TenantIsolation`, `select(tenant=...)`). Scopes could not do this:
  `RequireScopes` abstains for a tool declaring none, so an untagged tool stayed visible to
  every tenant. This rule decides on every tool and denies whenever it is not certain — a
  request that declares no tenant sees nothing tenant-owned, rather than everything.
- Every operational feature is configurable: `drift`, `aliases`, `savings`,
  `retrieval.on_error` / `retrieval.fallback`, and per-agent `tenant_isolation`. All off by
  default. A `SavingsTally` attached to a broker records every selection itself, so no call
  site has to remember.
- **Tool drift detection** (`DriftGuard`). The prompt-injection risk that matters is not
  a hostile description on day one, it is one that changes *later* on a server you already
  approved. The refresh diff already computed this to decide what to re-embed; it is now a
  security control. With `quarantine=True` a tool whose description or schema changed keeps
  serving its **last approved** version until a human accepts the new one. Privilege changes
  (risk tier, scopes) are reported but never held back, because serving a *lower* risk tier
  than the server now claims is the dangerous direction.
- **Policy reachability and diff** (`toolbroker diff`, `diff_policies`). Nobody can review
  `*/delete_*` → `*/delete_user` by reading it; they need to know what it grants. Policy is
  deterministic over a known catalogue, so the answer is static and belongs in CI.
  `--fail-on-high-risk` is the gate. `MaxTools` is reported separately rather than folded
  into reach (it bounds a response, not the catalogue) and `MinScore` is reported as
  undecidable rather than silently ignored.
- **Savings accounting** (`SavingsTally`). The 193x claim, on your traffic instead of our
  benchmark. States its counterfactual, and marks the report *estimated* unless every entry
  carried a real provider token count — a figure that silently mixes the two is not
  auditable.
- **Learned aliases** (`AliasLearner`). Attacks the largest measured failure mode: 8.3% of
  benchmark queries never reach their tool at any depth because of vocabulary, not ranking.
  Records confirmed query-to-call pairs and feeds the confident ones into the index. Three
  bounds against the feedback loop: only confirmed calls are learned, a phrase must recur
  `min_count` times before it is indexed, and each tool contributes at most `max_aliases`
  phrases.
- **Configurable degradation** (`ResilientRetriever`). There was no store-failure path at
  all: a Qdrant restart meant the agent got zero tools and no error. Now `fail`, `empty` or
  `fallback` — a lexical fallback needs no external service, so it survives the outage that
  took vector search down. The degraded path is filtered by exactly the same policy, and a
  test asserts it.
- **Environment substitution in config files.** `${VAR}` and `${VAR:-default}`, so a
  config with a Postgres DSN and an API key in it can still be committed. A missing
  required variable raises before validation, naming the variable and its path in the
  file, instead of handing a driver the literal string `${VAR}`.
- **Hooks attachable from config**, by dotted path, with `options` turning the target
  into a factory. The interesting handlers — audit sinks, query normalisers — belong to
  the deployment rather than to each service that embeds the broker.
- `Tracer` protocol plus `set_tracer()`/`get_tracer()`: OpenTelemetry is now one
  implementation rather than the only one, so Datadog, Sentry or an in-house collector
  plug in the same way a store does. A tracer that raises is swapped for the no-op and
  logged, never propagated — observability is not worth an outage.

### Changed

- **Renamed from ToolScope to ToolBroker.** `toolscope` was already taken on PyPI by a
  project doing the same thing — [ilya-kolchinsky/toolscope](https://github.com/ilya-kolchinsky/toolscope),
  Red Hat, Apache-2.0. Nothing had been published under the old name, so there is no
  migration path to provide: `ToolScope` is now `ToolBroker`, `toolscope-*` plugins are
  `toolbroker-*`, entry-point groups are `toolbroker.*`, and environment variables are
  `TOOLBROKER_*`.

- `retrieval.mode` was `Literal["semantic", "keyword", "hybrid"]` and is now any plugin
  name or dotted path; the three built-ins are unchanged. `retrieval.rerankers` is new,
  and replaces a hardcoded empty list. Retrievers, rerankers and tracers are handed the
  store, embedder and hook manager **only if their constructor names them**, so a plugin
  never has to accept collaborators it does not want.
- `EnrichmentConfig.name_weight` default 2 → 1. Measured: better for both the lexical and
  the semantic embedder at all three catalogue sizes. See
  [benchmarking](docs/benchmarking.md#tuning-with-evidence).

### Security

Eight policy bypasses found and fixed in pre-release review. None were shipped. Full
detail in [SECURITY.md](SECURITY.md).

- Deny and allow globs matched case-sensitively, so `*/delete_*` did not match
  `DeleteUser`.
- Names and namespaces accepted surrounding whitespace, so `"admin "` evaded `admin/*`.
- `AgentPolicy` fell back to a permissive engine for an unregistered agent name, so a
  typo granted the whole catalogue. Added `strict_agents`.
- The MCP proxy's `describe_tool` ignored policy, disclosing descriptions and full
  parameter schemas for tools a client could not call.
- Proxy error messages named denied tools, making them an enumeration oracle.
- `first_match` combining short-circuited `TenantIsolation`: an `allow` glob earlier in the
  rule list granted another tenant's tool before the boundary ran. Isolation rules are now
  `mandatory` — always evaluated, deny always final, whatever the combining mode.
- Drift detection ignored tag changes, though `DenyTags` and `RequireTags` gate on them, so
  a server dropping `destructive` walked through a rule silently.
- A renamed tool arrived as an addition rather than a change, walking past quarantine.
  `trust_on_first_use=False` now holds new arrivals instead of merely declining to approve
  them.

### Fixed

- `PgVectorStore` registered the pgvector type adapter before creating the extension, so
  it could not bootstrap a database that did not already have it — the first-run case.
- Negative benchmark cases that returned tools were scored with a perfect reciprocal
  rank.
- `bench/e2e.py` reported a retrieval miss — a case where no correct tool reached the
  model — as `expected []`, which counted correctly but made the two causes of failure
  indistinguishable in the report. Retrieval misses are now counted and named, and the
  table carries an `accuracy if retrieved` column separating what retrieval lost from
  what the caller lost.
- `config.build()` assigned the retrieval pipeline directly, silently discarding the
  usage booster. Replaced with `ToolBroker.set_retriever()`.
- `ToolBroker` ignored a store passed in empty, because `store or default` is falsy for a
  store defining `__len__`.
- Source discovery was lazy, deferring connection errors to whenever a caller happened to
  iterate.
