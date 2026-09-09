# Concepts

## The pipeline

```
 Sources ──▶ Index ──▶ Store          Policy
 (MCP,       (enrich,  (vectors)        │
  OpenAPI,    embed)       │            │
  Python fns,              ▼            ▼
  static JSON)         Retriever ──▶ Filtered set ──▶ Adapter ──▶ framework tools
                       (semantic/
                        hybrid/rerank)
```

Six interfaces, all in `toolbroker.protocols`. Each is a `typing.Protocol`, so you
satisfy one with any object that has the right methods — including one that already
exists in your codebase.

| Interface | Responsibility | Default |
|---|---|---|
| `Source` | Where tool definitions come from | Python functions, static JSON, MCP, OpenAPI |
| `Embedder` | Text to vectors | `HashingEmbedder`, or fastembed if installed |
| `Store` | Holds records, answers nearest-neighbour queries | `InMemoryStore` |
| `Retriever` | Query to ranked candidates | `SemanticRetriever` |
| `Policy` | Which candidates an agent may see | `PolicyEngine` (allow everything) |
| `Adapter` | Tools to a framework's shape | OpenAI, Anthropic, LangGraph |

## Tools

A `Tool` is immutable data. It carries what is needed to rank, filter, and render —
never to execute:

```python
Tool(
    name="issue_refund",
    namespace="billing",  # id becomes "billing/issue_refund"
    description="Refund a customer payment",
    input_schema={...},  # JSON Schema
    tags=frozenset({"write"}),
    risk=RiskTier.MEDIUM,  # LOW < MEDIUM < HIGH < CRITICAL
    cost=CostTier.LOW,
    required_scopes=frozenset({"payments:write"}),
)
```

`namespace` exists so two MCP servers can both expose `search` without colliding.

## Enrichment: what actually gets embedded

A raw MCP description is often one terse line — not enough signal to separate 500
tools. Before embedding, ToolBroker folds in the name (split into words), the namespace,
the tags, the parameter names and their descriptions, and any examples:

```python
from toolbroker.index.enrich import build_index_text

print(build_index_text(tool))
```

Field weighting is done by **repetition**, not vector arithmetic. That works with every
embedding model, and you can read the exact string that was embedded.

Tune it with `EnrichmentConfig(name_weight=3, include_examples=False, ...)`.

## Retrieval

- **Semantic** (default) — dense vectors.
- **Keyword** — BM25. Catches exact identifiers: someone typing `stripe_refund` wants
  that tool, not the five things semantically near it.
- **Hybrid** — both, fused. Default fusion is reciprocal rank fusion, which combines
  *ranks* rather than scores. BM25 scores are unbounded and cosine scores are not, so
  comparing them numerically is meaningless; comparing their ranks is not.

Rerankers run after retrieval on a shortlist: `UsageBooster` (tools that actually get
called), `LLMReranker` (you supply the scoring callable — ToolBroker ships no LLM client
and reads no API key).

```python
from toolbroker.retrieve import (
    HybridRetriever,
    KeywordRetriever,
    RetrievalPipeline,
    SemanticRetriever,
)

pipeline = RetrievalPipeline(
    HybridRetriever([SemanticRetriever(store, embedder), KeywordRetriever(store)]),
    [UsageBooster(weight=0.1)],
    overfetch=4.0,
)
```

`overfetch` matters: without it a reranker can only reorder what the retriever already
ranked highly, which defeats the point.

## Filters vs. policy

Both remove tools, for different reasons.

**Filters** narrow the search *before* scoring, so `k` stays meaningful — asking for 5
billing tools returns 5, not whatever survives filtering the global top 5.

```python
broker.select("refund", k=5, namespaces=["billing"], tags=["write"])
```

**Policy** decides what an agent is *allowed* to see, after retrieval, and records why.
Use filters for relevance, policy for permission. See [Policy](policy.md).

## Selections are explanations

`select()` returns a `Selection`, never a bare list:

```python
selection.tools  # the tools, best first
selection.hits  # with scores and per-signal components
selection.firings  # which policy rules fired, on what, and why
selection.exclusions  # what was dropped, at which stage, and why
selection.timings_ms  # where the time went
selection.explain()  # all of the above, readable
```

A policy that can only say "denied" is not auditable, and a developer who cannot see
why their tool vanished will work around the policy rather than with it.

## Keeping the catalogue current

A process that indexes once at startup serves a stale catalogue forever. MCP servers
gain and lose tools; an OpenAPI spec gets redeployed. `refresh()` re-discovers and
applies only what changed:

```python
report = broker.refresh()
report.summary()  # '+1 -0 ~1 =38 embedded 1'
```

`index()` rebuilds. `refresh()` diffs, and the difference is what it costs:

| what changed | re-embedded? | rewritten? |
|---|---|---|
| new tool | yes | yes |
| description or schema changed | yes | yes |
| only risk, scopes, or tags changed | **no** | yes |
| nothing | no | no |

The third row matters. Policy reads `risk` and `required_scopes`, so a tool whose risk
tier changed must be rewritten — but its indexed text did not change, so the vector is
still correct and no embedding call is needed. With a paid embedder that is the whole
point.

### A source that fails keeps its tools

If an MCP server is unreachable during a refresh, its tools are **not** removed:

```python
report = broker.refresh()
report.failed_sources  # (('mcp:github', 'connection refused'),)
report.removed  # () -- nothing was pruned from the failed source
```

Deleting them would mean a thirty-second outage silently strips capabilities from every
agent, with no error anywhere a caller would look. Pruning only ever happens within
sources that answered.

### In a long-running process

```python
from toolbroker import PeriodicRefresher

with PeriodicRefresher(broker, interval=300, on_refresh=my_metrics) as refresher:
    ...
```

Polling rather than push, deliberately: it needs no protocol support, survives a server
that never sends notifications, and costs almost nothing because the sync only embeds
what changed. The thread is a daemon, a failing refresh is logged and retried rather
than killing the loop, and `stop()` interrupts the wait instead of waiting out the
interval.

## Learning from what gets called

Retrieval sees a description. Usage sees reality: the tool whose description reads best
is not always the one that works, and a catalogue's users discover that long before its
author does.

ToolBroker never executes a tool, so it cannot observe calls. You report them:

```python
from toolbroker import ToolBroker, UsageTracker

broker = ToolBroker(usage=UsageTracker(), usage_weight=0.1)

selection = broker.select("look up a customer", k=5)
# ... your framework runs the loop, the model calls one tool ...
broker.record_use("crm/find_customer_records")
```

`record_use` is a no-op when usage boosting is off, so instrumenting call sites is safe
before you decide whether to enable it.

### Two properties that keep it from doing harm

**The boost is bounded.** It adds at most `usage_weight` of a hit's own score, which
guarantees two tools more than that far apart in relative score cannot swap places,
however lopsided their usage. Without that bound, usage boosting is a feedback loop —
popular tools rank higher, get called more, rank higher still — that entrenches whatever
was popular first and starves every tool added afterwards.

**Counts decay.** Exponentially, on a half-life (thirty days by default), applied lazily
on read so there is no sweep for anyone to forget to schedule. A tool that was heavily
used last quarter and abandoned since stops dominating on its own.

### Is your usage signal good enough?

```bash
uv run python bench/usage.py                        # synthetic signals
uv run python bench/usage.py --usage my_usage.json  # your own counts
```

Boosting is a bet on your feedback. Measured on the reference catalogue, baseline
recall@5 = 0.702:

| signal | weight | recall@5 | delta |
|---|---|---|---|
| informed *(usage tracks the labels)* | 0.05 | 0.739 | +0.037 |
| informed | **0.10** | **0.757** | **+0.056** |
| informed | 0.20 | 0.752 | +0.051 |
| informed | 0.40 | 0.751 | +0.049 |
| random *(usage unrelated to relevance)* | 0.05 | 0.689 | −0.013 |
| random | **0.10** | 0.649 | **−0.053** |
| random | 0.20 | 0.540 | −0.161 |
| random | 0.40 | 0.322 | −0.380 |

The asymmetry is the point, and it is why `usage_weight` defaults to **0.1**. The gain
from a good signal *saturates* around 0.1 — going to 0.4 buys nothing. The damage from a
bad signal keeps growing: −0.05 at 0.1, −0.38 at 0.4. A higher weight is all downside.

So: record the tool that actually **worked**, not every tool the model tried. If you
cannot tell the difference, leave boosting off — no signal beats a misleading one.

### Persistence

```python
tracker = UsageTracker.load("usage.json")  # empty if absent or corrupt
broker = ToolBroker(usage=tracker)
...
tracker.save("usage.json")
```

Timestamps are stored alongside the counts, so decay survives a restart: a process that
was down for a month comes back with month-old counts, not fresh ones. Writes are
atomic, and an unreadable file yields an empty tracker rather than stopping the service —
a corrupt usage file should cost ranking quality, not availability.

In YAML:

```yaml
usage:
  enabled: true
  weight: 0.1
  half_life_days: 30
  path: ./usage.json
```

## Tools nothing can reach

A catalogue can be correctly configured and still contain tools no agent is ever handed.
Nothing fails and no error is raised — the tool is simply never selected, which is the
hardest kind of problem to notice.

```bash
toolbroker doctor -c toolbroker.yaml
```

It asks the catalogue about itself, so it needs no labelled data:

* **Shadowed** — running a tool's own description as a query does not return it in the
  top `k`. That description is about the best query anyone could write for it, so if it
  loses there, a real user's rougher phrasing has no chance. Tools sharing a description
  are queried once between them, so the cost is one embedding per *distinct* description.
* **Thin** — the description is empty or too short to carry any signal.
* **Closest pairs** — the tightest neighbours in the catalogue, ranked. When two tools
  sit close enough, which one a query gets is close to arbitrary, and no ranking change
  fixes it; the descriptions have to say what actually differs. Pairs sharing a
  byte-identical description are marked, because those are a certainty rather than a
  suspicion.

### Why pairs are ranked and not flagged

There is no default margin, for the same reason there is no default `MinScore`. Tools
with identical descriptions sit 0.02–0.06 apart under bge-small and 0.07–0.20 apart under
the hashing embedder — the same defect, three times the number. Normalising by the
catalogue's own spread does not rescue it either: a catalogue that is 96% duplicates and
one with no duplicates at all produce nearly the same distribution shape, so nothing
internal to the catalogue tells you which you have.

So the report ranks pairs and lets you read down until they stop looking interchangeable.
That boundary is your embedder's margin, and passing it as `--margin` turns the ranking
into a verdict.

### What it cannot tell you

It reads the catalogue against itself, so it finds tools that collide or carry no signal.
It cannot find a tool whose description is unique, clear, and phrased unlike anything a
user would ever type — which on this project's own benchmark is the *largest* category of
retrieval failure. `bounce the pods` never reaches `restart_service` at any depth, and no
property of the catalogue predicts that.

Only real queries expose it, which is what `--samples` is for:

```bash
toolbroker doctor -c toolbroker.yaml --samples last_week_of_prompts.txt
```

That adds the tools none of your traffic reaches. If the sample is representative, those
tools are paying for a place in the catalogue they never earn.

## Running it for real

Four things that only a component sitting in the middle, across time, can do.

### When a tool changes underneath you

The prompt-injection risk people discuss is a hostile description on the day you connect a
server. The one that gets you is a description that changes *later*, on a server you
already reviewed. Nothing else in the stack compares the catalogue to its previous state —
ToolBroker does, on every refresh, to decide what to re-embed.

```python
guard = DriftGuard(quarantine=True, path="approved.json")
broker = ToolBroker(drift=guard)

report = broker.refresh()
for change in report.changes:
    alert(change.describe())
```

With `quarantine=True` a tool whose description or schema changed keeps serving its **last
approved** version until a human accepts it. The agent keeps working, on text somebody
signed off, and the change stays pending rather than becoming the new baseline on the next
refresh.

Privilege changes — risk tier, required scopes — are reported but **never** held back.
Serving a *lower* risk tier than the server now claims is the dangerous direction.

### What a policy change actually grants

Nobody can review `deny: ["*/delete_*"] → ["*/delete_user"]` by reading it. Policy is
deterministic over a known catalogue, so the answer is static:

```bash
toolbroker diff main.yaml pr.yaml --fail-on-high-risk
```

```
Policy change against 4 tools:

agent `support`
  ⚠ + billing/delete_order  (risk: high)

⚠ This change grants at least one HIGH risk tool.
```

`MaxTools` is reported separately rather than folded into reach — it bounds one response,
not the catalogue. `MinScore` is reported as **undecidable**, because it depends on a score
that only exists once there is a query; quietly ignoring it would overstate what an agent
can do.

### What retrieval saved, on your traffic

```python
tally = SavingsTally(price_per_million=3.0)
tally.record(selection, broker.tools())
print(tally.summary())
```

The report states its counterfactual — every tool on every turn, which is what an agent
with one tool list actually does — and marks itself *estimated* unless every entry carried
a real provider token count. A figure that silently mixes measured and estimated numbers is
not auditable.

### Learning how people actually ask

The largest measured failure mode is vocabulary, not ranking: `bounce the pods` never
reaches `restart_service` at any depth. The broker sees the pair nothing else does — the
query a user typed, and the tool the framework says was called.

```python
aliases = AliasLearner()
broker.hooks.register(Event.TRANSFORM_INDEX_TEXT, aliases.enrich)
aliases.record("bounce the pods", "infra/restart_service")  # confirmed call only
```

Three bounds keep it from becoming a feedback loop: only **confirmed calls** are learned,
never the tool that merely ranked first; a phrase must recur `min_count` times before it
enters the index; and each tool contributes at most `max_aliases` phrases, so no tool wins
by accumulating text. Counts decay, so last year's vocabulary fades.

### When the store is down

```python
broker.set_retriever(
    ResilientRetriever(primary, fallback=KeywordRetriever(store), on_error="fallback")
)
```

`fail` raises, `empty` returns nothing deliberately so the caller can say "temporarily
unavailable", and `fallback` tries a second retriever — a lexical one needs no external
service, so it survives the outage that took vector search down. Check `.degraded` and
alert on it: a silently degraded retriever answers every request with worse tools and
nothing ever fails.

**A degraded path returns fewer or worse tools. It never returns tools policy would have
refused.** The wrapper sits inside retrieval, upstream of the policy engine, so every
result takes the same path — and a test asserts it, because "graceful degradation" that
widens access is a privilege-escalation bug wearing a resilience costume.

## Determinism

Ties break by tool id, never by insertion order or dict iteration, so results are
stable across runs, Python versions, and store backends. The `HashingEmbedder` uses
blake2b rather than `hash()`, whose seed is randomised per process — identical text
must produce identical vectors in a restarted process, or a re-index would silently
rank differently than the process that built it.
