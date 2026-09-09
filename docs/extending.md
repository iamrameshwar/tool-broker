# Extending ToolBroker

Every extension point is a `typing.Protocol` with a default implementation and a plugin
entry point. Built-in components register through exactly the same mechanism you will
use, so "write your own" is a real instruction rather than an aspiration.

## Secrets stay out of the file

A config file is meant to be committed — that is the argument for having one, since agent
permissions should be reviewed like any other change. A real deployment's config also
holds a DSN and an API key, and those cannot be. So values come from the environment:

```yaml
store:
  name: pgvector
  options:
    dsn: ${TOOLBROKER_PG_DSN}                 # required
    table: ${TOOLBROKER_TABLE:-tools}         # with a default
sources:
  - {type: mcp, name: github, url: "${GITHUB_MCP_URL:-http://localhost:8080}"}
```

`${VAR}` is required and raises if unset, naming the variable *and* where in the file it
appeared — rather than quietly handing your database driver the literal string `${VAR}`.
`${VAR:-default}` falls back, using the shell syntax people already know, and an empty
value counts as set exactly as it does in the shell. `$${...}` escapes to a literal.

Substitution walks the parsed structure, not the raw text, so it cannot corrupt YAML and
a value containing a colon or a newline is still safe.

## Resolution order

For any component name:

1. an object registered at runtime with `toolbroker.registry.register(...)`
2. an installed entry point in the matching group
3. a dotted import path — `my_pkg.module:ClassName`

```python
from toolbroker.registry import GROUP_STORES, available, register

register(GROUP_STORES, "mystore", MyStore)
available(GROUP_STORES)  # ['memory', 'mystore']
```

The third route is the one that matters if you installed the package rather than cloning
it: **you do not have to publish anything, or register anything, to use your own
component.** A dotted path in a config file is enough.

```yaml
store:
  name: my_pkg.storage:PineconeStore
  options: {index: agent-tools, namespace: prod}
embedder:
  name: my_pkg.models:InternalEmbedder
  options: {endpoint: https://models.internal/v1}
retrieval:
  mode: my_pkg.retrieval:GraphRetriever
  rerankers:
    - {name: my_pkg.retrieval:CrossEncoder, options: {model: bge-reranker-base}}
observability:
  tracer: {name: my_pkg.tracing:DatadogTracer, options: {service: agent-gateway}}
```

The operational features are configured the same way, and every one is off by default:

```yaml
drift:
  enabled: true
  quarantine: true              # serve the last approved version on a change
  path: approved.json
aliases:
  enabled: true
  min_count: 3                  # a phrase must recur before it is indexed
  path: aliases.json
savings:
  enabled: true                 # every selection is recorded automatically
  price_per_million: 3.0
retrieval:
  on_error: fallback            # fail | empty | fallback
  fallback: {name: keyword}     # needs no external service
policy:
  agents:
    tenant_app: {tenant_isolation: true, shared_tools: false}
```

### Groups

| Group | Protocol | Built in |
|---|---|---|
| `toolbroker.embedders` | `Embedder` | `hashing`, `fastembed` |
| `toolbroker.stores` | `Store` | `memory` |
| `toolbroker.retrievers` | `Retriever` | `semantic`, `keyword`, `hybrid` |
| `toolbroker.rerankers` | `Reranker` | `usage`, `llm` |
| `toolbroker.adapters` | `Adapter` | `openai`, `anthropic` |
| `toolbroker.sources` | `Source` | `python`, `json`, `mcp`, `openapi` |
| `toolbroker.tracers` | `Tracer` | `otel`, `null` |
| `toolbroker.hooks` | any callable | — |

### What your constructor is handed

Retrievers, rerankers and tracers need collaborators a YAML file cannot express — a
semantic retriever wants the store *and* the embedder, a lexical one only the store, a
third-party one may want neither. Rather than force every plugin to accept every
collaborator, ToolBroker passes each one **only if your constructor names it**, or all of
them if you take `**kwargs`:

```python
class GraphRetriever:
    def __init__(self, store, *, hops: int = 2):  # gets `store`, not `embedder`
        ...
```

Anything under `options` is always passed and **wins over an injected value**, so you can
override a collaborator we would otherwise supply.

## Write your own Store

```python
from collections.abc import Sequence
from toolbroker.types import Filters, Hit, ToolRecord


class MyStore:
    def __init__(self, dim: int) -> None:
        self._dim = dim

    @property
    def dim(self) -> int: ...
    def upsert(self, records: Sequence[ToolRecord]) -> None: ...
    def delete(self, tool_ids: Sequence[str]) -> int: ...
    def search(self, vector, k: int, filters: Filters | None = None) -> list[Hit]: ...
    def get(self, tool_id: str) -> ToolRecord | None: ...
    def all_records(self) -> Sequence[ToolRecord]: ...
    def clear(self) -> None: ...
    def __len__(self) -> int: ...
```

Two contract details that are easy to get wrong:

- **Apply filters before scoring**, not after, so `k` stays meaningful.
- **Break ties deterministically** by tool id. `toolbroker.determinism.stable_sort` does
  this for you.

Then prove it:

```python
from toolbroker.testing import StoreConformanceSuite


class TestMyStore(StoreConformanceSuite):
    @pytest.fixture
    def store(self):
        return MyStore(dim=self.DIM)
```

Twenty-odd behaviours, verified. If the suite is missing a guarantee your store had to
make, that is a gap in the contract — please open an issue.

## Write your own Embedder

```python
class MyEmbedder:
    @property
    def dim(self) -> int: ...
    @property
    def id(self) -> str: ...  # include the model name; it is a cache key
    def embed(self, texts) -> list[list[float]]: ...
    def embed_query(self, text) -> list[float]: ...
```

`embed_query` is separate from `embed` because asymmetric models prefix queries and
documents differently. If yours is symmetric, delegate.

Wrap it in `CachedEmbedder` to avoid recomputing across re-indexes — essential for a
paid API, where every re-index is a bill.

Then prove it:

```python
from toolbroker.testing import EmbedderConformanceSuite


class TestMyEmbedder(EmbedderConformanceSuite):
    @pytest.fixture
    def embedder(self):
        return MyEmbedder()
```

Fourteen behaviours, including the ones that are easy to get wrong: results line up
positionally with inputs, a text embeds the same alone as in a batch, an empty string
does not crash, and vectors are finite. `id` must encode anything that changes the
output — model, width, prefixes — because it is a cache key, and reusing a vector across
such a change silently corrupts the index.

## Contracts you can run

Every extension point with a registry group has an executable contract. Subclass, supply
the component, and the shared behaviour is verified for you:

```python
from toolbroker.testing import RetrieverConformanceSuite


class TestGraphRetriever(RetrieverConformanceSuite):
    @pytest.fixture
    def retriever(self, populated_store):
        return GraphRetriever(populated_store)
```

| Suite | Checks, among others |
|---|---|
| `StoreConformanceSuite` | filters, an empty allow-set matching nothing, upsert semantics |
| `EmbedderConformanceSuite` | dimensions, unicode, very long text, real indexing |
| `AdapterConformanceSuite` | one entry per tool, name safety, bound callables |
| `RetrieverConformanceSuite` | `k` honoured, ordering, filters narrow, no duplicates |
| `RerankerConformanceSuite` | invents nothing, drops nothing, does not mutate its input |
| `TracerConformanceSuite` | nesting, and that an exception propagates unchanged |

**Ranking quality is deliberately not tested.** Which tools a retriever returns is its whole
reason for existing, and [API stability](stability.md) is explicit that retrieval results
are not an API. What the contract fixes is the *shape* of the answer, because every stage
downstream depends on it.

The suites are checked against deliberate violations, not only against implementations that
already pass — a retriever that ignores filters, a reranker that mutates its input, a tracer
that swallows exceptions. A contract nobody has tried to break is a contract nobody knows
the strength of.

## Write your own Retriever, Policy, Adapter, Source

If your framework wants a **pydantic model** rather than a JSON Schema dict — CrewAI
does — use `toolbroker.schema.model_from_schema` rather than writing another converter:

```python
from toolbroker.schema import model_from_schema

Args = model_from_schema(tool.input_schema, name="IssueRefundArgs")
```

It handles the parts that bite: `order-id` becomes `order_id` with an alias, a parameter
called `schema` or `class` is mangled rather than shadowing a pydantic or Python name,
enums become `Literal`, nested objects become nested models, and anything unmappable
degrades to a permissive field. A tool the model can still call with loose typing beats
one that vanished because its schema used a construct nobody anticipated.

Adapters that bind an implementation, rather than only rendering a schema, should use
`toolbroker.adapters.bind_all` instead of reaching into `tool.metadata` themselves. It
resolves the three cases every adapter faces — local callable, remote via an `invoker`,
or unbindable — and produces the same error message across frameworks:

```python
from toolbroker.adapters import bind_all

bindings = bind_all(tools, invoker=self._invoker, adapter="MyAdapter")
for binding in bindings:
    result = await binding.acall({"arg": 1})  # awaits only if the function is async
```

```python
class MyRetriever:
    def retrieve(self, query: str, k: int, filters=None) -> list[Hit]: ...


class MyPolicy:
    def evaluate(self, hits, *, agent=None, scopes=frozenset(), query="") -> PolicyResult: ...


class MyAdapter:
    id = "myframework"

    def render(self, tools) -> Any: ...


class MySource:
    id = "mysource"

    def discover(self) -> Iterable[Tool]: ...
```

A `Policy` must populate `firings` and `exclusions`, not just filter. Callers rely on
being able to explain a denial.

Prefer subclassing `BaseSource` for sources — it stamps `source_id` onto every tool and
makes discovery eager, so connection errors surface at the call rather than at some
later iteration.

## Worked examples

Two stores in this repo are built exactly the way yours would be — separate
packages, separate dependencies, nothing special-cased in the core:

| Package | Backend | Notable |
|---|---|---|
| [`toolbroker-qdrant`](https://github.com/iamrameshwar/tool-broker/tree/main/packages/toolbroker-qdrant) | Qdrant | Translates filters into Qdrant's filter language; risk stored as an ordinal so a ceiling is a range query |
| [`toolbroker-chroma`](https://github.com/iamrameshwar/tool-broker/tree/main/packages/toolbroker-chroma) | Chroma | Chroma metadata holds only scalars, so each tag becomes a boolean key — which is what lets tag *exclusion* run server-side |
| [`toolbroker-pgvector`](https://github.com/iamrameshwar/tool-broker/tree/main/packages/toolbroker-pgvector) | PostgreSQL | Filters compile to `WHERE` clauses; tags are a `text[]` with a GIN index and risk an ordinal, so a ceiling is a range scan |

Both pass `StoreConformanceSuite` with no overrides. Read either one before writing a
third; between them they cover most of the shapes a backend comes in.

Three adapters are built the same way:

| Package | Framework | Notable |
|---|---|---|
| [`toolbroker-langgraph`](https://github.com/iamrameshwar/tool-broker/tree/main/packages/toolbroker-langgraph) | LangChain / LangGraph | Emits `StructuredTool`, which both `ToolNode` and `bind_tools` accept |
| [`toolbroker-openai-agents`](https://github.com/iamrameshwar/tool-broker/tree/main/packages/toolbroker-openai-agents) | OpenAI Agents SDK | Owns the JSON-argument parse, so a malformed payload names the tool instead of surfacing a bare decode error |
| [`toolbroker-claude-agent`](https://github.com/iamrameshwar/tool-broker/tree/main/packages/toolbroker-claude-agent) | Claude Agent SDK | Wraps results as MCP content, and derives `allowed_tools` from the same selection so permissions cannot drift |
| [`toolbroker-crewai`](https://github.com/iamrameshwar/tool-broker/tree/main/packages/toolbroker-crewai) | CrewAI | Generates a pydantic `args_schema` per tool, since CrewAI will not take a JSON Schema dict |

They live outside the core on purpose: a framework's breaking change should force a
release of *that adapter*, not of ToolBroker.

And two embedders:

| Package | Backend | Notable |
|---|---|---|
| [`toolbroker-ollama`](https://github.com/iamrameshwar/tool-broker/tree/main/packages/toolbroker-ollama) | Local models via Ollama | No HTTP dependency at all — the endpoint is one JSON POST, so the stdlib covers it |
| [`toolbroker-openai-embed`](https://github.com/iamrameshwar/tool-broker/tree/main/packages/toolbroker-openai-embed) | OpenAI, Azure OpenAI | Looks widths up instead of probing, so construction costs nothing; reorders results by index, which the API says you must |

Both encode the width into `id`, so a cache built at one dimensionality can never be
reused at another.

Writing the Chroma one found a real gap: the suite did not test that an **empty
allow-set matches nothing**, and a backend that drops an empty clause silently widens
the query to the whole catalogue. That test now exists, and every store is held to it.
If your backend forces a similar decision, say so in the PR — the contract is meant to
grow that way.

## Publishing a plugin

```toml
# pyproject.toml of toolbroker-mystore
[project.entry-points."toolbroker.stores"]
mystore = "toolbroker_mystore:MyStore"
```

Installed, it is addressable everywhere a name is accepted, including YAML:

```yaml
store:
  name: mystore
  options: {url: "http://localhost:6333"}
```

Entry-point groups: `toolbroker.embedders`, `toolbroker.stores`, `toolbroker.retrievers`,
`toolbroker.adapters`, `toolbroker.sources`, `toolbroker.policies`.

## Hooks

For behaviour that does not fit an interface — redacting a description before indexing,
boosting tools your users actually click, mirroring denials into your own audit log.

| Event | Kind | Use |
|---|---|---|
| `BEFORE_DISCOVERY` / `AFTER_DISCOVERY` | observe / transform | Inspect or rewrite the discovered set |
| `TRANSFORM_TOOL` | transform, may `DROP` | Rewrite or remove a tool before indexing |
| `TRANSFORM_INDEX_TEXT` | transform | Change what gets embedded |
| `AFTER_INDEXING` | observe | Metrics, cache warming |
| `BEFORE_RETRIEVAL` / `TRANSFORM_QUERY` | observe / transform | Query expansion or rewriting |
| `TRANSFORM_HITS` / `AFTER_RETRIEVAL` | transform / observe | Custom scoring or logging |
| `BEFORE_POLICY` / `AFTER_POLICY` | observe | Audit trails |
| `AFTER_SELECTION` | transform | Final adjustments |

```python
from toolbroker import DROP, Event


@broker.hooks.on(Event.TRANSFORM_TOOL)
def hide_internal(tool):
    return DROP if "internal" in tool.tags else tool


@broker.hooks.on(Event.AFTER_POLICY)
def audit(result, agent):
    for exclusion in result.exclusions:
        audit_log.write(agent=agent, tool=exclusion.tool_id, reason=exclusion.reason)
```

Returning `None` leaves the value unchanged; returning `DROP` removes the item. Those
are deliberately different — a handler that only cares about some tools returns `None`
for the rest, and that must not be mistaken for "delete this tool".

A failing handler is logged and skipped rather than taking down retrieval. Use
`HookManager(strict=True)` in tests to make failures loud.

### Attaching hooks from config

The interesting handlers usually belong to the deployment rather than the application —
mirror every selection into an audit log, normalise queries through an in-house service —
and a platform team should be able to add those without a code change in each service
that embeds the broker.

```yaml
hooks:
  strict: false
  after_retrieval:
    - {name: my_pkg.audit:record}
  transform_query:
    - {name: my_pkg.nlp:make_normaliser, options: {lang: en}}
```

Without `options` the resolved object *is* the handler. With them it is **called** with
those options and the result is the handler, so a parameterised handler is a small
factory rather than a closure you have to find somewhere to define.

**Get the signature right for the kind of event**, because the two are not
interchangeable and the mismatch shows up as a logged `TypeError` at runtime rather than
at load:

```python
def record(**kwargs):  # observe: return value ignored
    audit_log.write(**kwargs)


def make_normaliser(lang="en"):  # transform: value comes first, and you return it
    def normalise(query, **kwargs):
        return query.strip().lower()

    return normalise
```

Set `strict: true` to re-raise instead of logging — useful in staging, rarely what you
want in production, where a broken audit hook should not take selection down with it.

## Observability

### Logging: nothing to plug in

ToolBroker logs through the standard library under the `toolbroker` logger namespace and
**never configures the root logger**. So any provider that speaks `logging` — Datadog,
Sentry, structlog, CloudWatch, a plain file — is wired up by your application the way you
already wire up every other library:

```python
import logging

logging.getLogger("toolbroker").addHandler(my_provider_handler)
```

`configure_logging` exists for applications and the CLI that want a sensible default, and
is never called on import:

```python
from toolbroker.observability import configure_logging

configure_logging("DEBUG", json_output=True)  # one JSON object per line
```

### Tracing: a one-method Protocol

Tracing has no `logging`-shaped standard, so it is a plugin point. The Protocol is
deliberately one method, because every APM vendor can satisfy it and a larger interface
would be picking a winner:

```python
from collections.abc import Iterator
from contextlib import contextmanager


class DatadogTracer:
    def __init__(self, service: str = "toolbroker") -> None:
        from ddtrace import tracer

        self._tracer, self._service = tracer, service

    @contextmanager
    def span(self, name: str, **attributes: object) -> Iterator[None]:
        with self._tracer.trace(name, service=self._service) as current:
            for key, value in attributes.items():
                current.set_tag(f"toolbroker.{key}", value)
            yield
```

Install it in code or in config:

```python
from toolbroker.observability import set_tracer

set_tracer(DatadogTracer(service="agent-gateway"))
```

```yaml
observability:
  tracer: {name: my_pkg.tracing:DatadogTracer, options: {service: agent-gateway}}
```

`otel` is used automatically when `opentelemetry-api` is installed
(`pip install 'toolbroker[otel]'`) and is a no-op otherwise — tracing is never a hard
dependency. `set_tracer(None)` disables it.

!!! note "A tracer that raises is disabled, not propagated"
    A vendor SDK that throws on a misconfigured endpoint is a real way to cause an
    outage, and observability is never worth one. If your tracer raises while opening a
    span, ToolBroker logs a warning, swaps in the no-op tracer, and continues serving.
