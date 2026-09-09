# ToolBroker — Planning Document

**Status:** Phase 1 (v0.1) implemented — see §0. Phase 0 benchmark partially done.
**Owner:** Rameshwar
**Last updated:** 2026-09-09

> **One-liner:** Give your agent the right 5 tools out of 500, on any framework, with rules you control.

---

## 0. Implementation status

*Updated 2026-09-09. This section records what actually exists; the phase plan below is
unchanged.*

### Built and verified

| Area | Status |
|---|---|
| Six `Protocol` interfaces + plugin registry (entry points, runtime, dotted path) | done |
| Sources: Python functions, static JSON/JSONL, MCP (stdio + HTTP, SDK 2.x), OpenAPI 3.x | done |
| Embedders: offline `HashingEmbedder` (default), fastembed, disk-backed cache | done |
| Store: `InMemoryStore` (numpy fast path, pure-Python fallback) | done |
| Plugin packages: 3 stores + 4 adapters + 2 embedders, all conformance-clean | done |
| Shared `adapters.bind_all` (invocation) and `schema.model_from_schema` (JSON Schema to pydantic) | done |
| `toolbroker.risk` name-based classifier, shared by MCP and Python sources | done |
| Retrieval: semantic, BM25, hybrid (RRF + weighted), pipeline, usage boost, LLM rerank | done |
| Policy: 10 rule types, deny-overrides/first-match, per-agent routing, dry run, strict agents | done |
| `toolbroker calibrate`: derive a score floor from the catalogue, no labels needed | done |
| Adapters: OpenAI, Anthropic (core) + LangGraph, OpenAI Agents SDK, Claude Agent SDK, CrewAI (packages) | done |
| MCP proxy (`search_tools` / `describe_tool` / `call_tool`, call-time re-authorization) | done |
| CLI: `index`, `query`, `bench`, `calibrate`, `serve --refresh` | done |
| Hooks (12 events), YAML/pydantic config, OTel spans, JSON logs, determinism | done |
| Benchmark: 40 hand-authored tools + **290 hand-written labelled queries**, 8 categories | done |
| End-to-end harness: model picks a tool, full catalogue vs. top-k | done (offline caller verified; provider paths unrun) |
| `bench/ablate.py` (enrichment), `bench/threshold.py` (score floor), Wilson CIs, per-category scoring | done |
| `toolbroker.testing` conformance suites for all six extension points | done |
| uv workspace monorepo; plugins install editable and run in CI | done |
| 1134 tests, `mypy --strict` clean over core + 9 plugins, ruff clean, CI matrix | done |
| Docs site (8 pages), 9 runnable examples, CONTRIBUTING, CHANGELOG, issue/PR templates | done |

Core is **8.8k executable lines**, split 4.8k retrieval-and-policy hot path and
3.9k operational tooling. That is well past the original ~3k cap; see
§"Settling the line-count cap" for the decision, which is a deliberate revision
rather than an oversight.

### The numbers as they stand

290 hand-written labelled queries (25 negative), `k=5`, synthetic catalogue. Recall@5:

| embedder | 50 tools | 200 tools | 1000 tools |
|---|---|---|---|
| hashing (offline) | 0.515 | 0.466 | 0.431 |
| fastembed bge-small | **0.749** | **0.702** | **0.668** |

Recall@5 degrades as the catalogue grows — the effect the project exists to address,
and it does not vanish just because retrieval was added. The embedder dominates every
other choice: lexical to semantic is worth ~0.24, and no other knob has come close.

End to end (offline lexical caller, `k=5`, fastembed retrieval), full catalogue vs.
top-k:

| tools | full accuracy | toolbroker accuracy | full tokens | toolbroker tokens |
|---|---|---|---|---|
| 50 | 0.334 | 0.438 | 2,145 | 217 |
| 200 | 0.324 | 0.441 | 8,128 | 212 |
| 1000 | 0.338 | **0.445** | 40,618 | **210** |

Retrieval is worth ~0.11 at every size, at 193× less context per turn. Absolute numbers
are meaningless — the caller is a deterministic stand-in, not a model. Separately,
OpenAI refuses any request carrying more than 128 tools, so past a couple hundred tools
the full-catalogue baseline is not worse, it is unavailable.

*(The first version of this section reported 42 templated queries, where the same
comparison read 0.262 vs 0.286 and `full` appeared to degrade with catalogue size. That
degradation was noise and has been withdrawn; see §"The labelled set changed three
conclusions".)*

### Splitting the loss by whose fault it is

A single end-to-end accuracy cannot say whether a failure was retrieval's or the
model's, and the two want opposite fixes. `bench/e2e.py` now separates them: a
**retrieval miss** is a case where no correct tool reached the model at all.

| tools | accuracy | retrieval misses | accuracy if retrieved |
|---|---|---|---|
| 50 | 0.438 | 37 / 290 (12.8%) | 0.502 |
| 200 | 0.441 | 50 / 290 (17.2%) | 0.533 |
| 1000 | 0.445 | 61 / 290 (21.0%) | **0.563** |

The flat headline was hiding two trends that cancel. Misses climb steadily with
catalogue size, while the caller does *better* per shortlist at every step — five tools
drawn from a thousand are a cleaner choice than five drawn from fifty.

That settles where the effort goes. Recall is the binding constraint: at a thousand
tools it caps end-to-end accuracy at 0.79 whatever the model does, and each point of
recall recovered is worth roughly half a point end to end. The part that limits the
result is the part this library controls.

Until this split existed the harness reported a retrieval miss as `expected []` — the
gold tool was filtered out before the comparison, so the report showed an empty label
and a caller that looked merely indecisive. It counted correctly and explained nothing.

### Configurable for people who install it, not clone it

"Everything is a swappable interface" was true of the parts I had reached for and false
in three places nobody would find until they hit one. An audit against the actual
question — *what can someone change without forking?* — separated the claim from the
reality.

Already true, and worth stating so it does not get "fixed": **embedder, store and sources
already accept a plain dotted path**, so any model and any database works with no entry
point and no registration. **Logging needs no plugin point at all** — the library logs
through the stdlib `toolbroker` namespace and never configures the root logger, so
Datadog, Sentry and structlog attach a handler the way they do for every other library.
Adding an abstraction there would have been worse than nothing.

Three things were genuinely closed:

* **`retrieval.mode` was a closed `Literal`.** The `toolbroker.retrievers` entry-point
  group existed and the config file refused to use it — a contradiction sitting in plain
  sight. Now any plugin name or dotted path.
* **Rerankers had no registry group.** `Reranker` is an advertised Protocol, and the
  config hardcoded an empty list, so the one extension point people most want to reach
  for could not be named at all.
* **Tracing was welded to OpenTelemetry** in a module-level global evaluated at import.
  There is now a one-method `Tracer` protocol — deliberately one method, because every
  APM vendor can satisfy it and a larger interface would be picking a winner.

The design question that took the longest was how a plugin gets its collaborators. A
semantic retriever needs the store *and* the embedder, a lexical one only the store, a
third-party one may want neither, and a YAML file cannot express any of them. Forcing
every plugin to accept every collaborator would push our problem into their signature.
So `registry.construct()` inspects the constructor and passes each collaborator **only if
it is named**, or everything if it takes `**kwargs` — and anything in `options` wins, so
a user can always override what would be injected.

One deliberate hardening: a tracer that raises is swapped for the no-op and logged rather
than propagated. A vendor SDK that throws on a misconfigured endpoint is a real way to
cause an outage, and observability is never worth one.

Two further gaps turned up once the question became "what does a *deployment* need",
and the first one quietly made the YAML story unusable in production.

**Secrets could not be kept out of the file.** The argument for having a config file at
all is that it gets committed and reviewed like any other change — and a real one holds
a Postgres DSN, an API key, an internal MCP endpoint. There was no substitution, so the
choice was committing secrets or not using the file. `${VAR}` and `${VAR:-default}` now
resolve before validation. The detail worth the effort is the failure: a missing
required variable raises naming the variable *and* its path in the file, rather than
handing a database driver the literal string `${VAR}` and failing later somewhere
unrecognisable.

**Hooks were Python-only.** Twelve events, and the handlers people most want — mirror
every selection into an audit log, normalise queries through an in-house service — belong
to the deployment rather than to each service embedding the broker. They now attach by
dotted path, with `options` turning the target into a factory so a parameterised handler
is a small function rather than a closure someone has to find a home for.

Writing the tests caught the trap a config user will hit: `after_selection` is a
*transform* and `after_retrieval` is an *observer*, and their handler signatures differ.
My first handler had the wrong shape and failed as a logged `TypeError` at runtime rather
than at load. That is documented now with both signatures side by side, which is cheaper
than the hour it would otherwise cost somebody.

Verified the way it will actually be used: a throwaway package the core has never heard
of, no entry points and no registration, wired in through dotted paths in one YAML file —
custom embedder, custom reranker, custom tracer, two hooks and per-agent policy, with the
environment supplying every secret.

### Turning "not our problem" into a feature

The ceiling work ended on an honest but unsatisfying note: 8.3% of queries are
unreachable because the tool's description is nothing like how anyone would ask for it,
and that belongs to the catalogue rather than the retriever. True, and useless to whoever
owns the catalogue, because nothing told them *which* tools were in trouble.

`toolbroker doctor` does. It asks the catalogue about itself, the way `calibrate` does
for score floors, and needs no labelled data:

* **Shadowed** — run a tool's own description as a query; if the tool is not in the top
  `k`, nothing rougher will find it either.
* **Thin** — empty or near-empty descriptions.
* **Closest pairs** — the tightest neighbours, ranked, with byte-identical descriptions
  marked.

Two things went wrong on the way, and both were worth the detour.

**The first design could not fire.** The shadow check searched the store with the tool's
own indexed vector, which is elegant — no embedding calls — and vacuous, because a vector
is maximally similar to itself, so every tool ranked first by construction. The tests
caught it: a catalogue of six identical tools produced no findings at all. The fix is to
query with the *description* rather than the indexed vector, which costs one embedding
per distinct description and asks a question that can actually fail.

**The twin threshold could not be defaulted.** Tools with byte-identical descriptions sit
0.02 to 0.06 apart under bge-small and 0.07 to 0.20 apart under the hashing embedder —
the same defect, three times the number. The obvious rescue was to normalise by the
catalogue's own spread, and that fails too: a catalogue that is 96% duplicates and one
with none produce nearly the same distribution shape, so nothing internal says which you
have. So pairs are **ranked, not judged**, and the operator reads down until they stop
looking interchangeable. That is the same conclusion `calibrate` reached about score
floors, arrived at independently, which is mild evidence it is the right shape for this
kind of problem.

The `identical_description` marker earns its place here: it turns a suspicion into a
certainty without needing any threshold at all.

### Settling the line-count cap

I recommended six features into core over two sessions and never once checked them
against the repo's own rule: *core capped at ~3k lines; everything else lives in plugins*.
Measuring afterwards is the wrong order, and the number is 8.8k.

The split matters more than the total:

| | lines |
|---|---|
| retrieval + policy hot path (catalog, sources, index, retrieve, policy, adapters, types) | 4,772 |
| operational tooling (config, CLI, testing, proxy, bench, drift, diagnose, aliases, savings, reachability, calibrate, conversation, aio, usage, refresh) | 3,863 |

The obvious remedy is a `toolbroker-ops` package. Working through it, that turns out to be
the wrong move for the wrong reason. The plugin precedent is **dependency isolation** —
CrewAI earned its own package because it drags in 53 dependencies, and the write-up says
so explicitly. Every candidate here is pure Python with zero dependencies, so extracting
them would shrink one number and buy nothing: same install, same review surface, same
import graph, plus a second package to release.

So the cap gets replaced by the constraint it was a proxy for:

> **Core carries one runtime dependency (pydantic) and no framework coupling. Anything
> heavy lives in a plugin.**

That is still true, has never been violated, and is checkable — the `minimal-install` CI
job already enforces it by running the quickstart against a bare `pip install .`. A line
count was measuring the wrong thing: it would have been satisfied by moving code without
changing what a user installs.

Two honest caveats. The hot path alone is 4.8k, so this is not merely reclassification —
retrieval and policy really did grow, mostly from the policy layer going from a few rules
to ten plus tenancy. And a rule replaced after it starts biting deserves suspicion; the
argument above should be read as a decision to defend, not a settled fact.

### Contracts for the rest of the extension points

Store, Adapter and Embedder had executable contracts. Retriever, Reranker and Tracer had
registry groups, docs telling people to implement them, and no safety net — and this
project's own record says both existing suites had holes that surfaced only when somebody
wrote a real plugin against them.

All three now exist, and every built-in runs through them: five retrievers including the
resilient wrapper in *both* its healthy and degraded configurations, two rerankers, two
tracers. Eighty-one checks, all green on the first run.

That last part was the worrying bit. A suite that passes everything immediately is either
correct or toothless, and there is no way to tell from the green. So the suites were then
run against seven deliberately broken components — a retriever that ignores filters, one
that ignores `k`, one that returns unsorted results, rerankers that drop, mutate and invent
hits, and a tracer that swallows exceptions. Fourteen failures, each pinned to the test
that should have caught it. The same method the API snapshot got: verify against real
breaks, not against things that already pass.

One judgement call worth recording: **ranking quality is deliberately not tested**. Which
tools a retriever returns is its entire reason for existing, and `docs/stability.md`
already says retrieval results are not an API. A shared contract can fix the *shape* of the
answer — `k` honoured, ordering, filters narrowing rather than widening, no duplicates —
and fixing anything more would freeze the thing plugins exist to vary.

### Housekeeping that found something

The `.venv` had survived the rename with the old absolute path baked into every console
script, so `pytest` and `mypy` only ran via `python -m`. Rebuilding it moved the toolchain
from pytest 8 to 9 and mypy 1 to 2, and Python from 3.12 to **3.14** — a version the CI
matrix does not test. Everything passed anyway: 846 core and 288 plugin tests, `mypy
--strict` clean, ruff clean. 3.14 is now in the matrix and the classifiers, on that
evidence rather than on optimism. The two remaining warnings come from `chromadb`, not from
here.

### Three more bypasses, in code written the same week

SECURITY.md's five findings came from writing the attack rather than the document. Having
just added a lot of gating code — quarantine, tenant isolation, degraded retrieval — not
doing that again would have been inconsistent. Three hypotheses, all three confirmed.

**`first_match` short-circuited tenant isolation.** The combining mode returns on the
first rule with an opinion, and the config builder appends isolation last, so an `allow`
glob granted another tenant's tool before the boundary was ever consulted. The narrow fix
is to reorder; the right fix is that **a security guarantee which depends on rule order is
not a guarantee**. `Rule` now carries `mandatory`, isolation sets it, and the engine
evaluates mandatory rules first and treats their deny as final under either mode. Tested
with the rule at both ends of the list.

**Drift ignored tag changes.** The privilege comparison covered risk and required scopes.
`DenyTags` and `RequireTags` gate on tags, so a server quietly dropping `destructive`
walked straight through a rule the operator believed was protecting them, with nothing
reported anywhere. Exactly the shape of finding #1 — a rule that looks applied and is not.

**A rename walked past quarantine.** A renamed tool has a different id, so the diff sees a
removal plus an addition rather than a change, and quarantine only guarded changes. Worse,
`trust_on_first_use=False` did not actually close it: it declined to *approve* new tools
but still indexed and served them, so the flag looked like a control and was not one. It
now genuinely holds arrivals.

The last one is the useful lesson. Two of the three were features that existed and did not
do what their names implied, which is the same failure mode as the case-sensitivity bug —
and the reason to write the attack every time the gating surface grows, rather than once
before release.

### Tenancy, and why scopes were not already it

One deployment serving several customers is the case a platform team asks about first, and
the honest answer had been "use scopes or namespaces". Looking properly, that answer was
wrong in a way worth writing down.

`RequireScopes` **abstains** for a tool that declares no scopes. For a capability check
that is right — a tool needing nothing should not be blocked by a rule about scopes. For an
isolation boundary it is exactly backwards: a tool nobody remembered to tag stays visible
to every tenant, and nothing anywhere indicates it. Same shape as the case-sensitivity
bypass in SECURITY.md — a rule that looks applied and is not.

`TenantIsolation` therefore *decides* on every tool rather than abstaining, and denies
whenever it is not certain. The row that matters is the third:

| tool's tenant | request's tenant | outcome |
|---|---|---|
| acme | acme | allowed |
| acme | globex | denied |
| acme | *not supplied* | **denied** |
| none | anything | allowed only if `shared_tools` |

An undeclared tenant must never mean "all tenants". Failing open there would hand every
customer's tools to a caller that simply forgot the argument, which is precisely the
mistake this exists to make impossible. `shared_tools: false` extends the same reasoning to
untagged tools, for deployments where forgetting to tag is likelier than deliberate sharing.

Two limits stated rather than hidden. `tenant` is a per-call argument, so it had to be
plumbed through `select`, the `Policy` protocol and both engines — the API snapshot caught
the signature change, which is the entire point of having it. And **the MCP proxy has no
tenant dimension**: it exposes one catalogue over one connection and the protocol carries
no tenant identity to enforce. Run a proxy per tenant, or use the library directly.

### Making the operational features configurable

The five features above shipped as Python-only, which contradicted the configurability
work two sections earlier — a platform team that installs the package rather than cloning
it could not switch any of them on. They are now `drift`, `aliases`, `savings`,
`retrieval.on_error` / `retrieval.fallback` and per-agent `tenant_isolation`, every one off
by default.

One small design win fell out. A `SavingsTally` attached to a broker records every
selection itself, because `select()` is the only place that knows both what was sent and
what could have been — and asking every call site to remember guarantees some forget.

### Five things only a middleman can do

A question worth asking of any library: what can it do that its competitors structurally
cannot? For a component that sits between the catalogue and the agent and watches across
time, the answer is more than it looked.

**Tool drift detection.** SECURITY.md is right that a hostile description cannot be
filtered — but that framing hides the attack that actually works, which is a description
that changes *later* on a server already reviewed. Nothing else in a stack compares the
catalogue to its previous state. ToolBroker does, on every refresh, and had been throwing
the comparison away after using it to decide what to re-embed. It is now a control: with
quarantine on, a changed tool keeps serving its last approved text until a human accepts
the new one. Privilege changes are deliberately exempt — holding one back could mean
serving a *lower* risk tier than the server now claims, which is the wrong direction to
fail in.

**Policy reachability and diff.** The plan has always said a config file exists so agent
permissions can be reviewed like any other change. That was aspirational: nobody can review
`*/delete_*` → `*/delete_user` by reading it. Policy is deterministic over a known
catalogue, so `toolbroker diff` answers it statically and `--fail-on-high-risk` makes it a
CI gate. Two things it refuses to fake: `MaxTools` is reported separately because it bounds
a response rather than the catalogue, and `MinScore` is reported as *undecidable* rather
than silently ignored, since ignoring it would overstate an agent's reach.

**Savings accounting.** The 193x figure is measured on a synthetic catalogue, and nobody
approving a budget cares. The broker sees both numbers on every request. The care went into
not producing a number people would distrust: the counterfactual is printed, and the report
marks itself estimated unless *every* entry carried a real provider count.

**Learned aliases.** The ceiling work ended by calling the 8.3% a description problem
belonging to the catalogue. True, and the broker can still fix it, because it sees the pair
nobody else does — the query typed, and the tool the framework reports as called. Three
bounds against the obvious feedback loop, one of them borrowed wholesale from the usage
booster.

**Configurable degradation.** There was no store-failure path at all. A Qdrant restart
handed the agent zero tools with no error, which to a model does not look like an outage —
it looks like a world with no capabilities in it. Now `fail`, `empty` or `fallback`, with
the security property stated and tested: a degraded path returns fewer or worse tools, and
never tools policy would have refused.

Two bugs the tests caught, both worth recording. The alias threshold compared a
decay-adjusted weight against an integer, so the third of three observations landed at
0.999999999999 and `min_count=3` never fired — and whether it fired depended on clock
granularity, so it would have presented as flakiness rather than as a rule that plainly did
not work. And the policy diff's no-change path never mentioned undecidable rules, so a
reviewer adding a score floor would have been shown "No change in reach" and nothing else.

### Where the recall ceiling actually is

The loss decomposition said recall was the binding constraint, so the next question was
how much of it is recoverable. `bench/ceiling.py` answers it by measuring how deep you
have to fetch before the right tool is in the pool at all — because a reranker cannot
promote a tool it never received.

| tools | @5 | @20 | @100 | @200 | unreachable |
|---|---|---|---|---|---|
| 50 | 0.860 | 0.958 | 1.000 | 1.000 | 0 (0.0%) |
| 200 | 0.811 | 0.906 | 0.992 | 0.992 | 2 (0.8%) |
| 1000 | 0.770 | 0.838 | 0.917 | 0.917 | 22 (8.3%) |

At a thousand tools a perfect reranker over 100 overfetched candidates would reach 0.917.
That 0.147 gap is the entire prize for any reranking strategy — worth knowing before
paying for a cross-encoder or an LLM call on every query, and the first number I would
want if someone proposed one.

The column that mattered more is the last. **8.3% of queries never surface their tool at
any depth**, and @200 equals @100 exactly, so those tools are not sitting just outside the
window — there is no signal at all. `bounce the pods` → `restart_service`,
`someone left the company today` → `deactivate_user`. Six of the 22 are jargon and six are
indirect phrasing. No `k`, no fusion, no reranker recovers them, because nothing about
"Deactivate a user account." is near that sentence in any embedding space.

That is a description problem, and it belongs to the catalogue rather than the retriever.
Saying so costs nothing and is the truth; a retrieval library that claims to close that
part of the gap is overselling.

Two levers were measured and declined. **Raising `k`** flattens worst exactly where the
catalogue is biggest (+0.098 at 50 tools from k=5 to k=20, only +0.068 at 1000), so it
quadruples context to buy least. **Fusing BM25** trails pure semantic at 50 and 200 under
every weighting tested and only crosses over at 1000 (best fusion 0.785 vs 0.770) — real,
but worth about one query in a hundred while costing more than that at the other two
sizes, so it is documented as advice for large catalogues rather than moved into the
default.

And one methodological scare that came to nothing. The 1000-tool catalogue's filler had
**200 distinct descriptions across 1000 entries** — six-way duplicates, visibly crowding
the top 5 for exactly the queries that were failing. It looked like the whole scaling
result might be an artifact of the generator. Rebuilding the filler with 1000 distinct
descriptions moved recall@5 by +0.011 for the semantic embedder and −0.008 for the
lexical one: noise at this sample size. The degradation with catalogue size is real and
survives the most obvious attack on how the corpus is built. Worth the afternoon to know
that before someone else asked.

### The name was taken — by the same idea

`toolscope` on PyPI is
[ilya-kolchinsky/toolscope](https://github.com/ilya-kolchinsky/toolscope), Red Hat,
Apache-2.0, released 2026-02-04: *"per-prompt automatic tool selection for LLM tool
calling"*, framed exactly as this plan frames it — "as the number of tools grows, LLMs
become worse at selecting the right one". Pluggable embedders, Milvus or in-memory,
LangChain/LangGraph/FastMCP adapters, cross-encoder reranking. Small so far (7 stars, 8
commits) but real and active.

Two consequences pointing in opposite directions.

**The name had to go.** Same name plus same problem domain is not a collision anyone can
live with, and publishing `toolbroker-qdrant`-style plugins under a name someone else
owns would read as appropriation. Renamed throughout to **ToolBroker**, free on PyPI and
as a GitHub org.

**The thesis is externally validated.** Someone at Red Hat independently reached the same
conclusion and shipped. §Risks listed "the frameworks ship this natively" as the top
threat; this is a variant of it, and the mitigation already written — *be the policy and
control layer they will not build* — turns out to be exactly right, because **their
project is retrieval-only**. No policy, no scopes, no risk tiers, no audit trail, no
call-time authorization. The differentiator stopped being a bet and became an observable
gap.

The rename touched 165 files, and the one genuine hazard was that `scope` is both the
catalogue variable and the *permissions* vocabulary (`scopes`, `--scope`,
`required_scopes`). Protecting the prose phrase "in scope" then silently corrupted
`for source in scope.sources`, which the test suite caught. The API surface snapshot
also fired on its first real use, refusing the changed signatures until they were
regenerated deliberately — which is the entire point of it.

### Preparing for an API freeze without declaring one

Committing to a public surface is the user's call, not mine, so the work here was to make
that call cheap: audit the surface for anything embarrassing to freeze, then make drift
impossible to miss.

Three things the audit found, all now fixed:

* **The extension protocols were not exported.** Every docs page says "implement this
  Protocol", and `from toolbroker import Store` failed — users had to reach into
  `toolbroker.protocols` while every other important name was top-level.
* **`PgVectorStore._drop()` was private but necessary.** Nine test call sites reaching for
  a private method is the API saying something is missing. Now `drop()`.
* **`__all__` was unsorted**, so its diffs would have been noise.

`tests/api/public_api.json` now snapshots every export, public method, property and
signature, checked on every run. Verified against real breaks: it catches a name dropped
from `__all__` and a renamed keyword argument, with a message telling you how to
regenerate it deliberately. Those are the changes that hurt after a freeze — invisible in
review, and a broken import for somebody.

`docs/stability.md` states what counts as public, which change needs which version bump,
and one thing worth being explicit about: **retrieval results are not an API**. A release
may change which tools a query returns. Freezing rankings would freeze the library's
ability to improve; what is guaranteed instead is that every such change ships with
before-and-after numbers.

### pgvector, and a CI check that the tests actually ran

The third store, and the first that needs a real service. Postgres is the right default
for most teams: a tool catalogue is hundreds to low thousands of rows, rarely worth a
second piece of infrastructure with its own backups, upgrades, and on-call rota.

The conformance suite passed unmodified on the first run — three backends with
completely different filter languages, one contract, no exemptions.

That first run was also misleading. It passed against a container where I had created the
pgvector extension by hand while probing the API; the store registered the vector type
adapter *before* creating the extension, so it could not bootstrap a fresh database — the
only case that matters, and exactly what the CI job does. Rebuilding the container from
scratch caught it. A green run on a hand-prepared environment proves less than it looks
like it does.

Two decisions worth recording. HNSW rather than IVFFlat, because IVFFlat needs a training
step and performs silently badly until there are enough rows to build it properly — which
a tool catalogue may never reach. And the test suite **skips** without
``TOOLBROKER_PG_DSN`` so a contributor without Docker is not blocked, which creates a new
hazard: a CI job where the service failed to start would report "53 skipped" and pass.
The CI job therefore asserts that nothing skipped.

### Five policy bypasses, found by looking for them

A security review of a library whose entire job is gating access is not paperwork, so I
went looking for bypasses rather than writing a document about hypothetical ones. Five
were real, all pre-release, all now fixed with regression tests. Written up in
[SECURITY.md](SECURITY.md).

The worst was case sensitivity. `DenyTools(["*/delete_*"])` used `fnmatchcase`, so an MCP
server exposing `DeleteUser` — which plenty of APIs generate — sailed straight through a
rule the operator believed was protecting them. Nothing anywhere indicated the rule was
inert. Deny and allow patterns now match case-insensitively, with an opt-out.

The other four:

* Names and namespaces accepted surrounding whitespace, so `"admin "` evaded `admin/*`
  while looking identical to `"admin"` in every log an operator might check. Now
  stripped, with control characters rejected outright.
* `AgentPolicy` fell back to a permissive engine for an *unregistered* agent name, so a
  typo in `agent="support"` granted the entire catalogue. There is now `strict_agents`,
  and even the permissive path warns.
* The MCP proxy's `describe_tool` ignored policy entirely: a client denied a tool by
  search could still fetch its description and full parameter schema.
* The "did you mean" hint was an enumeration oracle — guessing a substring confirmed the
  existence and exact id of tools policy was meant to hide. Denied and nonexistent tools
  now produce an identical client-facing error, with the real reason logged server-side.

The finding I could not fix is worth stating plainly: **prompt injection through tool
descriptions**. A hostile MCP server puts instructions in a description and that text
reaches the model; no filter detects this reliably. What ToolBroker can do is bound the
blast radius, because policy is enforced in Python rather than by asking the model
nicely. A hijacked model still only reaches tools the operator permitted. That is the
honest framing, and it is a better argument for the policy layer than any feature list.

### CrewAI, and the converter it forced

CrewAI takes `args_schema` as a pydantic **model class**, not a JSON Schema dict, so
supporting it meant building a converter. That went into the core rather than the plugin,
because it is framework-agnostic and the next adapter with the same requirement should
not write a second one.

The conversion is deliberately forgiving: a catalogue is assembled from MCP servers and
OpenAPI specs written by people who never expected this code to read them, so an
unmappable construct becomes a permissive `Any` field rather than raising. A tool the
model can still call with loose typing beats a tool that disappeared.

The parts that actually bite are name collisions: `order-id` is not an identifier, and a
parameter called `schema` or `class` would shadow a pydantic or Python name. Both are
mangled with an alias so the wire format is unchanged.

The 53-dependency install that made me hesitate turned out to be the argument *for*
doing it — it lives entirely in its own package and reaches nothing else. That is what
the plugin architecture is for.

### Usage boosting, and the bound that makes it safe

`UsageBooster` existed but nothing recorded, persisted, or decayed usage. Completing it
meant deciding how to stop it becoming a feedback loop, because the naive version is
actively harmful: popular tools rank higher, get called more, rank higher still, and
every tool added after launch starves.

Two bounds:

* **The boost is capped** at ``usage_weight`` of a hit's own score. That gives a hard,
  testable guarantee — two tools more than that far apart in relative score cannot swap,
  however lopsided their usage. A million recorded calls does not let a wrong tool
  overtake a clearly right one.
* **Counts decay** exponentially on a half-life, applied lazily on read so there is no
  sweep to schedule and forget. Timestamps persist, so a process down for a month comes
  back with month-old counts rather than fresh ones.

Wiring it up also caught a latent bug: `config.build()` assigned `broker._pipeline`
directly, which would have silently discarded the usage booster. There is now a
`set_retriever()` that re-attaches it, because the symptom — boosting quietly stopping —
is invisible.

### Live refresh, and the failure mode that shaped it

`refresh()` re-discovers and applies only the diff. The interesting decision was what to
do when a source is unreachable.

The obvious implementation removes tools that are no longer reported. That means a
thirty-second MCP outage silently strips capabilities from every agent using the
catalogue — no exception, no failed request, just an agent that quietly stops being able
to do things. Pruning is therefore scoped to sources that actually answered, and the
failures are reported on the refresh report so a caller can alert on them.

The second decision was cost. A tool whose risk tier changed must be rewritten so policy
sees it, but its indexed text has not changed, so the vector is still valid. Separating
"updated" from "embedded" means a metadata change costs zero embedding calls, which with
a hosted embedder is the difference between a cheap poll and one nobody enables.

Polling rather than push notifications: no protocol support required, works with servers
that never send `tools/list_changed`, and cheap enough not to matter given the diffing.
A persistent subscription remains possible later; it is not a prerequisite.

### Closing the gap the benchmark opened

The 0.000 negative-query score was a product gap, not just a benchmark result: nothing
stopped the retriever returning five confident guesses for a question no tool could
serve. Fixing it needed a score floor, and picking a floor needed labelled data most
users will not have.

`toolbroker calibrate` derives one from the catalogue alone — run queries that certainly
match nothing, measure what they score, put the floor above them. Two things it surfaces
that a single recommended number would have hidden:

* **Noise and genuine queries overlap.** On the reference catalogue, noise reaches 0.572
  and the weakest genuine query scores 0.509. No floor separates them; the user is
  choosing which error to make.
* **Score scales differ wildly between embedders.** Noise tops out near 0.57 with
  bge-small and near 0.20 with the hashing embedder, on the same catalogue. A threshold
  copied from a docs page is meaningless. This is the concrete reason no default ships.

My first implementation measured only the top-1 score and reported that a 0.545 floor
cost 2% of sample queries. That was wrong: a floor filters the whole top ``k``, and the
real figure was 13% of queries losing at least one candidate. The report now separates
"emptied" from "thinned" for exactly that reason.

### The labelled set changed three conclusions

Growing the labelled set from 42 templated paraphrases to 290 hand-written queries across
eight categories moved the resolution from 0.024 per query to 0.003. Three things
followed immediately:

1. **A default changed.** `name_weight` went 2 to 1 — better for both embedders at all
   three catalogue sizes. The old set called this noise; it was not.
2. **A claim was withdrawn.** "Full-catalogue accuracy degrades as the catalogue grows"
   does not reproduce. It was noise at 42 queries. What survives is that retrieval beats
   full by ~0.11 at every size, at 193x less context.
3. **A product gap surfaced.** Negative queries — where no tool applies — fail
   **100%** of the time by default, because nothing stops the retriever returning its
   five best guesses. `bench/threshold.py` now exists to pick a floor, and no default is
   shipped because the right value depends on the embedder.

The per-category breakdown is the part that matters: paraphrase scores 1.000 and
everything resembling real speech scores ~0.62. A benchmark of paraphrases alone would
have reported success and taught nothing.

### A bigger embedder bought almost nothing

With the Ollama plugin in place the benchmark could finally compare a small local model
against a large one. The jump that matters is lexical to semantic (0.31 to 0.79 recall).
Past that, a 4B-parameter 2560-dim model **ties** bge-small on recall@5 at 200 tools and
costs ~34x the latency. It ranks better — MRR is consistently higher — but recall is what
decides whether the agent can succeed when you send the top 5 anyway.

Practical consequence: `toolbroker[fastembed]` stays the recommended default, and the
docs now say so with numbers rather than assertion. Caveat, again §9.3: a synthetic
catalogue of templated descriptions may not be hard enough to separate two good models.

### The adapter contract had a blind spot

`AdapterConformanceSuite` could only ever test half the adapters it claimed to cover.
Its fixture tools carried no callable, so any adapter that *binds an implementation*
rather than only rendering a schema failed the suite by construction. It also asserted
`render(tools) == render(tools)`, which no adapter embedding a fresh closure can
satisfy.

Both are fixed: fixture tools carry a callable, and repeatability is compared by tool
name. All five adapters now pass it unmodified. Same lesson as the store suite — a
contract only covers what someone has actually implemented against it.

### The conformance suite earned its keep

Writing the Chroma store surfaced a hole in the contract: nothing tested that an **empty
allow-set matches nothing**. Chroma's filter language rejects `{"$in": []}`, so the
obvious encoding failed loudly — but a backend that instead *dropped* the empty clause
would have silently returned the whole catalogue for a filter written to restrict it.
That is the dangerous direction, and it would not have been caught by any test that
existed.

The test now lives in `StoreConformanceSuite`, so all three stores are held to it. This
is the argument for shipping the contract as executable tests rather than prose.

### Measured, then left alone

An enrichment ablation (7 variants × 3 catalogue sizes) found **no significant effect**:
every variant landed within one query of the default, in both directions and
non-monotonically. The default is unchanged.

That null result is the strongest argument for §9.3: with 42 labelled queries the
benchmark cannot resolve the tuning decisions people will want to make with it. The
harness now prints its own resolution so nobody reads a +0.02 as an improvement.

### Deliberate deviations from the plan

1. **fastembed is not the zero-config default.** It downloads a model on first use,
   which is neither offline nor instant, so it cannot be what a first-time user hits.
   The default is a deterministic offline hashing embedder, with fastembed auto-selected
   when installed. The docs are explicit that it is lexical-only and roughly half as
   accurate.
2. **MCP pinned to `>=2,<3`.** The 1.x `ClientSession` API and 2.x `Client` API differ
   enough that supporting both would cost more than it is worth pre-1.0.
3. **mypy runs at `python_version = "3.12"`** because numpy's bundled stubs use
   3.12-only syntax. The 3.10 floor is verified by CI running tests on 3.10 and by
   ruff's `py310` target.
4. **A `test` extra was added** so plugin authors can import `toolbroker.testing`.

### Open questions from §9, now answered

- **§9.6 (MCP proxy semantics)** — resolved as proposed: forwarding a single call the
  client already decided to make is transport; driving a loop is not. Implemented,
  documented, and enforced by re-authorizing every forwarded call against policy.

### Still open — blocking or shaping what comes next

- **§9.1 Name availability** — **checked 2026-09-09, and clear.** `toolbroker` and all
  nine `toolbroker-*` plugin names return 404 on PyPI, so every name this repo builds is
  unclaimed. `github.com/toolbroker` is free as an org. Two caveats before publishing:
  ~~the URLs point at an account that does not exist~~ — **resolved 2026-09-09**: the
  project publishes from `github.com/iamrameshwar/tool-broker` and every URL now points
  there. The domain is still unchecked, and is not on the critical path.
- **§9.2 Code provenance** — **closed.** Clean-room by rule: every line here was written
  from scratch. Phase 0 is "build and prove", not "extract and prove".
- **§9.4 "correct tool" definition** — **closed.** Set-recall@k with multiple
  acceptable answers, plus negative cases where the right move is to call nothing.
- **§9.3 Benchmark corpus** — **closed by scope, not by doing it.** A curated
  real-catalogue corpus does not belong in this repo; it moves to a separate **MCP Hub**
  project. See the second hard rule in §2. The synthetic corpus stays, labelled as
  synthetic in the docs.
- **§9.5 Model baselines** — harness built (`bench/e2e.py`), numbers not collected.
  The full-vs-retrieval comparison now runs end to end with a deterministic offline
  caller; the OpenAI and Anthropic callers are written but **have not been run against a
  live API** — no key has been used in this repo. Collecting real numbers on GPT, Claude,
  and one open model is the remaining Phase 0 work, and it is the launch content.

---

## 1. Problem statement

Agent frameworks assume the tool list fits in the prompt. Past ~100 tools, three things break:

1. **Selection accuracy collapses.** The model picks a plausible-but-wrong tool, or the right tool's description gets lost in a 40k-token tool block.
2. **Cost and latency scale with the catalogue, not the task.** Every turn re-sends every schema.
3. **There is no control layer.** Nothing says "this agent may never call `delete_*`", "this agent gets at most 8 tools", or "these tools require an approval scope".

MCP made it trivial to *attach* hundreds of tools and did nothing to help you *choose* among them. ToolBroker is the retrieval + policy layer that sits between a tool catalogue and an agent's context window.

**What ToolBroker is not:** an agent framework, an orchestrator, an execution runtime. It never runs an agent loop. It answers one question — *given this query and this policy, which tools go in the context?* — and hands back tool definitions in whatever shape the caller's framework wants.

---

## 2. Design principles

| Principle | What it means concretely |
|---|---|
| **Framework-agnostic core** | The library never invokes a tool or drives a loop. It returns tool definitions; the caller's framework calls them. Core has no framework dependency. |
| **Everything is a swappable interface** | Embedder, store, retriever, reranker, policy, adapter, source — each a `typing.Protocol` with a default impl and a plugin entry point. |
| **Zero-config works, full-config possible** | `pip install toolbroker` + fastembed + in-memory index must work **offline in 20 lines**. The same object is drivable from YAML/pydantic for teams that want it declarative. |
| **Control at every stage** | Hooks before/after discovery, indexing, retrieval, policy. Deterministic mode (fixed seeds, cached embeddings) for tests. |
| **Transparent decisions** | Every retrieval returns *why*: per-tool scores, which rules fired, what got filtered and by what. No black box. |

**Hard rule:** ToolBroker never executes agent loops. Any feature request that implies orchestration is out of scope by definition, not by priority.

**Second hard rule:** ToolBroker does not curate tool catalogues. It is a connected middleman — it reaches real catalogues at runtime through `MCPSource` and `OpenAPISource`, so vendoring one duplicates the product at build time, imports third-party content of unclear licence into an Apache-2.0 repo, and ships a snapshot that is stale within weeks. Catalogues and corpora belong to **MCP Hub**, a separate project. The synthetic `bench/` corpus stays and stays labelled synthetic; if the "these are tools the author wrote" objection needs answering, the answer is a recipe that points the tool at real servers, not a corpus committed here.

---

## 3. Architecture

```
 Sources ──▶ Index ──▶ Store          Policy
 (MCP,       (enrich,  (vectors)        │
  OpenAPI,    embed)       │            │
  Python fns,              ▼            ▼
  static JSON)         Retriever ──▶ Filtered set ──▶ Adapter ──▶ framework tools
                       (semantic/                                (LangGraph, OpenAI,
                        hybrid/rerank)                            Claude SDK, raw JSON)
                             │
                       Observability (OTel spans, JSON logs)
```

### 3.1 Components

**Sources** — where tool definitions come from. MCP servers (stdio/HTTP) first; then OpenAPI specs, plain Python functions, static JSON. Pluggable.

**Index** — normalises and enriches each tool: name, description, parameter schema, examples, tags, namespace. Embedder plug: fastembed default (local, offline); OpenAI, Voyage, Azure, Ollama, sentence-transformers, custom.

**Store** — in-memory default (numpy, no server). Plugins: Qdrant, Chroma, pgvector, Pinecone, LanceDB, Redis.

**Retriever** — semantic default. Composable pipeline: hybrid (BM25 + vector), LLM rerank, tag/namespace filters, usage-frequency boost.

**Policy** — per-agent allow/deny, scopes, `max_tools`, cost/risk tiers, dry-run mode. Policies as code or YAML; custom evaluators pluggable. Policy is applied *after* retrieval and *before* adapter, and its decisions are recorded in the result.

**Adapters** — LangGraph/LangChain, OpenAI Agents SDK, Claude Agent SDK, CrewAI, Google ADK, raw function-calling (OpenAI/Anthropic JSON schemas). Plus **MCP proxy mode**: ToolBroker exposes itself as a single MCP server with `search_tools` / `call_tool`, so any MCP client gets it with zero code.

**Observability** — OpenTelemetry spans, optional Langfuse/LangSmith exporters, JSON logs.

**CLI** — `toolbroker index`, `toolbroker query "..."`, `toolbroker bench`, `toolbroker serve`.

### 3.2 Core interfaces (sketch — to be frozen at v0.1)

```python
class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[Vector]: ...
    @property
    def dim(self) -> int: ...


class Store(Protocol):
    def upsert(self, records: Sequence[ToolRecord]) -> None: ...
    def search(self, vector: Vector, k: int, filters: Filters | None) -> list[Hit]: ...


class Retriever(Protocol):
    def retrieve(self, query: str, k: int, ctx: RetrievalContext) -> list[Hit]: ...


class Policy(Protocol):
    def evaluate(self, hits: list[Hit], ctx: PolicyContext) -> PolicyResult: ...


class Adapter(Protocol):
    def render(self, tools: list[Tool]) -> Any: ...


class Source(Protocol):
    def discover(self) -> Iterable[Tool]: ...
```

Every retrieval returns a `Selection` carrying `tools`, `scores`, `rules_fired`, `filtered_out` (with reasons), and `timing`.

### 3.3 Repo layout

```
toolbroker/                 # core, target <3k lines
  sources/                 # mcp, python_fn, static_json
  index/                   # enrichment, embedders (fastembed default)
  store/                   # in-memory default
  retrieve/                # semantic default, pipeline primitives
  policy/                  # allow/deny, scopes, limits
  adapters/                # raw schema, langgraph
  proxy/                   # MCP proxy server
  cli/
packages/
  toolbroker-qdrant/ toolbroker-chroma/ toolbroker-pgvector/ ...
  toolbroker-openai/ toolbroker-ollama/ ...
  toolbroker-crewai/ toolbroker-claude-sdk/ ...
bench/                     # benchmark harness + labelled query sets
docs/                      # mkdocs-material
```

Monorepo. Core stays small on purpose: every plugin is a contributor-sized, independently ownable task.

---

## 4. Phases

### Phase 0 — Build and prove (weeks 1–2)
- **Clean-room implementation.** No code is extracted from any prior or client codebase;
  everything here is written from scratch under Apache-2.0. This removes the licensing
  and ownership question entirely rather than answering it.
- Build benchmark sets: **50 / 200 / 1000 tools, 300 labelled queries**.
- Record baseline accuracy **with and without** retrieval on GPT, Claude, and one open model.
- **Exit criteria:** reproducible benchmark, committed numbers, chart-ready data. This benchmark *is* the launch content.

### Phase 1 — MVP, v0.1 (weeks 3–6)
MCP + Python-function sources · fastembed embedder · in-memory store · semantic retriever · basic allow/deny policy · LangGraph adapter + raw schema adapter · MCP proxy mode · CLI `index`/`query`.
Docs: README, 20-line quickstart, one LangGraph example, one "no framework" example.
Ship to PyPI. **Public at this point, not before.**
- **Exit criteria:** offline quickstart works on a clean machine in <5 min; benchmark runs from the published package.

### Phase 2 — Pluggability, v0.3 (weeks 7–11)
Plugin entry points + `toolbroker-contrib` pattern · Qdrant, Chroma, pgvector stores · OpenAI/Ollama embedders · hybrid retriever · LLM reranker · OpenAPI source · OpenAI Agents + Claude Agent SDK + CrewAI adapters · OTel tracing · YAML config · hook system · deterministic mode.
Docs site (mkdocs-material) with a **"write your own X"** page for every interface.
- **Exit criteria:** a stranger can add a store plugin without touching core.

### Phase 3 — Community and control, v0.6 (weeks 12–16)
Per-agent scopes and risk tiers *(done)* · live index refresh when MCP servers change *(done)* · usage-based boosting *(done)* · `toolbroker bench` so anyone can benchmark their own tools · TypeScript port scoped (community-led if possible).
Public roadmap, RFC process, 15+ good-first-issues, monthly release cadence.

### Phase 4 — v1.0 (weeks 17–24)
API freeze *(prepared, not declared)* · stability guarantees *(documented — docs/stability.md)* · security review *(done — SECURITY.md)* · adapter conformance test suite *(done)* · case studies from real deployments (anonymised unless the operator opts in).

---

## 5. Repo and governance

- **Licence:** Apache-2.0.
- **Stack:** Python 3.10+, uv, ruff, pytest, typed everything (`mypy --strict` on core).
- **Monorepo:** `toolbroker` core + `toolbroker-<plugin>` packages.
- `CONTRIBUTING.md` with a **10-minute dev setup**, `CODEOWNERS`, issue templates.
- CI on every adapter against real-but-mocked MCP servers.
- Core carries **one** runtime dependency (pydantic) and no framework coupling;
  anything heavy lives in a plugin. This replaces the original "~3k lines" cap —
  see §0.

---

## 6. Launch and growth

- **Benchmark blog:** *"Tool selection accuracy collapses past 100 tools — here's the fix"*, with charts. That's the HN / Reddit / LinkedIn post.
- PixelLie video walkthrough; LearnWithRam carousel on the architecture.
- Submit to MCP registries, LangChain integrations list, awesome-mcp lists. Open PRs adding ToolBroker examples to the frameworks' own docs where they accept them.
- Weekly changelog posts for the first two months; respond to every issue within 24h.

---

## 7. Success metrics

| Milestone | Target |
|---|---|
| Month 1 | 500 stars · 3 external PRs · listed in 2 registries |
| Month 3 | 5 stores/embedders contributed by others · 1 production user not recruited by us · 3 inbound conversations that started from the repo |
| Month 6 | v1.0 · TS port started · 20+ contributors |

---

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Anthropic or the frameworks ship this natively (MCP is already moving toward tool search) | Be the **reference implementation that supports the spec**, plus the policy/control layer they won't build. Track the MCP spec actively; treat native tool-search as an upstream *source*, not a competitor. |
| Scope creep into orchestration | Hard rule in §2: ToolBroker never executes agent loops. Enforce in review. |
| One R&D person | Phase 1 is the only phase that must be done solo. Phase 2 is designed so each plugin is a contributor-sized task. |
| Benchmark seen as self-serving | Publish harness + labelled data + raw results; make `toolbroker bench` runnable on anyone's own tools (Phase 3). Report the cases where retrieval *loses*. |

---

## 9. Open questions (need answers before/at Phase 0)

1. ~~Naming.~~ **Settled: renamed from ToolScope to ToolBroker.** `toolscope` was already
   taken on PyPI by [ilya-kolchinsky/toolscope](https://github.com/ilya-kolchinsky/toolscope)
   (Red Hat, Apache-2.0, Feb 2026) — *the same idea under the same name*: semantic
   retrieval to filter tools before they reach the model. Availability verified
   2026-09-09: `toolbroker` and all nine `toolbroker-*` plugin names are unclaimed on
   PyPI, and `toolbroker` is free as a GitHub org. Domain still unchecked.
2. ~~Code provenance.~~ **Settled: clean-room only.** No code from any prior or client
   codebase enters this repo. Everything is written from scratch under Apache-2.0. This is
   a standing constraint on contributions, not a one-time check — see CONTRIBUTING.
3. ~~Benchmark corpus.~~ **Settled: out of scope here.** Realistic catalogues move to
   **MCP Hub**, a separate project; this repo keeps a synthetic corpus and says so. See
   the second hard rule in §2.
4. **"Correct tool" definition.** Exact-match on a single gold tool, or set-recall@k with multiple acceptable answers? Affects every number we publish.
5. **Open model choice** for the baseline (Llama / Qwen / Mistral) and where it runs.
6. **MCP proxy semantics.** Does `call_tool` proxy execution through ToolBroker (breaking "never executes")? Proposed resolution: proxying a *single tool call* on behalf of an MCP client is transport, not orchestration — allowed; driving a loop is not. Confirm and write into the docs.

---

## 10. Immediate next steps

1. Answer Q1 (name availability) and Q2 (code provenance).
2. Freeze the v0.1 interface sketch in §3.2.
3. Scaffold the repo: `uv` workspace, ruff/pytest/mypy config, Apache-2.0, CONTRIBUTING, CI skeleton, stubbed Protocols.
4. Stand up the benchmark harness before the retriever, so Phase 1 is measured from day one.
