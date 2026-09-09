# Contributing to ToolBroker

## 10-minute dev setup

```bash
git clone https://github.com/iamrameshwar/tool-broker
cd toolbroker
curl -LsSf https://astral.sh/uv/install.sh | sh    # if you don't have uv
uv sync --all-extras
uv run pytest -q
```

That's it. If `uv run pytest` doesn't pass on a clean clone, that's a bug — please open
an issue.

### The checks CI runs

```bash
uv run ruff check .            # lint
uv run ruff format .           # format
uv run --with mypy mypy        # types (strict)
uv run pytest -q               # core tests
uv run pytest packages/toolbroker-qdrant -q      # one plugin at a time
```

CI runs the core suite and **each plugin in its own job**, which is also how you should
run them locally.

!!! note "Why not `pytest tests packages`?"
    Running every plugin in one process loads a lot of native extensions —
    onnxruntime, Chroma's Rust bindings, libpq — and roughly one run in four ends with
    `libc++abi: recursive_mutex lock failed` **after all tests have passed**, aborting
    the process at interpreter shutdown with exit code 134.

    It is a teardown ordering race between unrelated C++ libraries, not a ToolBroker bug:
    the suite reports `899 passed` first, and no smaller combination reproduces it. We
    checked ours anyway — the Postgres tests leak no connections and no tables. CI never
    hits it because no job loads that combination. If you see it locally, run the
    packages separately.

### Tests that need a service

Only `toolbroker-pgvector` does. Without `TOOLBROKER_PG_DSN` its suite skips, so you are
not blocked:

```bash
docker run -d -e POSTGRES_PASSWORD=toolbroker -e POSTGRES_DB=toolbroker \
  -p 55432:5432 pgvector/pgvector:pg16
TOOLBROKER_PG_DSN=postgresql://postgres:toolbroker@localhost:55432/toolbroker \
  uv run pytest packages/toolbroker-pgvector
```

Skipping is convenient and dangerous: a CI job whose service failed to start would
report "56 skipped" and pass. The pgvector job therefore asserts that nothing skipped.

## Provenance: clean-room only

Every line in this repo is written from scratch under Apache-2.0. **Do not paste code
from a prior employer, a client codebase, or any project whose licence you have not
checked** — not into a PR, not into an issue, not as "just a sketch".

If a PR contains code you did not write, say where it came from and under what licence.
Unattributed third-party code will be removed and the PR closed, regardless of quality.
This is not bureaucracy: a licensing problem found after adoption is unfixable without
a rewrite.

## The two architectural rules

**ToolBroker never executes agent loops.** It selects tools and returns their
definitions; the caller's framework calls them. A PR that adds an execution loop,
a planner, or an orchestrator will be declined no matter how good it is — that scope
belongs to the frameworks we integrate with.

**ToolBroker never curates tool catalogues.** It reaches real catalogues at runtime
through `MCPSource` and `OpenAPISource`; it does not ship them. A PR that vendors a
tool corpus, scrapes public MCP servers into this repo, or adds a curated catalogue
will be declined — that scope belongs to **MCP Hub**, a separate project. The synthetic
corpus under `bench/` exists to make the harness runnable offline and is labelled as
synthetic; it is not a catalogue and should not grow into one. This is also a
provenance rule: third-party tool descriptions carry the same licensing question as
third-party code, and the answer has to be no for the same reason.

Forwarding a *single* tool call in MCP proxy mode is transport, not orchestration, and
is in scope.

## Where to add things

The core is capped at roughly 3k lines. Everything else belongs in a plugin, which is
why every extension point is an entry point rather than an `if` branch in the core.

| You want to add | Put it in | Register under |
|---|---|---|
| A vector store (Pinecone, LanceDB, …) | `packages/toolbroker-<name>/` | `toolbroker.stores` |
| An embedder (OpenAI, Voyage, Ollama, …) | `packages/toolbroker-<name>/` | `toolbroker.embedders` |
| A framework adapter (CrewAI, ADK, …) | `packages/toolbroker-<name>/` | `toolbroker.adapters` |
| A tool source (OpenAPI, gRPC, …) | `packages/toolbroker-<name>/` | `toolbroker.sources` |
| A retrieval strategy | core, if it has no new dependency | `toolbroker.retrievers` |
| A policy rule | core `policy/rules.py`, if it needs no I/O | `toolbroker.policies` |

### Writing a plugin

Implement the relevant `Protocol` from `toolbroker.protocols` — you do not need to
subclass anything — then prove it satisfies the contract:

```python
from toolbroker.testing import StoreConformanceSuite


class TestMyStore(StoreConformanceSuite):
    @pytest.fixture
    def store(self):
        return MyStore(dim=self.DIM)
```

If the conformance suite is missing a guarantee your store had to make, that's a gap
in the contract — please say so in the PR.

## Code standards

- **Typed everything.** `mypy --strict` runs over `src/toolbroker`.
- **Docstrings on public API.** Google style, enforced by ruff's `D` rules.
- **Comments explain *why*.** The code already says what. A comment that restates the
  line below it will be asked about in review.
- **No new core dependencies.** The core has one (pydantic) and that is a feature. If
  your change needs a library, it needs an extra or a plugin.
- **Tests describe behaviour, not implementation.** `test_filters_apply_before_k`
  survives a refactor; `test_search_calls_matrix_similarities` does not.

## Changing the public API

A snapshot of every export, public method, property, and signature lives in
`tests/api/public_api.json` and is checked on every run. If your change moves it, the
test fails with what moved.

Intentional changes are made by regenerating it:

```bash
uv run python tests/api/surface.py --write
```

That puts the diff in your pull request, where a reviewer can see exactly what callers
are being asked to absorb. Do not regenerate it to silence a failure you did not intend —
that is the failure doing its job.

See [API stability](docs/stability.md) for what counts as public and which changes need
which version bump.

## Pull requests

1. One logical change per PR.
2. Include tests. New behaviour without a test will be asked for one.
3. If you changed the selection pipeline, run `toolbroker bench` before and after and put
   the numbers in the description. Retrieval changes that improve one query and quietly
   regress ten are the failure mode this project has to avoid.
4. Update the docs page for anything user-facing.

## Reporting a retrieval bug

The most useful bug report is a failing benchmark case:

```jsonl
{"query": "the thing I asked", "expected": ["namespace/the_tool_i_wanted"], "note": "got X instead"}
```

Attach that plus your tool catalogue (`toolbroker index --out tools.json`) and the
problem is reproducible in one command.
