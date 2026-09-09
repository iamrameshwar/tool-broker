# API stability

## Current status: pre-1.0, nothing is frozen

ToolBroker is `0.1.0.dev0`. The public API can change in any release, and it still does.
This page describes the guarantees that will apply **from 1.0**, and what is being done
now so that those guarantees are cheap to keep rather than aspirational.

If you are building on ToolBroker today, pin an exact version.

## What counts as public

Public, and covered by the guarantees below from 1.0:

- Everything in `toolbroker.__all__` — importable as `from toolbroker import X`
- The six extension protocols: `Source`, `Embedder`, `Store`, `Retriever`, `Reranker`,
  `Policy`, `Adapter`
- `toolbroker.testing` — the conformance suites plugin authors build against
- The `toolbroker` CLI: its subcommands, flags, and exit codes
- The YAML config schema
- Entry-point group names: `toolbroker.stores`, `toolbroker.embedders`, and the rest

Internal, and changeable at any time:

- Anything with a leading underscore, at any depth
- Anything not re-exported from `toolbroker/__init__.py`, unless listed above
- Log message wording, and the exact text of exception messages
- The `bench/` directory — it is research tooling, not a product surface

!!! note "Reaching into a submodule"
    `from toolbroker.retrieve import HybridRetriever` works and will keep working, but
    module *layout* is not itself a guarantee. If you depend on something that is not
    top-level and not listed above, open an issue and it can be promoted.

## The guarantees, from 1.0

Semantic versioning, read strictly:

| Change | Requires |
|---|---|
| Removing a public name | major |
| Removing or renaming a parameter | major |
| Making an optional parameter required | major |
| Narrowing an accepted type | major |
| Changing a documented default | major |
| Adding an optional keyword parameter | minor |
| Adding a name to `__all__` | minor |
| Adding a method to a class | minor |
| Adding a method to a **protocol** | major — it breaks every implementer |
| Fixing behaviour that contradicts documentation | patch |
| Changing log or exception wording | patch |

### Deprecation

Nothing public is removed without one minor release of warning. A deprecated name keeps
working, emits `DeprecationWarning` naming its replacement, and is documented in the
changelog under **Deprecated** before it moves to **Removed**.

### Retrieval results are not an API

A release may change which tools a query returns — a better default, a fixed bug, an
improved enrichment weighting. Rankings are a quality property, not a contract, and
freezing them would freeze the library's ability to improve.

What *is* guaranteed is that changes are measured and reported. Any change to the
selection pipeline ships with before-and-after numbers from `bench/`, and the
[benchmarking](benchmarking.md) page records the ones already made — including a default
that changed and two candidate changes that were declined because they helped one
embedder and hurt another.

Pin an embedder and a ToolBroker version if you need reproducible rankings, and use
`toolbroker bench` on your own catalogue to see what a upgrade would do before taking it.

## How this is enforced

A snapshot of the entire public surface — every export, every public method and
property, every signature — lives in
[`tests/api/public_api.json`](https://github.com/iamrameshwar/tool-broker/blob/main/tests/api/public_api.json)
and is checked on every run.

The changes that hurt after a freeze are never the deliberate ones. They are a renamed
keyword argument, a property quietly becoming a method, a name dropped from `__all__`
during a refactor — invisible in review, and a broken import for somebody. The test
turns each of those into a failure.

Intentional changes are made by regenerating the snapshot:

```bash
uv run python tests/api/surface.py --write
```

That puts the diff in the pull request, where a reviewer can see exactly what callers are
being asked to absorb.

## Plugin compatibility

Plugin packages declare `toolbroker>=X`. A plugin built against 1.x keeps working across
1.x, because adding a method to a protocol is a major change precisely so that it cannot
silently break implementers.

The conformance suites are the practical guarantee: `toolbroker.testing` grows as the
contract is clarified, and a plugin that passes them on 1.x will pass on 1.y.

## Python support

Python 3.10 and up. A minor release may add support for a new Python version. Dropping
one follows the same rule as any removal: one minor release of deprecation warning
first, and never before that version reaches end of life upstream.
