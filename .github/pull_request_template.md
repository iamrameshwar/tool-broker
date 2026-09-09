## What this changes

<!-- One paragraph. -->

## Why

<!-- What problem this solves. Link the issue if there is one. -->

## Checklist

- [ ] `uv run pytest -q` passes
- [ ] `uv run ruff check . && uv run ruff format --check .` passes
- [ ] `uv run --with mypy mypy` passes
- [ ] Tests added for new behaviour
- [ ] Docs updated for user-facing changes

## If this touches the selection pipeline

Benchmark numbers before and after. A change that improves one query and quietly
regresses ten is the failure mode this project has to avoid.

```
before:
after:
```

## Scope check

- [ ] This does not add an agent execution loop, planner, or orchestrator
