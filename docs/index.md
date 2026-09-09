# ToolBroker

**Broker the right 5 tools out of 500 to your agent, on any framework, with rules you
control.**

Agent frameworks assume your tool list fits in the prompt. Past roughly a hundred
tools it does not:

- **Selection accuracy collapses.** The model picks a plausible-but-wrong tool, or the
  right description is lost in a 40k-token block.
- **Cost and latency scale with the catalogue, not the task.** Every turn re-sends
  every schema.
- **There is no control layer.** Nothing says *this agent may never call `delete_*`*,
  *this agent gets at most eight tools*, or *these tools require an approval scope*.

MCP made it trivial to *attach* hundreds of tools and did nothing to help you *choose*
among them.

ToolBroker answers one question: **given this query and this policy, which tools go in
the context?**

## What it is not

Not an agent framework, an orchestrator, or an execution runtime. **It never executes
agent loops.** That is an architectural rule, not a roadmap item — the frameworks it
integrates with own that job.

Forwarding a single tool call in [MCP proxy mode](mcp-proxy.md) is transport, not
orchestration, and is in scope.

## Design principles

| Principle | In practice |
|---|---|
| Framework-agnostic core | The core has no framework dependency and one runtime dependency (pydantic). |
| Everything is swappable | Six `Protocol` interfaces, each with a default and a plugin entry point. |
| Zero-config works, full-config possible | `pip install toolbroker` runs offline in 20 lines; the same object is drivable from YAML. |
| Control at every stage | [Hooks](extending.md#hooks) around discovery, indexing, retrieval, and policy. |
| Transparent decisions | Every selection carries its scores, rule firings, and exclusions. |

## Next

- [Quickstart](quickstart.md) — running in five minutes
- [Concepts](concepts.md) — the six interfaces and how they fit
- [Policy](policy.md) — the part frameworks will not build
