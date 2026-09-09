# Security

ToolBroker decides which tools an agent can see and call. That makes it a control
surface: a bug here does not crash a service, it quietly grants privilege. This
document says what it defends against, what it does not, and how to report a problem.

## Reporting a vulnerability

Please **do not** open a public issue. Email `trivedirameshwar@gmail.com` with
`[toolbroker security]` in the subject. Expect an acknowledgement within 72 hours.

Include a reproduction if you can — the most useful report is a policy configuration
plus a tool definition that gets through it.

## Threat model

**Trusted:** the operator who writes the policy, the process ToolBroker runs in, and the
code that calls `select()`.

**Untrusted:** everything that arrives through a source. MCP servers, OpenAPI specs, and
static JSON files supply tool names, descriptions, and schemas written by people who
never expected this library to read them — and sometimes by people who did.

**Partly trusted:** the agent. A well-behaved agent asks for what it needs. A
compromised or prompt-injected one asks for whatever the attacker wants, which is
precisely why policy is enforced outside the model rather than by asking it nicely.

## What ToolBroker defends against

### A model choosing a tool it should not have

Policy is evaluated in Python, after retrieval and before rendering. The model never
sees a denied tool, so it cannot call one — no instruction in the prompt, and no
instruction injected into the prompt, changes that.

This is the meaningful mitigation for prompt injection. ToolBroker cannot stop a hostile
tool description from hijacking a model's reasoning. It can ensure that a hijacked model
still only reaches tools the operator permitted, which turns "the agent did something
catastrophic" into "the agent did something useless".

Get the most from it by keeping the permitted set small and specific:

```python
PolicyEngine(
    [MaxRisk(RiskTier.LOW), DenyTools(["*/delete*", "*/drop*"]), RequireScopes()],
    [MinScore(0.55), MaxTools(limit=5)],
)
```

### Direct calls that bypass search

The [MCP proxy](docs/mcp-proxy.md) re-evaluates policy on every `describe_tool` and
`call_tool`. Search-time filtering is not authorization: a client can name any id it
likes, including one it learned somewhere else.

### Enumeration of tools you cannot see

A denied tool and a nonexistent one produce the *same* client-facing error, and
suggestions are drawn only from tools the caller may already see. The real reason is
logged server-side, so an operator debugging a legitimate agent still gets a straight
answer.

### One tenant reading another's tools

`TenantIsolation` reads a tool's owner from `tool.metadata[tenant_key]` and denies
whenever it is not certain the caller should see it — including when the request declares
no tenant at all. It is marked `mandatory`, so it is evaluated whatever the combining mode
and wherever it sits in the rule list.

Two limits to know:

* **The tenant tag is only as trustworthy as its source.** Metadata arrives from MCP
  servers and OpenAPI specs. If one tenant controls a server, it controls what that server
  claims. Assign tenancy yourself — a `TRANSFORM_TOOL` hook that stamps it by `source_id`
  is the reliable way — rather than trusting what a server asserts about itself.
* **The MCP proxy has no tenant dimension.** It serves one catalogue over one connection
  and the protocol carries no tenant identity to enforce. Run a proxy per tenant, or use
  the library directly, where `tenant=` is a per-call argument.

### Naming tricks in tool identifiers

Deny and allow patterns match case-insensitively, surrounding whitespace is stripped
from names and namespaces, and control characters are rejected outright. See
[Fixed issues](#fixed-issues) for why each of those is there.

## What ToolBroker does not defend against

**Prompt injection through tool descriptions.** A hostile MCP server can put
instructions in a description, and that text reaches the model. No filter reliably
detects this. Bound the damage with policy, and treat a new MCP server the way you would
treat a new dependency.

**A malicious tool implementation.** ToolBroker never executes anything. If your
framework calls a tool, whatever that tool does is between it and your framework.

**Confidentiality of the catalogue itself.** Tool names and descriptions are held in
memory and written to whatever store you configure. A vector store you do not control
sees them.

**Resource exhaustion from a hostile source.** A server returning a million tools, or
one description of a gigabyte, will consume memory. Bound it at the source if the source
is not yours.

**Anything a hook does.** Hooks run arbitrary caller code inside the pipeline. They are
an extension point, not a sandbox.

### Tools that change after you approved them

The injection risk people discuss is a hostile description on the day you connect a
server. The one that gets you is a description that changes *later*. `DriftGuard` compares
every refresh against the last approved content and, with `quarantine=True`, keeps serving
the approved version until a human accepts the new one.

Know what it does and does not cover:

* **Changes to known tools** are guarded. Description, schema, risk, scopes and tags.
* **Arrivals are not**, by default. `trust_on_first_use` approves a tool the first time it
  is seen, because the alternative is a catalogue that starts empty. A renamed tool arrives
  as a *new* tool, so a hostile server can rename its way past change-quarantine. Set
  `trust_on_first_use=False` to hold arrivals too, and approve them deliberately.
* **Privilege changes are reported but never held.** Holding one back could mean serving a
  tool at a *lower* risk tier than the server now claims, which is the wrong way to fail.

## Hardening checklist

- [ ] `strict_agents: true` — an unregistered agent name is a typo, and the permissive
      fallback hands it the whole catalogue.
- [ ] `MaxRisk` rather than only name patterns. A glob written as `*/delete_*` genuinely
      does not match `deleteUser`; risk tiers classify a *category* of danger, and
      `classify_by_name` lowercases before matching.
- [ ] `RequireScopes()` in every agent's rule list, so scope-gated tools stay gated.
- [ ] `MinScore` — without a floor, a query no tool can serve still returns five
      confident suggestions. Find your threshold with `toolbroker calibrate`.
- [ ] `MaxTools` — a smaller permitted set is a smaller blast radius.
- [ ] Namespace every source, so two servers cannot collide on a tool name.
- [ ] Run new policies with `dry_run=True` first and read the firings.
- [ ] Log `selection.exclusions` and `selection.firings` where you can audit them.
- [ ] Pin your MCP servers. A server that silently gains a `delete_*` tool is a supply
      chain event.
- [ ] `DriftGuard(quarantine=True)` with a persisted `path`, so a description that changes
      after you approved it is held rather than served.
- [ ] `trust_on_first_use=False` where a hostile server is in the threat model — without
      it, a rename walks past change-quarantine as a new tool.
- [ ] `TenantIsolation(shared_tools=False)` in a multi-tenant deployment, so a tool nobody
      remembered to tag is invisible rather than universal.
- [ ] Assign tenancy yourself with a `TRANSFORM_TOOL` hook rather than trusting the
      `tenant` a server claims about its own tools.
- [ ] Alert on `ResilientRetriever.degraded`. A silently degraded retriever answers every
      request with worse tools and nothing ever fails.

## Fixed issues

Found during the pre-1.0 review of this repository. None were released.

| # | Issue | Why it mattered |
|---|---|---|
| 1 | Deny/allow globs matched case-sensitively | `*/delete_*` did not match `DeleteUser`, which plenty of APIs generate. Silent, with nothing to indicate the rule was inert. |
| 2 | Whitespace and control characters allowed in names | A namespace of `"admin "` evaded `admin/*` while looking identical to `"admin"` in every log an operator might check. |
| 3 | `AgentPolicy` failed open on an unregistered agent | A typo in `agent="support"` granted the entire catalogue. |
| 4 | `describe_tool` ignored policy | A client denied a tool by search could still fetch its description and full parameter schema. |
| 5 | "Did you mean" hints leaked denied tools | Guessing a substring confirmed the existence and exact id of tools policy was meant to hide. |
| 6 | `first_match` short-circuited tenant isolation | An `allow` glob earlier in the list granted another tenant's tool before the boundary was consulted. A security guarantee that depends on rule order is not a guarantee, so isolation rules are now `mandatory`: always evaluated, and their deny is final under either combining mode. |
| 7 | Drift detection ignored tag changes | `DenyTags` and `RequireTags` gate on tags, so a server quietly dropping `destructive` walked through a rule the operator believed was protecting them, with nothing reported. Tags now count as a privilege change alongside risk and scopes. |
| 8 | Renaming a tool walked past quarantine | A renamed tool has a different id, so the diff sees a removal and an addition rather than a change — and quarantine, which guarded only changes, never fired. `trust_on_first_use=False` now genuinely holds new arrivals instead of merely declining to approve them. |

Issues 6 to 8 were found the same way as 1 to 5: by writing the attack rather
than the document. All three were in code added the same week, which is the
argument for doing this every time the gating surface grows rather than once
before release.

Each has a regression test in
[`tests/unit/test_policy_bypass.py`](tests/unit/test_policy_bypass.py) and
[`tests/integration/test_proxy.py`](tests/integration/test_proxy.py).

## Supported versions

Pre-1.0. Fixes land on `main` and in the next release; there are no backports yet.
