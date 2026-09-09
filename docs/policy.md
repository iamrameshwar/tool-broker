# Policy

Retrieval decides what is *relevant*. Policy decides what an agent is *allowed* to see.

## Many customers, one deployment

Scopes cannot isolate tenants, and it is worth being precise about why.
`RequireScopes` **abstains** for a tool that declares none — correct for a capability
check, and exactly wrong for an isolation boundary, because a tool nobody remembered to
tag stays visible to everyone. Isolation has to decide on every tool.

```yaml
policy:
  agents:
    tenant_app:
      tenant_isolation: true
      tenant_key: tenant      # read from tool.metadata[tenant_key]
      shared_tools: false     # untagged tools are invisible, not universal
```

```python
broker.select("export the ledger", k=5, agent="tenant_app", tenant="acme")
```

`TenantIsolation` denies whenever it is not certain the caller should see a tool:

| tool's tenant | request's tenant | outcome |
|---|---|---|
| `acme` | `acme` | allowed |
| `acme` | `globex` | denied — belongs to another tenant |
| `acme` | *not supplied* | denied — the request declared no tenant |
| *none* | anything | allowed only when `shared_tools` is true |

The third row is the one that matters. An undeclared tenant must never mean "all
tenants", or a caller that forgot to pass one sees every customer's tools. And
`shared_tools: false` is the safer setting wherever forgetting to tag a tool is the
likelier mistake than deliberately sharing it.

!!! warning "The MCP proxy has no tenant dimension"
    `toolbroker serve` exposes one catalogue over one connection, and MCP carries no
    tenant identity for it to enforce. Run a proxy per tenant, or use the library
    directly, where `tenant=` is a per-call argument.

## Rules

Two kinds. `Rule` judges one tool; `SelectionRule` judges the set.

```python
from toolbroker import (
    AllowTools,
    DenyTools,
    DenyTags,
    RequireTags,
    RequireScopes,
    MaxRisk,
    MaxCost,
    MaxTools,
    MinScore,
    PredicateRule,
    PolicyEngine,
    RiskTier,
)

engine = PolicyEngine(
    rules=[
        DenyTools(["*/delete_*", "admin/*"]),  # globs match the full namespace/name id
        MaxRisk(RiskTier.MEDIUM),
        RequireScopes(),
        DenyTags(frozenset({"internal"})),
    ],
    selection_rules=[
        MinScore(0.25),
        MaxTools(limit=8, keep_pinned=frozenset({"core/search"})),
    ],
)
```

`MinScore` matters more than it looks, and the benchmark shows exactly how much.
Without a floor, **every** query that no tool can serve still returns five confident
suggestions — negative-query accuracy is 0.000. The model then uses one of them.

ToolBroker ships no default threshold, because the useful value tracks your embedder's
score distribution. Find yours with `bench/threshold.py`; for bge-small on the reference
catalogue the curve looks like this:

| floor | positive queries | negative queries |
|---|---|---|
| 0.00 | 0.785 | 0.000 |
| 0.50 | 0.774 | 0.320 |
| 0.55 | 0.725 | 0.720 |
| 0.60 | 0.600 | 0.840 |

Pick from your own curve based on which error you would rather make.

### Calibrating without labelled data

You do not need a labelled set to pick a floor. `toolbroker calibrate` runs queries that
certainly match nothing, measures what they score against *your* catalogue, and puts the
floor above the noise:

```bash
toolbroker calibrate -c toolbroker.yaml --samples my_real_queries.txt
```

```text
top-1 score when nothing should match:
  min 0.457   median 0.505   max 0.572

  floor  blocks noise  samples emptied  samples thinned
-------------------------------------------------------
  0.505          48%               0%               2%
  0.565          88%               3%              20%
  0.572          96%               3%              22%

Noise reached 0.572 and your weakest genuine query scored 0.509.
The distributions overlap, so no floor separates them cleanly.
```

Three things worth reading there.

**"Thinned" is the number people forget.** A floor chosen from best-match scores looks
free — 3% emptied — while quietly deleting the third-ranked tool that was sometimes the
right answer. That is the 20%.

**The distributions overlap.** They almost always do. No floor separates noise from
genuine queries cleanly, so this is a choice about which error costs you more, not a
number to look up.

**Score scales are not comparable between embedders.** On the same catalogue, noise
tops out around 0.57 with bge-small and around 0.20 with the offline hashing embedder.
A threshold copied from someone else's setup is meaningless — which is exactly why
ToolBroker ships no default.

See [Benchmarking](benchmarking.md#finding-your-score-floor) for the labelled version of
the same sweep.

## Combining

Default is **deny-overrides**: every rule gets a say, and a single deny wins. That is
the IAM model reviewers already know, and it fails closed — a misordered rule list
cannot accidentally grant access.

`combining="first_match"` gives ordered, escape-early semantics. It is opt-in because
getting the order wrong is silent.

`default=Decision.DENY` turns the whole thing into an allowlist.

## Per-agent policies

One process usually hosts several agents with different privileges:

```python
from toolbroker import AgentPolicy

broker.set_policy(
    AgentPolicy(
        {
            "support": PolicyEngine([MaxRisk(RiskTier.LOW), RequireScopes()], [MaxTools(limit=3)]),
            "oncall": PolicyEngine(
                [MaxRisk(RiskTier.HIGH), DenyTools(["*/delete_*"])], [MaxTools(limit=5)]
            ),
        }
    )
)

broker.select("production is down", agent="support")
```

An unknown agent gets the default engine, which allows everything — an unconfigured
agent behaves as if there were no policy rather than silently receiving zero tools.

## Scopes

Scopes are the caller's credentials, checked against each tool's `required_scopes`:

```python
broker.select("refund this customer", agent="oncall", scopes={"payments:write"})
```

## Dry run

This is how a policy gets rolled out: run it against real traffic, read the firings,
then turn it on.

```python
engine = PolicyEngine([MaxRisk(RiskTier.LOW)], dry_run=True)
selection = broker.select("...")
selection.dry_run  # True
selection.exclusions  # what *would* have been removed
len(selection)  # unchanged — nothing was actually enforced
```

## YAML

Agent permissions are the kind of thing that should be reviewed like any other config:

```yaml
policy:
  default_k: 5
  agents:
    support:
      max_tools: 4
      max_risk: low
      deny: ["*/delete_*"]
      deny_tags: [internal]
      min_score: 0.2
      require_scopes: true
    auditor:
      default: deny            # allowlist mode
      allow: ["reporting/*", "logs/query_*"]
```

Unknown keys are an error, not a silent default. A typo in a policy file must fail
loudly — the alternative failure mode is an agent quietly running unrestricted.

## Custom rules

```python
from toolbroker.policy.rules import Rule, RuleContext
from toolbroker.types import Decision


class BusinessHoursOnly(Rule):
    name = "business_hours"

    def check(self, tool, context: RuleContext):
        if "after_hours_forbidden" not in tool.tags:
            return None  # abstain: no opinion on this tool
        if 9 <= datetime.now().hour < 17:
            return Decision.ALLOW, "within business hours"
        return Decision.DENY, "outside business hours"
```

Abstaining is a first-class outcome. A rule about billing tools should have no opinion
about search tools, and saying so explicitly keeps the combining logic simple.

For one-off logic there is `PredicateRule(predicate=lambda tool, ctx: ...)`.

## Hardening

Three defaults are worth changing in production, all covered in
[SECURITY.md](https://github.com/iamrameshwar/tool-broker/blob/main/SECURITY.md):

**Turn on `strict_agents`.** An agent name that is not registered is nearly always a
typo, and the permissive fallback hands it the whole catalogue:

```yaml
policy:
  strict_agents: true
```

**Prefer risk tiers to name patterns for categories of danger.** A glob written as
`*/delete_*` genuinely does not match `deleteUser` — that is the pattern doing what it
says. `MaxRisk` combined with `classify_by_name` catches both, because it lowercases and
matches on the leading verb.

**Set a `MinScore` floor.** Without one, a query no tool can serve still returns five
confident suggestions. See [Calibrating without labelled data](#calibrating-without-labelled-data).

## Policy applies at call time too

In [MCP proxy mode](mcp-proxy.md) the policy is re-evaluated when a tool is actually
invoked. Search-time filtering is not authorization: a client can call `call_tool` with
any id it likes, including one it learned elsewhere.
