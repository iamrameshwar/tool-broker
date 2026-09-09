"""Per-agent policy: two agents, one catalogue, different privileges.

python examples/03_policy.py
"""

from toolbroker import (
    AgentPolicy,
    DenyTools,
    MaxRisk,
    MaxTools,
    PolicyEngine,
    RequireScopes,
    RiskTier,
    Tool,
    ToolBroker,
)

CATALOGUE = [
    Tool(name="search_tickets", namespace="ops", description="Search support tickets"),
    Tool(name="read_logs", namespace="ops", description="Read recent service logs"),
    Tool(
        name="restart_service",
        namespace="ops",
        description="Restart a running production service",
        risk=RiskTier.HIGH,
    ),
    Tool(
        name="delete_database",
        namespace="ops",
        description="Permanently drop a production database",
        risk=RiskTier.CRITICAL,
    ),
    Tool(
        name="issue_refund",
        namespace="billing",
        description="Refund a customer payment",
        risk=RiskTier.MEDIUM,
        required_scopes=frozenset({"payments:write"}),
    ),
]

broker = ToolBroker()
broker.index(CATALOGUE)

broker.set_policy(
    AgentPolicy(
        {
            # A support bot: read-only, small context, never destructive.
            "support": PolicyEngine(
                [MaxRisk(RiskTier.LOW), DenyTools(["*/delete_*"]), RequireScopes()],
                [MaxTools(limit=3)],
            ),
            # An on-call agent: may act, but still never drops a database.
            "oncall": PolicyEngine(
                [MaxRisk(RiskTier.HIGH), DenyTools(["*/delete_*"]), RequireScopes()],
                [MaxTools(limit=5)],
            ),
        }
    )
)

query = "production is down, fix it"
for agent in ("support", "oncall"):
    selection = broker.select(query, k=5, agent=agent)
    print(f"{agent:>8}: {list(selection.tool_ids)}")

print()
print("why support could not restart anything:")
support = broker.select(query, k=5, agent="support")
for exclusion in support.exclusions:
    if exclusion.stage.value == "policy":
        print(f"  {exclusion.tool_id}: {exclusion.reason}")

print()
print("scopes unlock gated tools:")
for scopes in ((), ("payments:write",)):
    selection = broker.select("refund this customer's payment", k=3, agent="oncall", scopes=scopes)
    held = ", ".join(scopes) or "none"
    print(f"  scopes={held:<16} -> {list(selection.tool_ids)}")
