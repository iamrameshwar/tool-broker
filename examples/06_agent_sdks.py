"""One catalogue, three agent frameworks.

Needs `pip install toolbroker-openai-agents toolbroker-claude-agent`.
Nothing here calls a model.

    python examples/06_agent_sdks.py
"""

from toolbroker import MaxRisk, PolicyEngine, RiskTier, ToolBroker, classify_by_name


def search_orders(customer_email: str, limit: int = 10) -> list[str]:
    """Find recent orders placed by a customer.

    Args:
        customer_email: The customer's email address.
        limit: Maximum number of orders to return.
    """
    return []


def issue_refund(order_id: str, amount_cents: int, reason: str) -> dict:
    """Return money to a customer for a specific order.

    Args:
        order_id: The order to refund.
        amount_cents: How much to refund, in cents.
        reason: Why the refund is being issued.
    """
    return {"refunded": amount_cents}


def delete_customer(customer_id: str) -> None:
    """Permanently erase a customer record.

    Args:
        customer_id: The customer to delete.
    """


broker = ToolBroker()
# Opt into name-based risk inference, so `delete_*` is HIGH without anyone
# having to remember to annotate it.
broker.add_functions(
    [search_orders, issue_refund, delete_customer], risk_classifier=classify_by_name
)
broker.index()

# The same policy governs every framework below. That is the point: the control
# layer does not move when the framework does.
broker.set_policy(PolicyEngine([MaxRisk(RiskTier.MEDIUM)]))

query = "customer is angry and wants their money back"
selection = broker.select(query, k=3)
print(f"query: {query!r}")
print("selected:", list(selection.tool_ids))
print()

# --- provider-native JSON, no framework ---------------------------------
print("OpenAI function-calling:")
for entry in broker.render("openai", selection):
    print("  ", entry["function"]["name"])

print("Anthropic tool use:")
for entry in broker.render("anthropic", selection):
    print("  ", entry["name"])
print()

# --- OpenAI Agents SDK ---------------------------------------------------
try:
    from toolbroker_openai_agents import OpenAIAgentsAdapter

    tools = OpenAIAgentsAdapter().render(selection.tools)
    print("OpenAI Agents SDK:")
    for tool in tools:
        print(f"   FunctionTool {tool.name} (strict={tool.strict_json_schema})")
    # agent = Agent(name="support", instructions="...", tools=tools)
except ImportError:
    print("OpenAI Agents SDK: not installed")
print()

# --- Claude Agent SDK ----------------------------------------------------
try:
    from toolbroker_claude_agent import ClaudeAgentAdapter

    adapter = ClaudeAgentAdapter()
    server = adapter.create_server(selection.tools, name="tools")
    allowed = adapter.allowed_tool_names(selection.tools, server="tools")
    print("Claude Agent SDK:")
    print(f"   in-process MCP server: {server['name']}")
    print(f"   allowed_tools: {allowed}")
    # options = ClaudeAgentOptions(mcp_servers={"tools": server}, allowed_tools=allowed)
except ImportError:
    print("Claude Agent SDK: not installed")
print()

# --- CrewAI --------------------------------------------------------------
try:
    from toolbroker_crewai import CrewAIAdapter

    crew_tools = CrewAIAdapter().render(selection.tools)
    print("CrewAI:")
    for tool in crew_tools:
        fields = ", ".join(tool.args_schema.model_fields) or "no arguments"
        print(f"   BaseTool {tool.name} ({fields})")
    # agent = Agent(role="Support", goal="...", backstory="...", tools=crew_tools)
except ImportError:
    print("CrewAI: not installed")
print()

# --- the tool policy blocked, in every framework at once -----------------
print("policy blocked:")
for exclusion in selection.exclusions:
    if exclusion.stage.value == "policy":
        print(f"   {exclusion.tool_id}: {exclusion.reason}")
