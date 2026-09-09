"""Teaching a catalogue to say "no tool fits".

By default a retriever always returns its k best guesses, so a question no tool
can answer still produces confident suggestions and the model uses one. This
shows the failure, then fixes it with a floor derived from the catalogue itself.

    python examples/07_saying_no.py
"""

from toolbroker import MinScore, PolicyEngine, ToolBroker, calibrate_floor


def issue_refund(order_id: str, amount_cents: int) -> dict:
    """Refund a payment.

    Args:
        order_id: The order to refund.
        amount_cents: How much to refund, in cents.
    """
    return {}


def get_stock_level(sku: str) -> int:
    """Units on hand for a SKU.

    Args:
        sku: The stock keeping unit.
    """
    return 0


def restart_service(name: str) -> None:
    """Restart a running service.

    Args:
        name: The service to restart.
    """


def query_logs(service: str, since: str) -> list[str]:
    """Search application logs over a time range.

    Args:
        service: Which service to search.
        since: ISO timestamp to search from.
    """
    return []


broker = ToolBroker()
broker.add_functions([issue_refund, get_stock_level, restart_service, query_logs])
broker.index()

UNANSWERABLE = "who won the 1974 world cup"
REAL = "the customer wants their money back"

print("--- with no floor ---")
print(f"  {REAL!r}")
print(f"    -> {list(broker.select(REAL, k=2).tool_ids)}")
print(f"  {UNANSWERABLE!r}")
print(f"    -> {list(broker.select(UNANSWERABLE, k=2).tool_ids)}")
print("  ...which the model will happily call.")
print()

# Calibrate: run queries that certainly match nothing, see what they score,
# and put the floor above them. No labelled data required.
report = calibrate_floor(broker, samples=[REAL, "how much stock is left", "restart the api"])
print(report.summary())
print()

floor = report.recommended(percentile=95)
broker.set_policy(PolicyEngine(selection_rules=[MinScore(floor)]))

print(f"--- with MinScore({floor:.3f}) ---")
for query in (REAL, UNANSWERABLE):
    selection = broker.select(query, k=2)
    answer = list(selection.tool_ids) or "(no tool fits)"
    print(f"  {query!r}\n    -> {answer}")

print()
print("the refusal is explained, not silent:")
for exclusion in broker.select(UNANSWERABLE, k=2).exclusions:
    if exclusion.stage.value == "policy":
        print(f"  {exclusion.tool_id}: {exclusion.reason}")
