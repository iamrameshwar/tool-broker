"""Zero config, no framework, works offline.

python examples/01_quickstart.py
"""

from toolbroker import ToolBroker


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
    return {}


def check_inventory(sku: str) -> int:
    """Return how many units of a SKU are currently in stock.

    Args:
        sku: The stock keeping unit to look up.
    """
    return 0


def rotate_credentials(service_account: str) -> None:
    """Rotate the API keys for a service account.

    Args:
        service_account: The account whose keys should be rotated.
    """


broker = ToolBroker()
broker.add_functions([search_orders, issue_refund, check_inventory, rotate_credentials])
broker.index()

selection = broker.select("customer is angry and wants their money back", k=2)

print(selection.explain())
print()
print("tools you would put in the prompt:", list(selection.tool_ids))
