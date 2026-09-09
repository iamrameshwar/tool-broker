"""The hand-authored benchmark catalogue.

Forty tools across eight domains, written the way real MCP servers write them:
terse, inconsistent, and full of near-neighbours. The previous generator
produced descriptions that were uniform and verbose, which made retrieval look
easier than it is.

Deliberate difficulty in here:

* **Confusable pairs within a domain** — ``void_invoice`` and
  ``cancel_shipment`` both answer to "cancel it"; ``adjust_stock`` and
  ``transfer_stock`` both move inventory.
* **Cross-domain lexical traps** — ``deactivate_user`` and ``close_ticket``
  both "close" something; ``rollback_release`` and ``void_invoice`` both undo.
* **Terse descriptions** — one line, no examples, as a real server ships them.

Every tool here appears in all three catalogue sizes, so the scaling curve
measures added distractors rather than which tools happened to survive
truncation.
"""

from __future__ import annotations

from typing import Any

# (name, description, tags, risk)
Entry = tuple[str, str, list[str], str]

CATALOGUE: dict[str, list[Entry]] = {
    "billing": [
        ("issue_refund", "Refund a payment.", ["write", "money"], "high"),
        ("create_invoice", "Raise a new invoice for an account.", ["write", "money"], "medium"),
        ("void_invoice", "Void an invoice that has not been paid.", ["write", "money"], "high"),
        ("list_payments", "Payments received on an account.", ["read", "money"], "low"),
        (
            "update_subscription",
            "Change an account's plan or billing interval.",
            ["write", "money"],
            "medium",
        ),
    ],
    "inventory": [
        ("get_stock_level", "Units on hand for a SKU.", ["read"], "low"),
        ("adjust_stock", "Correct the recorded quantity for a SKU.", ["write"], "medium"),
        ("list_low_stock", "SKUs below their reorder point.", ["read"], "low"),
        ("transfer_stock", "Move units between warehouses.", ["write"], "medium"),
        ("reserve_stock", "Hold units against an order.", ["write"], "medium"),
    ],
    "shipping": [
        ("create_shipment", "Book a shipment with a carrier.", ["write"], "medium"),
        ("track_shipment", "Current status and location of a shipment.", ["read"], "low"),
        ("cancel_shipment", "Cancel a shipment before pickup.", ["write"], "high"),
        ("list_carriers", "Carriers available for a destination.", ["read"], "low"),
        ("estimate_delivery", "Predicted arrival date for a route.", ["read"], "low"),
    ],
    "identity": [
        ("create_user", "Add a user account.", ["write"], "medium"),
        ("deactivate_user", "Disable a user account.", ["write"], "high"),
        ("reset_password", "Send a password reset link.", ["write"], "medium"),
        ("rotate_api_key", "Issue a new API key and revoke the old one.", ["write"], "high"),
        ("list_permissions", "Roles and permissions granted to a user.", ["read"], "low"),
    ],
    "observability": [
        ("query_logs", "Search application logs over a time range.", ["read"], "low"),
        ("get_metrics", "Time series for a service metric.", ["read"], "low"),
        ("list_alerts", "Alerts currently firing.", ["read"], "low"),
        ("silence_alert", "Mute an alert for a period.", ["write"], "medium"),
        ("get_trace", "Fetch a distributed trace by id.", ["read"], "low"),
    ],
    "infrastructure": [
        ("restart_service", "Restart a running service.", ["write"], "high"),
        ("scale_deployment", "Change the replica count.", ["write"], "medium"),
        ("list_instances", "Compute instances in a region.", ["read"], "low"),
        ("rollback_release", "Revert a service to its previous release.", ["write"], "high"),
        ("deploy_release", "Roll out a build to an environment.", ["write"], "high"),
    ],
    "crm": [
        ("search_contacts", "Find contacts by name, email, or company.", ["read"], "low"),
        ("create_deal", "Open a sales opportunity.", ["write"], "medium"),
        ("log_activity", "Record a call, email, or meeting.", ["write"], "low"),
        ("merge_contacts", "Combine duplicate contact records.", ["write"], "high"),
        ("update_contact", "Edit fields on a contact.", ["write"], "medium"),
    ],
    "support": [
        ("search_tickets", "Find support tickets.", ["read"], "low"),
        ("create_ticket", "Open a support ticket.", ["write"], "low"),
        ("escalate_ticket", "Raise a ticket's priority and reassign it.", ["write"], "medium"),
        ("close_ticket", "Mark a ticket resolved.", ["write"], "medium"),
        ("list_sla_breaches", "Tickets past their response deadline.", ["read"], "low"),
    ],
}

# Filler exists so catalogue size can grow without the labelled tools changing.
# These are the tools a real hub accumulates that nobody ever queries for.
FILLER_VERBS = ["get", "list", "create", "update", "delete", "sync", "validate", "export"]
FILLER_NOUNS = [
    "report",
    "schema",
    "webhook",
    "template",
    "workspace",
    "label",
    "policy",
    "snapshot",
    "dataset",
    "pipeline",
    "connector",
    "quota",
    "region",
    "tag",
    "audit",
    "backup",
    "session",
    "token",
    "rule",
    "channel",
]


def core_tools() -> list[dict[str, Any]]:
    """Return the forty labelled tools."""
    tools: list[dict[str, Any]] = []
    for namespace, entries in CATALOGUE.items():
        for name, description, tags, risk in entries:
            tools.append(
                {
                    "name": name,
                    "namespace": namespace,
                    "description": description,
                    "tags": [*tags, namespace],
                    "risk": risk,
                }
            )
    return tools


def filler_tools(count: int) -> list[dict[str, Any]]:
    """Return ``count`` plausible but unlabelled tools."""
    tools: list[dict[str, Any]] = []
    index = 0
    while len(tools) < count:
        verb = FILLER_VERBS[index % len(FILLER_VERBS)]
        noun = FILLER_NOUNS[(index // len(FILLER_VERBS)) % len(FILLER_NOUNS)]
        suffix = index // (len(FILLER_VERBS) * len(FILLER_NOUNS))
        name = f"{verb}_{noun}" if suffix == 0 else f"{verb}_{noun}_v{suffix}"
        tools.append(
            {
                "name": name,
                "namespace": f"service_{index % 15:02d}",
                "description": f"{verb.capitalize()} a {noun}.",
                "tags": ["filler"],
                "risk": "high" if verb == "delete" else "low",
            }
        )
        index += 1
    return tools


def core_tool_ids() -> set[str]:
    """Identifiers of every labelled tool."""
    return {f"{tool['namespace']}/{tool['name']}" for tool in core_tools()}
