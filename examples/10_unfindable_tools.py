"""Finding the tools an agent will never be handed.

A catalogue can be wired up correctly and still contain tools that are, in
practice, invisible: two servers expose near-identical operations, or a tool's
description is too thin to match anything. Nothing errors. The agent just never
picks them, and you find out months later.

    python examples/10_unfindable_tools.py
"""

from toolbroker import ToolBroker, diagnose_catalogue
from toolbroker.sources import PythonFunctionSource


def create_ticket(subject: str, body: str) -> dict:
    """Open a new support ticket for a customer.

    Args:
        subject: One-line summary.
        body: Full description of the problem.
    """
    return {}


def open_ticket(title: str, details: str) -> dict:
    """Open a new support ticket for a customer.

    Args:
        title: One-line summary.
        details: Full description of the problem.
    """
    return {}


def track_shipment(tracking_number: str) -> dict:
    """Where a parcel is right now, by its carrier tracking number.

    Args:
        tracking_number: The carrier's tracking number.
    """
    return {}


def sync(target: str) -> dict:
    """Sync it.

    Args:
        target: What to sync.
    """
    return {}


broker = ToolBroker()
# Two servers, and each was written without knowing the other existed. That is
# the normal case once a catalogue is assembled from more than one place.
broker.add_source(
    PythonFunctionSource(
        [create_ticket, track_shipment], source_id="helpdesk", namespace="helpdesk"
    )
)
broker.add_source(
    PythonFunctionSource([open_ticket, sync], source_id="legacy_crm", namespace="legacy_crm")
)
broker.index()

report = diagnose_catalogue(broker, k=1)
print(report.summary())

print()
print("=" * 70)
print("The ranking above jumps from 10% to 36%, so anything under ~15% is")
print("'too close' for this embedder. Passing that turns the ranking into a")
print("verdict:")
print()

flagged = diagnose_catalogue(broker, k=1, margin=0.15)
for finding in flagged.unhealthy:
    print(f"  {finding.tool_id:<28} {', '.join(finding.issues)}")

print()
print("`helpdesk/create_ticket` and `legacy_crm/open_ticket` do the same thing")
print("and say so in the same words, so which one an agent gets is close to a")
print("coin flip. No ranking change fixes that — one of them should be deleted,")
print("or the descriptions should say what actually differs.")
