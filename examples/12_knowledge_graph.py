"""Wiring an existing knowledge graph into tool selection.

If you already run a knowledge graph, you hold three things ToolBroker does not
and cannot infer: what your entities *are*, how your tools *relate*, and who may
reach what. This shows the three seams that matter, using a dictionary in place
of whatever you actually run — Neo4j, RDF, a Postgres edge table.

    python examples/12_knowledge_graph.py

The point to take away is the last one: a graph can *add* tools to a candidate
set, and it still cannot widen access, because every seam here runs upstream of
the policy engine.
"""

from toolbroker import DenyTools, Event, PolicyEngine, Tool, ToolBroker
from toolbroker.types import Hit

# Fifteen tools, not five. A six-tool catalogue cannot demonstrate anything
# about selection: at k=5 you get almost everything back whatever you do.
CATALOGUE = [
    Tool(name="issue_refund", namespace="billing", description="Refund a customer payment."),
    Tool(name="void_invoice", namespace="billing", description="Cancel an issued invoice."),
    Tool(name="create_invoice", namespace="billing", description="Raise an invoice for a client."),
    Tool(name="list_payments", namespace="billing", description="List payments on an account."),
    Tool(name="delete_ledger", namespace="billing", description="Delete the general ledger."),
    Tool(name="update_subscription", namespace="billing", description="Change a subscription."),
    Tool(name="log_activity", namespace="crm", description="Record an interaction with a contact."),
    Tool(name="search_contacts", namespace="crm", description="Find a contact by name or email."),
    Tool(name="update_contact", namespace="crm", description="Edit a contact record."),
    Tool(name="get_stock_level", namespace="inventory", description="Units on hand for a SKU."),
    Tool(name="adjust_stock", namespace="inventory", description="Correct a recorded quantity."),
    Tool(name="track_shipment", namespace="shipping", description="Locate a parcel in transit."),
    Tool(name="cancel_shipment", namespace="shipping", description="Stop a parcel before pickup."),
    Tool(name="create_ticket", namespace="support", description="Open a support ticket."),
    Tool(name="close_ticket", namespace="support", description="Close a resolved ticket."),
]
BY_ID = {tool.id: tool for tool in CATALOGUE}

# --- stand-ins for your graph ---------------------------------------------

#: Entities your users name in shorthand, and what they resolve to. A graph
#: knows "BER-01" is the Berlin warehouse; an embedding never will.
ENTITIES = {
    "berlin warehouse": "units on hand stock level SKU",
    "that depot": "units on hand stock level SKU",
}

#: Tools that belong together *procedurally*, which is the part vector search
#: cannot reach. Note the pair chosen: refunding a customer and logging the
#: interaction have almost no words in common and sit in different namespaces,
#: so no embedder will put them near each other — but every support process
#: does both. A graph knows that; similarity never will.
#:
#: `delete_ledger` is in here to prove a separate point further down.
COUNTERPARTS = {
    "billing/issue_refund": ["crm/log_activity", "billing/delete_ledger"],
}

broker = ToolBroker()


@broker.hooks.on(Event.TRANSFORM_QUERY)
def resolve_entities(query: str, **_) -> str:
    """Seam 1 — expand shorthand the graph can resolve and retrieval cannot."""
    lowered = query.lower()
    for phrase, expansion in ENTITIES.items():
        if phrase in lowered:
            return f"{query}\n{expansion}"
    return query


@broker.hooks.on(Event.TRANSFORM_HITS)
def expand_counterparts(hits, **_):
    """Seam 2 — make sure a task's counterpart tool travels with it.

    An agent about to refund a customer will need to log the interaction.
    Retrieval returns k independent tools and has no notion that a task needs
    a *set*.

    Two cases, and getting only one of them is how this silently does nothing:

    * **Boost.** On a small catalogue the pipeline overfetches enough to have
      already seen the counterpart — it is in the candidate list, simply ranked
      below the cut. Appending a duplicate achieves nothing; it has to be
      re-scored.
    * **Add.** On a large catalogue the counterpart never entered the pool at
      all, so there is nothing to re-score and it must be inserted.

    A graph neighbour is scored just under the hit that pulled it in. That is
    the editorial claim being made: a known counterpart of the best match beats
    the fifth-best fuzzy match. Score it under the whole result instead and
    truncation to `k` removes it again.
    """
    by_id = {hit.id: hit for hit in hits}
    promoted: dict[str, float] = {}

    for hit in hits:
        for neighbour in COUNTERPARTS.get(hit.id, []):
            promoted[neighbour] = max(promoted.get(neighbour, 0.0), hit.score * 0.99)

    expanded = [
        Hit(
            tool=hit.tool,
            score=max(hit.score, promoted.pop(hit.id, 0.0)),
            components=hit.components,
        )
        for hit in hits
    ]
    # Whatever is left in `promoted` was never retrieved at all.
    expanded.extend(
        Hit(tool=BY_ID[tool_id], score=score)
        for tool_id, score in promoted.items()
        if tool_id in BY_ID and tool_id not in by_id
    )

    # Re-sort: truncation to `k` downstream is positional.
    expanded.sort(key=lambda hit: hit.score, reverse=True)
    return expanded


broker.index(CATALOGUE)

print("=" * 70)
print("1. The graph resolves an entity retrieval could not")
print("=" * 70)
print("  'what is in that depot' ->", broker.select("what is in that depot", k=1).tool_ids)
print("  -> 'that depot' means nothing to an embedder. The graph made it mean")
print("     warehouse BER-01, and the stock tool became reachable.")

print()
print("=" * 70)
print("2. The graph adds the counterpart retrieval never scored")
print("=" * 70)
QUERY = "give this customer their money back"

# Printed so the claim is checkable rather than asserted: this is what
# retrieval alone returns, with the graph hook disabled.
broker.hooks.unregister(Event.TRANSFORM_HITS, expand_counterparts)
bare = broker.select(QUERY, k=5).tool_ids
deep = broker.select(QUERY, k=8).tool_ids
broker.hooks.register(Event.TRANSFORM_HITS, expand_counterparts)
withgraph = broker.select(QUERY, k=5).tool_ids

print(f"  retrieval alone, k=5 : {bare}")
print(f"  with the graph, k=5  : {withgraph}")
print()
print(f"  log_activity in the top 5 without the graph? {'crm/log_activity' in bare}")
print(f"  ... in the top 8 without the graph?          {'crm/log_activity' in deep}")
print("  -> and it never will be. Nothing about refunding a payment resembles")
print("     recording a contact interaction, so no ranking change reaches it.")
print("     Every support process does both anyway, and the graph knows that.")

print()
print("=" * 70)
print("3. And it still cannot widen access")
print("=" * 70)
broker.set_policy(PolicyEngine([DenyTools(["*/delete_*"])]))
guarded = broker.select(QUERY, k=5)
print("  selected:", guarded.tool_ids)
denied = [exclusion.tool_id for exclusion in guarded.exclusions if "deny_tools" in exclusion.reason]
print("  refused by policy:", denied)
print()
print("  The graph offered `billing/delete_ledger` as a counterpart and policy")
print("  refused it. Every seam here runs *upstream* of the policy engine, so an")
print("  external system can widen the candidate set without widening access.")
print("  That is what makes it safe to let a graph inject tools at all.")
