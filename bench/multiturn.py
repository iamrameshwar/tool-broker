"""Does folding conversation history into the query help, and how much?

Every other harness here scores a single sentence. Real agents do not get one:
by turn five the user says "cancel that one" and the referent was named three
turns ago. Selecting on the last message alone is the known weakness of this
whole approach, and until now it was unmeasured.

    uv run python bench/multiturn.py
    uv run python bench/multiturn.py --sizes 200 --turns 0 1 2 3 --weights 1 2 3

Each case is a short conversation whose **final turn is deliberately
under-specified** — a pronoun, an ellipsis, a follow-up — with the referent
sitting in earlier turns. The `last turn only` row is the honest baseline: what
the library did before `toolbroker.conversation` existed.

Read the sweep for the shape, not the peak. The interesting question is not
which cell wins by one query, it is whether more history keeps helping or starts
diluting the turn that actually asks for something.

**A limit of this corpus, stated up front:** every conversation here has exactly
one prior user turn. The ``2 turns``, ``3 turns`` and ``4 turns`` rows are
therefore identical to ``1 turn`` *by construction* — there is nothing further
back to include. They are printed anyway because their agreeing is the check
that the depth control does what it claims, but do not read them as evidence
that depth stops helping. Resolving that needs longer conversations than these.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from toolbroker import ToolBroker
from toolbroker.conversation import contextual_query
from toolbroker.index.embedders import HashingEmbedder
from toolbroker.sources import StaticJSONSource

DATA = Path(__file__).parent / "data"
DEFAULT_SIZES = (50, 200, 1000)
DEFAULT_TURNS = (0, 1, 2, 3, 4)
DEFAULT_WEIGHTS = (1, 2, 3)
K = 5

#: Hand-written conversations. The last turn is always the one being served,
#: and always insufficient on its own — that is the point of the set.
CASES: list[tuple[list[tuple[str, str]], list[str]]] = [
    (
        [
            ("user", "what is sitting in the Berlin warehouse"),
            ("assistant", "Berlin currently holds 412 distinct SKUs."),
            ("user", "how many of the blue ones are left there"),
        ],
        ["inventory/get_stock_level"],
    ),
    (
        [
            ("user", "customer 8812 says their parcel is late"),
            ("assistant", "I can look that up."),
            ("user", "where is it right now"),
        ],
        ["shipping/track_shipment"],
    ),
    (
        [
            ("user", "we charged this customer twice for order 4471"),
            ("user", "give them their money back"),
        ],
        ["billing/issue_refund"],
    ),
    (
        [
            ("user", "the checkout service is throwing 500s"),
            ("assistant", "Error rate is 4% over the last ten minutes."),
            ("user", "just bounce it"),
        ],
        ["infrastructure/restart_service"],
    ),
    (
        [
            ("user", "a new engineer starts on Monday"),
            ("user", "set that up please"),
        ],
        ["identity/create_user"],
    ),
    (
        [
            ("user", "we shipped order 331 to the wrong address"),
            ("user", "stop it before it leaves"),
        ],
        ["shipping/cancel_shipment"],
    ),
    (
        [
            ("user", "this ticket has been open eleven days"),
            ("assistant", "It is assigned to the tier-one queue."),
            ("user", "get someone senior on it"),
        ],
        ["support/escalate_ticket"],
    ),
    (
        [
            ("user", "the invoice we sent Acme has the wrong amount"),
            ("user", "kill it"),
        ],
        ["billing/void_invoice"],
    ),
    (
        [
            ("user", "black friday starts in two hours"),
            ("assistant", "Current capacity is four replicas."),
            ("user", "we are going to need more"),
        ],
        ["infrastructure/scale_deployment"],
    ),
    (
        [
            ("user", "a contractor's engagement ended on Friday"),
            ("user", "make sure they cannot get in any more"),
        ],
        ["identity/deactivate_user"],
    ),
    (
        [
            ("user", "someone committed an API key to a public repo"),
            ("user", "deal with it"),
        ],
        ["identity/rotate_api_key"],
    ),
    (
        [
            ("user", "the customer wants to move to the annual plan"),
            ("user", "go ahead"),
        ],
        ["billing/update_subscription"],
    ),
    (
        [
            ("user", "I just got off a call with the Acme buyer"),
            ("user", "make a note of that"),
        ],
        ["crm/log_activity"],
    ),
    (
        [
            ("user", "we keep running out of the 14-inch model"),
            ("user", "what else is close to running out"),
        ],
        ["inventory/list_low_stock"],
    ),
    (
        [
            ("user", "latency on the API has been climbing all week"),
            ("user", "show me the trend"),
        ],
        ["observability/get_metrics"],
    ),
    (
        [
            ("user", "my phone has been going off all night about the disk alert"),
            ("user", "make it stop, we know about it"),
        ],
        ["observability/silence_alert"],
    ),
    (
        [
            ("user", "one specific checkout hung yesterday afternoon"),
            ("assistant", "Do you have a request id?"),
            ("user", "yes, follow it all the way through"),
        ],
        ["observability/get_trace"],
    ),
    (
        [
            ("user", "the release we pushed this morning broke search"),
            ("user", "put it back how it was"),
        ],
        ["infrastructure/rollback_release"],
    ),
    (
        [
            ("user", "we have two records for the same person in the CRM"),
            ("user", "combine them"),
        ],
        ["crm/merge_contacts"],
    ),
    (
        [
            ("user", "an employee is locked out before a client call"),
            ("user", "sort it out quickly"),
        ],
        ["identity/reset_password"],
    ),
    (
        [
            ("user", "the bug this customer reported went out in today's release"),
            ("user", "we can wrap it up now"),
        ],
        ["support/close_ticket"],
    ),
    (
        [
            ("user", "legal is asking about response times last quarter"),
            ("user", "which ones did we miss"),
        ],
        ["support/list_sla_breaches"],
    ),
    (
        [
            ("user", "we need to bill Acme for last month's consulting"),
            ("user", "draw one up"),
        ],
        ["billing/create_invoice"],
    ),
    (
        [
            ("user", "has this customer actually paid us for order 91"),
            ("user", "check"),
        ],
        ["billing/list_payments"],
    ),
    (
        [
            ("user", "we are moving units from the London depot to Berlin"),
            ("user", "do it"),
        ],
        ["inventory/transfer_stock"],
    ),
    (
        [
            ("user", "the recorded count for this SKU is wrong"),
            ("user", "fix the number"),
        ],
        ["inventory/adjust_stock"],
    ),
    (
        [
            ("user", "a customer emailed support last week and heard nothing"),
            ("user", "find it"),
        ],
        ["support/search_tickets"],
    ),
    (
        [
            ("user", "their job title changed to VP of Operations"),
            ("user", "update the record"),
        ],
        ["crm/update_contact"],
    ),
    (
        [
            ("user", "what is running in eu-west right now"),
            ("user", "list them"),
        ],
        ["infrastructure/list_instances"],
    ),
    (
        [
            ("user", "customers are reporting errors but our dashboard is green"),
            ("user", "go and read what the servers actually said"),
        ],
        ["observability/query_logs"],
    ),
]


def load(size: int, embedder: Any) -> ToolBroker:
    """Build an indexed catalogue at ``size``."""
    broker = ToolBroker(embedder=embedder, cache_embeddings=True)
    broker.add_source(StaticJSONSource(DATA / f"tools_{size}.json"))
    broker.index()
    return broker


def recall(broker: ToolBroker, *, turns: int, weight: int) -> float:
    """Share of conversations whose gold tool reaches the top ``K``."""
    hits = 0
    for conversation, expected in CASES:
        query = (
            conversation[-1][1]
            if turns == 0
            else contextual_query(conversation, max_turns=turns, weight=weight)
        )
        if set(expected) & {tool.id for tool in broker.select(query, k=K).tools}:
            hits += 1
    return hits / len(CASES)


def main() -> None:
    """Sweep history depth and current-turn weight."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--turns", type=int, nargs="+", default=list(DEFAULT_TURNS))
    parser.add_argument("--weights", type=int, nargs="+", default=list(DEFAULT_WEIGHTS))
    parser.add_argument("--out", type=Path, help="write a markdown table here")
    args = parser.parse_args()

    if not (DATA / f"tools_{args.sizes[0]}.json").exists():
        raise SystemExit("run `python bench/generate.py` first")

    try:
        from toolbroker.index.embedders.fastembed import FastEmbedEmbedder

        embedder: Any = FastEmbedEmbedder()
    except Exception as exc:
        print(f"fastembed unavailable ({exc}); using the lexical embedder\n")
        embedder = HashingEmbedder(dim=512)

    print(f"{len(CASES)} conversations, k={K}, embedder {getattr(embedder, 'id', embedder)}")
    print(f"one case is worth {1 / len(CASES):.3f} recall — read the shape, not the peak\n")

    rows: list[str] = []
    for size in args.sizes:
        broker = load(size, embedder)
        header = f"{'history':>9}" + "".join(f"{'weight ' + str(w):>12}" for w in args.weights)
        print(f"catalogue: {size} tools")
        print(header)
        print("-" * len(header))
        for turns in args.turns:
            label = "last only" if turns == 0 else f"{turns} turn{'s' if turns > 1 else ''}"
            cells = []
            for weight in args.weights:
                score = recall(broker, turns=turns, weight=weight)
                cells.append(f"{score:>12.3f}")
                rows.append(f"| {size} | {label} | {weight} | {score:.3f} |")
            print(f"{label:>9}" + "".join(cells))
        print()

    if args.out:
        args.out.write_text(
            "# Multi-turn selection\n\n"
            f"{len(CASES)} conversations whose final turn is under-specified, k={K}.\n"
            f"`last only` is the baseline: the query as the library saw it before\n"
            "`toolbroker.conversation` existed.\n\n"
            "| tools | history | weight | recall@5 |\n|---|---|---|---|\n" + "\n".join(rows) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
