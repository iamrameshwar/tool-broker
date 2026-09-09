"""Find the ceiling on recall, and how much of it reranking could ever reach.

A reranker cannot promote a tool it never received. So before adding one —
cross-encoder, LLM, or anything else — it is worth knowing how deep you have to
fetch before the right tool is even in the pool, and what share of queries never
put it there at any depth.

    uv run python bench/ceiling.py
    uv run python bench/ceiling.py --sizes 1000 --depths 5 20 100 --misses

Three numbers come out of this, and they answer different questions:

* **recall@5** is what ships — the tools the model actually sees.
* **recall@N** is the ceiling for a reranker that overfetches ``N`` and reorders
  them. A perfect reranker turns recall@N into recall@5; a real one gets some
  fraction of the gap.
* **unreachable** is the share of queries whose tool never appears at any depth.
  No reranking, fusion, or larger ``k`` recovers those. They are a description
  problem, not a retrieval problem, and the only fix is on the catalogue side.

Read the gap between recall@5 and recall@N as the budget available to a
reranker, and `unreachable` as the part of the problem that retrieval tuning
cannot touch however much effort goes into it.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from toolbroker import ToolBroker
from toolbroker.bench import BenchmarkCase
from toolbroker.index.embedders import HashingEmbedder
from toolbroker.sources import StaticJSONSource

DATA = Path(__file__).parent / "data"
DEFAULT_SIZES = (50, 200, 1000)
DEFAULT_DEPTHS = (5, 10, 20, 50, 100, 200)


def load(size: int, embedder: Any) -> tuple[ToolBroker, list[BenchmarkCase]]:
    """Build an indexed catalogue and its labelled query set."""
    broker = ToolBroker(embedder=embedder, cache_embeddings=True)
    broker.add_source(StaticJSONSource(DATA / f"tools_{size}.json"))
    broker.index()
    cases = [
        BenchmarkCase.model_validate(json.loads(line))
        for line in (DATA / f"queries_{size}.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return broker, cases


def deepest_ranks(
    broker: ToolBroker,
    cases: list[BenchmarkCase],
    depth: int,
) -> list[tuple[BenchmarkCase, int | None]]:
    """Return each positive case with the best rank any of its tools reached.

    ``None`` means no expected tool appeared within ``depth`` — the case is
    unreachable at this depth, and every shallower depth misses it too.
    """
    ranked: list[tuple[BenchmarkCase, int | None]] = []
    for case in cases:
        if case.is_negative:
            continue
        expected = set(case.expected)
        ids = [tool.id for tool in broker.select(case.query, k=depth).tools]
        positions = [index + 1 for index, tool_id in enumerate(ids) if tool_id in expected]
        ranked.append((case, min(positions) if positions else None))
    return ranked


def main() -> None:
    """Measure the recall ceiling at each catalogue size."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--depths", type=int, nargs="+", default=list(DEFAULT_DEPTHS))
    parser.add_argument(
        "--misses",
        action="store_true",
        help="list the unreachable queries and their categories",
    )
    args = parser.parse_args()

    if not (DATA / f"tools_{args.sizes[0]}.json").exists():
        raise SystemExit("run `python bench/generate.py` first")

    try:
        from toolbroker.index.embedders.fastembed import FastEmbedEmbedder

        embedder: Any = FastEmbedEmbedder()
    except Exception as exc:
        print(f"fastembed unavailable ({exc}); using the lexical embedder\n")
        embedder = HashingEmbedder(dim=512)

    depths = sorted(set(args.depths))
    print(f"embedder: {getattr(embedder, 'id', embedder)}   positives only\n")
    columns = "".join(f"{'@' + str(d):>9}" for d in depths)
    header = f"{'tools':>6}{columns}{'unreachable':>14}"
    print(header)
    print("-" * len(header))

    unreachable_by_size: dict[int, list[BenchmarkCase]] = {}
    for size in args.sizes:
        broker, cases = load(size, embedder)
        # One deep query per case, then slice it. Re-querying at every depth
        # would multiply the cost for an identical answer.
        deepest = min(max(depths), size)
        ranks = deepest_ranks(broker, cases, deepest)
        total = len(ranks)
        cells = "".join(
            f"{sum(1 for _, rank in ranks if rank is not None and rank <= d) / total:>9.3f}"
            for d in depths
        )
        stranded = [case for case, rank in ranks if rank is None]
        unreachable_by_size[size] = stranded
        print(f"{size:>6}{cells}   {len(stranded):>3}/{total} ({len(stranded) / total:>5.1%})")

    print()
    for size in args.sizes:
        stranded = unreachable_by_size[size]
        if not stranded:
            continue
        categories = Counter(case.category or "?" for case in stranded)
        listed = ", ".join(f"{name} {count}" for name, count in categories.most_common())
        print(f"{size:>5} tools — unreachable by category: {listed}")
        if args.misses:
            for case in stranded:
                print(f"        [{case.category or '?':<11}] {case.query}  → {case.expected}")

    print(
        "\nThe gap between @5 and the deepest column is what a reranker could win.\n"
        "The unreachable share is what it could not: those tools never enter the\n"
        "pool at any depth, so the fix is a better description, not a better ranker."
    )


if __name__ == "__main__":
    main()
