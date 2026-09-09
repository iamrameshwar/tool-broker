"""Find the score floor that lets your catalogue say "no tool fits".

Without a floor a retriever always returns its ``k`` best guesses, so a query
no tool can serve still produces five confident suggestions and the model uses
one. Adding :class:`~toolbroker.policy.rules.MinScore` fixes that, but the right
threshold depends on your embedder's score distribution and your catalogue —
which is why ToolBroker ships no default value and ships this instead.

    uv run python bench/threshold.py
    uv run python bench/threshold.py --sizes 200 --k 5

Read the table by asking what you are optimising. A high floor protects against
confident nonsense at the cost of recall; a low floor does the reverse. The
row where positive and negative accuracy cross is usually a defensible starting
point, and the ``overall`` peak is usually slightly below it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from toolbroker import MinScore, PolicyEngine, ToolBroker
from toolbroker.bench import BenchmarkCase, run_benchmark
from toolbroker.index.embedders import HashingEmbedder
from toolbroker.sources import StaticJSONSource

DATA = Path(__file__).parent / "data"
DEFAULT_FLOORS = (0.0, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)


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


def main() -> None:
    """Sweep score floors and print the trade-off."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[200])
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--floors", type=float, nargs="+", default=list(DEFAULT_FLOORS))
    args = parser.parse_args()

    if not (DATA / f"tools_{args.sizes[0]}.json").exists():
        raise SystemExit("run `python bench/generate.py` first")

    try:
        from toolbroker.index.embedders.fastembed import FastEmbedEmbedder

        embedder: Any = FastEmbedEmbedder()
    except Exception as exc:
        print(f"fastembed unavailable ({exc}); using the lexical embedder\n")
        embedder = HashingEmbedder(dim=512)

    for size in args.sizes:
        broker, cases = load(size, embedder)
        positives = [case for case in cases if not case.is_negative]
        negatives = [case for case in cases if case.is_negative]

        print(f"catalogue: {size} tools   embedder: {getattr(embedder, 'id', embedder)}")
        print(f"cases: {len(positives)} positive, {len(negatives)} negative, k={args.k}\n")
        header = f"{'floor':>7}{'overall':>9}{'positive':>10}{'negative':>10}{'precision':>11}"
        print(header)
        print("-" * len(header))

        for floor in args.floors:
            broker.set_policy(
                PolicyEngine(selection_rules=[MinScore(floor)]) if floor else PolicyEngine()
            )
            overall = run_benchmark(broker, cases, k=args.k)
            positive = run_benchmark(broker, positives, k=args.k)
            negative = run_benchmark(broker, negatives, k=args.k)
            print(
                f"{floor:>7.2f}{overall.hit_rate:>9.3f}{positive.hit_rate:>10.3f}"
                f"{negative.hit_rate:>10.3f}{overall.precision_at_k:>11.3f}"
            )
        print()

    print(
        "No default is shipped: the useful floor tracks your embedder's score\n"
        "distribution, so a value tuned here would be wrong for a different model."
    )


if __name__ == "__main__":
    main()
