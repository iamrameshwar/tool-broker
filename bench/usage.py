"""Is your usage signal good enough to boost on?

Usage boosting is a bet: it helps if what you record correlates with what was
actually right, and hurts if it does not. The size of both effects is worth
measuring before turning it on in front of users.

    uv run python bench/usage.py                        # synthetic signals
    uv run python bench/usage.py --usage my_usage.json  # your own counts

Two synthetic signals are simulated for reference:

``informed``
    Usage matches the labels — people called the tool they needed. The best
    case, and the one the feature is designed for.
``random``
    Usage is unrelated to relevance. The realistic failure: counts dominated by
    one noisy integration, or recording every tool the model *tried* rather
    than the one that worked.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from toolbroker import ToolBroker, UsageTracker
from toolbroker.bench import BenchmarkCase, run_benchmark
from toolbroker.sources import StaticJSONSource

DATA = Path(__file__).parent / "data"
WEIGHTS = (0.05, 0.10, 0.20, 0.40)
SEED = 7


def load_cases(size: int) -> list[BenchmarkCase]:
    """Read the labelled query set."""
    return [
        BenchmarkCase.model_validate(json.loads(line))
        for line in (DATA / f"queries_{size}.jsonl").read_text().splitlines()
        if line.strip()
    ]


def build(size: int, embedder: Any, weight: float, tracker: UsageTracker | None) -> ToolBroker:
    """Index a catalogue with the given boost configuration."""
    broker = ToolBroker(
        embedder=embedder, cache_embeddings=True, usage=tracker, usage_weight=weight
    )
    broker.add_source(StaticJSONSource(DATA / f"tools_{size}.json"))
    broker.index()
    return broker


def informed_usage(cases: list[BenchmarkCase], rng: random.Random) -> UsageTracker:
    """Counts that track the labels: people called what they needed."""
    tracker = UsageTracker()
    for case in cases:
        for tool_id in case.expected:
            tracker.record(tool_id, rng.randint(1, 20))
    return tracker


def random_usage(tool_ids: list[str], rng: random.Random) -> UsageTracker:
    """Counts unrelated to relevance."""
    tracker = UsageTracker()
    for tool_id in rng.sample(tool_ids, min(20, len(tool_ids))):
        tracker.record(tool_id, rng.randint(50, 500))
    return tracker


def main() -> None:
    """Compare recall with and without boosting, under each signal."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--usage", type=Path, help="a usage file written by UsageTracker.save()")
    args = parser.parse_args()

    if not (DATA / f"tools_{args.size}.json").exists():
        raise SystemExit("run `python bench/generate.py` first")

    try:
        from toolbroker.index.embedders.fastembed import FastEmbedEmbedder

        embedder: Any = FastEmbedEmbedder()
    except Exception as exc:
        from toolbroker.index.embedders import HashingEmbedder

        print(f"fastembed unavailable ({exc}); using the lexical embedder\n")
        embedder = HashingEmbedder(dim=512)

    cases = load_cases(args.size)
    rng = random.Random(SEED)
    baseline = run_benchmark(build(args.size, embedder, 0.0, None), cases, k=args.k)

    signals: list[tuple[str, UsageTracker]] = []
    if args.usage:
        signals.append(("yours", UsageTracker.load(args.usage, missing_ok=False)))
    else:
        tool_ids = [tool.id for tool in build(args.size, embedder, 0.0, None).tools()]
        signals.append(("informed", informed_usage(cases, rng)))
        signals.append(("random", random_usage(tool_ids, rng)))

    print(f"catalogue: {args.size} tools, {len(cases)} labelled queries, k={args.k}")
    print(f"baseline recall@{args.k} with no boosting: {baseline.recall_at_k:.3f}\n")
    header = f"{'signal':<12}{'weight':>8}{'recall':>9}{'delta':>9}"
    print(header)
    print("-" * len(header))

    for name, tracker in signals:
        for weight in WEIGHTS:
            result = run_benchmark(build(args.size, embedder, weight, tracker), cases, k=args.k)
            delta = result.recall_at_k - baseline.recall_at_k
            print(f"{name:<12}{weight:>8.2f}{result.recall_at_k:>9.3f}{delta:>+9.3f}")
        print()

    print(
        "Boosting is a bet on your feedback signal. Record the tool that actually\n"
        "worked, not every tool the model tried, and keep the weight low enough that\n"
        "a bad signal costs less than a good one gains."
    )


if __name__ == "__main__":
    main()
