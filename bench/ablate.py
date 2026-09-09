"""Measure the effect of enrichment settings on retrieval accuracy.

CONTRIBUTING asks for before-and-after numbers on any change to the selection
pipeline. This is the tool that produces them: it re-indexes the benchmark
catalogue under each variant and reports recall, so a tuning decision is a
measurement rather than an opinion.

    uv run python bench/ablate.py
    uv run python bench/ablate.py --sizes 200 --k 5
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from toolbroker import EnrichmentConfig, ToolBroker
from toolbroker.bench import BenchmarkCase, run_benchmark
from toolbroker.index.embedders import HashingEmbedder
from toolbroker.sources import StaticJSONSource

DATA = Path(__file__).parent / "data"


@dataclass(frozen=True)
class Variant:
    """One enrichment configuration to measure."""

    label: str
    config: EnrichmentConfig


VARIANTS = (
    Variant("baseline (name x1)", EnrichmentConfig()),
    Variant("name x0 (drop name)", EnrichmentConfig(include_name=False)),
    Variant("name x2", EnrichmentConfig(name_weight=2)),
    Variant("name x3", EnrichmentConfig(name_weight=3)),
    Variant("description x2", EnrichmentConfig(description_weight=2)),
    Variant("no namespace line", EnrichmentConfig(include_namespace=False)),
    Variant("no parameters", EnrichmentConfig(include_parameters=False)),
    Variant("description only", EnrichmentConfig(include_name=False, include_parameters=False)),
)


def load_cases(size: int) -> list[BenchmarkCase]:
    """Read the labelled query set for ``size``."""
    return [
        BenchmarkCase.model_validate(json.loads(line))
        for line in (DATA / f"queries_{size}.jsonl").read_text().splitlines()
        if line.strip()
    ]


def main() -> None:
    """Run every variant against every catalogue size."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[50, 200, 1000])
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    if not (DATA / f"tools_{args.sizes[0]}.json").exists():
        raise SystemExit("run `python bench/generate.py` first")

    try:
        from toolbroker.index.embedders.fastembed import FastEmbedEmbedder

        embedder: Any = FastEmbedEmbedder()
    except Exception as exc:
        print(f"fastembed unavailable ({exc}); using the lexical embedder\n")
        embedder = HashingEmbedder(dim=512)

    header = f"{'variant':<24}" + "".join(f"{size:>10}" for size in args.sizes) + f"{'mean':>10}"
    print(f"recall@{args.k}   embedder: {getattr(embedder, 'id', embedder)}\n")
    print(header)
    print("-" * len(header))

    case_count = len(load_cases(args.sizes[0]))
    resolution = 1.0 / case_count if case_count else 0.0

    baseline: list[float] | None = None
    for variant in VARIANTS:
        scores: list[float] = []
        for size in args.sizes:
            broker = ToolBroker(
                embedder=embedder, enrichment=variant.config, cache_embeddings=False
            )
            broker.add_source(StaticJSONSource(DATA / f"tools_{size}.json"))
            broker.index()
            scores.append(run_benchmark(broker, load_cases(size), k=args.k).recall_at_k)

        mean = sum(scores) / len(scores)
        if baseline is None:
            baseline = scores
            delta = ""
        else:
            base_mean = sum(baseline) / len(baseline)
            delta = f"  ({mean - base_mean:+.3f})"
        row = f"{variant.label:<24}" + "".join(f"{score:>10.3f}" for score in scores)
        print(f"{row}{mean:>10.3f}{delta}")

    # Without this line someone will read a +0.02 as an improvement and ship it.
    print(
        f"\nresolution: {case_count} labelled queries, so one query flipping moves "
        f"recall by {resolution:.3f}."
    )
    print(
        "Treat any delta smaller than that as noise, and prefer a change that moves "
        "in the same direction at every catalogue size."
    )


if __name__ == "__main__":
    main()
