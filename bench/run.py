"""Score retrieval across catalogue sizes.

    uv run python bench/generate.py
    uv run python bench/run.py

Runs each available embedder against each catalogue size, so the numbers show
both the scaling curve and what a real embedding model buys over the offline
lexical default.

Optional embedders are opted into by environment, because one needs a running
server and the other costs money:

    TOOLBROKER_BENCH_OLLAMA_MODEL=nomic-embed-text uv run python bench/run.py
    OPENAI_API_KEY=sk-... uv run python bench/run.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from toolbroker import ToolBroker
from toolbroker.bench import BenchmarkCase, run_benchmark
from toolbroker.index.embedders import HashingEmbedder
from toolbroker.sources import StaticJSONSource

DATA = Path(__file__).parent / "data"
SIZES = (50, 200, 1000)
K = 5


def embedders() -> list[tuple[str, object]]:
    """Return every embedder available in this environment.

    Each one that cannot be constructed is reported and skipped, so the same
    command produces the best comparison the machine can offer rather than
    failing on a missing optional dependency.
    """
    available: list[tuple[str, object]] = [("hashing (offline)", HashingEmbedder(dim=512))]

    try:
        from toolbroker.index.embedders.fastembed import FastEmbedEmbedder

        available.append(("fastembed bge-small", FastEmbedEmbedder()))
    except Exception as exc:
        print(f"skipping fastembed: {exc}\n")

    ollama_model = os.environ.get("TOOLBROKER_BENCH_OLLAMA_MODEL")
    if ollama_model:
        try:
            from toolbroker_ollama import OllamaEmbedder

            available.append((f"ollama {ollama_model}", OllamaEmbedder(model=ollama_model)))
        except Exception as exc:
            print(f"skipping ollama: {exc}\n")

    if os.environ.get("OPENAI_API_KEY"):
        try:
            from toolbroker_openai_embed import OpenAIEmbedder

            model = os.environ.get("TOOLBROKER_BENCH_OPENAI_MODEL", "text-embedding-3-small")
            available.append((f"openai {model}", OpenAIEmbedder(model=model)))
        except Exception as exc:
            print(f"skipping openai: {exc}\n")

    return available


def main() -> None:
    """Run the suite and print a table."""
    if not (DATA / f"tools_{SIZES[0]}.json").exists():
        raise SystemExit("run `python bench/generate.py` first")

    rows: list[tuple[str, int, float, float, float, float]] = []
    for label, embedder in embedders():
        for size in SIZES:
            broker = ToolBroker(embedder=embedder, cache_embeddings=True)
            broker.add_source(StaticJSONSource(DATA / f"tools_{size}.json"))
            broker.index()

            cases = [
                BenchmarkCase.model_validate(json.loads(line))
                for line in (DATA / f"queries_{size}.jsonl").read_text().splitlines()
                if line.strip()
            ]
            result = run_benchmark(broker, cases, k=K)
            rows.append(
                (
                    label,
                    size,
                    result.recall_at_k,
                    result.mrr,
                    result.p95_latency_ms,
                    result.context_reduction,
                )
            )

    # Sized to the widest label so a long model tag does not shift the columns.
    width = max([22, *(len(row[0]) for row in rows)]) + 2
    header = (
        f"{'embedder':<{width}}{'tools':>7}{f'recall@{K}':>10}"
        f"{'MRR':>8}{'p95 ms':>9}{'ctx saved':>11}"
    )
    print(header)
    print("-" * len(header))
    for label, size, recall, mrr, p95, reduction in rows:
        print(
            f"{label:<{width}}{size:>7}{recall:>10.3f}"
            f"{mrr:>8.3f}{p95:>9.2f}{reduction * 100:>10.1f}%"
        )


if __name__ == "__main__":
    main()
