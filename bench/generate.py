"""Write the benchmark catalogues and labelled query set to ``bench/data/``.

The forty labelled tools from :mod:`corpus` appear in **every** catalogue size,
so the scaling curve measures added distractors rather than which tools happened
to survive truncation. Only the filler count changes.

Deterministic: a fixed seed means the same catalogue every run, so two people
comparing numbers are comparing the same thing.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from corpus import core_tools, filler_tools
from queries import cases, category_counts

HERE = Path(__file__).parent
DATA = HERE / "data"
SIZES = (50, 200, 1000)
SEED = 20260908


def main() -> None:
    """Write ``tools_<size>.json`` and ``queries_<size>.jsonl`` for each size."""
    DATA.mkdir(exist_ok=True)
    core = core_tools()
    labelled = cases()

    for size in SIZES:
        if size < len(core):
            raise SystemExit(
                f"size {size} is smaller than the {len(core)} labelled tools; "
                "every labelled tool must appear at every size"
            )
        tools = [*core, *filler_tools(size - len(core))]
        # Shuffled so position carries no signal, but seeded per size so the
        # file is byte-stable across runs.
        random.Random(SEED + size).shuffle(tools)

        (DATA / f"tools_{size}.json").write_text(
            json.dumps(tools, indent=2) + "\n", encoding="utf-8"
        )
        (DATA / f"queries_{size}.jsonl").write_text(
            "\n".join(json.dumps(case) for case in labelled) + "\n", encoding="utf-8"
        )
        print(f"wrote {len(tools):>5} tools and {len(labelled)} queries for size {size}")

    print()
    print("query categories:")
    for name, count in category_counts().items():
        print(f"  {name:<12}{count:>4}")


if __name__ == "__main__":
    main()
