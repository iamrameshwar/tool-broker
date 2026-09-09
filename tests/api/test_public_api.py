"""The public API must not change by accident.

ToolBroker is heading for an API freeze, and the changes that hurt after one are
never the deliberate ones. They are a renamed keyword argument, a property
quietly becoming a method, a name dropped from ``__all__`` during a refactor —
each invisible in review and each a broken import for somebody.

This test snapshots the surface and fails when it moves. Intentional changes
are made by regenerating the snapshot, which puts the diff in the pull request
where a reviewer can see exactly what callers are being asked to absorb:

    uv run python tests/api/surface.py --write
"""

from __future__ import annotations

import json

import pytest

from .surface import SNAPSHOT, describe


@pytest.fixture(scope="module")
def recorded() -> dict:
    if not SNAPSHOT.exists():  # pragma: no cover - first run only
        pytest.skip("no snapshot yet; run `python tests/api/surface.py --write`")
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def current() -> dict:
    return describe()


def _explain(kind: str, names: set[str]) -> str:
    listed = ", ".join(sorted(names))
    return (
        f"{kind}: {listed}\n\n"
        "If this is intentional, regenerate the snapshot so the diff is reviewable:\n"
        "    uv run python tests/api/surface.py --write"
    )


def test_no_exports_were_removed(recorded, current):
    """A name leaving ``__all__`` breaks an import somewhere."""
    lost = set(recorded["exports"]) - set(current["exports"])
    assert not lost, _explain("exports removed", lost)


def test_no_exports_were_added_unnoticed(recorded, current):
    """Adding to the public surface is a commitment; make it deliberate."""
    added = set(current["exports"]) - set(recorded["exports"])
    assert not added, _explain("exports added", added)


def test_no_public_members_were_removed(recorded, current):
    """A method or property disappearing breaks a caller."""
    lost: set[str] = set()
    for name, entry in recorded["api"].items():
        if entry.get("kind") != "class":
            continue
        now = current["api"].get(name)
        if now is None:
            lost.add(name)
            continue
        lost |= {
            f"{name}.{member}"
            for member in entry.get("members", {})
            if member not in now.get("members", {})
        }
    assert not lost, _explain("public members removed", lost)


def test_no_signatures_changed(recorded, current):
    """A renamed or reordered parameter is a silent break at the call site."""
    changed: set[str] = set()
    for name, entry in recorded["api"].items():
        now = current["api"].get(name)
        if now is None:
            continue
        if entry.get("kind") == "function" and entry.get("signature") != now.get("signature"):
            changed.add(f"{name}{entry.get('signature')} -> {now.get('signature')}")
        for member, signature in entry.get("members", {}).items():
            current_signature = now.get("members", {}).get(member)
            if current_signature is not None and current_signature != signature:
                changed.add(f"{name}.{member}: {signature} -> {current_signature}")
    assert not changed, _explain("signatures changed", changed)


def test_exports_are_sorted(current):
    """``__all__`` stays sorted, so its diffs stay readable."""
    assert current["exports"] == sorted(current["exports"])


def test_every_export_resolves(current):
    """Nothing in ``__all__`` is a dangling name."""
    import toolbroker

    missing = [name for name in toolbroker.__all__ if not hasattr(toolbroker, name)]
    assert not missing, f"exported but absent: {missing}"


def test_the_extension_interfaces_are_importable():
    """Every doc page says "implement this Protocol"; they must be reachable."""
    import toolbroker

    for name in ("Source", "Embedder", "Store", "Retriever", "Reranker", "Policy", "Adapter"):
        assert hasattr(toolbroker, name), f"toolbroker.{name} is not exported"


def test_version_matches_the_packaging_metadata():
    """`__version__` and the distribution must agree.

    They drifted once, in the direction that is hardest to spot: `pip show`
    reported 0.1.0 while `toolbroker --version` reported 0.1.0.dev0, because the
    version was written down in two places and only one was bumped.
    """
    from importlib import metadata
    from pathlib import Path

    import tomllib

    import toolbroker

    declared = tomllib.loads(
        Path(__file__).resolve().parents[2].joinpath("pyproject.toml").read_text()
    )["project"]["version"]
    assert toolbroker.__version__ == declared
    assert metadata.version("toolbroker") == declared
