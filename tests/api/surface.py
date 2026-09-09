"""Compute a canonical description of the public API.

Used by the surface test to detect changes. Kept separate from the test so it
can also regenerate the snapshot:

    uv run python tests/api/surface.py --write
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

SNAPSHOT = Path(__file__).parent / "public_api.json"

# Inherited plumbing that says nothing about our design.
_NOISE = frozenset(
    {
        "model_config",
        "model_fields",
        "model_computed_fields",
        "model_extra",
        "model_fields_set",
        "model_post_init",
        "copy",
        "dict",
        "json",
        "parse_file",
        "parse_obj",
        "parse_raw",
        "schema",
        "schema_json",
        "update_forward_refs",
        "construct",
        "from_orm",
        "mro",
        "name",
        "value",
    }
)


def _signature(obj: Any) -> str:
    """Return a readable signature, or an empty string when unavailable."""
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return ""


def _members(cls: type) -> dict[str, str]:
    """Return the public methods and properties of ``cls`` with signatures."""
    members: dict[str, str] = {}
    for name, value in vars(cls).items():
        if name.startswith("_") or name in _NOISE:
            continue
        if isinstance(value, property):
            members[name] = "property"
        elif callable(value):
            members[name] = _signature(value)
        else:
            members[name] = type(value).__name__
    # Pydantic models describe their shape through fields rather than slots.
    fields = getattr(cls, "model_fields", None)
    if isinstance(fields, dict):
        for name in fields:
            members.setdefault(f"field:{name}", "field")
    return dict(sorted(members.items()))


def describe() -> dict[str, Any]:
    """Return the full public surface of the ``toolbroker`` package."""
    import toolbroker

    surface: dict[str, Any] = {"exports": sorted(toolbroker.__all__), "api": {}}
    for name in sorted(toolbroker.__all__):
        if name.startswith("__"):
            continue
        obj = getattr(toolbroker, name)
        if inspect.isclass(obj):
            surface["api"][name] = {
                "kind": "class",
                "bases": [b.__name__ for b in obj.__bases__ if b is not object],
                "members": _members(obj),
            }
        elif inspect.isfunction(obj):
            surface["api"][name] = {"kind": "function", "signature": _signature(obj)}
        else:
            surface["api"][name] = {"kind": type(obj).__name__}
    return surface


def main() -> None:
    """Print or rewrite the snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="rewrite the snapshot file")
    args = parser.parse_args()

    payload = json.dumps(describe(), indent=2, sort_keys=True) + "\n"
    if args.write:
        SNAPSHOT.write_text(payload, encoding="utf-8")
        print(f"wrote {SNAPSHOT}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
