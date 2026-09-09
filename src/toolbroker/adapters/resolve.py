"""Adapter lookup by name."""

from __future__ import annotations

from typing import Any

from ..protocols import Adapter
from ..registry import GROUP_ADAPTERS, create


def get_adapter(name: str, /, **kwargs: Any) -> Adapter:
    """Instantiate the adapter registered under ``name``."""
    adapter: Adapter = create(GROUP_ADAPTERS, name, **kwargs)
    return adapter
