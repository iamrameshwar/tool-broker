"""Tools defined in a JSON or JSONL file.

Useful for fixtures, benchmarks, and catalogues exported from somewhere else.
Accepts either a bare list of tool objects or ``{"tools": [...]}``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..errors import SourceError
from ..types import Tool
from .base import BaseSource


class StaticJSONSource(BaseSource):
    """Reads tool definitions from a ``.json`` or ``.jsonl`` file."""

    def __init__(self, path: str | Path, *, source_id: str | None = None) -> None:
        """Point the source at ``path``."""
        self._path = Path(path)
        super().__init__(source_id or f"json:{self._path.stem}")

    @property
    def path(self) -> Path:
        """The file this source reads."""
        return self._path

    def _load_payload(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            raise SourceError(f"tool file not found: {self._path}")
        text = self._path.read_text(encoding="utf-8")
        if self._path.suffix == ".jsonl":
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SourceError(f"{self._path} is not valid JSON: {exc}") from exc
        if isinstance(data, dict):
            data = data.get("tools", [])
        if not isinstance(data, list):
            raise SourceError(f"{self._path} must contain a list of tools or {{'tools': [...]}}")
        return data

    def _discover(self) -> Iterable[Tool]:
        for index, entry in enumerate(self._load_payload()):
            try:
                yield Tool.model_validate(entry)
            except ValidationError as exc:
                raise SourceError(f"{self._path}[{index}] is not a valid tool: {exc}") from exc
