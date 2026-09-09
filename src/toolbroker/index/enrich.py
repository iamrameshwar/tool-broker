"""Turning a tool definition into the text we actually embed.

A raw MCP description is often one terse line, which is not enough signal to
separate 500 tools. We fold in the name (split into words), the namespace, the
tags, the parameter names and their descriptions, and any usage examples.

Field weighting is done by repetition rather than by vector arithmetic: it
works with every embedding model, stays inspectable, and a user can read the
exact string that was embedded. The default weights are measured rather than
assumed — see ``bench/ablate.py``, which is also how you would justify changing
them.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..types import JSONSchema, Tool

_WORD_SPLIT = re.compile(r"[_\-./]+|(?<=[a-z0-9])(?=[A-Z])")


class EnrichmentConfig(BaseModel):
    """Controls which parts of a tool contribute to its index text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    include_name: bool = True
    include_namespace: bool = True
    include_description: bool = True
    include_tags: bool = True
    include_parameters: bool = True
    include_examples: bool = True
    # Measured, not guessed. Repeating the name dilutes the description without
    # adding signal: name_weight=1 beat 2 and 3 for both the lexical and the
    # semantic embedder, at every catalogue size (bench/ablate.py). Dropping the
    # name entirely is much worse for lexical retrieval, so 1 is the peak.
    name_weight: int = Field(default=1, ge=0, le=10)
    description_weight: int = Field(default=1, ge=0, le=10)
    max_parameters: int = Field(default=12, ge=0)
    max_description_chars: int = Field(default=1000, ge=0)


def humanize(identifier: str) -> str:
    """Split ``get_user_by_id`` / ``getUserById`` into ``get user by id``."""
    parts = [part for part in _WORD_SPLIT.split(identifier) if part]
    return " ".join(parts).lower()


def describe_schema(schema: JSONSchema, *, max_parameters: int = 12) -> str:
    """Summarise a JSON Schema as searchable text.

    Parameter *names* carry real signal — a tool taking ``invoice_id`` is about
    invoices even when its description does not say so.
    """
    if not schema:
        return ""
    properties = schema.get("properties")
    if not isinstance(properties, Mapping) or not properties:
        return ""
    required = set(schema.get("required") or ())
    fragments: list[str] = []
    for name, spec in list(properties.items())[:max_parameters]:
        piece = humanize(str(name))
        if isinstance(spec, Mapping):
            description = spec.get("description")
            if isinstance(description, str) and description:
                piece = f"{piece}: {description}"
            elif isinstance(spec.get("type"), str):
                piece = f"{piece} ({spec['type']})"
        if name in required:
            piece = f"{piece} [required]"
        fragments.append(piece)
    return "parameters: " + "; ".join(fragments)


def build_index_text(tool: Tool, config: EnrichmentConfig | None = None) -> str:
    """Return the text that will be embedded for ``tool``."""
    settings = config or EnrichmentConfig()
    lines: list[str] = []

    if settings.include_name and settings.name_weight:
        readable = humanize(tool.name)
        line = f"{tool.name} ({readable})" if readable != tool.name else tool.name
        lines.extend([line] * settings.name_weight)

    if settings.include_namespace and tool.namespace != "default":
        lines.append(f"namespace: {humanize(tool.namespace)}")

    if settings.include_description and tool.description and settings.description_weight:
        description = tool.description.strip()[: settings.max_description_chars]
        lines.extend([description] * settings.description_weight)

    if settings.include_tags and tool.tags:
        lines.append("tags: " + ", ".join(sorted(tool.tags)))

    if settings.include_parameters:
        summary = describe_schema(tool.input_schema, max_parameters=settings.max_parameters)
        if summary:
            lines.append(summary)

    if settings.include_examples and tool.examples:
        lines.append("examples: " + " | ".join(tool.examples))

    return "\n".join(lines)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, with identifier splitting applied.

    Shared by the hashing embedder and the BM25 retriever so lexical behaviour
    is consistent between them.
    """
    tokens: list[str] = []
    for raw in re.findall(r"[A-Za-z0-9_]+", text):
        tokens.extend(part for part in humanize(raw).split() if part)
    return tokens


def unique_stable(items: Sequence[Any]) -> list[Any]:
    """Deduplicate preserving first-seen order."""
    seen: set[Any] = set()
    result: list[Any] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
