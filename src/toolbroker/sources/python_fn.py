"""Plain Python functions as a tool source.

The zero-config path: decorate or pass functions and get a searchable
catalogue. Signatures become JSON Schema via pydantic; the docstring supplies
the description and per-parameter text.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Iterable, Sequence
from typing import Any, get_type_hints

from pydantic import TypeAdapter

from ..errors import SourceError
from ..risk import RiskClassifier
from ..types import CostTier, JSONSchema, RiskTier, Tool
from .base import BaseSource

_ARGS_SECTION = re.compile(
    r"^\s*(?:Args|Arguments|Parameters)\s*:\s*$", re.MULTILINE | re.IGNORECASE
)
_SECTION_BREAK = re.compile(
    r"^\s*(?:Returns|Raises|Yields|Examples?|Notes?)\s*:\s*$", re.MULTILINE | re.IGNORECASE
)
_ARG_LINE = re.compile(r"^\s*(\*{0,2}\w+)\s*(?:\([^)]*\))?\s*:\s*(.+)$")


def _split_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Return ``(summary, {param: description})`` from a Google-style docstring."""
    if not doc:
        return "", {}
    text = inspect.cleandoc(doc)
    args_match = _ARGS_SECTION.search(text)
    if not args_match:
        return _first_block(text), {}

    summary = _first_block(text[: args_match.start()])
    remainder = text[args_match.end() :]
    end_match = _SECTION_BREAK.search(remainder)
    if end_match:
        remainder = remainder[: end_match.start()]

    params: dict[str, str] = {}
    current: str | None = None
    for line in remainder.splitlines():
        if not line.strip():
            continue
        match = _ARG_LINE.match(line)
        if match:
            current = match.group(1).lstrip("*")
            params[current] = match.group(2).strip()
        elif current:
            params[current] = f"{params[current]} {line.strip()}"
    return summary, params


def _first_block(text: str) -> str:
    """Return the leading paragraph of ``text`` as a single line."""
    blocks = [block.strip() for block in text.strip().split("\n\n") if block.strip()]
    if not blocks:
        return ""
    return " ".join(line.strip() for line in blocks[0].splitlines())


def _schema_for(annotation: Any) -> JSONSchema:
    """Return the JSON Schema for one annotation, degrading to permissive."""
    if annotation is inspect.Parameter.empty:
        return {}
    try:
        schema = TypeAdapter(annotation).json_schema(ref_template="#/$defs/{model}")
    except Exception:
        return {}
    return dict(schema)


def build_input_schema(func: Callable[..., Any]) -> tuple[JSONSchema, dict[str, str]]:
    """Return ``(json_schema, param_descriptions)`` for ``func``'s signature."""
    signature = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}
    _, param_docs = _split_docstring(func.__doc__)

    properties: dict[str, JSONSchema] = {}
    required: list[str] = []
    defs: dict[str, Any] = {}

    for name, parameter in signature.parameters.items():
        if name in {"self", "cls"}:
            continue
        if parameter.kind in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}:
            continue
        schema = _schema_for(hints.get(name, parameter.annotation))
        defs.update(schema.pop("$defs", {}))
        if name in param_docs:
            schema["description"] = param_docs[name]
        if parameter.default is inspect.Parameter.empty:
            required.append(name)
        else:
            try:
                TypeAdapter(type(parameter.default)).dump_python(parameter.default, mode="json")
                schema["default"] = parameter.default
            except Exception:  # pragma: no cover - exotic defaults
                pass
        properties[name] = schema

    result: JSONSchema = {"type": "object", "properties": properties}
    if required:
        result["required"] = required
    if defs:
        result["$defs"] = defs
    return result, param_docs


def tool_from_function(
    func: Callable[..., Any],
    *,
    name: str | None = None,
    namespace: str = "python",
    description: str | None = None,
    tags: Iterable[str] = (),
    examples: Iterable[str] = (),
    risk: RiskTier = RiskTier.LOW,
    cost: CostTier = CostTier.FREE,
    required_scopes: Iterable[str] = (),
    source_id: str = "python",
) -> Tool:
    """Build a :class:`~toolbroker.types.Tool` from a Python callable.

    The function itself is kept in ``metadata["callable"]`` so an adapter can
    hand it back to the caller's framework. ToolBroker never invokes it.
    """
    if not callable(func):
        raise SourceError(f"{func!r} is not callable")
    summary, _ = _split_docstring(func.__doc__)
    schema, _ = build_input_schema(func)
    return Tool(
        name=name or func.__name__,
        namespace=namespace,
        description=description if description is not None else summary,
        input_schema=schema,
        tags=frozenset(tags),
        examples=tuple(examples),
        risk=risk,
        cost=cost,
        required_scopes=frozenset(required_scopes),
        source_id=source_id,
        metadata={"callable": func},
    )


class PythonFunctionSource(BaseSource):
    """Exposes a collection of Python callables as tools."""

    def __init__(
        self,
        functions: Sequence[Callable[..., Any]] = (),
        *,
        source_id: str = "python",
        namespace: str = "python",
        risk_classifier: RiskClassifier | None = None,
    ) -> None:
        """Create a source over ``functions``.

        Args:
            functions: Callables to expose as tools.
            source_id: Identifier stamped onto every tool.
            namespace: Namespace the tools land in.
            risk_classifier: Infer each tool's risk from its name. Off by
                default: this is code you own, so an explicit ``risk=`` on the
                tool beats a guess. Pass
                :func:`toolbroker.risk.classify_by_name` to opt in.
        """
        super().__init__(source_id)
        self._namespace = namespace
        self._classify = risk_classifier
        self._tools: list[Tool] = []
        for func in functions:
            self.add(func)

    def add(self, func: Callable[..., Any], **overrides: Any) -> Tool:
        """Register one callable and return the resulting tool."""
        overrides.setdefault("namespace", self._namespace)
        overrides.setdefault("source_id", self._id)
        if self._classify is not None and "risk" not in overrides:
            summary, _ = _split_docstring(func.__doc__)
            overrides["risk"] = self._classify(str(overrides.get("name") or func.__name__), summary)
        tool = tool_from_function(func, **overrides)
        self._tools.append(tool)
        return tool

    def tool(self, **overrides: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator that registers the decorated function and returns it unchanged."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.add(func, **overrides)
            return func

        return decorator

    def _discover(self) -> Iterable[Tool]:
        return list(self._tools)
