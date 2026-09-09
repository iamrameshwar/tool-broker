"""Turning a tool's JSON Schema into a pydantic model.

Most frameworks accept a JSON Schema dict. Some — CrewAI among them — want a
pydantic model class instead, and building one by hand for every tool in a
five-hundred-tool catalogue is not an option.

The conversion is deliberately forgiving. A tool catalogue is assembled from
MCP servers and OpenAPI specs written by people who never expected this code to
read them, so anything unrecognised degrades to a permissive ``Any`` field
rather than raising. A tool the model can still call with slightly loose typing
beats a tool that vanished because its schema used a construct we did not
anticipate.

    from toolbroker.schema import model_from_schema

    Args = model_from_schema(tool.input_schema, name="IssueRefundArgs")
"""

from __future__ import annotations

import keyword
import re
from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, create_model

from .types import JSONSchema

_IDENTIFIER = re.compile(r"[^0-9a-zA-Z_]")

#: Names pydantic reserves on a model. A tool parameter called ``schema`` is
#: legal JSON and would shadow one of these, so those get an alias instead.
_RESERVED = frozenset(
    {
        "model_config",
        "model_fields",
        "model_computed_fields",
        "model_extra",
        "model_fields_set",
        "schema",
        "json",
        "dict",
        "copy",
        "construct",
        "validate",
        "fields",
    }
)

_PRIMITIVES: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "null": type(None),
    "object": dict,
    "array": list,
}

MAX_DEPTH = 8


def safe_field_name(raw: str) -> tuple[str, str | None]:
    """Return ``(python_name, alias)`` for a JSON property name.

    The alias is ``None`` when the name was already usable as-is. When it is
    not — a hyphen, a leading digit, a Python keyword, a pydantic reserved
    name — the model gets a mangled attribute and keeps the original as an
    alias, so the wire format is unchanged.
    """
    cleaned = _IDENTIFIER.sub("_", raw)
    if cleaned and cleaned[0].isdigit():
        cleaned = f"f_{cleaned}"
    if keyword.iskeyword(cleaned) or cleaned in _RESERVED:
        cleaned = f"{cleaned}_"
    if not cleaned:
        cleaned = "field"
    return cleaned, (raw if cleaned != raw else None)


def _model_name(raw: str) -> str:
    """Return a class-name-safe version of ``raw``."""
    cleaned = _IDENTIFIER.sub("_", raw).strip("_")
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"Model_{cleaned}"
    return cleaned[:1].upper() + cleaned[1:]


def python_type(schema: Any, *, name: str = "Nested", depth: int = 0) -> Any:
    """Return the Python annotation for one JSON Schema node.

    Unrecognised or deeply nested constructs collapse to ``Any``: a permissive
    field still lets the tool be called, whereas raising would drop it.
    """
    if depth > MAX_DEPTH or not isinstance(schema, dict) or not schema:
        return Any

    # Unions come first: a node may carry both `anyOf` and a `type`.
    for key in ("anyOf", "oneOf"):
        options = schema.get(key)
        if isinstance(options, list) and options:
            members = tuple(
                python_type(option, name=f"{name}Option{index}", depth=depth + 1)
                for index, option in enumerate(options)
            )
            if any(member is Any for member in members):
                return Any
            return Union[members]  # noqa: UP007 - dynamic union needs the form

    enum = schema.get("enum")
    if isinstance(enum, list) and enum and all(isinstance(v, str | int | bool) for v in enum):
        return Literal[tuple(enum)]

    declared = schema.get("type")
    if isinstance(declared, list):
        # `type: [string, "null"]` is how optionality is written in older specs.
        members = tuple(
            python_type({**schema, "type": entry}, name=name, depth=depth + 1) for entry in declared
        )
        return Any if any(member is Any for member in members) else Union[members]  # noqa: UP007

    if declared == "object":
        properties = schema.get("properties")
        if isinstance(properties, dict) and properties:
            return model_from_schema(schema, name=_model_name(name), depth=depth + 1)
        return dict[str, Any]

    if declared == "array":
        items = schema.get("items")
        if isinstance(items, dict):
            return list[python_type(items, name=f"{name}Item", depth=depth + 1)]  # type: ignore[misc]
        return list[Any]

    if isinstance(declared, str):
        return _PRIMITIVES.get(declared, Any)

    return Any


def model_from_schema(
    schema: JSONSchema | None,
    *,
    name: str = "ToolArgs",
    depth: int = 0,
) -> type[BaseModel]:
    """Build a pydantic model from a JSON Schema object.

    Args:
        schema: A JSON Schema of ``type: object``. ``None`` or empty produces a
            model with no fields, which is the right shape for a tool that
            takes no arguments.
        name: Class name for the generated model.
        depth: Recursion guard for nested objects.

    Returns:
        A pydantic model class.
    """
    class_name = _model_name(name)
    properties = (schema or {}).get("properties")
    if not isinstance(properties, dict) or not properties:
        return create_model(class_name, __config__=ConfigDict(extra="allow"))

    required = set((schema or {}).get("required") or ())
    fields: dict[str, Any] = {}

    for raw_name, raw_spec in properties.items():
        spec = raw_spec if isinstance(raw_spec, dict) else {}
        field_name, alias = safe_field_name(str(raw_name))
        annotation = python_type(spec, name=f"{class_name}_{field_name}", depth=depth + 1)

        options: dict[str, Any] = {}
        description = spec.get("description")
        if isinstance(description, str) and description:
            options["description"] = description
        if alias is not None:
            options["alias"] = alias
            options["validation_alias"] = alias
            options["serialization_alias"] = alias

        if raw_name in required:
            default: Any = ...
        elif "default" in spec:
            default = spec["default"]
        else:
            default = None
            # Optional with no default has to admit None, or pydantic rejects
            # the very omission the schema says is allowed.
            annotation = annotation if annotation is Any else annotation | None

        fields[field_name] = (annotation, Field(default, **options))

    return create_model(
        class_name,
        __config__=ConfigDict(populate_by_name=True, extra="allow"),
        **fields,
    )
