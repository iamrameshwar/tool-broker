"""OpenAPI 3.x specifications as a tool source.

Every REST API with a spec becomes a searchable tool catalogue. One operation
becomes one tool: path and query parameters, headers, and the JSON request body
are merged into a single flat JSON Schema, because that is the shape
function-calling APIs accept.

The merge is lossy in one direction — a flat object cannot say *this field goes
in the path and that one in the body* — so the mapping is preserved in
``metadata["http"]``. A caller that wants to actually issue the request has
everything it needs there. ToolBroker, as always, does not issue it.

No network access: pass a parsed dict or a local file. Fetching a remote spec is
the caller's job, which keeps an HTTP client out of the core.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..errors import SourceError
from ..observability import get_logger
from ..types import CostTier, JSONSchema, RiskTier, Tool
from .base import BaseSource

logger = get_logger("sources.openapi")

HTTP_METHODS = ("get", "put", "post", "delete", "patch", "head", "options", "trace")

# A GET is not a DELETE. The method is a far better risk signal than anything
# in the description, so it is the default classifier.
METHOD_RISK = {
    "get": RiskTier.LOW,
    "head": RiskTier.LOW,
    "options": RiskTier.LOW,
    "trace": RiskTier.LOW,
    "post": RiskTier.MEDIUM,
    "put": RiskTier.MEDIUM,
    "patch": RiskTier.MEDIUM,
    "delete": RiskTier.HIGH,
}

_PATH_SEGMENT = re.compile(r"\{([^}]+)\}")
_NON_IDENTIFIER = re.compile(r"[^a-zA-Z0-9_]+")
_MAX_REF_DEPTH = 32


class _RefResolver:
    """Resolves local ``$ref`` pointers within one document.

    Only ``#/...`` pointers are followed. A remote ``$ref`` would mean fetching,
    and the recursion depth is capped so a self-referential schema — which is
    legal and common — cannot hang indexing.
    """

    def __init__(self, document: Mapping[str, Any]) -> None:
        """Bind the resolver to ``document``."""
        self._document = document

    def resolve(self, node: Any, depth: int = 0) -> Any:
        """Return ``node`` with local refs expanded."""
        if depth > _MAX_REF_DEPTH:
            # A cycle. Degrade to a permissive schema rather than recursing
            # forever or dropping the operation entirely.
            return {}
        if isinstance(node, Mapping):
            ref = node.get("$ref")
            if isinstance(ref, str):
                target = self._lookup(ref)
                if target is None:
                    return {}
                merged = self.resolve(target, depth + 1)
                extras = {key: value for key, value in node.items() if key != "$ref"}
                if extras and isinstance(merged, Mapping):
                    return {**merged, **self.resolve(extras, depth + 1)}
                return merged
            return {key: self.resolve(value, depth + 1) for key, value in node.items()}
        if isinstance(node, list):
            return [self.resolve(item, depth + 1) for item in node]
        return node

    def _lookup(self, ref: str) -> Any:
        if not ref.startswith("#/"):
            logger.debug("skipping non-local $ref", extra={"ref": ref})
            return None
        current: Any = self._document
        for raw in ref[2:].split("/"):
            key = raw.replace("~1", "/").replace("~0", "~")
            if isinstance(current, Mapping) and key in current:
                current = current[key]
            elif isinstance(current, list) and key.isdigit() and int(key) < len(current):
                current = current[int(key)]
            else:
                logger.debug("unresolvable $ref", extra={"ref": ref})
                return None
        return current


def operation_name(method: str, path: str, operation_id: str | None) -> str:
    """Derive a tool name for one operation.

    Prefers ``operationId`` when the spec provides one, because that is what the
    API's own docs call it. Otherwise builds a readable name from the method and
    path: ``GET /users/{id}/orders`` becomes ``get_users_by_id_orders``.
    """
    if operation_id:
        cleaned = _NON_IDENTIFIER.sub("_", operation_id).strip("_")
        if cleaned:
            return cleaned

    parts = [method.lower()]
    for segment in path.strip("/").split("/"):
        if not segment:
            continue
        placeholder = _PATH_SEGMENT.fullmatch(segment)
        if placeholder:
            parts.append("by")
            parts.append(_NON_IDENTIFIER.sub("_", placeholder.group(1)).strip("_"))
        else:
            parts.append(_NON_IDENTIFIER.sub("_", segment).strip("_"))
    return "_".join(part for part in parts if part) or method.lower()


def _server_url(document: Mapping[str, Any]) -> str:
    """Return the first declared server URL, or an empty string."""
    servers = document.get("servers")
    if isinstance(servers, Sequence) and servers:
        first = servers[0]
        if isinstance(first, Mapping):
            url = first.get("url")
            if isinstance(url, str):
                return url
    return ""


def _security_scopes(operation: Mapping[str, Any], document: Mapping[str, Any]) -> frozenset[str]:
    """Collect OAuth2 scopes an operation requires.

    Spec security requirements map cleanly onto ToolBroker scopes, so an API that
    already documents its permissions gets policy enforcement for free.
    """
    requirements = operation.get("security")
    if requirements is None:
        requirements = document.get("security")
    if not isinstance(requirements, Sequence):
        return frozenset()

    scopes: set[str] = set()
    for requirement in requirements:
        if not isinstance(requirement, Mapping):
            continue
        for values in requirement.values():
            if isinstance(values, Sequence) and not isinstance(values, str):
                scopes.update(str(value) for value in values)
    return frozenset(scopes)


class OpenAPISource(BaseSource):
    """Turns the operations in an OpenAPI 3.x document into tools."""

    def __init__(
        self,
        spec: Mapping[str, Any] | str | Path,
        *,
        namespace: str | None = None,
        source_id: str | None = None,
        tags: Iterable[str] = (),
        include_methods: Sequence[str] = HTTP_METHODS,
        include_deprecated: bool = False,
        body_content_type: str = "application/json",
        default_cost: CostTier = CostTier.LOW,
    ) -> None:
        """Create a source over ``spec``.

        Args:
            spec: A parsed document, or a path to a ``.json``/``.yaml`` file.
            namespace: Tool namespace. Defaults to a slug of the spec title.
            source_id: Source identifier. Defaults to ``openapi:<namespace>``.
            tags: Tags applied to every tool, on top of the spec's own.
            include_methods: Which HTTP methods to expose.
            include_deprecated: Include operations marked deprecated.
            body_content_type: Which request-body media type to read.
            default_cost: Cost tier stamped on every tool.
        """
        self._document = _load_spec(spec)
        self._resolver = _RefResolver(self._document)
        info = self._document.get("info")
        title = info.get("title", "api") if isinstance(info, Mapping) else "api"
        self._namespace = namespace or _slug(str(title))
        super().__init__(source_id or f"openapi:{self._namespace}")
        self._tags = frozenset(tags)
        self._methods = tuple(method.lower() for method in include_methods)
        self._include_deprecated = include_deprecated
        self._body_content_type = body_content_type
        self._default_cost = default_cost

    @property
    def namespace(self) -> str:
        """Namespace tools are placed in."""
        return self._namespace

    @property
    def document(self) -> Mapping[str, Any]:
        """The parsed specification."""
        return self._document

    def _discover(self) -> Iterable[Tool]:
        paths = self._document.get("paths")
        if not isinstance(paths, Mapping):
            raise SourceError("OpenAPI document has no 'paths' object")

        seen: set[str] = set()
        for path, item in paths.items():
            if not isinstance(item, Mapping):
                continue
            shared = item.get("parameters", [])
            for method in self._methods:
                operation = item.get(method)
                if not isinstance(operation, Mapping):
                    continue
                if operation.get("deprecated") and not self._include_deprecated:
                    continue
                tool = self._to_tool(str(path), method, operation, shared)
                if tool.name in seen:
                    # Two operations resolving to the same name means one would
                    # silently shadow the other at index time. Disambiguate here
                    # where we still know which is which.
                    tool = tool.model_copy(update={"name": f"{tool.name}_{method}"})
                seen.add(tool.name)
                yield tool

    def _to_tool(
        self,
        path: str,
        method: str,
        operation: Mapping[str, Any],
        shared_parameters: Any,
    ) -> Tool:
        """Build one tool from one operation."""
        name = operation_name(method, path, operation.get("operationId"))
        schema, locations, body_fields = self._build_schema(operation, shared_parameters)

        summary = str(operation.get("summary") or "").strip()
        detail = str(operation.get("description") or "").strip()
        description = " ".join(part for part in (summary, detail) if part)
        if not description:
            description = f"{method.upper()} {path}"

        spec_tags = operation.get("tags")
        tags = set(self._tags)
        if isinstance(spec_tags, Sequence) and not isinstance(spec_tags, str):
            tags.update(_slug(str(tag)) for tag in spec_tags)

        return Tool(
            name=name,
            namespace=self._namespace,
            description=description[:2000],
            input_schema=schema,
            tags=frozenset(tags),
            risk=METHOD_RISK.get(method, RiskTier.MEDIUM),
            cost=self._default_cost,
            required_scopes=_security_scopes(operation, self._document),
            source_id=self._id,
            metadata={
                "http": {
                    "method": method.upper(),
                    "path": path,
                    "server": _server_url(self._document),
                    # Where each flattened field belongs when the request is
                    # actually built: path, query, header, cookie, or body.
                    "parameter_in": locations,
                    "body_fields": body_fields,
                    "body_content_type": self._body_content_type,
                }
            },
        )

    def _build_schema(
        self, operation: Mapping[str, Any], shared_parameters: Any
    ) -> tuple[JSONSchema, dict[str, str], list[str]]:
        """Merge parameters and request body into one flat object schema."""
        properties: dict[str, Any] = {}
        required: list[str] = []
        locations: dict[str, str] = {}

        merged: list[Any] = []
        for group in (shared_parameters, operation.get("parameters")):
            if isinstance(group, Sequence) and not isinstance(group, str):
                merged.extend(group)

        for raw in merged:
            parameter = self._resolver.resolve(raw)
            if not isinstance(parameter, Mapping):
                continue
            field = parameter.get("name")
            location = parameter.get("in")
            if not isinstance(field, str) or location == "cookie":
                continue
            entry = dict(self._resolver.resolve(parameter.get("schema")) or {})
            description = parameter.get("description")
            if isinstance(description, str) and description:
                entry["description"] = description
            properties[field] = entry
            locations[field] = str(location or "query")
            if parameter.get("required"):
                required.append(field)

        body_fields: list[str] = []
        body_schema = self._body_schema(operation)
        if body_schema:
            body_properties = body_schema.get("properties")
            if isinstance(body_properties, Mapping):
                body_required = set(body_schema.get("required") or ())
                for field, entry in body_properties.items():
                    # A body field colliding with a query parameter is rare but
                    # real; suffix it rather than letting one overwrite the other.
                    key = f"body_{field}" if field in properties else str(field)
                    properties[key] = entry
                    locations[key] = "body"
                    body_fields.append(key)
                    if field in body_required:
                        required.append(key)
            else:
                # A non-object body (an array, a scalar) cannot be flattened, so
                # it is exposed as a single field.
                properties["body"] = body_schema
                locations["body"] = "body"
                body_fields.append("body")
                required.append("body")

        schema: JSONSchema = {"type": "object", "properties": properties}
        if required:
            schema["required"] = sorted(set(required))
        return schema, locations, body_fields

    def _body_schema(self, operation: Mapping[str, Any]) -> JSONSchema | None:
        """Return the resolved request-body schema, if there is one."""
        body = self._resolver.resolve(operation.get("requestBody"))
        if not isinstance(body, Mapping):
            return None
        content = body.get("content")
        if not isinstance(content, Mapping):
            return None
        media = content.get(self._body_content_type)
        if not isinstance(media, Mapping):
            # Fall back to any JSON-ish media type, e.g. application/merge-patch+json.
            for key, value in content.items():
                if isinstance(key, str) and "json" in key and isinstance(value, Mapping):
                    media = value
                    break
            else:
                return None
        schema = self._resolver.resolve(media.get("schema"))
        return dict(schema) if isinstance(schema, Mapping) else None


def _slug(value: str) -> str:
    """Turn a title into a namespace-safe identifier."""
    cleaned = _NON_IDENTIFIER.sub("_", value).strip("_").lower()
    return cleaned or "api"


def _load_spec(spec: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    """Load a specification from a mapping or a file path."""
    if isinstance(spec, Mapping):
        return spec

    path = Path(spec)
    if not path.exists():
        raise SourceError(f"OpenAPI spec not found: {path}")
    text = path.read_text(encoding="utf-8")

    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise SourceError(
                "YAML specs require pyyaml. Install with: pip install 'toolbroker[yaml]'"
            ) from exc
        loaded = yaml.safe_load(text)
    else:
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SourceError(f"{path} is not valid JSON: {exc}") from exc

    if not isinstance(loaded, Mapping):
        raise SourceError(f"{path} must contain an OpenAPI document at the top level")
    return loaded
