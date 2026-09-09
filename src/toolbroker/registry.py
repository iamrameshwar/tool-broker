"""Plugin resolution via ``importlib.metadata`` entry points.

Built-in components register through exactly the same mechanism third-party
packages use. Nothing in the core is special-cased, so "write your own store"
is a real instruction rather than an aspiration.

Resolution order for any component name:

1. an object explicitly registered at runtime via :func:`register`
2. an installed entry point in the matching group
3. a dotted import path (``my_pkg.module:ClassName``)
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
from collections.abc import Iterator, Mapping
from importlib import metadata
from typing import Any, TypeVar

from .errors import PluginError

T = TypeVar("T")

GROUP_EMBEDDERS = "toolbroker.embedders"
GROUP_STORES = "toolbroker.stores"
GROUP_RETRIEVERS = "toolbroker.retrievers"
GROUP_RERANKERS = "toolbroker.rerankers"
GROUP_ADAPTERS = "toolbroker.adapters"
GROUP_SOURCES = "toolbroker.sources"
GROUP_POLICIES = "toolbroker.policies"
GROUP_TRACERS = "toolbroker.tracers"
GROUP_HOOKS = "toolbroker.hooks"

ALL_GROUPS = (
    GROUP_EMBEDDERS,
    GROUP_STORES,
    GROUP_RETRIEVERS,
    GROUP_RERANKERS,
    GROUP_ADAPTERS,
    GROUP_SOURCES,
    GROUP_POLICIES,
    GROUP_TRACERS,
    GROUP_HOOKS,
)

_runtime: dict[str, dict[str, Any]] = {group: {} for group in ALL_GROUPS}


def register(group: str, name: str, obj: Any) -> None:
    """Register ``obj`` under ``name`` at runtime, shadowing any entry point."""
    if group not in _runtime:
        _runtime[group] = {}
    _runtime[group][name] = obj


def unregister(group: str, name: str) -> bool:
    """Remove a runtime registration; return whether one existed."""
    return _runtime.get(group, {}).pop(name, None) is not None


def _iter_entry_points(group: str) -> Iterator[metadata.EntryPoint]:
    yield from metadata.entry_points(group=group)


def available(group: str) -> list[str]:
    """Return every name resolvable in ``group``, runtime and entry points."""
    names = set(_runtime.get(group, {}))
    names.update(entry.name for entry in _iter_entry_points(group))
    return sorted(names)


def _load_dotted(path: str) -> Any:
    module_name, _, attribute = path.partition(":")
    if not attribute:
        module_name, _, attribute = path.rpartition(".")
    if not module_name or not attribute:
        raise PluginError(f"{path!r} is not a valid import path (expected 'module:Attribute')")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise PluginError(f"cannot import module {module_name!r}: {exc}") from exc
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise PluginError(f"{module_name!r} has no attribute {attribute!r}") from exc


def resolve(group: str, name: str) -> Any:
    """Resolve ``name`` in ``group`` to a class or factory.

    Raises:
        PluginError: if nothing matches, listing what is installed.
    """
    runtime_hit = _runtime.get(group, {}).get(name)
    if runtime_hit is not None:
        return runtime_hit

    for entry in _iter_entry_points(group):
        if entry.name == name:
            try:
                return entry.load()
            except Exception as exc:
                raise PluginError(
                    f"plugin {name!r} in {group!r} failed to load: {exc}. "
                    "Its optional dependencies may not be installed."
                ) from exc

    if ":" in name or "." in name:
        return _load_dotted(name)

    installed = available(group)
    hint = ", ".join(installed) if installed else "none"
    raise PluginError(f"no plugin {name!r} in {group!r}. Installed: {hint}")


def create(group: str, name: str, /, **kwargs: Any) -> Any:
    """Resolve ``name`` and instantiate it with ``kwargs``."""
    factory = resolve(group, name)
    try:
        return factory(**kwargs)
    except TypeError as exc:
        raise PluginError(f"cannot construct {group}:{name} with {sorted(kwargs)}: {exc}") from exc


def construct(
    group: str,
    name: str,
    options: Mapping[str, Any] | None = None,
    /,
    **offered: Any,
) -> Any:
    """Instantiate ``name``, passing only the ``offered`` arguments it accepts.

    Retrievers and rerankers need collaborators a config file cannot express:
    a semantic retriever wants the store *and* the embedder, a lexical one only
    the store, and a third-party one may want neither. Rather than force every
    plugin to accept every collaborator, this passes each one only if the
    constructor names it — or all of them if it takes ``**kwargs``.

    ``options`` come from the config file and are always passed; a name that
    appears in both wins from ``options``, so a user can override what would
    otherwise be injected.
    """
    factory = resolve(group, name)
    settings = dict(options or {})

    parameters: Mapping[str, inspect.Parameter] = {}
    # Builtins and some C extensions have no introspectable signature. Falling
    # back to "accepts nothing" is right: options still go through, and the
    # constructor's own TypeError is a better message than a guess.
    with contextlib.suppress(TypeError, ValueError):
        parameters = inspect.signature(factory).parameters

    takes_everything = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    accepted = {
        key: value
        for key, value in offered.items()
        if key not in settings and (takes_everything or key in parameters)
    }

    try:
        return factory(**accepted, **settings)
    except TypeError as exc:
        raise PluginError(
            f"cannot construct {group}:{name} with {sorted({**accepted, **settings})}: {exc}"
        ) from exc
