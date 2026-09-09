"""Environment substitution for config files.

A config file is meant to be checked into a repo — that is the whole argument
for having one, because agent permissions should be reviewed like any other
change. But a real deployment's config also holds a Postgres DSN, an API key,
and an internal MCP endpoint, and none of those can be committed.

So values come from the environment:

.. code-block:: yaml

    store:
      name: pgvector
      options:
        dsn: ${TOOLBROKER_PG_DSN}                    # required
        table: ${TOOLBROKER_TABLE:-tools}            # with a default
    sources:
      - {type: mcp, name: github, url: "${GITHUB_MCP_URL:-http://localhost:8080}"}

Three rules, and they exist to make the failure modes loud:

* ``${VAR}`` is **required**. A missing one raises, naming the variable and
  where in the file it appeared, rather than quietly producing the string
  ``"${VAR}"`` and failing later inside a database driver.
* ``${VAR:-default}`` falls back, using the shell syntax people already know.
  An empty value counts as set, matching the shell.
* ``$${...}`` escapes to a literal ``${...}``, for the rare value that really
  contains one.

Substitution walks the *parsed* structure rather than the raw text, so it
cannot corrupt YAML syntax and a value containing a colon or a newline is
still safe.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .errors import ConfigurationError

#: ``${NAME}`` or ``${NAME:-default}``, with ``$${`` escaping a literal.
_PATTERN = re.compile(
    r"""
    (?P<escape>\$\$)\{                       # $${...}  -> literal ${...}
    |
    \$\{
        (?P<name>[A-Za-z_][A-Za-z0-9_]*)     # variable name
        (?: :- (?P<default>[^}]*) )?         # optional :-default
    \}
    """,
    re.VERBOSE,
)


def interpolate(value: Any, *, env: Mapping[str, str] | None = None, path: str = "") -> Any:
    """Return ``value`` with every ``${VAR}`` replaced from the environment.

    Args:
        value: Any parsed config fragment — mapping, sequence, string, scalar.
        env: Where to read variables from. Defaults to ``os.environ``.
        path: Dotted location used in error messages. Supplied recursively.

    Returns:
        The same shape with strings substituted.

    Raises:
        ConfigurationError: if a ``${VAR}`` with no default is not set.
    """
    source = os.environ if env is None else env

    if isinstance(value, str):
        return _substitute(value, source, path)
    if isinstance(value, Mapping):
        return {
            key: interpolate(item, env=source, path=f"{path}.{key}" if path else str(key))
            for key, item in value.items()
        }
    # str is a Sequence, and so is bytes; both are handled or ignored above.
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [
            interpolate(item, env=source, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    return value


def _substitute(text: str, env: Mapping[str, str], path: str) -> str:
    """Replace every reference in one string."""
    missing: list[str] = []

    def replace(match: re.Match[str]) -> str:
        if match.group("escape"):
            return "${"
        name = match.group("name")
        default = match.group("default")
        if name in env:
            return env[name]
        if default is not None:
            return default
        missing.append(name)
        return ""

    result = _PATTERN.sub(replace, text)
    if missing:
        listed = ", ".join(sorted(set(missing)))
        where = f" (at {path})" if path else ""
        raise ConfigurationError(
            f"config references unset environment variable{'s' if len(set(missing)) > 1 else ''}: "
            f"{listed}{where}. Set it, or give a default with "
            f"${{{sorted(set(missing))[0]}:-some-value}}."
        )
    return result
