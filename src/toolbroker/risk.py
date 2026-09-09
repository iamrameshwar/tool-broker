"""Guessing how dangerous a tool is from its name.

A name-based guess is a starting point, not an authority. It exists because a
catalogue of five hundred tools arrives with no risk metadata at all, and
"everything is LOW until someone classifies it" is the worst possible default
for a layer whose job includes keeping an agent away from ``delete_*``.

Anything that actually matters should be set explicitly, in the tool definition
or in policy. This is the floor, not the ceiling.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from .types import RiskTier

#: Called as ``classifier(name, description)``.
RiskClassifier: TypeAlias = Callable[[str, str], RiskTier]

DESTRUCTIVE_PREFIXES = (
    "delete",
    "remove",
    "drop",
    "destroy",
    "purge",
    "revoke",
    "terminate",
    "wipe",
    "erase",
)

MUTATING_PREFIXES = (
    "create",
    "update",
    "write",
    "send",
    "post",
    "put",
    "patch",
    "set",
    "insert",
    "modify",
    "restart",
    "deploy",
)


def classify_by_name(name: str, description: str = "") -> RiskTier:
    """Infer a risk tier from a tool's name, conservatively.

    Reads only the leading verb: ``delete_customer`` is destructive,
    ``create_invoice`` mutates, ``list_customers`` does neither. The description
    is accepted for signature compatibility with custom classifiers and
    deliberately unused — prose is far too easy to mislead on.
    """
    del description
    lowered = name.lower()
    if lowered.startswith(DESTRUCTIVE_PREFIXES):
        return RiskTier.HIGH
    if lowered.startswith(MUTATING_PREFIXES):
        return RiskTier.MEDIUM
    return RiskTier.LOW


def always(tier: RiskTier) -> RiskClassifier:
    """Return a classifier that assigns ``tier`` to everything."""

    def classifier(name: str, description: str = "") -> RiskTier:
        del name, description
        return tier

    return classifier
