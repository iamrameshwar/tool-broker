"""Noticing when a tool changes underneath you.

The prompt-injection risk everyone talks about is a hostile tool description on
the day you connect a server. The one that actually gets people is a
description that changes *later*, on a server they already reviewed and
approved — because nothing anywhere is watching for it. The agent's behaviour
shifts, the tool list looks identical, and no request fails.

ToolBroker is the only component positioned to catch that, because it is the
only one that compares the catalogue against its previous state on every
refresh. :class:`~toolbroker.index.indexer.Indexer` already computes the diff
to decide what to re-embed; this turns the same comparison into a security
control.

Two fields matter, for two different reasons:

* **description** and **input_schema** reach the model, so a change to either
  is a change to what the model is being told. That is the injection surface.
* **risk** and **required_scopes** are read by policy, so a change to either
  moves a privilege boundary. That is the authorisation surface.

    guard = DriftGuard(quarantine=True, path=Path("approved.json"))
    broker = ToolBroker(drift=guard)
    report = broker.refresh()
    for change in report.changes:
        alert(change.describe())

With ``quarantine`` on, a tool whose sensitive fields changed keeps serving
its **last approved** version until a human accepts the new one. The agent
keeps working, with text somebody signed off on, which is the behaviour you
want at three in the morning.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from .observability import get_logger

if TYPE_CHECKING:
    from .types import Tool

logger = get_logger("drift")

#: How much of the digest to keep. Full SHA-256 in a config file is noise; this
#: is far beyond collision range for a catalogue of a few thousand tools.
DIGEST_LENGTH = 16


class ChangeKind(str, Enum):
    """Which surface a change moved."""

    #: Text the model reads. The prompt-injection surface.
    DESCRIPTION = "description"
    #: Parameter schema, including its descriptions. Also read by the model.
    SCHEMA = "schema"
    #: Risk tier or required scopes. The authorisation surface.
    PRIVILEGE = "privilege"


class ToolChange(BaseModel):
    """One field of one tool, before and after."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_id: str
    kind: ChangeKind
    before: str = ""
    after: str = ""
    #: Whether the new content was held back rather than served.
    quarantined: bool = False

    def describe(self) -> str:
        """One line suitable for an alert."""
        state = "QUARANTINED" if self.quarantined else "applied"
        return (
            f"[{state}] {self.tool_id} {self.kind.value} changed\n"
            f"  before: {_clip(self.before)}\n"
            f"   after: {_clip(self.after)}"
        )


def _clip(text: str, limit: int = 160) -> str:
    """Shorten for a log line, keeping it on one line."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def sensitive_digest(tool: Tool) -> str:
    """Return a stable digest of everything about ``tool`` that reaches a model.

    Deliberately excludes name and namespace: those are the tool's identity, and
    a change to either produces a different tool id, which the diff already sees
    as a removal plus an addition.
    """
    payload = json.dumps(
        {
            "description": tool.description,
            "input_schema": tool.input_schema,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:DIGEST_LENGTH]


class DriftGuard:
    """Detects, records, and optionally holds back changes to known tools."""

    def __init__(
        self,
        *,
        quarantine: bool = False,
        trust_on_first_use: bool = True,
        approved: Mapping[str, str] | None = None,
        path: str | Path | None = None,
    ) -> None:
        """Configure the guard.

        Args:
            quarantine: Keep serving the last approved version when a sensitive
                field changes, instead of accepting the new one.
            trust_on_first_use: Approve a tool's content the first time it is
                seen. On by default, because the alternative is a catalogue
                that starts empty and has to be approved tool by tool. Turn it
                off when the approved set is managed deliberately, and pair it
                with ``approved`` from config.
            approved: Known-good digests, ``{tool_id: digest}``. Values pinned
                here are the ones a change is compared against.
            path: Where to persist approvals. Without it, approvals last only
                as long as the process, so quarantine re-fires on restart.
        """
        self._quarantine = quarantine
        self._trust_on_first_use = trust_on_first_use
        self._path = Path(path) if path else None
        self._approved: dict[str, str] = dict(approved or {})
        if self._path and self._path.exists():
            self._approved.update(self._read(self._path))

    @property
    def quarantining(self) -> bool:
        """Whether changed content is held back rather than served."""
        return self._quarantine

    @property
    def approved(self) -> Mapping[str, str]:
        """The currently approved digest for each known tool."""
        return dict(self._approved)

    def is_approved(self, tool: Tool) -> bool:
        """Whether ``tool``'s current content matches its approved digest."""
        known = self._approved.get(tool.id)
        return known is not None and known == sensitive_digest(tool)

    def approve(self, tool: Tool) -> str:
        """Accept ``tool``'s current content, returning the digest recorded."""
        digest = sensitive_digest(tool)
        self._approved[tool.id] = digest
        return digest

    def approve_digest(self, tool_id: str, digest: str) -> None:
        """Accept a digest for ``tool_id`` without needing the tool itself."""
        self._approved[tool_id] = digest

    def forget(self, tool_id: str) -> bool:
        """Drop an approval, returning whether one existed."""
        return self._approved.pop(tool_id, None) is not None

    def inspect(self, previous: Tool, incoming: Tool) -> list[ToolChange]:
        """Return every sensitive change between two versions of one tool.

        Compares against the *approved* content where there is one, not merely
        against what was last stored. A change that was quarantined and never
        approved therefore keeps being reported until somebody decides, rather
        than becoming the new baseline on the next refresh.
        """
        changes: list[ToolChange] = []
        held = self._quarantine and not self.is_approved(incoming)

        if previous.description != incoming.description:
            changes.append(
                ToolChange(
                    tool_id=incoming.id,
                    kind=ChangeKind.DESCRIPTION,
                    before=previous.description,
                    after=incoming.description,
                    quarantined=held,
                )
            )
        if previous.input_schema != incoming.input_schema:
            changes.append(
                ToolChange(
                    tool_id=incoming.id,
                    kind=ChangeKind.SCHEMA,
                    before=json.dumps(previous.input_schema, sort_keys=True, default=str),
                    after=json.dumps(incoming.input_schema, sort_keys=True, default=str),
                    quarantined=held,
                )
            )
        if (
            previous.risk != incoming.risk
            or previous.required_scopes != incoming.required_scopes
            or previous.tags != incoming.tags
        ):
            # Never quarantined: holding back a privilege change could mean
            # serving a tool at a *lower* risk tier than the server now claims,
            # which is the dangerous direction. Report it and apply it.
            changes.append(
                ToolChange(
                    tool_id=incoming.id,
                    kind=ChangeKind.PRIVILEGE,
                    before=_privilege(previous),
                    after=_privilege(incoming),
                    quarantined=False,
                )
            )
        return changes

    def should_hold(self, incoming: Tool) -> bool:
        """Whether ``incoming`` must be held back rather than served."""
        return self._quarantine and not self.is_approved(incoming)

    def observe_new(self, tool: Tool) -> None:
        """Record a tool seen for the first time."""
        if self._trust_on_first_use:
            self.approve(tool)

    def holds_new_tools(self) -> bool:
        """Whether a tool seen for the first time is held pending approval.

        This is what closes the rename bypass. A tool renamed by a hostile
        server has a different id, so the diff sees a removal and an addition
        rather than a change — and quarantine, which only guards *changes*,
        never fires. With trust-on-first-use off, arrivals are held too, and
        the rename is stopped by the same gate.
        """
        return self._quarantine and not self._trust_on_first_use

    def accept(self, tool: Tool) -> None:
        """Record content that was served, so it becomes the baseline."""
        if not self._quarantine:
            self.approve(tool)

    def save(self, path: str | Path | None = None) -> Path:
        """Persist approvals so quarantine does not re-fire on restart."""
        target = Path(path) if path else self._path
        if target is None:
            raise ValueError("DriftGuard has no path; pass one to save()")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self._approved, indent=2, sort_keys=True), encoding="utf-8")
        return target

    @staticmethod
    def _read(path: Path) -> dict[str, str]:
        """Load approvals, tolerating a corrupt file rather than refusing to start."""
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("could not read approvals from %s: %s", path, exc)
            return {}
        if not isinstance(payload, dict):
            logger.warning("approvals file %s is not an object; ignoring", path)
            return {}
        return {str(key): str(value) for key, value in payload.items()}


def _privilege(tool: Tool) -> str:
    """Render the authorisation-relevant fields of ``tool``."""
    scopes = ",".join(sorted(tool.required_scopes)) or "none"
    tags = ",".join(sorted(tool.tags)) or "none"
    return f"risk={tool.risk.value} scopes={scopes} tags={tags}"
