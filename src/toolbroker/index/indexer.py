"""Builds store records from tools.

Kept separate from both the source and the store so the enrichment and
embedding steps can be tested, cached, and hooked without a live server or a
running database.

Two ways in. :meth:`Indexer.index` rebuilds; :meth:`Indexer.sync` diffs. Sync is
what a live catalogue wants: an MCP server that adds one tool should cost one
embedding call, not five hundred.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..drift import DriftGuard, ToolChange
from ..errors import DimensionMismatchError
from ..hooks import Event, HookManager, _Drop
from ..observability import get_logger, span
from ..protocols import Embedder, Store
from ..types import Tool, ToolRecord
from .enrich import EnrichmentConfig, build_index_text

logger = get_logger("indexer")


@dataclass(frozen=True, slots=True)
class RefreshReport:
    """What one incremental sync changed.

    Distinguishes *updated* from *re-embedded* on purpose. A tool whose risk
    tier or required scopes changed must be rewritten so policy sees the new
    values, but its index text did not change, so it needs no embedding call —
    and with a paid embedder that difference is the whole point.
    """

    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    updated: tuple[str, ...] = ()
    unchanged: int = 0
    embedded: int = 0
    #: Sources that failed during discovery, as ``(source_id, error)``. Their
    #: tools are deliberately left in place — see :meth:`Indexer.sync`.
    failed_sources: tuple[tuple[str, str], ...] = ()
    #: Sensitive changes to tools that already existed. Empty unless a
    #: :class:`~toolbroker.drift.DriftGuard` is attached.
    changes: tuple[ToolChange, ...] = ()
    #: Tools whose new content was held back pending approval.
    quarantined: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        """Whether anything moved."""
        return bool(self.added or self.removed or self.updated)

    @property
    def total(self) -> int:
        """Tools in the catalogue after the sync."""
        return len(self.added) + len(self.updated) + self.unchanged

    def summary(self) -> str:
        """One-line description, for logs."""
        parts = [
            f"+{len(self.added)}",
            f"-{len(self.removed)}",
            f"~{len(self.updated)}",
            f"={self.unchanged}",
            f"embedded {self.embedded}",
        ]
        if self.changes:
            parts.append(f"{len(self.changes)} sensitive change(s)")
        if self.quarantined:
            parts.append(f"{len(self.quarantined)} quarantined")
        if self.failed_sources:
            parts.append(f"{len(self.failed_sources)} source(s) failed")
        return " ".join(parts)


@dataclass(frozen=True, slots=True)
class IndexReport:
    """What one indexing pass did."""

    indexed: int
    skipped: int
    removed: int
    duplicates: tuple[str, ...] = ()

    @property
    def total(self) -> int:
        """Tools considered."""
        return self.indexed + self.skipped


class Indexer:
    """Enriches, embeds, and writes tools into a store."""

    def __init__(
        self,
        embedder: Embedder,
        store: Store,
        *,
        enrichment: EnrichmentConfig | None = None,
        hooks: HookManager | None = None,
        batch_size: int = 128,
        drift: DriftGuard | None = None,
    ) -> None:
        """Wire the embedder and store this indexer writes through."""
        if embedder.dim != store.dim:
            raise DimensionMismatchError(store.dim, embedder.dim)
        self._embedder = embedder
        self._store = store
        self._enrichment = enrichment or EnrichmentConfig()
        self._hooks = hooks or HookManager()
        self._batch_size = max(1, batch_size)
        self._drift = drift

    @property
    def store(self) -> Store:
        """The store being written to."""
        return self._store

    @property
    def embedder(self) -> Embedder:
        """The embedder being used."""
        return self._embedder

    def index(self, tools: Sequence[Tool], *, replace: bool = False) -> IndexReport:
        """Index ``tools``.

        Args:
            tools: The tools to index.
            replace: Clear the store first, so removed tools actually disappear.

        Returns:
            A report of what changed.
        """
        with span("toolbroker.index", tool_count=len(tools)):
            removed = 0
            if replace:
                removed = len(self._store)
                self._store.clear()

            deduped: dict[str, Tool] = {}
            duplicates: list[str] = []
            for tool in tools:
                transformed = self._hooks.transform_or_drop(Event.TRANSFORM_TOOL, tool)
                if isinstance(transformed, _Drop):
                    continue
                if transformed.id in deduped:
                    # Last writer wins, but say so: silently dropping a tool
                    # because two servers chose the same name is a debugging
                    # nightmare for whoever hits it.
                    duplicates.append(transformed.id)
                deduped[transformed.id] = transformed

            if duplicates:
                logger.warning(
                    "duplicate tool ids collapsed; namespace your sources to avoid this",
                    extra={"duplicates": sorted(set(duplicates))},
                )

            selected = list(deduped.values())
            skipped = len(tools) - len(selected)
            records: list[ToolRecord] = []

            for start in range(0, len(selected), self._batch_size):
                chunk = selected[start : start + self._batch_size]
                texts = [
                    self._hooks.transform(
                        Event.TRANSFORM_INDEX_TEXT,
                        build_index_text(tool, self._enrichment),
                        tool=tool,
                    )
                    for tool in chunk
                ]
                vectors = self._embedder.embed(texts)
                records.extend(
                    ToolRecord(tool=tool, text=text, vector=tuple(vector))
                    for tool, text, vector in zip(chunk, texts, vectors, strict=True)
                )

            if records:
                self._store.upsert(records)

            report = IndexReport(
                indexed=len(records),
                skipped=skipped,
                removed=removed,
                duplicates=tuple(sorted(set(duplicates))),
            )
            self._hooks.emit(Event.AFTER_INDEXING, report=report, store=self._store)
            logger.info(
                "indexing complete",
                extra={"indexed": report.indexed, "skipped": report.skipped},
            )
            return report

    # -- incremental ------------------------------------------------------

    def sync(
        self,
        tools: Sequence[Tool],
        *,
        prune: bool = True,
        known_sources: Sequence[str] | None = None,
        failed_sources: Sequence[tuple[str, str]] = (),
    ) -> RefreshReport:
        """Bring the store in line with ``tools``, touching only what changed.

        Args:
            tools: The catalogue as it is now.
            prune: Remove stored tools that are no longer present.
            known_sources: Only prune tools belonging to these sources. This is
                the safety valve: if an MCP server is unreachable, its tools are
                absent from ``tools`` but must **not** be deleted, or a thirty
                second outage silently strips the catalogue and the agent
                quietly loses capabilities with no error anywhere.
            failed_sources: Recorded on the report so callers can alert.

        Returns:
            A :class:`RefreshReport` naming exactly what moved.
        """
        with span("toolbroker.sync", tool_count=len(tools)):
            incoming = self._deduplicate(tools)
            existing = {record.id: record for record in self._store.all_records()}

            added: list[str] = []
            updated: list[str] = []
            unchanged = 0
            to_embed: list[tuple[Tool, str]] = []
            to_write: list[ToolRecord] = []

            changes: list[ToolChange] = []
            quarantined: list[str] = []

            for tool_id, tool in incoming.items():
                previous = existing.get(tool_id)

                if previous is not None and self._drift is not None:
                    found = self._drift.inspect(previous.tool, tool)
                    changes.extend(found)
                    if found and self._drift.should_hold(tool):
                        # Keep serving content a human approved. The record is
                        # left exactly as it is: no rewrite, no embedding call,
                        # and the change stays pending until somebody accepts it.
                        quarantined.append(tool_id)
                        unchanged += 1
                        logger.warning(
                            "quarantined change to %s; serving the approved version", tool_id
                        )
                        continue
                    if found:
                        self._drift.accept(tool)
                elif previous is None and self._drift is not None:
                    if self._drift.holds_new_tools():
                        # A renamed tool arrives as an addition, not a change,
                        # so quarantining changes alone would not stop it.
                        quarantined.append(tool_id)
                        logger.warning("quarantined new tool %s pending approval", tool_id)
                        continue
                    self._drift.observe_new(tool)

                text = self._index_text(tool)

                if previous is None:
                    added.append(tool_id)
                    to_embed.append((tool, text))
                elif previous.text != text:
                    # The embedded text changed, so the vector is stale.
                    updated.append(tool_id)
                    to_embed.append((tool, text))
                elif previous.tool != tool:
                    # Same text, different tool: risk, scopes, or tags moved.
                    # Policy reads those, so the record must be rewritten — but
                    # the vector is still correct, so no embedding call.
                    updated.append(tool_id)
                    to_write.append(ToolRecord(tool=tool, text=text, vector=previous.vector))
                else:
                    unchanged += 1

            for start in range(0, len(to_embed), self._batch_size):
                chunk = to_embed[start : start + self._batch_size]
                vectors = self._embedder.embed([text for _, text in chunk])
                to_write.extend(
                    ToolRecord(tool=tool, text=text, vector=tuple(vector))
                    for (tool, text), vector in zip(chunk, vectors, strict=True)
                )

            if to_write:
                self._store.upsert(to_write)

            removed: list[str] = []
            if prune:
                prunable = self._prunable(existing, incoming, known_sources)
                if prunable:
                    self._store.delete(prunable)
                    removed = prunable

            report = RefreshReport(
                added=tuple(sorted(added)),
                removed=tuple(sorted(removed)),
                updated=tuple(sorted(updated)),
                unchanged=unchanged,
                embedded=len(to_embed),
                failed_sources=tuple(failed_sources),
                changes=tuple(changes),
                quarantined=tuple(sorted(quarantined)),
            )
            self._hooks.emit(Event.AFTER_INDEXING, report=report, store=self._store)
            logger.info("sync complete", extra={"summary": report.summary()})
            return report

    @staticmethod
    def _prunable(
        existing: dict[str, ToolRecord],
        incoming: dict[str, Tool],
        known_sources: Sequence[str] | None,
    ) -> list[str]:
        """Return stored ids that should be deleted."""
        candidates = [tool_id for tool_id in existing if tool_id not in incoming]
        if known_sources is None:
            return sorted(candidates)
        # Only prune within sources that answered. A tool whose source failed
        # is missing because we could not ask, not because it is gone.
        answered = set(known_sources)
        return sorted(
            tool_id for tool_id in candidates if existing[tool_id].tool.source_id in answered
        )

    def _index_text(self, tool: Tool) -> str:
        """Return the text that would be embedded for ``tool``."""
        return self._hooks.transform(
            Event.TRANSFORM_INDEX_TEXT,
            build_index_text(tool, self._enrichment),
            tool=tool,
        )

    def _deduplicate(self, tools: Sequence[Tool]) -> dict[str, Tool]:
        """Apply tool hooks and collapse duplicate ids, last writer winning."""
        result: dict[str, Tool] = {}
        duplicates: list[str] = []
        for tool in tools:
            transformed = self._hooks.transform_or_drop(Event.TRANSFORM_TOOL, tool)
            if isinstance(transformed, _Drop):
                continue
            if transformed.id in result:
                duplicates.append(transformed.id)
            result[transformed.id] = transformed
        if duplicates:
            logger.warning(
                "duplicate tool ids collapsed; namespace your sources to avoid this",
                extra={"duplicates": sorted(set(duplicates))},
            )
        return result
