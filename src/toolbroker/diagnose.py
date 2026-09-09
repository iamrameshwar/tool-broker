"""Finding the tools in your catalogue that retrieval can never surface.

A catalogue can be perfectly configured and still contain tools no agent will
ever be handed, because retrieval cannot tell them apart from their neighbours
or has almost nothing to match on. Those tools are invisible in the way that is
hardest to notice: nothing fails, no error is raised, the agent simply never
uses them.

This module finds them without labelled data, the same way
:mod:`toolbroker.calibrate` derives a score floor without labelled data — by
asking the catalogue about itself.

Two probes, asking two different questions.

For **shadowing**, each tool's own description is run as a query. It deliberately
is *not* the tool's indexed vector: that vector includes the name, namespace and
tags, so searching with it would return the tool at rank 1 by self-similarity
every time and the check would be vacuous. The description alone is the closest
thing to a perfect user query, and a tool that cannot win on it will not win on
a user's rougher paraphrase. Tools sharing a description are queried once
between them, so the cost is one embedding per *distinct* description.

For **pairs**, the indexed vectors are compared directly. That costs no
embedding calls, and here self-similarity is the point: it is the fixed
baseline the gap to a neighbour is measured against.

What it reports:

* **Shadowed** — the tool is not in the top ``k`` when its own description is
  used as the query. Some other group of tools always looks at least as good.
  Threshold-free: ``k`` is the width you already selected at, not a tuning
  constant.
* **Thin** — the description is empty or so short there is nothing to match on.
* **Confusable pairs** — the closest neighbours in the catalogue, ranked. Which
  of a tight pair a query gets is close to arbitrary, and no ranking change
  fixes that; the descriptions have to say what differs.

Confusable pairs are **ranked rather than flagged**, and that is a deliberate
consequence of a measurement. On the reference catalogue, tools with byte-identical
descriptions sit 0.02 to 0.06 apart under bge-small but 0.07 to 0.20 apart under the
hashing embedder — the same defect, three times the number. Worse, normalising by
the catalogue's own spread does not rescue it: a catalogue that is 96% duplicates
and one with no duplicates at all produce nearly the same distribution shape, so
there is no internal statistic that says which you have. A default margin would
therefore be silently wrong on some embedder, which is the same reason no default
``MinScore`` ships. Pass ``margin`` once you have looked at the ranking and know
what your embedder calls "too close".

What it does **not** find, and this is the important caveat: a tool with a
unique, well-written description that nobody would ever phrase that way. On the
project's own benchmark that vocabulary gap accounts for more failures than
everything here combined — ``bounce the pods`` never reaches
``restart_service`` at any depth, and no property of the catalogue alone
predicts it. Closing that needs real queries, which is what ``samples`` is for.

    from toolbroker.diagnose import diagnose_catalogue

    report = diagnose_catalogue(broker)
    print(report.summary())
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from .catalog import ToolBroker

#: Below this many characters a description carries almost no signal. Chosen to
#: be forgiving: "Cancel a shipment before pickup." is 32 and perfectly fine.
#: Unlike a score margin this is embedder-independent, so it can have a default.
DEFAULT_MIN_DESCRIPTION = 16

#: How many of the closest pairs to report.
DEFAULT_PAIRS = 20

SHADOWED = "shadowed"
TWINNED = "twinned"
THIN = "thin-description"
UNDESCRIBED = "no-description"


class ToolPair(BaseModel):
    """Two tools and how far apart retrieval considers them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    left: str
    right: str
    #: Score distance as a share of a tool's self-similarity. Zero means the
    #: store cannot separate them at all.
    gap: float
    #: Whether the two carry byte-identical descriptions, which makes the pair
    #: a certainty rather than a suspicion.
    identical_description: bool = False


class ToolFinding(BaseModel):
    """What the catalogue says about one tool's findability."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_id: str
    issues: tuple[str, ...] = ()
    #: Rank the tool reached when its own description was used as the query.
    #: ``1`` is healthy; ``None`` means it did not appear in the top ``k`` at
    #: all, or that it has no description to query with.
    self_rank: int | None = None
    #: Tools that outranked it on its own description, best first.
    shadowed_by: tuple[str, ...] = ()
    #: Closest other tool, and the gap to it.
    nearest: str | None = None
    nearest_gap: float | None = None
    description_length: int = 0

    @property
    def healthy(self) -> bool:
        """Whether nothing was found against this tool."""
        return not self.issues


class CatalogueDiagnosis(BaseModel):
    """Findability of every tool in a catalogue."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    findings: tuple[ToolFinding, ...] = ()
    #: Closest pairs in the catalogue, tightest first.
    closest_pairs: tuple[ToolPair, ...] = ()
    catalogue_size: int = 0
    embedder: str = ""
    k: int = 5
    #: The margin used to flag twins, when one was supplied.
    margin: float | None = None
    #: Tools no sample query surfaced. Empty when no samples were supplied,
    #: which is not the same as "every tool is reachable".
    unreached: tuple[str, ...] = ()
    sample_count: int = 0

    @property
    def unhealthy(self) -> tuple[ToolFinding, ...]:
        """Findings with at least one issue, most issues first."""
        return tuple(
            sorted(
                (finding for finding in self.findings if finding.issues),
                key=lambda finding: (-len(finding.issues), finding.tool_id),
            )
        )

    def with_issue(self, issue: str) -> tuple[ToolFinding, ...]:
        """Findings carrying ``issue``."""
        return tuple(finding for finding in self.findings if issue in finding.issues)

    def summary(self) -> str:
        """Render the findings in readable form."""
        shadowed = self.with_issue(SHADOWED)
        twinned = self.with_issue(TWINNED)
        thin = self.with_issue(THIN) + self.with_issue(UNDESCRIBED)
        flagged = {finding.tool_id for finding in shadowed + twinned + thin}

        lines = [
            f"catalogue: {self.catalogue_size} tools    embedder: {self.embedder}",
            f"{len(flagged)} of {self.catalogue_size} tools flagged, k={self.k}",
            "",
        ]

        if shadowed:
            lines.append(
                f"SHADOWED — not in the top {self.k} for their own description ({len(shadowed)})"
            )
            lines.append("  A query describing these exactly does not reach them, so a")
            lines.append("  rougher real query will not either.")
            lines.extend(_shadowed_rows(shadowed))
            lines.append("")

        if twinned:
            lines.append(f"TWINNED — within the {self.margin:.1%} margin you set ({len(twinned)})")
            lines.extend(
                f"    {finding.tool_id:<44} {finding.nearest_gap or 0.0:>7.1%}   "
                f"twin: {finding.nearest}"
                for finding in twinned[:20]
            )
            if len(twinned) > 20:
                lines.append(f"    ... and {len(twinned) - 20} more")
            lines.append("")

        if thin:
            lines.append(f"THIN — nothing to match on ({len(thin)})")
            lines.extend(_thin_rows(thin))
            lines.append("")

        if self.closest_pairs:
            lines.append("CLOSEST PAIRS — ranked, not judged")
            lines.append("  Read down until the pairs stop looking interchangeable. That")
            lines.append("  boundary is your embedder's margin; pass it as `margin=`.")
            for pair in self.closest_pairs:
                mark = "  (identical description)" if pair.identical_description else ""
                lines.append(f"    {pair.gap:>7.1%}   {pair.left}  ~  {pair.right}{mark}")
            lines.append("")

        if self.sample_count:
            share = len(self.unreached) / self.catalogue_size if self.catalogue_size else 0.0
            lines.append(
                f"UNREACHED — not surfaced by any of your {self.sample_count} sample "
                f"queries ({len(self.unreached)}, {share:.0%})"
            )
            lines.append("  Only meaningful if the samples represent real traffic. If they")
            lines.append("  do, these tools are paying for a place in the catalogue they")
            lines.append("  never earn.")
            lines.extend(f"    {tool_id}" for tool_id in self.unreached[:20])
            if len(self.unreached) > 20:
                lines.append(f"    ... and {len(self.unreached) - 20} more")
            lines.append("")

        if not flagged and not self.unreached:
            lines.append("Nothing flagged: every tool is retrievable by its own description.")
            lines.append("")

        lines.append(_CAVEAT)
        return "\n".join(lines)


_CAVEAT = (
    "This reads the catalogue against itself, so it finds tools that collide or\n"
    "carry no signal. It cannot find a tool whose description is unique and clear\n"
    "but phrased unlike anything a user would type — the largest category of\n"
    "retrieval failure, and the one only real queries expose. Pass a sample of\n"
    "them as `samples` to see which tools your traffic never reaches."
)


def _shadowed_rows(findings: Sequence[ToolFinding], limit: int = 20) -> list[str]:
    """Render the shadowed table."""
    rows = []
    for finding in findings[:limit]:
        rank = "not found" if finding.self_rank is None else f"rank {finding.self_rank}"
        blockers = ", ".join(finding.shadowed_by[:3]) or "—"
        rows.append(f"    {finding.tool_id:<44} {rank:>10}   behind: {blockers}")
    if len(findings) > limit:
        rows.append(f"    ... and {len(findings) - limit} more")
    return rows


def _thin_rows(findings: Sequence[ToolFinding], limit: int = 20) -> list[str]:
    """Render the thin-description table."""
    rows = []
    for finding in findings[:limit]:
        what = "empty" if UNDESCRIBED in finding.issues else f"{finding.description_length} chars"
        rows.append(f"    {finding.tool_id:<44} {what:>10}")
    if len(findings) > limit:
        rows.append(f"    ... and {len(findings) - limit} more")
    return rows


def diagnose_catalogue(
    broker: ToolBroker,
    *,
    k: int = 5,
    samples: Sequence[str] = (),
    min_description: int = DEFAULT_MIN_DESCRIPTION,
    margin: float | None = None,
    pairs: int = DEFAULT_PAIRS,
) -> CatalogueDiagnosis:
    """Report which tools in ``broker`` retrieval cannot surface.

    Args:
        broker: An indexed catalogue.
        k: Selection width. A tool outside the top ``k`` for its own text is
            reported as shadowed, because that is the width it must win at.
        samples: Optional real queries. With them the report adds which tools
            nothing in the sample reaches; without them it can only speak about
            the catalogue's internal structure.
        min_description: Below this many characters a description is called
            thin.
        margin: Gap below which two tools are flagged as twins. ``None`` — the
            default — ranks the closest pairs instead of flagging any, because
            the right value depends on the embedder and no single number is
            correct across them.
        pairs: How many of the closest pairs to report.

    Returns:
        A :class:`CatalogueDiagnosis`.
    """
    store = broker.store
    records = list(store.all_records())

    # Tools sharing a description compete identically, so query once between
    # them. On a catalogue full of duplicates this is the difference between
    # one embedding call per tool and one per distinct description.
    by_description: dict[str, list[str]] = {}
    for record in records:
        description = record.tool.description.strip()
        if description:
            by_description.setdefault(description, []).append(record.id)

    ranked_by_description = {
        description: [hit.id for hit in broker.pipeline.retrieve(description, k)]
        for description in by_description
    }

    findings: list[ToolFinding] = []
    seen_pairs: dict[tuple[str, str], ToolPair] = {}

    for record in records:
        description = record.tool.description.strip()
        ranked = ranked_by_description.get(description, [])
        self_rank = ranked.index(record.id) + 1 if record.id in ranked else None

        # Pair analysis reads the indexed vectors directly: no embedding cost,
        # and self-similarity is the baseline the neighbour gap is measured on.
        neighbours = store.search(record.vector, 2)
        own_score = next((hit.score for hit in neighbours if hit.id == record.id), 0.0)
        others = [hit for hit in neighbours if hit.id != record.id]

        nearest = others[0] if others else None
        nearest_gap: float | None = None
        if nearest is not None and own_score:
            nearest_gap = max(0.0, (own_score - nearest.score) / abs(own_score))
            key = (min(record.id, nearest.id), max(record.id, nearest.id))
            existing = seen_pairs.get(key)
            if existing is None or nearest_gap < existing.gap:
                seen_pairs[key] = ToolPair(
                    left=key[0],
                    right=key[1],
                    gap=nearest_gap,
                    identical_description=(
                        bool(description) and description == nearest.tool.description.strip()
                    ),
                )

        issues: list[str] = []
        if not description:
            # Nothing to query with, so the shadow check cannot run and does not
            # need to: a tool with no description is already unfindable.
            issues.append(UNDESCRIBED)
        else:
            if self_rank is None:
                issues.append(SHADOWED)
            if len(description) < min_description:
                issues.append(THIN)
        if margin is not None and nearest_gap is not None and nearest_gap <= margin:
            issues.append(TWINNED)

        findings.append(
            ToolFinding(
                tool_id=record.id,
                issues=tuple(issues),
                self_rank=self_rank,
                shadowed_by=tuple(other for other in ranked if other != record.id)[:5],
                nearest=nearest.id if nearest is not None else None,
                nearest_gap=nearest_gap,
                description_length=len(description),
            )
        )

    unreached: tuple[str, ...] = ()
    if samples:
        reached = {hit.id for query in samples for hit in broker.pipeline.retrieve(query, k)}
        unreached = tuple(record.id for record in records if record.id not in reached)

    closest = tuple(sorted(seen_pairs.values(), key=lambda pair: (pair.gap, pair.left))[:pairs])

    return CatalogueDiagnosis(
        findings=tuple(findings),
        closest_pairs=closest,
        catalogue_size=len(records),
        embedder=broker.embedder.id,
        k=k,
        margin=margin,
        unreached=unreached,
        sample_count=len(samples),
    )
