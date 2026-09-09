"""A :class:`~toolbroker.protocols.Store` backed by Qdrant.

Two contract details drive most of the code here.

**Filters run server-side, before scoring.** Qdrant's own filter language does
the narrowing, so asking for five low-risk tools returns five rather than
whatever survives filtering the global top five. Translating ToolBroker's
:class:`~toolbroker.types.Filters` into that language is the bulk of this module.

**Ties break by tool id.** Qdrant returns floats; two identical vectors can come
back in either order across runs. The store re-sorts through
:func:`~toolbroker.determinism.stable_sort` so results are reproducible, which is
the same guarantee the in-memory store makes.
"""

from __future__ import annotations

import contextlib
import uuid
import warnings
from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient, models

from toolbroker._vectors import Vector
from toolbroker.determinism import stable_sort
from toolbroker.errors import DimensionMismatchError, StoreError
from toolbroker.types import Filters, Hit, RiskTier, Tool, ToolRecord

# Qdrant point ids must be an unsigned integer or a UUID, but tool ids are
# strings like "billing/issue_refund". uuid5 maps them deterministically, so the
# same tool always lands on the same point and upserts replace rather than
# duplicate.
_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

_DISTANCES = {
    "cosine": models.Distance.COSINE,
    "dot": models.Distance.DOT,
    "euclid": models.Distance.EUCLID,
    "manhattan": models.Distance.MANHATTAN,
}

_RISK_ORDER = [RiskTier.LOW, RiskTier.MEDIUM, RiskTier.HIGH, RiskTier.CRITICAL]


def point_id(tool_id: str) -> str:
    """Return the deterministic Qdrant point id for ``tool_id``."""
    return str(uuid.uuid5(_NAMESPACE, tool_id))


class QdrantStore:
    """Stores tool records in a Qdrant collection."""

    def __init__(
        self,
        dim: int,
        *,
        location: str | None = ":memory:",
        url: str | None = None,
        api_key: str | None = None,
        collection: str = "toolbroker",
        distance: str = "cosine",
        client: QdrantClient | None = None,
        recreate: bool = False,
        timeout: int | None = None,
        **client_kwargs: Any,
    ) -> None:
        """Connect to Qdrant and ensure the collection exists.

        Args:
            dim: Vector width. Must match the embedder.
            location: ``":memory:"`` for an in-process store, or a path for
                on-disk local mode. Ignored when ``url`` or ``client`` is given.
            url: Server URL, e.g. ``http://localhost:6333``.
            api_key: Qdrant Cloud API key.
            collection: Collection name.
            distance: ``cosine`` (default), ``dot``, ``euclid``, or ``manhattan``.
            client: An already-configured client. Takes precedence over
                ``location`` and ``url``, so a host application can share one
                connection pool.
            recreate: Drop and rebuild the collection on construction. Destroys
                data; off by default.
            timeout: Request timeout in seconds.
            **client_kwargs: Passed through to ``QdrantClient``.

        Raises:
            StoreError: If the collection exists with a different vector width.
        """
        if dim <= 0:
            raise ValueError("dim must be positive")
        if distance not in _DISTANCES:
            raise ValueError(f"distance must be one of {sorted(_DISTANCES)}")

        self._dim = dim
        self._collection = collection
        self._distance = _DISTANCES[distance]

        if client is not None:
            self._client = client
        elif url is not None:
            self._client = QdrantClient(url=url, api_key=api_key, timeout=timeout, **client_kwargs)
        else:
            self._client = QdrantClient(location=location, **client_kwargs)

        self._ensure_collection(recreate=recreate)

    # -- lifecycle --------------------------------------------------------

    @property
    def client(self) -> QdrantClient:
        """The underlying Qdrant client."""
        return self._client

    @property
    def collection(self) -> str:
        """Collection name."""
        return self._collection

    @property
    def dim(self) -> int:
        """Vector width this store was built for."""
        return self._dim

    def _ensure_collection(self, *, recreate: bool) -> None:
        """Create the collection, or verify an existing one matches."""
        exists = self._client.collection_exists(self._collection)
        if exists and recreate:
            self._client.delete_collection(self._collection)
            exists = False

        if exists:
            self._verify_dimension()
            return

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=models.VectorParams(size=self._dim, distance=self._distance),
        )
        # Payload indexes on the fields filters actually use. Without them
        # Qdrant scans the payload, which shows up as latency the moment a
        # catalogue passes a few thousand tools.
        with warnings.catch_warnings():
            # Local (in-process) Qdrant warns that indexes do nothing there.
            # True, and not actionable: the same code has to work against a
            # server, where they matter.
            warnings.simplefilter("ignore", UserWarning)
            for field, schema in (
                ("namespace", models.PayloadSchemaType.KEYWORD),
                ("tags", models.PayloadSchemaType.KEYWORD),
                ("tool_id", models.PayloadSchemaType.KEYWORD),
                ("risk_level", models.PayloadSchemaType.INTEGER),
            ):
                # Indexes are an optimisation, not a correctness requirement.
                with contextlib.suppress(Exception):
                    self._client.create_payload_index(
                        collection_name=self._collection,
                        field_name=field,
                        field_schema=schema,
                    )

    def _verify_dimension(self) -> None:
        """Fail loudly if an existing collection was built for another embedder."""
        size: Any = None
        with contextlib.suppress(Exception):  # shape varies by server version
            info = self._client.get_collection(self._collection)
            size = getattr(info.config.params.vectors, "size", None)
        if isinstance(size, int) and size != self._dim:
            raise StoreError(
                f"collection {self._collection!r} holds {size}-dimensional vectors but "
                f"this store expects {self._dim}. It was indexed with a different "
                "embedder; use a different collection or pass recreate=True."
            )

    # -- writes -----------------------------------------------------------

    def upsert(self, records: Sequence[ToolRecord]) -> None:
        """Insert or replace records, keyed by tool id."""
        if not records:
            return
        points: list[models.PointStruct] = []
        for record in records:
            if len(record.vector) != self._dim:
                raise DimensionMismatchError(self._dim, len(record.vector))
            points.append(
                models.PointStruct(
                    id=point_id(record.id),
                    vector=list(record.vector),
                    payload=self._payload(record),
                )
            )
        self._client.upsert(collection_name=self._collection, points=points, wait=True)

    @staticmethod
    def _payload(record: ToolRecord) -> dict[str, Any]:
        """Build the stored payload.

        The whole tool is serialised so records round-trip exactly, and the
        fields filters query are hoisted to the top level where Qdrant can index
        them.
        """
        tool = record.tool
        return {
            "tool_id": tool.id,
            "namespace": tool.namespace,
            "tags": sorted(tool.tags),
            "risk_level": _RISK_ORDER.index(tool.risk),
            "text": record.text,
            "tool": tool.model_dump(mode="json"),
        }

    def delete(self, tool_ids: Sequence[str]) -> int:
        """Remove records by tool id, returning how many existed."""
        if not tool_ids:
            return 0
        present = [tool_id for tool_id in tool_ids if self.get(tool_id) is not None]
        if not present:
            return 0
        self._client.delete(
            collection_name=self._collection,
            points_selector=models.PointIdsList(points=[point_id(t) for t in present]),
            wait=True,
        )
        return len(present)

    def clear(self) -> None:
        """Drop and recreate the collection."""
        self._client.delete_collection(self._collection)
        self._ensure_collection(recreate=False)

    # -- reads ------------------------------------------------------------

    def get(self, tool_id: str) -> ToolRecord | None:
        """Return one record by tool id."""
        found = self._client.retrieve(
            collection_name=self._collection,
            ids=[point_id(tool_id)],
            with_payload=True,
            with_vectors=True,
        )
        if not found:
            return None
        return self._to_record(found[0])

    def all_records(self) -> Sequence[ToolRecord]:
        """Return every record, paging through the collection."""
        records: list[ToolRecord] = []
        offset: Any = None
        while True:
            batch, offset = self._client.scroll(
                collection_name=self._collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            records.extend(self._to_record(point) for point in batch)
            if offset is None:
                break
        # Sorted so repeated calls agree; scroll order is not guaranteed stable.
        return tuple(sorted(records, key=lambda record: record.id))

    def search(self, vector: Vector, k: int, filters: Filters | None = None) -> list[Hit]:
        """Return the ``k`` highest-scoring records that satisfy ``filters``."""
        if k <= 0:
            return []
        if len(vector) != self._dim:
            raise DimensionMismatchError(self._dim, len(vector))

        response = self._client.query_points(
            collection_name=self._collection,
            query=list(vector),
            limit=k,
            query_filter=build_filter(filters),
            with_payload=True,
            with_vectors=False,
        )
        hits = [
            Hit(
                tool=self._to_tool(point.payload or {}),
                score=float(point.score),
                components={"vector": float(point.score)},
            )
            for point in response.points
        ]
        return stable_sort(hits)[:k]

    def __len__(self) -> int:
        """Number of stored records."""
        return int(self._client.count(self._collection, exact=True).count)

    # -- conversion -------------------------------------------------------

    @staticmethod
    def _to_tool(payload: dict[str, Any]) -> Tool:
        """Rebuild a tool from a stored payload."""
        data = payload.get("tool")
        if not isinstance(data, dict):
            raise StoreError("stored point is missing its 'tool' payload")
        return Tool.model_validate(data)

    def _to_record(self, point: Any) -> ToolRecord:
        """Rebuild a full record from a retrieved point."""
        payload = point.payload or {}
        vector = point.vector
        if not isinstance(vector, list):  # pragma: no cover - named-vector configs
            raise StoreError("expected a single unnamed vector per point")
        return ToolRecord(
            tool=self._to_tool(payload),
            text=str(payload.get("text", "")),
            vector=tuple(float(value) for value in vector),
        )

    def __repr__(self) -> str:
        """Show the collection and dimensionality."""
        return f"QdrantStore(collection={self._collection!r}, dim={self._dim})"


def build_filter(filters: Filters | None) -> models.Filter | None:
    """Translate ToolBroker filters into a Qdrant filter.

    Returns ``None`` when nothing is constrained, so Qdrant can skip filtering
    entirely rather than evaluating an empty clause.
    """
    if filters is None or filters.is_empty():
        return None

    must: list[models.Condition] = []
    must_not: list[models.Condition] = []

    if filters.namespaces is not None:
        must.append(
            models.FieldCondition(
                key="namespace", match=models.MatchAny(any=sorted(filters.namespaces))
            )
        )

    if filters.tool_ids is not None:
        must.append(
            models.FieldCondition(
                key="tool_id", match=models.MatchAny(any=sorted(filters.tool_ids))
            )
        )

    if filters.exclude_tool_ids:
        must_not.append(
            models.FieldCondition(
                key="tool_id", match=models.MatchAny(any=sorted(filters.exclude_tool_ids))
            )
        )

    if filters.tags_any is not None:
        must.append(
            models.FieldCondition(key="tags", match=models.MatchAny(any=sorted(filters.tags_any)))
        )

    if filters.tags_all is not None:
        # No "match all" primitive on a keyword array, so each required tag
        # becomes its own condition.
        must.extend(
            models.FieldCondition(key="tags", match=models.MatchValue(value=tag))
            for tag in sorted(filters.tags_all)
        )

    if filters.exclude_tags:
        must_not.append(
            models.FieldCondition(
                key="tags", match=models.MatchAny(any=sorted(filters.exclude_tags))
            )
        )

    if filters.max_risk is not None:
        # Risk is stored as an ordinal so the ceiling is a range query rather
        # than an enumeration of every permitted tier.
        must.append(
            models.FieldCondition(
                key="risk_level",
                range=models.Range(lte=float(_RISK_ORDER.index(filters.max_risk))),
            )
        )

    if not must and not must_not:
        return None
    return models.Filter(must=must or None, must_not=must_not or None)
