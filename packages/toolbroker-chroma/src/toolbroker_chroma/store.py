"""A :class:`~toolbroker.protocols.Store` backed by Chroma.

Chroma metadata holds only scalars, so a tool's tags cannot go in as a list.
Each tag becomes its own boolean key instead — ``tag_read: true``. That is not
merely a workaround: because Chroma's ``$ne`` also matches documents that lack
the key entirely, this encoding lets *every* ToolBroker filter, tag exclusion
included, run server-side **before** scoring. `k` therefore keeps meaning what
the caller expects, which is the contract the conformance suite checks.

The full tool is stored as JSON in a separate metadata field so records
round-trip exactly; the boolean tag keys exist only for filtering.
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

import chromadb

from toolbroker._vectors import Vector
from toolbroker.determinism import stable_sort
from toolbroker.errors import DimensionMismatchError, StoreError
from toolbroker.types import Filters, Hit, RiskTier, Tool, ToolRecord

_RISK_ORDER = [RiskTier.LOW, RiskTier.MEDIUM, RiskTier.HIGH, RiskTier.CRITICAL]

TAG_PREFIX = "tag_"
_UNSAFE = re.compile(r"[^a-zA-Z0-9_]+")

# Chroma rejects names outside this shape with an error most people meet at the
# worst possible moment, so it is validated up front.
_COLLECTION_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,510}[a-zA-Z0-9]$")

_SPACES = {"cosine", "l2", "ip"}

# Chroma rejects `{"$in": []}` outright, so "match nothing" needs a value that
# is legal but impossible. A tool id always looks like "namespace/name" and
# neither half may contain "/", so this can never be one.
_IMPOSSIBLE_TOOL_ID = "//toolbroker-never-matches"
_MATCH_NOTHING: dict[str, Any] = {"tool_id": {"$eq": _IMPOSSIBLE_TOOL_ID}}


def tag_key(tag: str) -> str:
    """Return the metadata key holding the flag for ``tag``."""
    return f"{TAG_PREFIX}{_UNSAFE.sub('_', tag)}"


class ChromaStore:
    """Stores tool records in a Chroma collection."""

    def __init__(
        self,
        dim: int,
        *,
        path: str | None = None,
        host: str | None = None,
        port: int = 8000,
        collection: str = "toolbroker",
        space: str = "cosine",
        client: Any | None = None,
        recreate: bool = False,
        **client_kwargs: Any,
    ) -> None:
        """Connect to Chroma and ensure the collection exists.

        Args:
            dim: Vector width. Must match the embedder.
            path: Directory for a persistent local store. Omit for ephemeral.
            host: Hostname of a Chroma server. Takes precedence over ``path``.
            port: Server port.
            collection: Collection name. Chroma requires 3-512 characters from
                ``[a-zA-Z0-9._-]``, starting and ending alphanumeric.
            space: Distance metric: ``cosine`` (default), ``l2``, or ``ip``.
            client: An already-configured Chroma client, so a host application
                can share one connection.
            recreate: Drop and rebuild the collection. Destroys data.
            **client_kwargs: Passed through to the Chroma client constructor.

        Raises:
            StoreError: If the collection name is invalid, or an existing
                collection was built with a different vector width.
        """
        if dim <= 0:
            raise ValueError("dim must be positive")
        if space not in _SPACES:
            raise ValueError(f"space must be one of {sorted(_SPACES)}")
        if not _COLLECTION_NAME.match(collection):
            raise StoreError(
                f"invalid Chroma collection name {collection!r}: it must be 3-512 "
                "characters from [a-zA-Z0-9._-], starting and ending alphanumeric"
            )

        self._dim = dim
        self._name = collection
        self._space = space

        if client is not None:
            self._client = client
        elif host is not None:
            self._client = chromadb.HttpClient(host=host, port=port, **client_kwargs)
        elif path is not None:
            self._client = chromadb.PersistentClient(path=path, **client_kwargs)
        else:
            self._client = chromadb.EphemeralClient(**client_kwargs)

        if recreate:
            self._drop()
        self._collection = self._open()
        self._verify_dimension()

    # -- lifecycle --------------------------------------------------------

    @property
    def client(self) -> Any:
        """The underlying Chroma client."""
        return self._client

    @property
    def collection(self) -> Any:
        """The underlying Chroma collection."""
        return self._collection

    @property
    def dim(self) -> int:
        """Vector width this store was built for."""
        return self._dim

    def _open(self) -> Any:
        return self._client.get_or_create_collection(
            name=self._name,
            metadata={"hnsw:space": self._space, "toolbroker_dim": self._dim},
        )

    def _drop(self) -> None:
        # Deleting something that does not exist is not an error here.
        with contextlib.suppress(Exception):
            self._client.delete_collection(self._name)

    def _verify_dimension(self) -> None:
        """Refuse a collection that was indexed by a different embedder.

        Querying it would return confident nonsense, which is worse than
        failing at construction.
        """
        metadata = getattr(self._collection, "metadata", None) or {}
        recorded = metadata.get("toolbroker_dim")
        if isinstance(recorded, int) and recorded != self._dim:
            raise StoreError(
                f"collection {self._name!r} holds {recorded}-dimensional vectors but this "
                f"store expects {self._dim}. It was indexed with a different embedder; "
                "use a different collection or pass recreate=True."
            )

    # -- writes -----------------------------------------------------------

    def upsert(self, records: Sequence[ToolRecord]) -> None:
        """Insert or replace records, keyed by tool id."""
        if not records:
            return
        ids: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict[str, Any]] = []
        documents: list[str] = []

        for record in records:
            if len(record.vector) != self._dim:
                raise DimensionMismatchError(self._dim, len(record.vector))
            ids.append(record.id)
            embeddings.append([float(value) for value in record.vector])
            metadatas.append(self._metadata(record.tool))
            # Chroma rejects an empty document, and an unembedded tool would
            # have no index text anyway.
            documents.append(record.text or record.id)

        self._collection.upsert(
            ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents
        )

    @staticmethod
    def _metadata(tool: Tool) -> dict[str, Any]:
        """Build the stored metadata: filterable scalars plus the full tool."""
        metadata: dict[str, Any] = {
            "tool_id": tool.id,
            "namespace": tool.namespace,
            "risk_level": _RISK_ORDER.index(tool.risk),
            "tool_json": tool.model_dump_json(),
        }
        for tag in tool.tags:
            metadata[tag_key(tag)] = True
        return metadata

    def delete(self, tool_ids: Sequence[str]) -> int:
        """Remove records by tool id, returning how many existed."""
        if not tool_ids:
            return 0
        existing = self._collection.get(ids=list(tool_ids), include=[])
        present = list(existing.get("ids") or [])
        if not present:
            return 0
        self._collection.delete(ids=present)
        return len(present)

    def clear(self) -> None:
        """Drop and recreate the collection."""
        self._drop()
        self._collection = self._open()

    # -- reads ------------------------------------------------------------

    def get(self, tool_id: str) -> ToolRecord | None:
        """Return one record by tool id."""
        found = self._collection.get(
            ids=[tool_id], include=["embeddings", "metadatas", "documents"]
        )
        ids = list(found.get("ids") or [])
        if not ids:
            return None
        return self._to_record(
            ids[0],
            _at(found, "metadatas", 0),
            _at(found, "documents", 0),
            _at(found, "embeddings", 0),
        )

    def all_records(self) -> Sequence[ToolRecord]:
        """Return every record, ordered by tool id."""
        found = self._collection.get(include=["embeddings", "metadatas", "documents"])
        records = [
            self._to_record(
                tool_id,
                _at(found, "metadatas", index),
                _at(found, "documents", index),
                _at(found, "embeddings", index),
            )
            for index, tool_id in enumerate(found.get("ids") or [])
        ]
        # Chroma does not promise a stable order; the contract does.
        return tuple(sorted(records, key=lambda record: record.id))

    def search(self, vector: Vector, k: int, filters: Filters | None = None) -> list[Hit]:
        """Return the ``k`` highest-scoring records that satisfy ``filters``."""
        if k <= 0:
            return []
        if len(vector) != self._dim:
            raise DimensionMismatchError(self._dim, len(vector))

        total = len(self)
        if total == 0:
            return []

        response = self._collection.query(
            query_embeddings=[[float(value) for value in vector]],
            # Chroma errors rather than clamping when n_results exceeds the
            # collection size.
            n_results=min(k, total),
            where=build_where(filters),
            include=["metadatas", "distances"],
        )

        ids = (response.get("ids") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]

        hits: list[Hit] = []
        for index, _tool_id in enumerate(ids):
            score = self._score(float(distances[index]))
            hits.append(
                Hit(
                    tool=self._to_tool(metadatas[index]),
                    score=score,
                    components={"vector": score},
                )
            )
        return stable_sort(hits)[:k]

    def _score(self, distance: float) -> float:
        """Convert a Chroma distance into a similarity, higher-is-better.

        Chroma reports cosine *distance*; ToolBroker ranks on similarity, and
        the two are the same measure with the sign flipped.
        """
        if self._space == "cosine":
            return 1.0 - distance
        if self._space == "ip":
            return -distance
        return -distance

    def __len__(self) -> int:
        """Number of stored records."""
        return int(self._collection.count())

    # -- conversion -------------------------------------------------------

    @staticmethod
    def _to_tool(metadata: Mapping[str, Any] | None) -> Tool:
        """Rebuild a tool from stored metadata."""
        raw = (metadata or {}).get("tool_json")
        if not isinstance(raw, str):
            raise StoreError("stored document is missing its 'tool_json' metadata")
        try:
            return Tool.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValueError) as exc:
            raise StoreError(f"stored tool metadata is not a valid tool: {exc}") from exc

    def _to_record(
        self,
        tool_id: str,
        metadata: Mapping[str, Any] | None,
        document: Any,
        embedding: Any,
    ) -> ToolRecord:
        """Rebuild a full record from Chroma's column-wise response."""
        del tool_id
        return ToolRecord(
            tool=self._to_tool(metadata),
            text=str(document or ""),
            vector=tuple(float(value) for value in (embedding if embedding is not None else ())),
        )

    def __repr__(self) -> str:
        """Show the collection and dimensionality."""
        return f"ChromaStore(collection={self._name!r}, dim={self._dim})"


def _at(response: Mapping[str, Any], key: str, index: int) -> Any:
    """Read one element from a Chroma column, tolerating absent columns.

    Chroma returns numpy arrays for embeddings, so truthiness checks are unsafe
    here — ``if column`` raises on a multi-element array.
    """
    column = response.get(key)
    if column is None:
        return None
    if index >= len(column):
        return None
    return column[index]


def build_where(filters: Filters | None) -> dict[str, Any] | None:
    """Translate ToolBroker filters into a Chroma ``where`` clause.

    Returns ``None`` when nothing is constrained. Every clause is expressible,
    so no filtering is deferred to the client — which is what keeps ``k``
    honest.
    """
    if filters is None or filters.is_empty():
        return None

    # An empty allow-set matches nothing, the same way Filters.matches does.
    # Letting the clause vanish would silently widen the query to everything,
    # which is the dangerous direction for something used as a guard rail.
    if (
        (filters.namespaces is not None and not filters.namespaces)
        or (filters.tool_ids is not None and not filters.tool_ids)
        or (filters.tags_any is not None and not filters.tags_any)
    ):
        return dict(_MATCH_NOTHING)

    clauses: list[dict[str, Any]] = []

    if filters.namespaces is not None:
        clauses.append({"namespace": {"$in": sorted(filters.namespaces)}})

    if filters.tool_ids is not None:
        clauses.append({"tool_id": {"$in": sorted(filters.tool_ids)}})

    if filters.exclude_tool_ids:
        clauses.append({"tool_id": {"$nin": sorted(filters.exclude_tool_ids)}})

    if filters.tags_any is not None:
        any_clauses: list[dict[str, Any]] = [
            {tag_key(tag): True} for tag in sorted(filters.tags_any)
        ]
        clauses.append(any_clauses[0] if len(any_clauses) == 1 else {"$or": any_clauses})

    if filters.tags_all is not None:
        clauses.extend({tag_key(tag): True} for tag in sorted(filters.tags_all))

    # `$ne True` also matches documents with no such key, which is exactly the
    # semantics exclusion needs.
    clauses.extend({tag_key(tag): {"$ne": True}} for tag in sorted(filters.exclude_tags))

    if filters.max_risk is not None:
        clauses.append({"risk_level": {"$lte": _RISK_ORDER.index(filters.max_risk)}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}
