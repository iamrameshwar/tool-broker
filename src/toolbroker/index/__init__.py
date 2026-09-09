"""Enrichment and embedding: turning tool definitions into searchable records."""

from .enrich import EnrichmentConfig, build_index_text, describe_schema
from .indexer import Indexer

__all__ = ["EnrichmentConfig", "Indexer", "build_index_text", "describe_schema"]
