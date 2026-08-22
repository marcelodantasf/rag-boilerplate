"""Immutable provider-neutral values shared by application ports."""

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
import json
import math
import re


JsonScalar = str | int | float | bool
Metadata = dict[str, JsonScalar]
RESERVED_METADATA_FIELDS = frozenset(
    {
        "tenant_id",
        "access_scope",
        "document_id",
        "document_version",
        "chunk_id",
        "chunk_index",
    }
)
_METADATA_KEY = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}")


class DistanceMetric(StrEnum):
    COSINE = "cosine"
    DOT = "dot"
    EUCLID = "euclid"


class CollectionState(StrEnum):
    BUILDING = "building"
    READY = "ready"
    ACTIVE = "active"
    RETIRED = "retired"
    FAILED = "failed"


class MetadataFieldType(StrEnum):
    KEYWORD = "keyword"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"


@dataclass(frozen=True, slots=True)
class MetadataField:
    name: str
    type: MetadataFieldType
    indexed: bool = True


class FilterOperator(StrEnum):
    EQ = "eq"
    IN = "in"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"


class FilterGroupOperator(StrEnum):
    ALL = "all"
    ANY = "any"
    NOT = "not"


@dataclass(frozen=True, slots=True)
class FilterCondition:
    field: str
    operator: FilterOperator
    value: JsonScalar | tuple[JsonScalar, ...]


@dataclass(frozen=True, slots=True)
class FilterGroup:
    operator: FilterGroupOperator
    clauses: tuple["FilterExpression", ...]


FilterExpression = FilterCondition | FilterGroup


@dataclass(frozen=True, slots=True)
class EmbeddingContract:
    model_id: str
    revision: str
    dimension: int
    normalized: bool
    distance_metric: DistanceMetric


@dataclass(frozen=True, slots=True)
class CollectionContract:
    collection_id: str
    generation_id: str
    physical_name: str
    embedding: EmbeddingContract
    index_schema_version: int = 1
    metadata_fields: tuple[MetadataField, ...] = ()
    isolation_policy: str = "shared"
    state: CollectionState = CollectionState.ACTIVE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    activated_at: datetime | None = None
    source_generation_id: str | None = None

    def with_state(self, state: CollectionState) -> "CollectionContract":
        return replace(self, state=state)


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    model_id: str
    revision: str
    dimension: int
    normalized: bool
    vectors: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class VectorPoint:
    point_id: str
    chunk_id: str
    document_id: str
    document_version: str
    chunk_index: int
    text: str
    metadata: Metadata
    vector: tuple[float, ...]
    embedding_model: str
    embedding_revision: str
    index_schema_version: int


@dataclass(frozen=True, slots=True)
class VectorMatch:
    chunk_id: str
    document_id: str
    document_version: str
    text: str
    metric_value: float
    metadata: Metadata


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    rank: int
    chunk_id: str
    document_id: str
    document_version: str
    text: str
    score: float
    metadata: Metadata


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    index: int
    text: str


@dataclass(frozen=True, slots=True)
class IngestResult:
    collection_id: str
    generation_id: str
    document_id: str
    document_version: str
    chunks_indexed: int
    status: str = "indexed"


@dataclass(frozen=True, slots=True)
class DeleteResult:
    collection_id: str
    generation_id: str
    document_id: str
    chunks_deleted: int
    status: str = "deleted"


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    collection_id: str
    generation_id: str
    query: str
    top_k: int
    results: tuple[RetrievalHit, ...]


@dataclass(frozen=True, slots=True)
class TraceContext:
    trace_id: str
    traceparent: str | None = None
    tracestate: str | None = None


@dataclass(frozen=True, slots=True)
class ReadinessResult:
    ready: bool
    code: str | None = None


@dataclass(frozen=True, slots=True)
class Verification:
    compatible: bool
    code: str | None = None

    @property
    def ready(self) -> bool:
        return self.compatible


@dataclass(frozen=True, slots=True)
class CatalogPage:
    items: tuple[CollectionContract, ...]
    next_cursor: str | None = None


def safe_metadata(value: dict[str, Any]) -> Metadata:
    """Copy metadata after enforcing the intentionally small public scalar grammar."""
    output: Metadata = {}
    if len(value) > 32:
        raise ValueError("metadata may contain at most 32 fields")
    for key, item in value.items():
        if not isinstance(key, str) or _METADATA_KEY.fullmatch(key) is None:
            raise ValueError("metadata keys have an invalid format")
        if key in RESERVED_METADATA_FIELDS or key.startswith("_"):
            raise ValueError("metadata key is reserved")
        if type(item) not in {str, int, float, bool}:
            raise ValueError("metadata values must be strings, numbers, or booleans")
        if isinstance(item, str) and len(item) > 512:
            raise ValueError("metadata string values may contain at most 512 characters")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("metadata numbers must be finite")
        output[key] = item
    if len(json.dumps(output, sort_keys=True, separators=(",", ":")).encode("utf-8")) > 16_384:
        raise ValueError("metadata canonical JSON may contain at most 16384 bytes")
    return output
