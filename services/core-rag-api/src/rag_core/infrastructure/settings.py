"""Typed, fail-fast environment configuration."""

import os
from dataclasses import dataclass

from rag_core.domain.models import DistanceMetric


PINNED_MODEL_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"


def _positive_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _positive_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} must be a number") from error
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, str(default).lower()).lower()
    if raw not in {"true", "false"}:
        raise ValueError(f"{name} must be true or false")
    return raw == "true"


@dataclass(frozen=True, slots=True)
class Settings:
    rag_port: int = 8000
    embedding_base_url: str = "http://embedding-api:8001"
    embedding_timeout_seconds: float = 10.0
    readiness_dependency_timeout_seconds: float = 1.0
    readiness_total_timeout_seconds: float = 3.0
    vector_db_url: str = "http://qdrant:6333"
    vector_db_api_key: str | None = None
    default_embedding_model: str = "all-MiniLM-L6-v2"
    embedding_revision: str = PINNED_MODEL_REVISION
    embedding_dimension: int = 384
    normalize_embeddings: bool = True
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    default_chunk_size: int = 1_000
    default_chunk_overlap: int = 200
    max_document_bytes: int = 262_144
    max_chunks_per_document: int = 512
    max_embedding_batch_items: int = 64
    max_search_top_k: int = 100
    max_query_bytes: int = 16_384
    max_metadata_fields: int = 32
    max_metadata_value_bytes: int = 1_024
    log_level: str = "INFO"
    otel_service_name: str = "rag-core-api"
    otel_exporter_otlp_endpoint: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        defaults = cls()
        metric_raw = os.getenv("DISTANCE_METRIC", defaults.distance_metric.value).lower()
        try:
            metric = DistanceMetric(metric_raw)
        except ValueError as error:
            raise ValueError("DISTANCE_METRIC must be cosine, dot, or euclid") from error
        settings = cls(
            rag_port=_positive_int("RAG_PORT", defaults.rag_port),
            embedding_base_url=os.getenv("EMBEDDING_BASE_URL", defaults.embedding_base_url),
            embedding_timeout_seconds=_positive_float(
                "EMBEDDING_TIMEOUT_SECONDS",
                float(os.getenv("EMBEDDING_TIMEOUT_MS", defaults.embedding_timeout_seconds * 1_000)) / 1_000,
            ),
            readiness_dependency_timeout_seconds=_positive_float(
                "READINESS_DEPENDENCY_TIMEOUT_SECONDS",
                defaults.readiness_dependency_timeout_seconds,
            ),
            readiness_total_timeout_seconds=_positive_float(
                "READINESS_TOTAL_TIMEOUT_SECONDS",
                defaults.readiness_total_timeout_seconds,
            ),
            vector_db_url=os.getenv("VECTOR_DB_URL", defaults.vector_db_url),
            vector_db_api_key=os.getenv("VECTOR_DB_API_KEY") or None,
            default_embedding_model=os.getenv("DEFAULT_EMBEDDING_MODEL", defaults.default_embedding_model),
            embedding_revision=os.getenv("EMBEDDING_MODEL_REVISION", defaults.embedding_revision),
            embedding_dimension=_positive_int("EMBEDDING_DIMENSION", defaults.embedding_dimension),
            normalize_embeddings=_bool("NORMALIZE_EMBEDDINGS", defaults.normalize_embeddings),
            distance_metric=metric,
            default_chunk_size=_positive_int("DEFAULT_CHUNK_SIZE", defaults.default_chunk_size),
            default_chunk_overlap=int(os.getenv("DEFAULT_CHUNK_OVERLAP", defaults.default_chunk_overlap)),
            max_document_bytes=_positive_int("MAX_DOCUMENT_BYTES", defaults.max_document_bytes),
            max_chunks_per_document=_positive_int("MAX_CHUNKS_PER_DOCUMENT", defaults.max_chunks_per_document),
            max_embedding_batch_items=_positive_int("MAX_EMBEDDING_BATCH_ITEMS", defaults.max_embedding_batch_items),
            max_search_top_k=_positive_int("MAX_SEARCH_TOP_K", defaults.max_search_top_k),
            max_query_bytes=_positive_int("MAX_QUERY_BYTES", defaults.max_query_bytes),
            max_metadata_fields=_positive_int("MAX_METADATA_FIELDS", defaults.max_metadata_fields),
            max_metadata_value_bytes=_positive_int("MAX_METADATA_VALUE_BYTES", defaults.max_metadata_value_bytes),
            log_level=os.getenv("LOG_LEVEL", defaults.log_level).upper(),
            otel_service_name=os.getenv("OTEL_SERVICE_NAME", defaults.otel_service_name),
            otel_exporter_otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or None,
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.embedding_base_url.startswith(("http://", "https://")):
            raise ValueError("EMBEDDING_BASE_URL must be an HTTP(S) URL")
        if not self.vector_db_url.startswith(("http://", "https://", ":memory:")):
            raise ValueError("VECTOR_DB_URL must be an HTTP(S) URL or :memory:")
        if not self.default_embedding_model.strip() or not self.embedding_revision.strip():
            raise ValueError("embedding model and immutable revision are required")
        if self.readiness_total_timeout_seconds < self.readiness_dependency_timeout_seconds:
            raise ValueError("READINESS_TOTAL_TIMEOUT_SECONDS must cover one dependency timeout")
        if self.default_chunk_overlap < 0 or self.default_chunk_overlap >= self.default_chunk_size:
            raise ValueError("DEFAULT_CHUNK_OVERLAP must be >= 0 and smaller than DEFAULT_CHUNK_SIZE")
        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("LOG_LEVEL is invalid")
        if not self.otel_service_name.strip():
            raise ValueError("OTEL_SERVICE_NAME must not be blank")
        if self.otel_exporter_otlp_endpoint and not self.otel_exporter_otlp_endpoint.startswith(("http://", "https://")):
            raise ValueError("OTEL_EXPORTER_OTLP_ENDPOINT must be an HTTP(S) URL")
