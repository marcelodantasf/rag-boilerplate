"""Product-level ingestion, deletion, and top-k retrieval use cases."""

from collections.abc import Iterable
from time import perf_counter
from typing import TypeVar

from rag_core.application.validation import ensure_compatible, validate_filter, validate_metadata, validate_resource_id
from rag_core.domain.chunking import chunk_text, content_version, normalize_text, stable_point_id
from rag_core.domain.errors import InvalidRequestError, LimitExceededError, NotFoundError, VectorStoreUnavailableError
from rag_core.domain.models import (
    CollectionContract,
    DeleteResult,
    DistanceMetric,
    FilterCondition,
    FilterExpression,
    FilterOperator,
    IngestResult,
    Metadata,
    RetrievalHit,
    RetrievalResult,
    TraceContext,
    VectorMatch,
    VectorPoint,
)
from rag_core.infrastructure.settings import Settings
from rag_core.infrastructure.instruments import (
    chunks_indexed,
    contract_mismatches,
    dependency_call,
    documents_indexed,
    embedding_batch_size,
    ingestion_duration,
    retrieval_duration,
    retrieval_no_match,
    retrieval_results,
    retrieval_top_k,
    safe_attributes,
    tracer,
)
from rag_core.ports.embedding import EmbeddingGateway
from rag_core.ports.vector_store import CollectionCatalog, VectorStore


class RagService:
    def __init__(self, *, catalog: CollectionCatalog, vector_store: VectorStore, embedding: EmbeddingGateway, settings: Settings):
        self._catalog = catalog
        self._store = vector_store
        self._embedding = embedding
        self._settings = settings

    async def ingest(
        self,
        *,
        collection_id: str,
        text: str,
        document_id: str,
        metadata: Metadata,
        trace_context: TraceContext,
        deadline_seconds: float,
    ) -> IngestResult:
        started = perf_counter()
        contract = await self._active_verified(collection_id, deadline_seconds)
        normalized = normalize_text(text)
        if not normalized:
            raise InvalidRequestError("Document content must not be blank", field="content")
        if len(normalized.encode("utf-8")) > self._settings.max_document_bytes:
            raise LimitExceededError(max_document_bytes=self._settings.max_document_bytes)
        validate_resource_id(document_id, "document_id")
        validate_metadata(metadata, contract)
        version = content_version(normalized)
        try:
            with tracer.start_as_current_span("rag.chunking"):
                chunks = chunk_text(
                    normalized,
                    collection_id=collection_id,
                    generation_id=contract.generation_id,
                    document_id=document_id,
                    version=version,
                    chunk_size=self._settings.default_chunk_size,
                    overlap=self._settings.default_chunk_overlap,
                )
        except ValueError as error:
            raise InvalidRequestError(str(error)) from error
        if len(chunks) > self._settings.max_chunks_per_document:
            raise LimitExceededError(max_chunks_per_document=self._settings.max_chunks_per_document)
        vectors: list[tuple[float, ...]] = []
        for batch in _batches(chunks, self._settings.max_embedding_batch_items):
            embedding_batch_size.record(len(batch))
            with dependency_call("embedding", "embed"):
                result = await self._embedding.embed(
                    [chunk.text for chunk in batch],
                    contract=contract.embedding,
                    deadline_seconds=deadline_seconds,
                    trace_context=trace_context,
                )
            try:
                ensure_compatible(contract, result)
            except Exception:
                contract_mismatches.add(
                    1, safe_attributes(operation="ingest", error_code="embedding_schema_mismatch")
                )
                raise
            vectors.extend(result.vectors)
        points = tuple(
            VectorPoint(
                point_id=stable_point_id(chunk.chunk_id),
                chunk_id=chunk.chunk_id,
                document_id=document_id,
                document_version=version,
                chunk_index=chunk.index,
                text=chunk.text,
                metadata=dict(metadata),
                vector=vector,
                embedding_model=contract.embedding.model_id,
                embedding_revision=contract.embedding.revision,
                index_schema_version=contract.index_schema_version,
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        )
        with dependency_call("vector_store", "upsert"):
            await self._store.upsert(contract, points, deadline_seconds=deadline_seconds)
        with dependency_call("vector_store", "delete_older_versions"):
            await self._store.delete_older_document_versions(
                contract,
                document_id,
                version,
                deadline_seconds=deadline_seconds,
            )
        documents_indexed.add(1)
        chunks_indexed.add(len(points))
        ingestion_duration.record((perf_counter() - started) * 1_000)
        return IngestResult(collection_id, contract.generation_id, document_id, version, len(points))

    async def delete(self, *, collection_id: str, document_id: str, deadline_seconds: float) -> DeleteResult:
        contract = await self._active_verified(collection_id, deadline_seconds)
        validate_resource_id(document_id, "document_id")
        with dependency_call("vector_store", "delete_document"):
            deleted = await self._store.delete_by_document(contract, document_id, deadline_seconds=deadline_seconds)
        return DeleteResult(collection_id, contract.generation_id, document_id, deleted)

    async def retrieve(
        self,
        *,
        collection_id: str,
        query: str,
        top_k: int,
        filter: FilterExpression | None,
        minimum_score: float,
        trace_context: TraceContext,
        deadline_seconds: float,
    ) -> RetrievalResult:
        started = perf_counter()
        contract = await self._active_verified(collection_id, deadline_seconds)
        normalized = normalize_text(query)
        if not normalized:
            raise InvalidRequestError("Query must not be blank", field="query")
        if len(normalized.encode("utf-8")) > self._settings.max_query_bytes:
            raise LimitExceededError(max_query_bytes=self._settings.max_query_bytes)
        if not 1 <= top_k <= self._settings.max_search_top_k:
            raise InvalidRequestError("top_k is outside the supported range", max=self._settings.max_search_top_k)
        if not 0 <= minimum_score <= 1:
            raise InvalidRequestError("minimum_score must be between 0 and 1")
        validate_filter(filter, contract)
        with dependency_call("embedding", "embed_query"):
            result = await self._embedding.embed(
                (normalized,),
                contract=contract.embedding,
                deadline_seconds=deadline_seconds,
                trace_context=trace_context,
            )
        try:
            ensure_compatible(contract, result)
        except Exception:
            contract_mismatches.add(
                1, safe_attributes(operation="retrieve", error_code="embedding_schema_mismatch")
            )
            raise
        with dependency_call("vector_store", "search"):
            matches = await self._store.search(
                contract,
                result.vectors[0],
                top_k=top_k,
                filter=filter,
                trusted_filter=FilterCondition("tenant_id", FilterOperator.EQ, "local"),
                deadline_seconds=deadline_seconds,
            )
        scored = [(_public_score(item, contract.embedding.distance_metric), item) for item in matches]
        scored.sort(key=lambda pair: (-pair[0], pair[1].chunk_id))
        hits = tuple(
            RetrievalHit(rank, item.chunk_id, item.document_id, item.document_version, item.text, score, dict(item.metadata))
            for rank, (score, item) in enumerate((pair for pair in scored if pair[0] >= minimum_score), start=1)
        )[:top_k]
        retrieval_top_k.record(top_k)
        retrieval_results.record(len(hits))
        if not hits:
            retrieval_no_match.add(1)
        retrieval_duration.record((perf_counter() - started) * 1_000)
        return RetrievalResult(collection_id, contract.generation_id, normalized, top_k, hits)

    async def _active_verified(self, collection_id: str, deadline_seconds: float) -> CollectionContract:
        validate_resource_id(collection_id, "collection_id")
        contract = await self._catalog.get_active(collection_id)
        if contract is None:
            raise NotFoundError("collection")
        with dependency_call("vector_store", "verify_collection"):
            verification = await self._store.verify_collection(contract, deadline_seconds=deadline_seconds)
        if not verification.ready:
            raise VectorStoreUnavailableError()
        return contract


T = TypeVar("T")


def _batches(values: tuple[T, ...], size: int) -> Iterable[tuple[T, ...]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def _public_score(match: VectorMatch, metric: DistanceMetric) -> float:
    if metric == DistanceMetric.EUCLID:
        score = 1.0 / (1.0 + max(match.metric_value, 0.0))
    else:
        score = (max(-1.0, min(1.0, match.metric_value)) + 1.0) / 2.0
    return round(max(0.0, min(1.0, score)), 6)
