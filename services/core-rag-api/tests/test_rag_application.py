import pytest

from rag_core.application.rag import RagService
from rag_core.domain.errors import EmbeddingSchemaMismatchError, InvalidRequestError
from rag_core.domain.models import FilterCondition, FilterOperator
from rag_core.domain.models import TraceContext
from rag_core.adapters.fakes import InMemoryVectorStore


TRACE = TraceContext("trace")


@pytest.mark.asyncio
async def test_ingest_is_deterministic_and_retry_safe(dependencies, settings) -> None:
    catalog, store, embedding = dependencies
    service = RagService(catalog=catalog, vector_store=store, embedding=embedding, settings=settings)
    first = await service.ingest(
        collection_id="handbook",
        document_id="leave-policy",
        text="Parental leave is sixteen weeks. " * 4,
        metadata={"department": "people", "year": 2026},
        trace_context=TRACE,
        deadline_seconds=1,
    )
    first_ids = set(store.points["rag-handbook-1"])
    second = await service.ingest(
        collection_id="handbook",
        document_id="leave-policy",
        text="Parental leave is sixteen weeks. " * 4,
        metadata={"department": "people", "year": 2026},
        trace_context=TRACE,
        deadline_seconds=1,
    )
    assert first == second
    assert set(store.points["rag-handbook-1"]) == first_ids
    assert len(first_ids) == first.chunks_indexed


@pytest.mark.asyncio
async def test_retrieve_is_one_product_operation_and_delete_removes_all(dependencies, settings) -> None:
    catalog, store, embedding = dependencies
    service = RagService(catalog=catalog, vector_store=store, embedding=embedding, settings=settings)
    await service.ingest(
        collection_id="handbook",
        document_id="leave-policy",
        text="Parental leave is sixteen weeks.",
        metadata={"department": "people", "year": 2026},
        trace_context=TRACE,
        deadline_seconds=1,
    )
    before_calls = len(embedding.calls)
    result = await service.retrieve(
        collection_id="handbook",
        query="parental leave",
        top_k=5,
        filter=FilterCondition("department", FilterOperator.EQ, "people"),
        minimum_score=0,
        trace_context=TRACE,
        deadline_seconds=1,
    )
    assert len(embedding.calls) == before_calls + 1
    assert result.results[0].document_id == "leave-policy"
    assert 0 <= result.results[0].score <= 1
    deleted = await service.delete(collection_id="handbook", document_id="leave-policy", deadline_seconds=1)
    assert deleted.chunks_deleted == 1
    assert not store.points["rag-handbook-1"]


@pytest.mark.asyncio
async def test_ingest_rejects_wrong_embedding_schema_before_write(dependencies, settings) -> None:
    catalog, store, embedding = dependencies
    original_embed = embedding.embed
    async def changed(*args, **kwargs):
        result = await original_embed(*args, **kwargs)
        return type(result)(result.model_id, "changed", result.dimension, result.normalized, result.vectors)
    embedding.embed = changed
    service = RagService(catalog=catalog, vector_store=store, embedding=embedding, settings=settings)
    with pytest.raises(EmbeddingSchemaMismatchError):
        await service.ingest(
            collection_id="handbook",
            document_id="doc",
            text="content",
            metadata={},
            trace_context=TRACE,
            deadline_seconds=1,
        )
    assert not store.points["rag-handbook-1"]


@pytest.mark.asyncio
async def test_retrieve_enforces_filter_and_top_k(dependencies, settings) -> None:
    catalog, store, embedding = dependencies
    service = RagService(catalog=catalog, vector_store=store, embedding=embedding, settings=settings)
    with pytest.raises(InvalidRequestError):
        await service.retrieve(
            collection_id="handbook",
            query="query",
            top_k=11,
            filter=None,
            minimum_score=0,
            trace_context=TRACE,
            deadline_seconds=1,
        )


@pytest.mark.asyncio
async def test_partial_cleanup_failure_replay_repairs_without_duplicates_or_stale_results(dependencies, settings) -> None:
    catalog, original_store, embedding = dependencies

    class FailCleanupOnceStore(InMemoryVectorStore):
        def __init__(self):
            super().__init__()
            self.fail_cleanup = False

        async def delete_older_document_versions(self, *args, **kwargs):
            if self.fail_cleanup:
                self.fail_cleanup = False
                raise RuntimeError("transient cleanup failure")
            return await super().delete_older_document_versions(*args, **kwargs)

    store = FailCleanupOnceStore()
    store.collections = dict(original_store.collections)
    store.points = {name: dict(points) for name, points in original_store.points.items()}
    service = RagService(catalog=catalog, vector_store=store, embedding=embedding, settings=settings)
    await service.ingest(
        collection_id="handbook",
        document_id="policy",
        text="Old complete policy.",
        metadata={},
        trace_context=TRACE,
        deadline_seconds=1,
    )
    store.fail_cleanup = True
    with pytest.raises(RuntimeError, match="cleanup"):
        await service.ingest(
            collection_id="handbook",
            document_id="policy",
            text="New complete policy with replacement details.",
            metadata={},
            trace_context=TRACE,
            deadline_seconds=1,
        )
    partial_count = len(store.points["rag-handbook-1"])
    repaired = await service.ingest(
        collection_id="handbook",
        document_id="policy",
        text="New complete policy with replacement details.",
        metadata={},
        trace_context=TRACE,
        deadline_seconds=1,
    )
    points = tuple(store.points["rag-handbook-1"].values())
    assert len(points) == repaired.chunks_indexed
    assert len(points) < partial_count
    assert {point.document_version for point in points} == {repaired.document_version}
    result = await service.retrieve(
        collection_id="handbook",
        query="replacement details",
        top_k=10,
        filter=None,
        minimum_score=0,
        trace_context=TRACE,
        deadline_seconds=1,
    )
    assert result.results
    assert {hit.document_version for hit in result.results} == {repaired.document_version}
