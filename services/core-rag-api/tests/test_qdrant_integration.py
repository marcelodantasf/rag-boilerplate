import os
from uuid import uuid4

import pytest

from rag_core.adapters.catalog import QdrantCollectionCatalog
from rag_core.adapters.fakes import FakeEmbeddingGateway
from rag_core.adapters.vector_store import QdrantVectorStore
from rag_core.application.collections import CollectionService
from rag_core.application.rag import RagService
from rag_core.domain.errors import ConflictError
from rag_core.domain.models import (
    CollectionState,
    DistanceMetric,
    EmbeddingContract,
    FilterCondition,
    FilterOperator,
    MetadataField,
    MetadataFieldType,
    TraceContext,
    VectorPoint,
)
from rag_core.infrastructure.settings import Settings


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_qdrant_create_ingest_filter_search_delete_and_catalog() -> None:
    base_url = os.getenv("QDRANT_TEST_URL")
    if not base_url:
        pytest.skip("QDRANT_TEST_URL is not configured")
    suffix = uuid4().hex[:12]
    collection_id = f"it-{suffix}"
    store = QdrantVectorStore(base_url)
    catalog = QdrantCollectionCatalog(base_url)
    embedding = FakeEmbeddingGateway()
    settings = Settings(embedding_dimension=3, default_chunk_size=1_000, default_chunk_overlap=200)
    collections = CollectionService(catalog, store)
    rag = RagService(catalog=catalog, vector_store=store, embedding=embedding, settings=settings)
    created = None
    generation = None
    try:
        created = await collections.create(
            collection_id=collection_id,
            embedding=EmbeddingContract(settings.default_embedding_model, settings.embedding_revision, 3, True, DistanceMetric.COSINE),
            index_schema_version=1,
            metadata_fields=(MetadataField("department", MetadataFieldType.KEYWORD),),
            isolation_policy="shared",
            deadline_seconds=5,
        )
        assert (await catalog.get_active(collection_id)) == created
        indexed = await rag.ingest(
            collection_id=collection_id,
            document_id="leave-policy",
            text="Parental leave lasts sixteen weeks.",
            metadata={"department": "people"},
            trace_context=TraceContext("integration-trace"),
            deadline_seconds=5,
        )
        assert indexed.chunks_indexed == 1
        result = await rag.retrieve(
            collection_id=collection_id,
            query="parental leave",
            top_k=3,
            filter=FilterCondition("department", FilterOperator.EQ, "people"),
            minimum_score=0,
            trace_context=TraceContext("integration-trace"),
            deadline_seconds=5,
        )
        assert result.results and result.results[0].document_id == "leave-policy"
        deleted = await rag.delete(collection_id=collection_id, document_id="leave-policy", deadline_seconds=5)
        assert deleted.chunks_deleted == 1
        generation = await collections.provision_generation(
            collection_id=collection_id,
            embedding=EmbeddingContract(settings.default_embedding_model, "generation-revision", 3, True, DistanceMetric.COSINE),
            index_schema_version=2,
            metadata_fields=(MetadataField("department", MetadataFieldType.KEYWORD),),
            deadline_seconds=5,
        )
        assert generation.state == CollectionState.READY
        assert (await catalog.get_active(collection_id)).generation_id == created.generation_id
        with pytest.raises(Exception) as stale:
            await collections.activate(collection_id, generation.generation_id, "gen_00000000000000000000000000", 5)
        assert stale.value.status_code == 412
        assert stale.value.code == "precondition_failed"
        activated = await collections.activate(collection_id, generation.generation_id, created.generation_id, 5)
        assert activated.state == CollectionState.ACTIVE
        predecessor = await catalog.get_generation(collection_id, created.generation_id)
        assert predecessor is not None and predecessor.state == CollectionState.RETIRED
        await catalog.update_state(collection_id, created.generation_id, CollectionState.RETIRED, CollectionState.READY)
        rollback = await collections.activate(collection_id, created.generation_id, generation.generation_id, 5)
        assert rollback.generation_id == created.generation_id
        retired_target = await catalog.get_generation(collection_id, generation.generation_id)
        assert retired_target is not None and retired_target.state == CollectionState.RETIRED
        await collections.retire(collection_id, created.generation_id, 5)
        retired_generations = await catalog.list_generations(collection_id)
        assert retired_generations
        assert all(
            item.state == CollectionState.RETIRED for item in retired_generations
        )
    finally:
        if generation is not None:
            await store.delete_collection(generation, deadline_seconds=5)
        if created is not None:
            await store.delete_collection(created, deadline_seconds=5)
        await embedding.close()
        await store.close()
        await catalog.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_qdrant_trusted_tenant_isolation_for_search_and_delete() -> None:
    base_url = os.getenv("QDRANT_TEST_URL")
    if not base_url:
        pytest.skip("QDRANT_TEST_URL is not configured")
    collection_id = f"tenant-it-{uuid4().hex[:10]}"
    tenant_a = QdrantVectorStore(base_url, trusted_tenant_id="tenant-a")
    tenant_b = QdrantVectorStore(base_url, trusted_tenant_id="tenant-b")
    catalog = QdrantCollectionCatalog(base_url)
    collections = CollectionService(catalog, tenant_a)
    contract = None
    embedding = EmbeddingContract("test-model", "test-revision", 3, True, DistanceMetric.COSINE)
    try:
        contract = await collections.create(
            collection_id=collection_id,
            embedding=embedding,
            index_schema_version=1,
            metadata_fields=(MetadataField("department", MetadataFieldType.KEYWORD),),
            isolation_policy="shared",
            deadline_seconds=5,
        )
        common = {
            "document_id": "shared-document",
            "document_version": "sha256:" + "a" * 64,
            "chunk_index": 0,
            "text": "Tenant-private content",
            "metadata": {"department": "people"},
            "vector": (1.0, 0.0, 0.0),
            "embedding_model": embedding.model_id,
            "embedding_revision": embedding.revision,
            "index_schema_version": 1,
        }
        await tenant_a.upsert(
            contract,
            (VectorPoint(point_id="tenant-a-point", chunk_id="tenant-a-chunk", **common),),
            deadline_seconds=5,
        )
        await tenant_b.upsert(
            contract,
            (VectorPoint(point_id="tenant-b-point", chunk_id="tenant-b-chunk", **common),),
            deadline_seconds=5,
        )
        trusted_a = FilterCondition("tenant_id", FilterOperator.EQ, "tenant-a")
        trusted_b = FilterCondition("tenant_id", FilterOperator.EQ, "tenant-b")
        matches_a = await tenant_a.search(contract, (1.0, 0.0, 0.0), top_k=10, filter=None, trusted_filter=trusted_a, deadline_seconds=5)
        matches_b = await tenant_b.search(contract, (1.0, 0.0, 0.0), top_k=10, filter=None, trusted_filter=trusted_b, deadline_seconds=5)
        assert {item.chunk_id for item in matches_a} == {"tenant-a-chunk"}
        assert {item.chunk_id for item in matches_b} == {"tenant-b-chunk"}
        assert await tenant_a.delete_by_document(contract, "shared-document", deadline_seconds=5) == 1
        remaining_b = await tenant_b.search(contract, (1.0, 0.0, 0.0), top_k=10, filter=None, trusted_filter=trusted_b, deadline_seconds=5)
        assert {item.chunk_id for item in remaining_b} == {"tenant-b-chunk"}
    finally:
        if contract is not None:
            await tenant_a.delete_collection(contract, deadline_seconds=5)
        await tenant_a.close()
        await tenant_b.close()
        await catalog.close()
