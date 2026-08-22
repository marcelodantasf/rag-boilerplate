from datetime import datetime, timezone

import pytest

from rag_core.adapters.fakes import FakeEmbeddingGateway, InMemoryCollectionCatalog, InMemoryVectorStore
from rag_core.domain.models import (
    CollectionContract,
    CollectionState,
    DistanceMetric,
    EmbeddingContract,
    MetadataField,
    MetadataFieldType,
)
from rag_core.infrastructure.settings import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(
        embedding_dimension=3,
        default_chunk_size=40,
        default_chunk_overlap=5,
        max_document_bytes=1_000,
        max_chunks_per_document=20,
        max_embedding_batch_items=2,
        max_search_top_k=10,
    )


@pytest.fixture
def contract(settings: Settings) -> CollectionContract:
    return CollectionContract(
        collection_id="handbook",
        generation_id="gen_1",
        physical_name="rag-handbook-1",
        embedding=EmbeddingContract(
            settings.default_embedding_model,
            settings.embedding_revision,
            settings.embedding_dimension,
            settings.normalize_embeddings,
            DistanceMetric.COSINE,
        ),
        metadata_fields=(
            MetadataField("department", MetadataFieldType.KEYWORD),
            MetadataField("year", MetadataFieldType.INTEGER),
            MetadataField("confidence", MetadataFieldType.FLOAT),
            MetadataField("published", MetadataFieldType.BOOLEAN),
        ),
        state=CollectionState.ACTIVE,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


@pytest.fixture
async def dependencies(contract: CollectionContract):
    catalog = InMemoryCollectionCatalog()
    store = InMemoryVectorStore()
    embedding = FakeEmbeddingGateway()
    await catalog.add_generation(contract)
    await store.create_collection(contract)
    await store.create_payload_indexes(contract)
    return catalog, store, embedding
