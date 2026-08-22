from dataclasses import replace

import pytest

from rag_core.adapters.fakes import InMemoryCollectionCatalog, InMemoryVectorStore
from rag_core.application.collections import CollectionService
from rag_core.domain.errors import CollectionAlreadyExistsError, ConflictError
from rag_core.domain.models import CollectionState, DistanceMetric, EmbeddingContract, MetadataField, MetadataFieldType


def values():
    return {
        "collection_id": "handbook",
        "embedding": EmbeddingContract("model", "revision-one", 3, True, DistanceMetric.COSINE),
        "index_schema_version": 1,
        "metadata_fields": (MetadataField("department", MetadataFieldType.KEYWORD),),
        "isolation_policy": "shared",
        "deadline_seconds": 1,
    }


@pytest.mark.asyncio
async def test_create_provisions_and_activates_collection() -> None:
    catalog = InMemoryCollectionCatalog()
    store = InMemoryVectorStore()
    service = CollectionService(catalog, store)
    created = await service.create(**values())
    assert created.state == CollectionState.ACTIVE
    assert await catalog.get_active("handbook") == created
    assert (await store.verify_collection(created, deadline_seconds=1)).ready
    with pytest.raises(CollectionAlreadyExistsError):
        await service.create(**values())


@pytest.mark.asyncio
async def test_generation_requires_explicit_activation_and_retains_rollback_generation() -> None:
    catalog = InMemoryCollectionCatalog()
    store = InMemoryVectorStore()
    service = CollectionService(catalog, store)
    first = await service.create(**values())
    generation_values = values()
    generation_values.pop("isolation_policy")
    generation = await service.provision_generation(**{**generation_values, "embedding": EmbeddingContract("model", "revision-two", 3, True, DistanceMetric.COSINE), "index_schema_version": 2})
    assert generation.state == CollectionState.READY
    assert await catalog.get_active("handbook") == first
    active = await service.activate("handbook", generation.generation_id, first.generation_id, 1)
    assert active.state == CollectionState.ACTIVE
    prior = await catalog.get_generation("handbook", first.generation_id)
    assert prior is not None and prior.state == CollectionState.RETIRED
    assert len(await service.inspect("handbook")) == 2


@pytest.mark.asyncio
async def test_retirement_retires_active_and_ready_generations() -> None:
    catalog = InMemoryCollectionCatalog()
    store = InMemoryVectorStore()
    service = CollectionService(catalog, store)
    active = await service.create(**values())
    generation_values = values()
    generation_values.pop("isolation_policy")
    ready = await service.provision_generation(
        **{
            **generation_values,
            "embedding": EmbeddingContract(
                "model", "revision-two", 3, True, DistanceMetric.COSINE
            ),
            "index_schema_version": 2,
        }
    )
    building = replace(
        ready,
        generation_id="gen_building",
        physical_name="rag-handbook-building",
        state=CollectionState.BUILDING,
    )
    failed = replace(
        ready,
        generation_id="gen_failed",
        physical_name="rag-handbook-failed",
        state=CollectionState.FAILED,
    )
    await catalog.add_generation(building)
    await catalog.add_generation(failed)

    await service.retire("handbook", active.generation_id, 1)

    generations = await service.inspect("handbook")
    assert {item.generation_id for item in generations} == {
        active.generation_id,
        ready.generation_id,
        building.generation_id,
        failed.generation_id,
    }
    states = {item.generation_id: item.state for item in generations}
    assert states[active.generation_id] is CollectionState.RETIRED
    assert states[ready.generation_id] is CollectionState.RETIRED
    assert states[building.generation_id] is CollectionState.RETIRED
    assert states[failed.generation_id] is CollectionState.FAILED


@pytest.mark.asyncio
async def test_generation_rejects_stale_activation_and_supports_revalidated_rollback() -> None:
    catalog = InMemoryCollectionCatalog()
    store = InMemoryVectorStore()
    service = CollectionService(catalog, store)
    first = await service.create(**values())
    generation_values = values()
    generation_values.pop("isolation_policy")
    target = await service.provision_generation(
        **{
            **generation_values,
            "embedding": EmbeddingContract("model", "revision-two", 3, True, DistanceMetric.COSINE),
            "index_schema_version": 2,
        }
    )
    with pytest.raises(Exception) as stale:
        await service.activate("handbook", target.generation_id, "gen_00000000000000000000000000", 1)
    assert stale.value.status_code == 412
    assert stale.value.code == "precondition_failed"
    await service.activate("handbook", target.generation_id, first.generation_id, 1)
    await catalog.update_state("handbook", first.generation_id, CollectionState.RETIRED, CollectionState.READY)
    rolled_back = await service.activate("handbook", first.generation_id, target.generation_id, 1)
    assert rolled_back.generation_id == first.generation_id
    assert rolled_back.state == CollectionState.ACTIVE
    retired_target = await catalog.get_generation("handbook", target.generation_id)
    assert retired_target is not None and retired_target.state == CollectionState.RETIRED
