"""Logical collection lifecycle orchestration."""

import hashlib
import secrets
import time
from time import perf_counter

from rag_core.application.validation import validate_resource_id
from rag_core.domain.errors import (
    CollectionAlreadyExistsError,
    ConflictError,
    GenerationNotReadyError,
    NotFoundError,
    PreconditionFailedError,
    VectorStoreUnavailableError,
)
from rag_core.domain.models import (
    CatalogPage,
    CollectionContract,
    CollectionState,
    EmbeddingContract,
    MetadataField,
)
from rag_core.ports.vector_store import CollectionCatalog, VectorStore
from rag_core.infrastructure.instruments import (
    dependency_call,
    generation_duration,
    generation_operations,
    safe_attributes,
    tracer,
)


class CollectionService:
    def __init__(self, catalog: CollectionCatalog, vector_store: VectorStore):
        self._catalog = catalog
        self._store = vector_store

    async def create(
        self,
        *,
        collection_id: str,
        embedding: EmbeddingContract,
        index_schema_version: int,
        metadata_fields: tuple[MetadataField, ...],
        isolation_policy: str,
        deadline_seconds: float,
    ) -> CollectionContract:
        validate_resource_id(collection_id, "collection_id")
        if await self._catalog.list_generations(collection_id):
            raise CollectionAlreadyExistsError()
        return await self._provision(
            collection_id=collection_id,
            embedding=embedding,
            index_schema_version=index_schema_version,
            metadata_fields=metadata_fields,
            isolation_policy=isolation_policy,
            source_generation_id=None,
            activate=True,
            deadline_seconds=deadline_seconds,
        )

    async def provision_generation(
        self,
        *,
        collection_id: str,
        embedding: EmbeddingContract,
        index_schema_version: int,
        metadata_fields: tuple[MetadataField, ...],
        deadline_seconds: float,
    ) -> CollectionContract:
        current = await self._require_active(collection_id)
        return await self._provision(
            collection_id=collection_id,
            embedding=embedding,
            index_schema_version=index_schema_version,
            metadata_fields=metadata_fields,
            isolation_policy=current.isolation_policy,
            source_generation_id=current.generation_id,
            activate=False,
            deadline_seconds=deadline_seconds,
        )

    async def _provision(self, *, activate: bool, deadline_seconds: float, **values: object) -> CollectionContract:
        started = perf_counter()
        operation = "create" if activate else "provision"
        generation_operations.add(1, safe_attributes(phase="building", operation=operation))
        _validate_contract(values["embedding"], values["metadata_fields"], str(values["isolation_policy"]))
        generation_id = _generation_id()
        logical = str(values["collection_id"])
        suffix = hashlib.sha256(f"{logical}:{generation_id}".encode()).hexdigest()[:12]
        contract = CollectionContract(
            generation_id=generation_id,
            physical_name=f"rag-{logical}-{suffix}",
            state=CollectionState.BUILDING,
            **values,  # type: ignore[arg-type]
        )
        await self._catalog.add_generation(contract)
        try:
            with dependency_call("vector_store", "create_collection"):
                await self._store.create_collection(contract)
            with dependency_call("vector_store", "create_payload_indexes"):
                await self._store.create_payload_indexes(contract)
            with dependency_call("vector_store", "verify_collection"):
                verification = await self._store.verify_collection(contract, deadline_seconds=deadline_seconds)
            if not verification.ready:
                raise VectorStoreUnavailableError()
            ready = await self._catalog.update_state(logical, generation_id, CollectionState.BUILDING, CollectionState.READY)
            if not activate:
                generation_operations.add(1, safe_attributes(phase="ready", operation=operation))
                generation_duration.record((perf_counter() - started) * 1_000, safe_attributes(operation=operation))
                return ready
            with dependency_call("vector_store", "activate_alias"):
                await self._store.activate_alias(
                    logical_id=logical,
                    previous=None,
                    target=ready,
                    deadline_seconds=deadline_seconds,
                )
            await self._catalog.set_active(logical, generation_id, None)
            active = await self._catalog.get_generation(logical, generation_id)
            if active is None:
                raise VectorStoreUnavailableError()
            generation_operations.add(1, safe_attributes(phase="active", operation=operation))
            generation_duration.record((perf_counter() - started) * 1_000, safe_attributes(operation=operation))
            return active
        except Exception:
            current = await self._catalog.get_generation(logical, generation_id)
            if current is not None and current.state == CollectionState.BUILDING:
                await self._catalog.update_state(logical, generation_id, CollectionState.BUILDING, CollectionState.FAILED)
            generation_operations.add(1, safe_attributes(phase="failed", operation=operation))
            raise

    async def activate(self, collection_id: str, generation_id: str, expected_active_generation_id: str, deadline_seconds: float) -> CollectionContract:
        target = await self._catalog.get_generation(collection_id, generation_id)
        if target is None:
            raise NotFoundError("collection generation")
        if target.state != CollectionState.READY:
            raise GenerationNotReadyError()
        previous = await self._require_active(collection_id)
        if previous.generation_id != expected_active_generation_id:
            raise PreconditionFailedError("The expected active generation is stale")
        with dependency_call("vector_store", "verify_collection"):
            verification = await self._store.verify_collection(target, deadline_seconds=deadline_seconds)
        if not verification.ready:
            raise VectorStoreUnavailableError()
        with tracer.start_as_current_span("rag.generation.activate"):
            with dependency_call("vector_store", "activate_alias"):
                await self._store.activate_alias(
                    logical_id=collection_id,
                    previous=previous,
                    target=target,
                    deadline_seconds=deadline_seconds,
                )
        await self._catalog.set_active(collection_id, generation_id, expected_active_generation_id)
        active = await self._catalog.get_generation(collection_id, generation_id)
        if active is None:
            raise VectorStoreUnavailableError()
        return active

    async def retire(self, collection_id: str, expected_active_generation_id: str, deadline_seconds: float):
        active = await self._require_active(collection_id)
        if active.generation_id != expected_active_generation_id:
            raise PreconditionFailedError("The If-Match generation is stale")
        retained_until = await self._catalog.retire_logical(
            collection_id, expected_active_generation_id
        )
        with dependency_call("vector_store", "retire_alias"):
            await self._store.retire_alias(collection_id, deadline_seconds=deadline_seconds)
        return retained_until

    async def list(self, *, limit: int, cursor: str | None) -> CatalogPage:
        return await self._catalog.list_logical(limit=limit, cursor=cursor)

    async def inspect(self, collection_id: str) -> tuple[CollectionContract, ...]:
        generations = await self._catalog.list_generations(collection_id)
        if not generations:
            raise NotFoundError("collection")
        return generations

    async def _require_active(self, collection_id: str) -> CollectionContract:
        contract = await self._catalog.get_active(collection_id)
        if contract is None:
            raise NotFoundError("collection")
        return contract


def _generation_id() -> str:
    alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
    value = (int(time.time() * 1_000) << 80) | secrets.randbits(80)
    characters = []
    for _ in range(26):
        characters.append(alphabet[value & 31])
        value >>= 5
    return "gen_" + "".join(reversed(characters))


def _validate_contract(embedding: object, metadata_fields: object, isolation_policy: str) -> None:
    if not isinstance(embedding, EmbeddingContract):
        raise ConflictError("Embedding contract is invalid")
    if embedding.distance_metric.value == "dot" and not embedding.normalized:
        raise ConflictError("Dot distance requires normalized embeddings")
    fields = tuple(metadata_fields)  # type: ignore[arg-type]
    if len(fields) > 32 or len({field.name for field in fields}) != len(fields):
        raise ConflictError("Metadata fields must be unique and limited to 32")
    if isolation_policy not in {"shared", "collection_per_tenant"}:
        raise ConflictError("Isolation policy is invalid")
