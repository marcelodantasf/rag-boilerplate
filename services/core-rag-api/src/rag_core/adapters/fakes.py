"""Deterministic in-memory adapters for application and contract tests."""

import math
import asyncio
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from rag_core.domain.errors import ConflictError, NotFoundError
from rag_core.domain.models import (
    CatalogPage,
    CollectionContract,
    CollectionState,
    EmbeddingContract,
    EmbeddingResult,
    FilterExpression,
    FilterGroup,
    FilterGroupOperator,
    FilterOperator,
    ReadinessResult,
    TraceContext,
    VectorMatch,
    VectorPoint,
    Verification,
)
from rag_core.ports.idempotency import IdempotencyResult, IdempotencyState


class FakeEmbeddingGateway:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.is_ready = True

    async def embed(self, texts: Sequence[str], *, contract: EmbeddingContract, deadline_seconds: float, trace_context: TraceContext) -> EmbeddingResult:
        self.calls.append(tuple(texts))
        vectors: list[tuple[float, ...]] = []
        for text in texts:
            values = [float(sum(ord(char) for char in text[index::contract.dimension]) % 101 + 1) for index in range(contract.dimension)]
            length = math.sqrt(sum(value * value for value in values))
            vectors.append(tuple(value / length for value in values) if contract.normalized else tuple(values))
        return EmbeddingResult(contract.model_id, contract.revision, contract.dimension, contract.normalized, tuple(vectors))

    async def ready(self, *, contract: EmbeddingContract, deadline_seconds: float, trace_context: TraceContext) -> ReadinessResult:
        return ReadinessResult(self.is_ready, None if self.is_ready else "embedding_unavailable")

    async def close(self) -> None:
        return None


class InMemoryCollectionCatalog:
    def __init__(self) -> None:
        self.contracts: list[CollectionContract] = []

    async def add_generation(self, contract: CollectionContract) -> None:
        existing = await self.get_generation(contract.collection_id, contract.generation_id)
        if existing == contract:
            return
        if existing is not None or any(item.physical_name == contract.physical_name for item in self.contracts):
            raise ConflictError("The collection generation already exists")
        self.contracts.append(contract)

    async def get_active(self, collection_id: str) -> CollectionContract | None:
        return next((item for item in self.contracts if item.collection_id == collection_id and item.state == CollectionState.ACTIVE), None)

    async def get_generation(self, collection_id: str, generation_id: str) -> CollectionContract | None:
        return next((item for item in self.contracts if item.collection_id == collection_id and item.generation_id == generation_id), None)

    async def list_logical(self, *, limit: int, cursor: str | None) -> CatalogPage:
        identifiers = sorted({item.collection_id for item in self.contracts})
        offset = int(cursor or 0)
        items: list[CollectionContract] = []
        for collection_id in identifiers[offset : offset + limit]:
            generations = await self.list_generations(collection_id)
            items.append(next((item for item in generations if item.state == CollectionState.ACTIVE), generations[-1]))
        next_offset = offset + len(items)
        return CatalogPage(tuple(items), str(next_offset) if next_offset < len(identifiers) else None)

    async def list_generations(self, collection_id: str) -> tuple[CollectionContract, ...]:
        return tuple(sorted((item for item in self.contracts if item.collection_id == collection_id), key=lambda item: item.created_at))

    async def update_state(self, collection_id: str, generation_id: str, expected_state: CollectionState, new_state: CollectionState, activated_at: datetime | None = None) -> CollectionContract:
        for index, contract in enumerate(self.contracts):
            if contract.collection_id == collection_id and contract.generation_id == generation_id:
                if contract.state != expected_state:
                    raise ConflictError("Collection generation state changed")
                updated = replace(contract, state=new_state, activated_at=activated_at or contract.activated_at)
                self.contracts[index] = updated
                return updated
        raise NotFoundError("collection generation")

    async def set_active(self, collection_id: str, target_generation_id: str, expected_active_generation_id: str | None) -> None:
        current = await self.get_active(collection_id)
        if (current.generation_id if current else None) != expected_active_generation_id:
            raise ConflictError("Active generation changed")
        for index, contract in enumerate(self.contracts):
            if contract.collection_id != collection_id:
                continue
            if contract.generation_id == target_generation_id:
                if contract.state != CollectionState.READY:
                    raise ConflictError("Target generation is not ready")
                self.contracts[index] = replace(contract, state=CollectionState.ACTIVE, activated_at=datetime.now(timezone.utc))
            elif contract.state == CollectionState.ACTIVE:
                self.contracts[index] = replace(contract, state=CollectionState.RETIRED)

    async def retire_logical(self, collection_id: str, expected_active_generation_id: str) -> datetime:
        current = await self.get_active(collection_id)
        if current is None or current.generation_id != expected_active_generation_id:
            raise ConflictError("Active generation changed")
        for index, contract in enumerate(self.contracts):
            if (
                contract.collection_id == collection_id
                and contract.state != CollectionState.FAILED
            ):
                self.contracts[index] = contract.with_state(CollectionState.RETIRED)
        return datetime.now(timezone.utc) + timedelta(days=7)

    async def ready(self, *, deadline_seconds: float) -> ReadinessResult:
        return ReadinessResult(True)

    async def close(self) -> None:
        return None


class InMemoryVectorStore:
    def __init__(self) -> None:
        self.collections: dict[str, CollectionContract] = {}
        self.points: dict[str, dict[str, VectorPoint]] = {}
        self.is_ready = True

    async def create_collection(self, contract: CollectionContract) -> None:
        existing = self.collections.get(contract.physical_name)
        if existing == contract:
            return
        if existing is not None:
            raise ConflictError("The physical collection already exists")
        self.collections[contract.physical_name] = contract
        self.points[contract.physical_name] = {}

    async def create_payload_indexes(self, contract: CollectionContract) -> None:
        return None

    async def verify_collection(self, contract: CollectionContract, *, deadline_seconds: float) -> Verification:
        existing = self.collections.get(contract.physical_name)
        compatible = existing is not None and (
            existing.collection_id,
            existing.generation_id,
            existing.embedding,
            existing.index_schema_version,
            existing.metadata_fields,
            existing.isolation_policy,
        ) == (
            contract.collection_id,
            contract.generation_id,
            contract.embedding,
            contract.index_schema_version,
            contract.metadata_fields,
            contract.isolation_policy,
        )
        return Verification(compatible)

    async def upsert(self, contract: CollectionContract, points: Sequence[VectorPoint], *, deadline_seconds: float) -> None:
        target = self.points.get(contract.physical_name)
        if target is None:
            raise NotFoundError("collection")
        for point in points:
            target[point.point_id] = point

    async def search(self, contract: CollectionContract, vector: Sequence[float], *, top_k: int, filter: FilterExpression | None, trusted_filter: FilterExpression, deadline_seconds: float) -> tuple[VectorMatch, ...]:
        candidates: list[VectorMatch] = []
        for point in self.points.get(contract.physical_name, {}).values():
            if filter is None or _matches(filter, point.metadata):
                metric = sum(left * right for left, right in zip(vector, point.vector, strict=True))
                candidates.append(VectorMatch(point.chunk_id, point.document_id, point.document_version, point.text, float(metric), dict(point.metadata)))
        return tuple(sorted(candidates, key=lambda item: item.metric_value, reverse=True)[:top_k])

    async def delete_by_document(self, contract: CollectionContract, document_id: str, *, deadline_seconds: float) -> int:
        return self._delete(contract, lambda point: point.document_id == document_id)

    async def delete_older_document_versions(self, contract: CollectionContract, document_id: str, keep_version: str, *, deadline_seconds: float) -> int:
        return self._delete(contract, lambda point: point.document_id == document_id and point.document_version != keep_version)

    def _delete(self, contract: CollectionContract, predicate) -> int:
        target = self.points.get(contract.physical_name)
        if target is None:
            raise NotFoundError("collection")
        keys = [key for key, point in target.items() if predicate(point)]
        for key in keys:
            del target[key]
        return len(keys)

    async def delete_collection(self, contract: CollectionContract, *, deadline_seconds: float) -> None:
        self.collections.pop(contract.physical_name, None)
        self.points.pop(contract.physical_name, None)

    async def activate_alias(self, logical_id: str, previous: CollectionContract | None, target: CollectionContract, *, deadline_seconds: float) -> None:
        return None

    async def retire_alias(self, logical_id: str, *, deadline_seconds: float) -> None:
        return None

    async def ready(self, *, deadline_seconds: float) -> ReadinessResult:
        return ReadinessResult(self.is_ready, None if self.is_ready else "vector_store_unavailable")

    async def close(self) -> None:
        return None


def _matches(expression: FilterExpression, metadata: dict[str, object]) -> bool:
    if isinstance(expression, FilterGroup):
        values = tuple(_matches(clause, metadata) for clause in expression.clauses)
        if expression.operator == FilterGroupOperator.ALL:
            return all(values)
        if expression.operator == FilterGroupOperator.ANY:
            return any(values)
        return not values[0]
    actual = metadata.get(expression.field)
    expected = expression.value
    if expression.operator == FilterOperator.EQ:
        return actual == expected
    if expression.operator == FilterOperator.IN:
        return actual in expected
    if actual is None or isinstance(expected, tuple):
        return False
    if expression.operator == FilterOperator.GT:
        return actual > expected  # type: ignore[operator]
    if expression.operator == FilterOperator.GTE:
        return actual >= expected  # type: ignore[operator]
    if expression.operator == FilterOperator.LT:
        return actual < expected  # type: ignore[operator]
    return actual <= expected  # type: ignore[operator]


class InMemoryIdempotencyStore:
    def __init__(self) -> None:
        self._records: dict[tuple[str, str], tuple[str, int | None, dict | None]] = {}
        self._lock = asyncio.Lock()

    async def begin(self, scope: str, key: str, request_hash: str, ttl: int) -> IdempotencyResult:
        async with self._lock:
            record = self._records.get((scope, key))
            if record is None:
                self._records[(scope, key)] = (request_hash, None, None)
                return IdempotencyResult(IdempotencyState.BEGUN)
            stored_hash, status, response = record
            if stored_hash != request_hash:
                return IdempotencyResult(IdempotencyState.CONFLICT)
            if status is not None and response is not None:
                return IdempotencyResult(IdempotencyState.REPLAY, status, dict(response))
            return IdempotencyResult(IdempotencyState.CONFLICT)

    async def complete(self, scope: str, key: str, status: int, response: dict) -> None:
        async with self._lock:
            request_hash, _, _ = self._records[(scope, key)]
            self._records[(scope, key)] = (request_hash, status, dict(response))

    async def abandon(self, scope: str, key: str) -> None:
        async with self._lock:
            self._records.pop((scope, key), None)

    async def close(self) -> None:
        self._records.clear()
