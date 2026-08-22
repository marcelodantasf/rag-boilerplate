"""Durable logical collection catalog persisted in a private Qdrant collection."""

from __future__ import annotations

import asyncio
import base64
import uuid
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any

from rag_core.domain.errors import (
    ConflictError,
    LimitExceededError,
    NotFoundError,
    VectorStoreUnavailableError,
)
from rag_core.domain.models import (
    CollectionContract,
    CollectionState,
    CatalogPage,
    DistanceMetric,
    EmbeddingContract,
    MetadataField,
    MetadataFieldType,
    ReadinessResult,
)

from rag_core.adapters.vector_store._errors import (
    QdrantAdapterError,
    QdrantContractError,
    QdrantResourceConflictError,
    QdrantTimeoutError,
)
from rag_core.adapters.vector_store._http import QdrantHttpClient

_CATALOG_COLLECTION = "__rag_core_catalog_v1"
_CATALOG_NAMESPACE = uuid.UUID("148fa566-774e-5d2a-873e-8bfde004d6bf")
_PAGE_SIZE = 256
_MAX_CATALOG_RESULTS = 10_000
_CATALOG_INDEXES = {
    "collection_id": "keyword",
    "generation_id": "keyword",
    "state": "keyword",
}


class QdrantCollectionCatalog:
    """Persist immutable generation contracts separately from chunk points."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 3.0,
        retention_seconds: float = 7 * 24 * 60 * 60,
        client: QdrantHttpClient | None = None,
    ) -> None:
        self._client = client or QdrantHttpClient(
            base_url, api_key=api_key, timeout_seconds=timeout_seconds
        )
        self._owns_client = client is None
        if retention_seconds <= 0:
            raise ValueError("retention_seconds must be positive")
        self._retention = timedelta(seconds=retention_seconds)
        self._initialized = False
        self._initialize_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()

    async def add_generation(self, contract: CollectionContract) -> None:
        await self._ensure_catalog()
        existing = await self.get_generation(
            contract.collection_id, contract.generation_id
        )
        if existing is not None:
            if existing == contract:
                return
            raise ConflictError("The collection generation already exists")
        try:
            await self._client.upsert_points(
                _CATALOG_COLLECTION,
                [
                    {
                        "id": _catalog_point_id(
                            contract.collection_id, contract.generation_id
                        ),
                        "vector": [0.0],
                        "payload": _serialize_contract(contract),
                    }
                ],
            )
        except QdrantTimeoutError as exc:
            raise VectorStoreUnavailableError(timeout=True) from exc
        except QdrantAdapterError as exc:
            raise VectorStoreUnavailableError() from exc

    async def get_active(self, collection_id: str) -> CollectionContract | None:
        await self._ensure_catalog()
        records = await self._scroll(
            {
                "must": [
                    {"key": "collection_id", "match": {"value": collection_id}},
                    {"key": "state", "match": {"value": CollectionState.ACTIVE.value}},
                ]
            },
            maximum=2,
        )
        if len(records) > 1:
            raise ConflictError("The logical collection has multiple active generations")
        return records[0] if records else None

    async def get_generation(
        self, collection_id: str, generation_id: str
    ) -> CollectionContract | None:
        await self._ensure_catalog()
        point_id = _catalog_point_id(collection_id, generation_id)
        try:
            points = await self._client.retrieve_points(
                _CATALOG_COLLECTION, [point_id]
            )
            if not points:
                return None
            if len(points) != 1:
                raise QdrantContractError("Catalog identity is not unique")
            contract = _deserialize_point(points[0])
            if (
                contract.collection_id != collection_id
                or contract.generation_id != generation_id
            ):
                raise QdrantContractError("Catalog identity does not match its key")
            return contract
        except QdrantTimeoutError as exc:
            raise VectorStoreUnavailableError(timeout=True) from exc
        except QdrantAdapterError as exc:
            raise VectorStoreUnavailableError() from exc

    async def list_logical(
        self, *, limit: int, cursor: str | None
    ) -> CatalogPage:
        if limit <= 0:
            raise ValueError("limit must be positive")
        await self._ensure_catalog()
        records = await self._scroll(
            {
                "must": [
                    {"key": "state", "match": {"value": CollectionState.ACTIVE.value}}
                ]
            }
        )
        ordered = sorted(records, key=lambda item: item.collection_id)
        after = _decode_cursor(cursor) if cursor else None
        if after is not None:
            ordered = [item for item in ordered if item.collection_id > after]
        page = ordered[:limit]
        next_cursor = None
        if len(ordered) > limit and page:
            next_cursor = _encode_cursor(page[-1].collection_id)
        return CatalogPage(tuple(page), next_cursor)

    async def list_generations(
        self, collection_id: str
    ) -> tuple[CollectionContract, ...]:
        await self._ensure_catalog()
        records = await self._scroll(
            {
                "must": [
                    {"key": "collection_id", "match": {"value": collection_id}}
                ]
            }
        )
        return tuple(
            sorted(records, key=lambda item: (item.created_at, item.generation_id))
        )

    async def update_state(
        self,
        collection_id: str,
        generation_id: str,
        expected_state: CollectionState,
        new_state: CollectionState,
        activated_at: object | None = None,
    ) -> CollectionContract:
        if activated_at is not None and not isinstance(activated_at, datetime):
            raise ValueError("activated_at must be a datetime")
        async with self._state_lock:
            current = await self.get_generation(collection_id, generation_id)
            if current is None:
                raise NotFoundError("collection generation")
            if current.state is not expected_state:
                raise ConflictError("The collection generation state changed")
            updated = replace(
                current,
                state=new_state,
                activated_at=activated_at or current.activated_at,
            )
            try:
                await self._client.upsert_points(
                    _CATALOG_COLLECTION,
                    [{
                        "id": _catalog_point_id(collection_id, generation_id),
                        "vector": [0.0],
                        "payload": _serialize_contract(updated),
                    }],
                )
                return updated
            except QdrantTimeoutError as exc:
                raise VectorStoreUnavailableError(timeout=True) from exc
            except QdrantAdapterError as exc:
                raise VectorStoreUnavailableError() from exc

    async def set_active(
        self,
        collection_id: str,
        target_generation_id: str,
        expected_active_generation_id: str | None,
    ) -> None:
        async with self._state_lock:
            active = await self.get_active(collection_id)
            actual_active_id = active.generation_id if active else None
            if actual_active_id != expected_active_generation_id:
                raise ConflictError("The active collection generation changed")
            target = await self.get_generation(collection_id, target_generation_id)
            if target is None:
                raise NotFoundError("collection generation")
            if target.state is not CollectionState.READY:
                raise ConflictError("The target collection generation is not ready")
            activated = replace(
                target,
                state=CollectionState.ACTIVE,
                activated_at=datetime.now(timezone.utc),
            )
            points = [_catalog_point(activated)]
            if active is not None and active.generation_id != target_generation_id:
                points.append(_catalog_point(active.with_state(CollectionState.RETIRED)))
            try:
                await self._client.upsert_points(_CATALOG_COLLECTION, points)
            except QdrantTimeoutError as exc:
                raise VectorStoreUnavailableError(timeout=True) from exc
            except QdrantAdapterError as exc:
                raise VectorStoreUnavailableError() from exc

    async def retire_logical(
        self, collection_id: str, expected_active_generation_id: str
    ) -> object:
        async with self._state_lock:
            active = await self.get_active(collection_id)
            if active is None:
                raise NotFoundError("logical collection")
            if active.generation_id != expected_active_generation_id:
                raise ConflictError("The active collection generation changed")
            retained_until = datetime.now(timezone.utc) + self._retention
            generations = await self.list_generations(collection_id)
            points: list[dict[str, Any]] = []
            for generation in generations:
                # Failed remains a truthful terminal failure state, but receives
                # the same cleanup retention timestamp as every other generation.
                retired = (
                    generation
                    if generation.state is CollectionState.FAILED
                    else generation.with_state(CollectionState.RETIRED)
                )
                payload = _serialize_contract(retired)
                payload["retained_until"] = retained_until.isoformat()
                points.append(
                    {
                        "id": _catalog_point_id(
                            generation.collection_id, generation.generation_id
                        ),
                        "vector": [0.0],
                        "payload": payload,
                    }
                )
            try:
                await self._client.upsert_points(_CATALOG_COLLECTION, points)
                return retained_until
            except QdrantTimeoutError as exc:
                raise VectorStoreUnavailableError(timeout=True) from exc
            except QdrantAdapterError as exc:
                raise VectorStoreUnavailableError() from exc

    async def ready(self, *, deadline_seconds: float) -> ReadinessResult:
        if deadline_seconds <= 0:
            return ReadinessResult(False, "catalog_unavailable")
        try:
            async with asyncio.timeout(deadline_seconds):
                if not await self._client.ready():
                    return ReadinessResult(False, "catalog_unavailable")
                await self._ensure_catalog()
                await self._client.scroll_points(_CATALOG_COLLECTION, limit=1)
                return ReadinessResult(True)
        except (TimeoutError, VectorStoreUnavailableError):
            return ReadinessResult(False, "catalog_unavailable")

    async def remove(self, collection_id: str, generation_id: str) -> None:
        await self._ensure_catalog()
        existing = await self.get_generation(collection_id, generation_id)
        if existing is None:
            raise NotFoundError("collection generation")
        try:
            await self._client.delete_points_by_ids(
                _CATALOG_COLLECTION,
                [_catalog_point_id(collection_id, generation_id)],
            )
        except QdrantTimeoutError as exc:
            raise VectorStoreUnavailableError(timeout=True) from exc
        except QdrantAdapterError as exc:
            raise VectorStoreUnavailableError() from exc

    async def close(self) -> None:
        if self._owns_client:
            await self._client.close()

    async def _ensure_catalog(self) -> None:
        if self._initialized:
            return
        async with self._initialize_lock:
            if self._initialized:
                return
            try:
                details = await self._client.get_collection(_CATALOG_COLLECTION)
                if details is None:
                    try:
                        await self._client.create_collection(
                            _CATALOG_COLLECTION,
                            vector_size=1,
                            distance="Cosine",
                        )
                    except QdrantResourceConflictError:
                        pass
                else:
                    _verify_catalog_vector(details)
                for field_name, field_type in _CATALOG_INDEXES.items():
                    await self._client.create_payload_index(
                        _CATALOG_COLLECTION,
                        field_name=field_name,
                        field_schema=field_type,
                    )
                self._initialized = True
            except ConflictError:
                raise
            except QdrantTimeoutError as exc:
                raise VectorStoreUnavailableError(timeout=True) from exc
            except QdrantAdapterError as exc:
                raise VectorStoreUnavailableError() from exc

    async def _scroll(
        self,
        query_filter: Mapping[str, Any],
        *,
        maximum: int = _MAX_CATALOG_RESULTS,
    ) -> list[CollectionContract]:
        output: list[CollectionContract] = []
        offset: str | int | None = None
        try:
            while len(output) < maximum:
                page_limit = min(_PAGE_SIZE, maximum - len(output))
                points, next_offset = await self._client.scroll_points(
                    _CATALOG_COLLECTION,
                    limit=page_limit,
                    offset=offset,
                    query_filter=query_filter,
                )
                output.extend(_deserialize_point(point) for point in points)
                if next_offset is None:
                    return output
                offset = next_offset
            if len(output) >= _MAX_CATALOG_RESULTS:
                raise LimitExceededError(
                    "The collection catalog result limit was exceeded",
                    max_results=_MAX_CATALOG_RESULTS,
                )
            return output
        except (LimitExceededError, ConflictError):
            raise
        except QdrantTimeoutError as exc:
            raise VectorStoreUnavailableError(timeout=True) from exc
        except QdrantAdapterError as exc:
            raise VectorStoreUnavailableError() from exc


def _catalog_point_id(collection_id: str, generation_id: str) -> str:
    return str(uuid.uuid5(_CATALOG_NAMESPACE, f"{collection_id}\0{generation_id}"))


def _serialize_contract(contract: CollectionContract) -> dict[str, Any]:
    return {
        "collection_id": contract.collection_id,
        "generation_id": contract.generation_id,
        "physical_name": contract.physical_name,
        "embedding_model": contract.embedding.model_id,
        "embedding_revision": contract.embedding.revision,
        "dimension": contract.embedding.dimension,
        "normalized": contract.embedding.normalized,
        "distance": contract.embedding.distance_metric.value,
        "index_schema_version": contract.index_schema_version,
        "metadata_fields": [
            {"name": field.name, "type": field.type.value, "indexed": field.indexed}
            for field in contract.metadata_fields
        ],
        "isolation_policy": contract.isolation_policy,
        "state": contract.state.value,
        "created_at": contract.created_at.isoformat(),
        "activated_at": contract.activated_at.isoformat() if contract.activated_at else None,
        "source_generation_id": contract.source_generation_id,
    }


def _deserialize_point(point: Mapping[str, Any]) -> CollectionContract:
    payload = point.get("payload")
    if not isinstance(payload, dict):
        raise QdrantContractError("Catalog payload is malformed")
    try:
        raw_schema = payload["metadata_fields"]
        if not isinstance(raw_schema, list):
            raise TypeError
        metadata_fields = tuple(
            MetadataField(
                name=_required_str(item, "name"),
                type=MetadataFieldType(_required_str(item, "type")),
                indexed=_required_bool(item, "indexed"),
            )
            for item in raw_schema
            if isinstance(item, Mapping)
        )
        if len(metadata_fields) != len(raw_schema):
            raise TypeError
        created_at = datetime.fromisoformat(_required_str(payload, "created_at"))
        if created_at.tzinfo is None:
            raise ValueError
        activated_raw = payload.get("activated_at")
        activated_at = (
            datetime.fromisoformat(activated_raw)
            if isinstance(activated_raw, str)
            else None
        )
        if activated_at is not None and activated_at.tzinfo is None:
            raise ValueError
        source_generation_id = payload.get("source_generation_id")
        if source_generation_id is not None and not isinstance(source_generation_id, str):
            raise TypeError
        return CollectionContract(
            collection_id=_required_str(payload, "collection_id"),
            generation_id=_required_str(payload, "generation_id"),
            physical_name=_required_str(payload, "physical_name"),
            embedding=EmbeddingContract(
                model_id=_required_str(payload, "embedding_model"),
                revision=_required_str(payload, "embedding_revision"),
                dimension=_required_int(payload, "dimension"),
                normalized=_required_bool(payload, "normalized"),
                distance_metric=DistanceMetric(_required_str(payload, "distance")),
            ),
            index_schema_version=_required_int(payload, "index_schema_version"),
            metadata_fields=metadata_fields,
            isolation_policy=_required_str(payload, "isolation_policy"),
            state=CollectionState(_required_str(payload, "state")),
            created_at=created_at,
            activated_at=activated_at,
            source_generation_id=source_generation_id,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise QdrantContractError("Catalog payload is malformed") from exc


def _required_str(payload: Mapping[str, Any], name: str) -> str:
    value = payload[name]
    if not isinstance(value, str) or not value:
        raise TypeError
    return value


def _required_int(payload: Mapping[str, Any], name: str) -> int:
    value = payload[name]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TypeError
    return value


def _required_bool(payload: Mapping[str, Any], name: str) -> bool:
    value = payload[name]
    if not isinstance(value, bool):
        raise TypeError
    return value


def _verify_catalog_vector(details: Mapping[str, Any]) -> None:
    try:
        vectors = details["config"]["params"]["vectors"]
        if vectors["size"] != 1 or str(vectors["distance"]).lower() != "cosine":
            raise ConflictError("The private collection catalog schema is incompatible")
    except (KeyError, TypeError) as exc:
        raise QdrantContractError("Catalog collection response is malformed") from exc


def _deserialize_catalog_record(point: Mapping[str, Any]) -> CollectionContract:
    """Compatibility alias kept private for focused adapter diagnostics."""
    return _deserialize_point(point)


def _catalog_point(contract: CollectionContract) -> dict[str, Any]:
    return {
        "id": _catalog_point_id(contract.collection_id, contract.generation_id),
        "vector": [0.0],
        "payload": _serialize_contract(contract),
    }


def _encode_cursor(collection_id: str) -> str:
    return base64.urlsafe_b64encode(collection_id.encode("utf-8")).decode("ascii")


def _decode_cursor(cursor: str) -> str:
    try:
        value = base64.b64decode(
            cursor.encode("ascii"), altchars=b"-_", validate=True
        ).decode("utf-8")
    except (ValueError, UnicodeError) as exc:
        raise ValueError("cursor is invalid") from exc
    if not value:
        raise ValueError("cursor is invalid")
    return value
