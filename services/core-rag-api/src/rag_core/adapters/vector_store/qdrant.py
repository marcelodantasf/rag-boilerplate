"""Qdrant implementation of the provider-neutral vector-store port."""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import math
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from rag_core.domain.errors import (
    ConflictError,
    InvalidRequestError,
    NotFoundError,
    VectorStoreUnavailableError,
)
from rag_core.domain.models import (
    CollectionContract,
    DistanceMetric,
    FilterExpression,
    Metadata,
    MetadataField,
    MetadataFieldType,
    ReadinessResult,
    VectorMatch,
    VectorPoint,
)

from ._errors import (
    QdrantAdapterError,
    QdrantContractError,
    QdrantResourceConflictError,
    QdrantResourceNotFoundError,
    QdrantTimeoutError,
)
from ._filters import translate_filter, validate_metadata
from ._http import QdrantHttpClient

_POINT_NAMESPACE = uuid.UUID("f04fa79d-1028-59cc-84df-bef297f430ff")
_SAFE_PROVIDER_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,127}$")
_SAFE_METADATA_FIELD = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")

_QDRANT_DISTANCE = {
    DistanceMetric.COSINE: "Cosine",
    DistanceMetric.DOT: "Dot",
    DistanceMetric.EUCLID: "Euclid",
}
_QDRANT_PAYLOAD_TYPE = {
    MetadataFieldType.KEYWORD: "keyword",
    MetadataFieldType.INTEGER: "integer",
    MetadataFieldType.FLOAT: "float",
    MetadataFieldType.BOOLEAN: "bool",
}
_REQUIRED_INDEXES = {
    "__tenant_id": "keyword",
    "document_id": "keyword",
    "embedding_model": "keyword",
    "embedding_revision": "keyword",
    "index_schema_version": "integer",
}


def _bounded_operation(method: Any) -> Any:
    """Enforce the remaining caller deadline across all nested adapter calls."""

    @functools.wraps(method)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        deadline = kwargs.get("deadline_seconds")
        if type(deadline) not in {float, int} or deadline <= 0:
            raise VectorStoreUnavailableError(timeout=True)
        try:
            async with asyncio.timeout(float(deadline)):
                return await method(*args, **kwargs)
        except TimeoutError as exc:
            raise VectorStoreUnavailableError(timeout=True) from exc

    return wrapper


class QdrantVectorStore:
    """Store chunk points without leaking Qdrant IDs or filters to callers."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 3.0,
        trusted_tenant_id: str = "local",
        client: QdrantHttpClient | None = None,
    ) -> None:
        if not trusted_tenant_id:
            raise ValueError("trusted_tenant_id must not be empty")
        self._client = client or QdrantHttpClient(
            base_url, api_key=api_key, timeout_seconds=timeout_seconds
        )
        self._owns_client = client is None
        self._trusted_tenant_id = trusted_tenant_id

    async def create_collection(self, contract: CollectionContract) -> None:
        _validate_contract(contract)
        try:
            existing = await self._client.get_collection(contract.physical_name)
            if existing is not None:
                if not _collection_is_compatible(existing, contract):
                    raise ConflictError(
                        "The physical collection has an incompatible vector schema"
                    )
            else:
                try:
                    await self._client.create_collection(
                        contract.physical_name,
                        vector_size=contract.embedding.dimension,
                        distance=_QDRANT_DISTANCE[contract.embedding.distance_metric],
                    )
                except QdrantResourceConflictError:
                    # A concurrent idempotent creator may have won the race.
                    existing = await self._client.get_collection(contract.physical_name)
                    if existing is None or not _collection_is_compatible(existing, contract):
                        raise ConflictError(
                            "The physical collection has an incompatible vector schema"
                        ) from None

        except ConflictError:
            raise
        except QdrantTimeoutError as exc:
            raise VectorStoreUnavailableError(timeout=True) from exc
        except QdrantAdapterError as exc:
            raise VectorStoreUnavailableError() from exc

    @_bounded_operation
    async def verify_collection(
        self, contract: CollectionContract, *, deadline_seconds: float
    ) -> ReadinessResult:
        _validate_contract(contract)
        try:
            details = await self._client.get_collection(contract.physical_name)
            if details is None:
                return ReadinessResult(False, "collection_not_found")
            if not _collection_is_compatible(details, contract):
                return ReadinessResult(False, "collection_incompatible")
            payload_schema = details.get("payload_schema")
            if not isinstance(payload_schema, dict):
                return ReadinessResult(False, "collection_incompatible")
            indexes_match = all(
                _payload_index_type(payload_schema.get(name)) == expected
                for name, expected in _expected_indexes(contract).items()
            )
            if not indexes_match:
                return ReadinessResult(False, "collection_incompatible")
            marker = await self._client.retrieve_points(
                contract.physical_name, [_schema_marker_id(contract)]
            )
            if len(marker) != 1 or not _marker_matches(marker[0], contract):
                return ReadinessResult(False, "collection_incompatible")
            return ReadinessResult(True)
        except QdrantTimeoutError as exc:
            raise VectorStoreUnavailableError(timeout=True) from exc
        except QdrantAdapterError as exc:
            raise VectorStoreUnavailableError() from exc

    async def create_payload_indexes(self, contract: CollectionContract) -> None:
        _validate_contract(contract)
        try:
            for field_name, field_type in _expected_indexes(contract).items():
                await self._client.create_payload_index(
                    contract.physical_name,
                    field_name=field_name,
                    field_schema=field_type,
                )
            await self._client.upsert_points(
                contract.physical_name,
                [{
                    "id": _schema_marker_id(contract),
                    "vector": [0.0] * contract.embedding.dimension,
                    "payload": {
                        "__record_type": "schema",
                        "__schema_fingerprint": _schema_fingerprint(contract),
                    },
                }],
            )
        except QdrantTimeoutError as exc:
            raise VectorStoreUnavailableError(timeout=True) from exc
        except QdrantAdapterError as exc:
            raise VectorStoreUnavailableError() from exc

    @_bounded_operation
    async def upsert(
        self,
        contract: CollectionContract,
        points: Sequence[VectorPoint],
        *,
        deadline_seconds: float,
    ) -> None:
        _validate_contract(contract)
        if not points:
            return
        verification = await self.verify_collection(
            contract, deadline_seconds=deadline_seconds
        )
        if not verification.ready:
            raise ConflictError("The vector collection is incompatible with its contract")

        encoded = [
            _encode_point(point, contract, trusted_tenant_id=self._trusted_tenant_id)
            for point in points
        ]
        try:
            await self._client.upsert_points(contract.physical_name, encoded)
        except QdrantTimeoutError as exc:
            raise VectorStoreUnavailableError(timeout=True) from exc
        except QdrantAdapterError as exc:
            raise VectorStoreUnavailableError() from exc

    @_bounded_operation
    async def search(
        self,
        contract: CollectionContract,
        vector: Sequence[float],
        *,
        top_k: int,
        filter: FilterExpression | None,
        trusted_filter: FilterExpression,
        deadline_seconds: float,
    ) -> tuple[VectorMatch, ...]:
        _validate_contract(contract)
        _validate_vector(vector, contract.embedding.dimension)
        if top_k <= 0:
            raise InvalidRequestError("top_k must be positive")
        verification = await self.verify_collection(
            contract, deadline_seconds=deadline_seconds
        )
        if not verification.ready:
            raise ConflictError("The vector collection is incompatible with its contract")
        consumer_filter = translate_filter(filter, contract)
        server_filter = _translate_trusted_filter(trusted_filter, contract)
        must: list[dict[str, Any]] = [
            {"key": "__record_type", "match": {"value": "chunk"}},
            {
                "key": "__tenant_id",
                "match": {"value": self._trusted_tenant_id},
            },
            server_filter or {},
        ]
        if consumer_filter:
            must.append(consumer_filter)
        qdrant_filter = {"must": must}
        try:
            raw_points = await self._client.search_points(
                contract.physical_name,
                vector=vector,
                limit=top_k,
                query_filter=qdrant_filter,
            )
            return tuple(_decode_vector_match(item, contract) for item in raw_points)
        except QdrantContractError as exc:
            raise VectorStoreUnavailableError() from exc
        except QdrantTimeoutError as exc:
            raise VectorStoreUnavailableError(timeout=True) from exc
        except QdrantAdapterError as exc:
            raise VectorStoreUnavailableError() from exc

    @_bounded_operation
    async def delete_by_document(
        self,
        contract: CollectionContract,
        document_id: str,
        *,
        deadline_seconds: float,
    ) -> int:
        _validate_contract(contract)
        if not document_id:
            raise InvalidRequestError("document_id must not be empty")
        verification = await self.verify_collection(
            contract, deadline_seconds=deadline_seconds
        )
        if not verification.ready:
            raise ConflictError("The vector collection is incompatible with its contract")
        query_filter = {
            "must": [
                {"key": "__record_type", "match": {"value": "chunk"}},
                {"key": "__tenant_id", "match": {"value": self._trusted_tenant_id}},
                {"key": "document_id", "match": {"value": document_id}},
            ]
        }
        try:
            count = await self._client.count_points(
                contract.physical_name, query_filter
            )
            if count:
                await self._client.delete_points_by_filter(
                    contract.physical_name, query_filter
                )
            return count
        except QdrantTimeoutError as exc:
            raise VectorStoreUnavailableError(timeout=True) from exc
        except QdrantAdapterError as exc:
            raise VectorStoreUnavailableError() from exc

    @_bounded_operation
    async def delete_older_document_versions(
        self,
        contract: CollectionContract,
        document_id: str,
        keep_version: str,
        *,
        deadline_seconds: float,
    ) -> int:
        _validate_contract(contract)
        if not document_id or not keep_version:
            raise InvalidRequestError("Document identity and version must not be empty")
        verification = await self.verify_collection(
            contract, deadline_seconds=deadline_seconds
        )
        if not verification.ready:
            raise ConflictError("The vector collection is incompatible with its contract")
        query_filter = {
            "must": [
                {"key": "__record_type", "match": {"value": "chunk"}},
                {"key": "__tenant_id", "match": {"value": self._trusted_tenant_id}},
                {"key": "document_id", "match": {"value": document_id}},
            ],
            "must_not": [
                {"key": "document_version", "match": {"value": keep_version}}
            ],
        }
        try:
            count = await self._client.count_points(contract.physical_name, query_filter)
            if count:
                await self._client.delete_points_by_filter(
                    contract.physical_name, query_filter
                )
            return count
        except QdrantTimeoutError as exc:
            raise VectorStoreUnavailableError(timeout=True) from exc
        except QdrantAdapterError as exc:
            raise VectorStoreUnavailableError() from exc

    @_bounded_operation
    async def delete_collection(
        self, contract: CollectionContract, *, deadline_seconds: float
    ) -> None:
        _validate_contract(contract)
        try:
            await self._client.delete_collection(contract.physical_name)
        except QdrantResourceNotFoundError as exc:
            raise NotFoundError("collection generation") from exc
        except QdrantTimeoutError as exc:
            raise VectorStoreUnavailableError(timeout=True) from exc
        except QdrantAdapterError as exc:
            raise VectorStoreUnavailableError() from exc

    @_bounded_operation
    async def activate_alias(
        self,
        *,
        logical_id: str,
        previous: CollectionContract | None,
        target: CollectionContract,
        deadline_seconds: float,
    ) -> None:
        _validate_contract(target)
        if previous is not None:
            _validate_contract(previous)
        verification = await self.verify_collection(
            target, deadline_seconds=deadline_seconds
        )
        if not verification.ready:
            raise ConflictError("The target collection generation is incompatible")
        alias_name = _active_alias_name(logical_id)
        actions: list[dict[str, Any]] = []
        if previous is not None and previous.physical_name != target.physical_name:
            actions.append({"delete_alias": {"alias_name": alias_name}})
        actions.append(
            {
                "create_alias": {
                    "collection_name": target.physical_name,
                    "alias_name": alias_name,
                }
            }
        )
        try:
            await self._client.update_aliases(actions)
        except QdrantResourceConflictError as exc:
            raise ConflictError("The collection alias could not be activated") from exc
        except QdrantTimeoutError as exc:
            raise VectorStoreUnavailableError(timeout=True) from exc
        except QdrantAdapterError as exc:
            raise VectorStoreUnavailableError() from exc

    @_bounded_operation
    async def retire_alias(
        self, logical_id: str, *, deadline_seconds: float
    ) -> None:
        try:
            await self._client.update_aliases(
                [{"delete_alias": {"alias_name": _active_alias_name(logical_id)}}]
            )
        except QdrantResourceNotFoundError:
            return
        except QdrantTimeoutError as exc:
            raise VectorStoreUnavailableError(timeout=True) from exc
        except QdrantAdapterError as exc:
            raise VectorStoreUnavailableError() from exc

    async def ready(self, *, deadline_seconds: float) -> ReadinessResult:
        if deadline_seconds <= 0:
            return ReadinessResult(False, "vector_store_timeout")
        try:
            async with asyncio.timeout(deadline_seconds):
                ready = await self._client.ready()
        except TimeoutError:
            return ReadinessResult(False, "vector_store_timeout")
        return ReadinessResult(
            ready,
            None if ready else "vector_store_unavailable",
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.close()


def _validate_contract(contract: CollectionContract) -> None:
    if not _SAFE_PROVIDER_NAME.fullmatch(contract.physical_name):
        raise InvalidRequestError("The collection generation has an invalid storage name")
    if contract.embedding.dimension <= 0:
        raise InvalidRequestError("Collection dimension must be positive")
    if contract.index_schema_version <= 0:
        raise InvalidRequestError("Index schema version must be positive")
    if contract.embedding.distance_metric not in _QDRANT_DISTANCE:
        raise InvalidRequestError("Distance metric is unsupported")
    if (
        contract.embedding.distance_metric is DistanceMetric.DOT
        and not contract.embedding.normalized
    ):
        raise InvalidRequestError("Dot distance requires normalized embeddings")
    names: set[str] = set()
    for field in contract.metadata_fields:
        if not _SAFE_METADATA_FIELD.fullmatch(field.name):
            raise InvalidRequestError("Metadata schema contains an invalid field name")
        if field.name in names:
            raise InvalidRequestError("Metadata schema contains a duplicate field name")
        names.add(field.name)


def _expected_indexes(contract: CollectionContract) -> dict[str, str]:
    expected = dict(_REQUIRED_INDEXES)
    expected["__record_type"] = "keyword"
    expected["__schema_fingerprint"] = "keyword"
    expected.update(
        {
            f"metadata.{field.name}": _QDRANT_PAYLOAD_TYPE[field.type]
            for field in contract.metadata_fields
            if field.indexed
        }
    )
    return expected


def _collection_is_compatible(
    details: Mapping[str, Any], contract: CollectionContract
) -> bool:
    config = details.get("config")
    if not isinstance(config, Mapping):
        return False
    params = config.get("params")
    if not isinstance(params, Mapping):
        return False
    vectors = params.get("vectors")
    if not isinstance(vectors, Mapping):
        return False
    size = vectors.get("size")
    distance = vectors.get("distance")
    return size == contract.embedding.dimension and str(distance).lower() == _QDRANT_DISTANCE[
        contract.embedding.distance_metric
    ].lower()


def _payload_index_type(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    data_type = value.get("data_type")
    if isinstance(data_type, str):
        return data_type.lower()
    params = value.get("params")
    if isinstance(params, Mapping) and isinstance(params.get("type"), str):
        return str(params["type"]).lower()
    return None


def _encode_point(
    point: VectorPoint,
    contract: CollectionContract,
    *,
    trusted_tenant_id: str,
) -> dict[str, Any]:
    _validate_vector(point.vector, contract.embedding.dimension)
    if point.embedding_model != contract.embedding.model_id:
        raise ConflictError("Point embedding model does not match the collection")
    if point.embedding_revision != contract.embedding.revision:
        raise ConflictError("Point embedding revision does not match the collection")
    if point.index_schema_version != contract.index_schema_version:
        raise ConflictError("Point schema version does not match the collection")
    if not point.point_id or not point.chunk_id or not point.document_id:
        raise InvalidRequestError("Point identities must not be empty")
    if point.chunk_index < 0:
        raise InvalidRequestError("Chunk index must not be negative")
    metadata = validate_metadata(point.metadata, contract)
    return {
        "id": _provider_point_id(point.point_id),
        "vector": list(point.vector),
        "payload": {
            "__record_type": "chunk",
            "__tenant_id": trusted_tenant_id,
            "chunk_id": point.chunk_id,
            "document_id": point.document_id,
            "document_version": point.document_version,
            "chunk_index": point.chunk_index,
            "text": point.text,
            "metadata": metadata,
            "embedding_model": point.embedding_model,
            "embedding_revision": point.embedding_revision,
            "index_schema_version": point.index_schema_version,
        },
    }


def _validate_vector(vector: Sequence[float], dimension: int) -> None:
    if len(vector) != dimension:
        raise ConflictError("Vector dimension does not match the collection")
    if any(type(value) not in {float, int} or not math.isfinite(value) for value in vector):
        raise InvalidRequestError("Vectors must contain only finite numbers")


def _decode_vector_match(
    value: Mapping[str, Any], contract: CollectionContract
) -> VectorMatch:
    payload = value.get("payload")
    score = value.get("score")
    if not isinstance(payload, dict) or type(score) not in {float, int}:
        raise QdrantContractError("Qdrant search point is malformed")
    chunk_id = payload.get("chunk_id")
    document_id = payload.get("document_id")
    document_version = payload.get("document_version")
    text = payload.get("text")
    metadata = payload.get("metadata", {})
    if (
        not isinstance(chunk_id, str)
        or not isinstance(document_id, str)
        or not isinstance(document_version, str)
        or not isinstance(text, str)
        or not isinstance(metadata, dict)
        or not math.isfinite(score)
    ):
        raise QdrantContractError("Qdrant search payload is malformed")
    safe_metadata: Metadata = {}
    for key, item in metadata.items():
        if not isinstance(key, str) or type(item) not in {str, int, float, bool}:
            raise QdrantContractError("Qdrant search metadata is malformed")
        safe_metadata[key] = item
    metric_value = float(score)
    if contract.embedding.distance_metric is DistanceMetric.EUCLID:
        metric_value = max(0.0, -metric_value)
    return VectorMatch(
        chunk_id=chunk_id,
        document_id=document_id,
        document_version=document_version,
        text=text,
        metric_value=metric_value,
        metadata=safe_metadata,
    )


def _provider_point_id(application_point_id: str) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, application_point_id))


def _active_alias_name(collection_id: str) -> str:
    digest = hashlib.sha256(collection_id.encode("utf-8")).hexdigest()[:32]
    return f"rag_active_{digest}"


def _schema_fingerprint(contract: CollectionContract) -> str:
    fields = [
        {"name": field.name, "type": field.type.value, "indexed": field.indexed}
        for field in contract.metadata_fields
    ]
    value = {
        "model_id": contract.embedding.model_id,
        "revision": contract.embedding.revision,
        "dimension": contract.embedding.dimension,
        "normalized": contract.embedding.normalized,
        "distance_metric": contract.embedding.distance_metric.value,
        "index_schema_version": contract.index_schema_version,
        "metadata_fields": fields,
        "isolation_policy": contract.isolation_policy,
    }
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _schema_marker_id(contract: CollectionContract) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, f"schema\0{contract.physical_name}"))


def _marker_matches(point: Mapping[str, Any], contract: CollectionContract) -> bool:
    payload = point.get("payload")
    return (
        isinstance(payload, Mapping)
        and payload.get("__record_type") == "schema"
        and payload.get("__schema_fingerprint") == _schema_fingerprint(contract)
    )


def _translate_trusted_filter(
    expression: FilterExpression, contract: CollectionContract
) -> dict[str, Any]:
    """Translate server-owned isolation fields outside the public metadata schema."""

    trusted_contract = replace(
        contract,
        metadata_fields=(
            MetadataField("tenant_id", MetadataFieldType.KEYWORD, indexed=True),
        ),
    )
    translated = translate_filter(expression, trusted_contract)
    if translated is None:
        raise InvalidRequestError("Trusted isolation filter is required")
    return _rewrite_trusted_filter_keys(translated)


def _rewrite_trusted_filter_keys(value: Any) -> Any:
    if isinstance(value, list):
        return [_rewrite_trusted_filter_keys(item) for item in value]
    if isinstance(value, dict):
        output = {
            key: _rewrite_trusted_filter_keys(item) for key, item in value.items()
        }
        if output.get("key") == "metadata.tenant_id":
            output["key"] = "__tenant_id"
        return output
    return value
