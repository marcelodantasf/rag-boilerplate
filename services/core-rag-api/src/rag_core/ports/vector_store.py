"""Vector persistence and logical collection catalog ports."""

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from rag_core.domain.models import (
    CollectionContract,
    CollectionState,
    FilterExpression,
    CatalogPage,
    ReadinessResult,
    Verification,
    VectorMatch,
    VectorPoint,
)


class VectorStore(Protocol):
    async def create_collection(self, contract: CollectionContract) -> None: ...

    async def verify_collection(
        self, contract: CollectionContract, *, deadline_seconds: float
    ) -> Verification: ...

    async def create_payload_indexes(self, contract: CollectionContract) -> None: ...

    async def upsert(
        self, contract: CollectionContract, points: Sequence[VectorPoint], *, deadline_seconds: float
    ) -> None: ...

    async def search(
        self,
        contract: CollectionContract,
        vector: Sequence[float],
        *,
        top_k: int,
        filter: FilterExpression | None,
        trusted_filter: FilterExpression,
        deadline_seconds: float,
    ) -> tuple[VectorMatch, ...]: ...

    async def delete_by_document(
        self, contract: CollectionContract, document_id: str, *, deadline_seconds: float
    ) -> int: ...

    async def delete_older_document_versions(
        self, contract: CollectionContract, document_id: str, keep_version: str, *, deadline_seconds: float
    ) -> int: ...

    async def delete_collection(self, contract: CollectionContract, *, deadline_seconds: float) -> None: ...

    async def activate_alias(
        self,
        *,
        logical_id: str,
        previous: CollectionContract | None,
        target: CollectionContract,
        deadline_seconds: float,
    ) -> None: ...

    async def retire_alias(self, logical_id: str, *, deadline_seconds: float) -> None: ...

    async def ready(self, *, deadline_seconds: float) -> ReadinessResult: ...

    async def close(self) -> None: ...


class CollectionCatalog(Protocol):
    async def add_generation(self, contract: CollectionContract) -> None: ...

    async def get_active(self, collection_id: str) -> CollectionContract | None: ...

    async def get_generation(
        self, collection_id: str, generation_id: str
    ) -> CollectionContract | None: ...

    async def list_logical(self, *, limit: int, cursor: str | None) -> CatalogPage: ...

    async def list_generations(
        self, collection_id: str
    ) -> tuple[CollectionContract, ...]: ...

    async def update_state(
        self,
        collection_id: str,
        generation_id: str,
        expected_state: CollectionState,
        new_state: CollectionState,
        activated_at: datetime | None = None,
    ) -> CollectionContract: ...

    async def set_active(
        self, collection_id: str, target_generation_id: str, expected_active_generation_id: str | None
    ) -> None: ...

    async def retire_logical(self, collection_id: str, expected_active_generation_id: str) -> datetime: ...

    async def ready(self, *, deadline_seconds: float) -> ReadinessResult: ...

    async def close(self) -> None: ...
