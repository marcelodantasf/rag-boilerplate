"""Stable application errors mapped to the public error envelope."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(eq=False)
class CoreError(Exception):
    code: str
    message: str
    status_code: int
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)


class InvalidRequestError(CoreError):
    def __init__(self, message: str = "The request is invalid", **details: Any):
        super().__init__("invalid_request", message, 422, details)


class LimitExceededError(CoreError):
    def __init__(self, message: str = "A request limit was exceeded", **details: Any):
        super().__init__("limit_exceeded", message, 413, details)


class NotFoundError(CoreError):
    def __init__(self, resource: str = "resource"):
        super().__init__("not_found", f"The requested {resource} was not found", 404)


class ConflictError(CoreError):
    def __init__(self, message: str, **details: Any):
        super().__init__("conflict", message, 409, details)


class IdempotencyConflictError(CoreError):
    def __init__(self):
        super().__init__(
            "idempotency_conflict",
            "The idempotency key is already bound to a different request",
            409,
        )


class PreconditionFailedError(CoreError):
    def __init__(self, message: str = "A request precondition did not match current state"):
        super().__init__("precondition_failed", message, 412)


class CollectionAlreadyExistsError(CoreError):
    def __init__(self):
        super().__init__(
            "collection_already_exists",
            "The logical collection already exists",
            409,
        )


class GenerationNotReadyError(CoreError):
    def __init__(self):
        super().__init__(
            "generation_not_ready",
            "The collection generation is not ready for activation",
            409,
        )


class EmbeddingSchemaMismatchError(CoreError):
    def __init__(self, **details: Any):
        super().__init__(
            "embedding_schema_mismatch",
            "The embedding result is incompatible with the collection",
            409,
            details,
        )


class EmbeddingUnavailableError(CoreError):
    def __init__(self, *, timeout: bool = False):
        super().__init__(
            "embedding_timeout" if timeout else "embedding_unavailable",
            "The embedding service timed out" if timeout else "The embedding service is unavailable",
            504 if timeout else 503,
        )


class EmbeddingContractError(CoreError):
    def __init__(self):
        super().__init__(
            "embedding_contract_violation",
            "The embedding service returned an invalid response",
            502,
        )


class VectorStoreUnavailableError(CoreError):
    def __init__(self, *, timeout: bool = False):
        super().__init__(
            "vector_store_timeout" if timeout else "vector_store_unavailable",
            "The vector store timed out" if timeout else "The vector store is unavailable",
            504 if timeout else 503,
        )
