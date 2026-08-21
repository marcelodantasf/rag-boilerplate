"""Embedding engine port owned by the application layer."""

from typing import Protocol

from embedding_api.domain.models import EmbeddingBatch, ModelCapability


class EmbeddingEngine(Protocol):
    def capabilities(self) -> tuple[ModelCapability, ...]: ...

    def embed(self, model_id: str, inputs: list[str]) -> EmbeddingBatch: ...

    def warmup(self, model_id: str) -> None: ...
