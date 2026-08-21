"""Deterministic test double for transport/application tests."""

import hashlib
import math

from embedding_api.domain.models import EmbeddingBatch, ModelCapability


class DeterministicFakeEngine:
    """Produces stable vectors without pretending to test semantic quality."""

    def __init__(self, model_id: str = "fake-general-purpose", dimension: int = 8):
        self.embed_calls: list[tuple[str, tuple[str, ...]]] = []
        self._capability = ModelCapability(
            model_id=model_id,
            revision="test-only",
            dimension=dimension,
            max_tokens_per_input=10_000,
            normalized=True,
        )

    def capabilities(self) -> tuple[ModelCapability, ...]:
        return (self._capability,)

    def warmup(self, model_id: str) -> None:
        return None

    def embed(self, model_id: str, inputs: list[str]) -> EmbeddingBatch:
        self.embed_calls.append((model_id, tuple(inputs)))
        vectors = tuple(self._vector(value) for value in inputs)
        return EmbeddingBatch(
            model_id=model_id,
            vectors=vectors,
            input_tokens=sum(len(value.split()) for value in inputs),
        )

    def _vector(self, value: str) -> tuple[float, ...]:
        digest = hashlib.sha256(value.encode("utf-8")).digest()
        raw = [float(digest[index] - 127.5) for index in range(self._capability.dimension)]
        norm = math.sqrt(sum(component * component for component in raw)) or 1.0
        return tuple(component / norm for component in raw)
