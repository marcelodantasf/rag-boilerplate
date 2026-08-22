"""Embed-text use case and provider-independent contract enforcement."""

import math

from embedding_api.domain.errors import (
    EmbeddingContractViolationError,
    InputTooLargeError,
    InvalidInputError,
    UnsupportedModelError,
)
from embedding_api.domain.models import EmbedResult, ModelCapability
from embedding_api.infrastructure.settings import Settings
from embedding_api.ports.engine import EmbeddingEngine


class EmbedTexts:
    def __init__(self, engine: EmbeddingEngine, settings: Settings):
        self._engine = engine
        self._settings = settings

    def execute(self, model_id: str, inputs: list[str]) -> EmbedResult:
        capability = self._capability(model_id)
        self._validate_inputs(inputs)

        batch = self._engine.embed(model_id, inputs)
        self._validate_output(batch.model_id, batch.vectors, inputs, capability)
        return EmbedResult(
            model_id=batch.model_id,
            dimension=capability.dimension,
            vectors=batch.vectors,
            input_tokens=batch.input_tokens,
        )

    def _capability(self, model_id: str) -> ModelCapability:
        for capability in self._engine.capabilities():
            if capability.model_id == model_id:
                return capability
        raise UnsupportedModelError(details={"model": model_id})

    def _validate_inputs(self, inputs: list[str]) -> None:
        if not inputs or len(inputs) > self._settings.max_batch_items:
            raise InvalidInputError(
                details={"max_batch_items": self._settings.max_batch_items}
            )

        total_bytes = 0
        for index, value in enumerate(inputs):
            if not isinstance(value, str) or not value.strip():
                raise InvalidInputError(details={"index": index})
            try:
                item_bytes = len(value.encode("utf-8"))
            except UnicodeEncodeError as error:
                raise InvalidInputError(
                    "Input must be valid UTF-8 text.", details={"index": index}
                ) from error
            if item_bytes > self._settings.max_input_bytes:
                raise InputTooLargeError(
                    details={
                        "index": index,
                        "actual_bytes": item_bytes,
                        "max_input_bytes": self._settings.max_input_bytes,
                    }
                )
            total_bytes += item_bytes

        if total_bytes > self._settings.max_total_input_bytes:
            raise InputTooLargeError(
                details={
                    "actual_bytes": total_bytes,
                    "max_total_input_bytes": self._settings.max_total_input_bytes,
                }
            )

    @staticmethod
    def _validate_output(
        actual_model_id: str,
        vectors: tuple[tuple[float, ...], ...],
        inputs: list[str],
        capability: ModelCapability,
    ) -> None:
        if actual_model_id != capability.model_id or len(vectors) != len(inputs):
            raise EmbeddingContractViolationError()
        for vector in vectors:
            if len(vector) != capability.dimension:
                raise EmbeddingContractViolationError()
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in vector
            ):
                raise EmbeddingContractViolationError()
