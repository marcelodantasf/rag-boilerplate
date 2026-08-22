"""Local Sentence Transformers adapter."""

from collections.abc import Sequence
from typing import Any

from embedding_api.domain.errors import (
    EmbeddingEngineUnavailableError,
    InputTooLargeError,
    UnsupportedModelError,
)
from embedding_api.domain.models import EmbeddingBatch, ModelCapability
from embedding_api.infrastructure.settings import Settings


class SentenceTransformerEngine:
    def __init__(self, settings: Settings):
        self._settings = settings
        try:
            # Keep this optional runtime dependency out of import-time code so the
            # application/domain layers and test doubles remain lightweight.
            from sentence_transformers import SentenceTransformer

            self._model: Any = SentenceTransformer(
                settings.model_source,
                revision=settings.model_revision,
                device=settings.inference_device,
                cache_folder=settings.model_cache_dir,
                trust_remote_code=False,
            )
            dimension = int(self._model.get_sentence_embedding_dimension())
            max_tokens = int(self._model.max_seq_length)
        except Exception as error:
            raise EmbeddingEngineUnavailableError() from error

        if dimension != settings.expected_dimension:
            raise ValueError(
                f"configured model dimension is {dimension}, expected "
                f"{settings.expected_dimension}"
            )
        self._capability = ModelCapability(
            model_id=settings.model_id,
            revision=settings.model_revision,
            dimension=dimension,
            max_tokens_per_input=max_tokens,
            normalized=settings.normalize_embeddings,
        )

    def capabilities(self) -> tuple[ModelCapability, ...]:
        return (self._capability,)

    def warmup(self, model_id: str) -> None:
        self.embed(model_id, ["embedding engine warmup"])

    def embed(self, model_id: str, inputs: list[str]) -> EmbeddingBatch:
        if model_id != self._capability.model_id:
            raise UnsupportedModelError(details={"model": model_id})

        token_lengths = self._token_lengths(inputs)
        for index, token_count in enumerate(token_lengths):
            if token_count > self._capability.max_tokens_per_input:
                raise InputTooLargeError(
                    "Input exceeds the model token limit; it was not truncated.",
                    details={
                        "index": index,
                        "actual_tokens": token_count,
                        "max_tokens": self._capability.max_tokens_per_input,
                    },
                )
        total_tokens = sum(token_lengths)
        if total_tokens > self._settings.max_batch_tokens:
            raise InputTooLargeError(
                "Batch exceeds the configured token limit.",
                details={
                    "actual_tokens": total_tokens,
                    "max_batch_tokens": self._settings.max_batch_tokens,
                },
            )

        try:
            encoded = self._model.encode(
                inputs,
                batch_size=min(self._settings.engine_batch_size, len(inputs)),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self._settings.normalize_embeddings,
            )
            vectors = tuple(
                tuple(float(value) for value in vector.tolist()) for vector in encoded
            )
        except Exception as error:
            raise EmbeddingEngineUnavailableError() from error

        return EmbeddingBatch(
            model_id=self._capability.model_id,
            vectors=vectors,
            input_tokens=total_tokens,
        )

    def _token_lengths(self, inputs: Sequence[str]) -> list[int]:
        try:
            tokenized = self._model.tokenizer(
                list(inputs),
                add_special_tokens=True,
                padding=False,
                truncation=False,
            )
            return [len(token_ids) for token_ids in tokenized["input_ids"]]
        except Exception as error:
            raise EmbeddingEngineUnavailableError() from error
