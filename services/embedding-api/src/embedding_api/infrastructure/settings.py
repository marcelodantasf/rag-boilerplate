"""Typed environment configuration with fail-fast validation."""

import os
from dataclasses import dataclass


PINNED_MODEL_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    model_id: str = "all-MiniLM-L6-v2"
    model_source: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_revision: str = PINNED_MODEL_REVISION
    expected_dimension: int = 384
    inference_device: str = "cpu"
    model_cache_dir: str | None = None
    normalize_embeddings: bool = True
    engine_batch_size: int = 32
    max_batch_items: int = 64
    max_input_bytes: int = 65_536
    max_total_input_bytes: int = 262_144
    max_batch_tokens: int = 8_192
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        defaults = cls()
        # DEFAULT_EMBEDDING_MODEL is retained as a scaffold-compatible alias.
        model_id = os.getenv(
            "DEFAULT_MODEL_ID",
            os.getenv("DEFAULT_EMBEDDING_MODEL", defaults.model_id),
        )
        model_source = os.getenv("MODEL_SOURCE", defaults.model_source)
        model_revision = os.getenv("MODEL_REVISION", defaults.model_revision)
        inference_device = os.getenv("INFERENCE_DEVICE", defaults.inference_device)
        model_cache_dir = os.getenv("MODEL_CACHE_DIR") or None
        normalize_raw = os.getenv("NORMALIZE_EMBEDDINGS", "true").lower()
        if normalize_raw not in {"true", "false"}:
            raise ValueError("NORMALIZE_EMBEDDINGS must be true or false")
        settings = cls(
            model_id=model_id,
            model_source=model_source,
            model_revision=model_revision,
            expected_dimension=_positive_int(
                "EXPECTED_DIMENSION", defaults.expected_dimension
            ),
            inference_device=inference_device,
            model_cache_dir=model_cache_dir,
            normalize_embeddings=normalize_raw == "true",
            engine_batch_size=_positive_int(
                "ENGINE_BATCH_SIZE", defaults.engine_batch_size
            ),
            max_batch_items=_positive_int(
                "MAX_BATCH_ITEMS", defaults.max_batch_items
            ),
            max_input_bytes=_positive_int(
                "MAX_INPUT_BYTES", defaults.max_input_bytes
            ),
            max_total_input_bytes=_positive_int(
                "MAX_TOTAL_INPUT_BYTES", defaults.max_total_input_bytes
            ),
            max_batch_tokens=_positive_int(
                "MAX_BATCH_TOKENS", defaults.max_batch_tokens
            ),
            log_level=os.getenv("LOG_LEVEL", defaults.log_level).upper(),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.model_id.strip() or not self.model_source.strip():
            raise ValueError("model identity and source must not be empty")
        if not self.model_revision.strip():
            raise ValueError("MODEL_REVISION must pin an immutable model revision")
        if self.max_total_input_bytes < self.max_input_bytes:
            raise ValueError("MAX_TOTAL_INPUT_BYTES must be >= MAX_INPUT_BYTES")
        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")
