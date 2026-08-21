"""Provider-independent embedding domain models."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelCapability:
    model_id: str
    revision: str
    dimension: int
    max_tokens_per_input: int
    normalized: bool


@dataclass(frozen=True, slots=True)
class EmbeddingBatch:
    """Engine output. Vector position corresponds to input position."""

    model_id: str
    vectors: tuple[tuple[float, ...], ...]
    input_tokens: int


@dataclass(frozen=True, slots=True)
class EmbedResult:
    model_id: str
    dimension: int
    vectors: tuple[tuple[float, ...], ...]
    input_tokens: int
