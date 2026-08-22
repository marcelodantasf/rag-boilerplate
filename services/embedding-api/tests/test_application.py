from __future__ import annotations

import math

import pytest

from conftest import make_settings
from embedding_api.application.embed import EmbedTexts
from embedding_api.domain.errors import (
    EmbeddingContractViolationError,
    InputTooLargeError,
    InvalidInputError,
    UnsupportedModelError,
)
from embedding_api.domain.models import EmbeddingBatch, ModelCapability


class StubEngine:
    def __init__(self, batch: EmbeddingBatch):
        self.batch = batch
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.capability = ModelCapability(
            model_id="test-model",
            revision="test-revision",
            dimension=3,
            max_tokens_per_input=100,
            normalized=False,
        )

    def capabilities(self) -> tuple[ModelCapability, ...]:
        return (self.capability,)

    def warmup(self, model_id: str) -> None:
        return None

    def embed(self, model_id: str, inputs: list[str]) -> EmbeddingBatch:
        self.calls.append((model_id, tuple(inputs)))
        return self.batch


def service_for(batch: EmbeddingBatch) -> tuple[EmbedTexts, StubEngine]:
    engine = StubEngine(batch)
    settings = make_settings(
        model_id="test-model",
        expected_dimension=3,
        max_batch_items=3,
        max_input_bytes=8,
        max_total_input_bytes=12,
    )
    return EmbedTexts(engine, settings), engine


def valid_batch(**overrides: object) -> EmbeddingBatch:
    values: dict[str, object] = {
        "model_id": "test-model",
        "vectors": ((0.1, 0.2, 0.3),),
        "input_tokens": 2,
    }
    values.update(overrides)
    return EmbeddingBatch(**values)


def test_execute_routes_model_and_preserves_engine_result() -> None:
    service, engine = service_for(valid_batch())

    result = service.execute("test-model", ["hello"])

    assert engine.calls == [("test-model", ("hello",))]
    assert result.model_id == "test-model"
    assert result.dimension == 3
    assert result.vectors == ((0.1, 0.2, 0.3),)
    assert result.input_tokens == 2


def test_unknown_model_is_rejected_before_engine_call() -> None:
    service, engine = service_for(valid_batch())

    with pytest.raises(UnsupportedModelError) as caught:
        service.execute("other-model", ["hello"])

    assert caught.value.details == {"model": "other-model"}
    assert engine.calls == []


@pytest.mark.parametrize("inputs", [[], [""], ["   "], ["ok", 4]])
def test_invalid_input_values_are_rejected(inputs: list[object]) -> None:
    service, engine = service_for(valid_batch())

    with pytest.raises(InvalidInputError):
        service.execute("test-model", inputs)  # type: ignore[arg-type]

    assert engine.calls == []


def test_batch_item_limit_is_enforced_before_engine_call() -> None:
    service, engine = service_for(valid_batch())

    with pytest.raises(InvalidInputError) as caught:
        service.execute("test-model", ["a", "b", "c", "d"])

    assert caught.value.details == {"max_batch_items": 3}
    assert engine.calls == []


def test_per_item_limit_counts_utf8_bytes() -> None:
    service, engine = service_for(valid_batch())

    # Four accented characters are 8 UTF-8 bytes and are accepted.
    service.execute("test-model", ["éééé"])
    assert len(engine.calls) == 1

    with pytest.raises(InputTooLargeError) as caught:
        service.execute("test-model", ["ééééa"])

    assert caught.value.details == {
        "index": 0,
        "actual_bytes": 9,
        "max_input_bytes": 8,
    }


def test_total_request_byte_limit_is_enforced() -> None:
    batch = valid_batch(vectors=((0.1, 0.2, 0.3), (0.4, 0.5, 0.6)))
    service, engine = service_for(batch)

    with pytest.raises(InputTooLargeError) as caught:
        service.execute("test-model", ["1234567", "7654321"])

    assert caught.value.details == {
        "actual_bytes": 14,
        "max_total_input_bytes": 12,
    }
    assert engine.calls == []


@pytest.mark.parametrize(
    "batch",
    [
        valid_batch(model_id="wrong-model"),
        valid_batch(vectors=()),
        valid_batch(vectors=((0.1, 0.2),)),
        valid_batch(vectors=((0.1, math.nan, 0.3),)),
        valid_batch(vectors=((0.1, math.inf, 0.3),)),
        valid_batch(vectors=((0.1, True, 0.3),)),
        valid_batch(vectors=((0.1, "bad", 0.3),)),
    ],
    ids=[
        "wrong-model",
        "wrong-count",
        "wrong-dimension",
        "nan",
        "infinity",
        "boolean",
        "nonnumeric",
    ],
)
def test_invalid_engine_output_is_rejected(batch: EmbeddingBatch) -> None:
    service, _ = service_for(batch)

    with pytest.raises(EmbeddingContractViolationError):
        service.execute("test-model", ["hello"])


def test_integer_components_are_valid_numeric_output() -> None:
    service, _ = service_for(valid_batch(vectors=((1, 0, -1),)))

    result = service.execute("test-model", ["hello"])

    assert result.vectors == ((1, 0, -1),)
