from __future__ import annotations

import pytest

from conftest import make_settings
from embedding_api.adapters.sentence_transformer import SentenceTransformerEngine
from embedding_api.domain.errors import (
    EmbeddingEngineUnavailableError,
    InputTooLargeError,
    UnsupportedModelError,
)


class ArrayRow:
    def __init__(self, values: list[float]):
        self.values = values

    def tolist(self) -> list[float]:
        return self.values


class RecordingModel:
    max_seq_length = 6

    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        self.tokenizer_calls: list[tuple[list[str], dict[str, object]]] = []
        self.encode_calls: list[tuple[list[str], dict[str, object]]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def tokenizer(self, inputs: list[str], **kwargs):
        self.tokenizer_calls.append((inputs, kwargs))
        return {"input_ids": [list(range(len(value.split()) + 2)) for value in inputs]}

    def encode(self, inputs: list[str], **kwargs):
        self.encode_calls.append((inputs, kwargs))
        return [ArrayRow([float(index), 0.5, -0.5]) for index, _ in enumerate(inputs)]


@pytest.fixture
def model_holder(monkeypatch):
    holder: dict[str, RecordingModel] = {}

    def constructor(*args, **kwargs):
        model = RecordingModel(*args, **kwargs)
        holder["model"] = model
        return model

    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", constructor)
    return holder


def adapter_settings(**overrides):
    values = {
        "model_id": "public-model",
        "model_source": "org/pinned-model",
        "model_revision": "abc123",
        "expected_dimension": 3,
        "engine_batch_size": 2,
        "max_batch_tokens": 20,
    }
    values.update(overrides)
    return make_settings(**values)


def test_adapter_loads_exact_config_and_advertises_capability(model_holder) -> None:
    settings = adapter_settings(
        inference_device="mps", model_cache_dir="/tmp/model-cache"
    )

    engine = SentenceTransformerEngine(settings)
    model = model_holder["model"]

    assert model.init_args == ("org/pinned-model",)
    assert model.init_kwargs == {
        "revision": "abc123",
        "device": "mps",
        "cache_folder": "/tmp/model-cache",
        "trust_remote_code": False,
    }
    assert engine.capabilities()[0].model_id == "public-model"
    assert engine.capabilities()[0].revision == "abc123"
    assert engine.capabilities()[0].dimension == 3
    assert engine.capabilities()[0].max_tokens_per_input == 6
    assert engine.capabilities()[0].normalized is True


def test_batch_uses_one_encode_call_and_does_not_request_truncation(model_holder) -> None:
    engine = SentenceTransformerEngine(adapter_settings())
    inputs = ["one", "two words", "three words here"]

    batch = engine.embed("public-model", inputs)
    model = model_holder["model"]

    assert model.tokenizer_calls == [
        (
            inputs,
            {"add_special_tokens": True, "padding": False, "truncation": False},
        )
    ]
    assert model.encode_calls == [
        (
            inputs,
            {
                "batch_size": 2,
                "show_progress_bar": False,
                "convert_to_numpy": True,
                "normalize_embeddings": True,
            },
        )
    ]
    assert batch.model_id == "public-model"
    assert batch.vectors == (
        (0.0, 0.5, -0.5),
        (1.0, 0.5, -0.5),
        (2.0, 0.5, -0.5),
    )
    assert batch.input_tokens == 12


def test_item_over_model_token_limit_is_rejected_without_encode(model_holder) -> None:
    engine = SentenceTransformerEngine(adapter_settings())

    with pytest.raises(InputTooLargeError) as caught:
        engine.embed("public-model", ["one two three four five"])

    assert caught.value.message == (
        "Input exceeds the model token limit; it was not truncated."
    )
    assert caught.value.details == {"index": 0, "actual_tokens": 7, "max_tokens": 6}
    assert model_holder["model"].encode_calls == []
    assert model_holder["model"].tokenizer_calls[0][1]["truncation"] is False


def test_total_batch_token_limit_is_rejected_without_encode(model_holder) -> None:
    engine = SentenceTransformerEngine(adapter_settings(max_batch_tokens=7))

    with pytest.raises(InputTooLargeError) as caught:
        engine.embed("public-model", ["one two", "three four"])

    assert caught.value.message == "Batch exceeds the configured token limit."
    assert caught.value.details == {"actual_tokens": 8, "max_batch_tokens": 7}
    assert model_holder["model"].encode_calls == []


def test_adapter_rejects_unknown_public_model(model_holder) -> None:
    engine = SentenceTransformerEngine(adapter_settings())

    with pytest.raises(UnsupportedModelError):
        engine.embed("other", ["text"])

    assert model_holder["model"].tokenizer_calls == []
    assert model_holder["model"].encode_calls == []


def test_loaded_dimension_must_match_configuration(model_holder) -> None:
    with pytest.raises(ValueError, match="dimension is 3, expected 4"):
        SentenceTransformerEngine(adapter_settings(expected_dimension=4))


def test_model_loading_failure_is_normalized(monkeypatch) -> None:
    import sentence_transformers

    def fail(*args, **kwargs):
        raise RuntimeError("provider secret that must not cross the boundary")

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", fail)

    with pytest.raises(EmbeddingEngineUnavailableError) as caught:
        SentenceTransformerEngine(adapter_settings())

    assert "provider secret" not in caught.value.message


def test_tokenizer_failure_is_normalized(model_holder) -> None:
    engine = SentenceTransformerEngine(adapter_settings())

    def fail(*args, **kwargs):
        raise RuntimeError("raw input leaked here")

    model_holder["model"].tokenizer = fail

    with pytest.raises(EmbeddingEngineUnavailableError) as caught:
        engine.embed("public-model", ["sensitive text"])

    assert caught.value.message == "The embedding engine is temporarily unavailable."


def test_encode_failure_is_normalized(model_holder) -> None:
    engine = SentenceTransformerEngine(adapter_settings())

    def fail(*args, **kwargs):
        raise RuntimeError("internal provider stack")

    model_holder["model"].encode = fail

    with pytest.raises(EmbeddingEngineUnavailableError) as caught:
        engine.embed("public-model", ["safe"])

    assert caught.value.message == "The embedding engine is temporarily unavailable."
