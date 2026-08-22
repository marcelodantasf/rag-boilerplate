from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from embedding_api.adapters.sentence_transformer import SentenceTransformerEngine
from embedding_api.infrastructure.settings import PINNED_MODEL_REVISION, Settings
from embedding_api.transport.http import create_app


def cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True)) / (
        math.sqrt(sum(a * a for a in left))
        * math.sqrt(sum(b * b for b in right))
    )


@pytest.fixture(scope="module")
def real_engine() -> SentenceTransformerEngine:
    return SentenceTransformerEngine(Settings())


def test_pinned_real_model_capability(real_engine: SentenceTransformerEngine) -> None:
    capability = real_engine.capabilities()[0]

    assert capability.model_id == "all-MiniLM-L6-v2"
    assert capability.revision == PINNED_MODEL_REVISION
    assert capability.dimension == 384
    assert capability.max_tokens_per_input > 0
    assert capability.normalized is True


def test_real_model_embeds_mixed_documents_as_ordered_normalized_batch(
    real_engine: SentenceTransformerEngine,
) -> None:
    inputs = [
        "Article 1. Every person has the right to education.",
        "Whisk eggs and sugar, then fold in flour before baking.",
        "Gradient descent updates parameters using the loss derivative.",
        "教育は未来への架け橋です。",
    ]

    batch = real_engine.embed("all-MiniLM-L6-v2", inputs)

    assert batch.model_id == "all-MiniLM-L6-v2"
    assert batch.input_tokens >= len(inputs)
    assert len(batch.vectors) == len(inputs)
    assert all(len(vector) == 384 for vector in batch.vectors)
    assert all(math.isfinite(value) for vector in batch.vectors for value in vector)
    for vector in batch.vectors:
        assert math.sqrt(sum(value * value for value in vector)) == pytest.approx(
            1.0, abs=2e-5
        )
    assert len(set(batch.vectors)) == len(inputs)


def test_real_model_is_deterministic_and_preserves_input_order(
    real_engine: SentenceTransformerEngine,
) -> None:
    inputs = ["A bread recipe", "A constitutional law excerpt", "A research paper"]

    first = real_engine.embed("all-MiniLM-L6-v2", inputs)
    reversed_batch = real_engine.embed("all-MiniLM-L6-v2", list(reversed(inputs)))
    second = real_engine.embed("all-MiniLM-L6-v2", inputs)

    for left, right in zip(first.vectors, second.vectors, strict=True):
        assert left == pytest.approx(right, abs=1e-6)
    for expected, actual in zip(
        reversed(first.vectors), reversed_batch.vectors, strict=True
    ):
        assert expected == pytest.approx(actual, abs=1e-6)


def test_real_model_has_tolerant_semantic_sanity(
    real_engine: SentenceTransformerEngine,
) -> None:
    batch = real_engine.embed(
        "all-MiniLM-L6-v2",
        [
            "Bake the bread dough in a hot oven.",
            "Put the loaf in the oven until the bread is baked.",
            "Quantum entanglement correlates measurements of distant particles.",
        ],
    )

    related = cosine(batch.vectors[0], batch.vectors[1])
    unrelated = cosine(batch.vectors[0], batch.vectors[2])

    assert related > unrelated + 0.15
    assert related > 0.5


def test_real_model_rejects_overlong_text_without_silent_truncation(
    real_engine: SentenceTransformerEngine,
) -> None:
    from embedding_api.domain.errors import InputTooLargeError

    oversized = "token " * (real_engine.capabilities()[0].max_tokens_per_input + 20)

    with pytest.raises(InputTooLargeError, match="not truncated"):
        real_engine.embed("all-MiniLM-L6-v2", [oversized])


def test_real_model_serves_end_to_end_http_batch_contract(
    real_engine: SentenceTransformerEngine,
) -> None:
    app = create_app(settings=Settings(), engine=real_engine)
    inputs = [
        "A statute establishes a legal obligation.",
        "Simmer the soup over low heat.",
    ]

    with TestClient(app) as client:
        response = client.post(
            "/v1/embeddings",
            json={"model": "all-MiniLM-L6-v2", "input": inputs},
            headers={"x-trace-id": "real-model-integration"},
        )

    assert response.status_code == 200
    assert response.headers["x-trace-id"] == "real-model-integration"
    body = response.json()
    assert body["model"] == "all-MiniLM-L6-v2"
    assert body["dimension"] == 384
    assert body["usage"]["input_count"] == 2
    assert body["usage"]["input_tokens"] >= 2
    assert [item["index"] for item in body["vectors"]] == [0, 1]
    assert all(len(item["embedding"]) == 384 for item in body["vectors"])
