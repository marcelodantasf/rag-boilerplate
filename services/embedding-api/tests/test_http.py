from __future__ import annotations

import math
import json
import logging
import re

import pytest
from fastapi.testclient import TestClient

from conftest import make_settings
from embedding_api.adapters.fake import DeterministicFakeEngine
from embedding_api.domain.models import EmbeddingBatch, ModelCapability
from embedding_api.transport.http import create_app
from embedding_api.infrastructure.observability import REQUEST_LOGGER_NAME


MODEL = "fake-general-purpose"


def assert_success_contract(body: dict[str, object], count: int) -> None:
    assert set(body) == {"model", "dimension", "vectors", "usage"}
    assert body["model"] == MODEL
    assert body["dimension"] == 8
    vectors = body["vectors"]
    assert isinstance(vectors, list)
    assert len(vectors) == count
    for index, vector in enumerate(vectors):
        assert set(vector) == {"index", "embedding"}
        assert vector["index"] == index
        assert len(vector["embedding"]) == 8
        assert all(
            isinstance(value, float) and math.isfinite(value)
            for value in vector["embedding"]
        )
    usage = body["usage"]
    assert set(usage) == {"input_count", "input_tokens"}
    assert usage["input_count"] == count
    assert isinstance(usage["input_tokens"], int)


def assert_structured_error(response, status: int, code: str) -> dict[str, object]:
    assert response.status_code == status
    body = response.json()
    assert body["code"] == code
    assert isinstance(body["message"], str) and body["message"]
    assert re.fullmatch(r"[0-9a-f]{32}", body["trace_id"])
    assert response.headers["x-trace-id"] == body["trace_id"]
    return body


def test_single_document_is_normalized_to_one_vector(client, fake_engine) -> None:
    response = client.post(
        "/v1/embeddings",
        json={"model": MODEL, "input": "Any free-form document."},
    )

    assert response.status_code == 200
    assert_success_contract(response.json(), 1)
    assert fake_engine.embed_calls == [
        (MODEL, ("Any free-form document.",)),
    ]


def test_list_is_sent_to_engine_in_one_ordered_batch(client, fake_engine) -> None:
    inputs = ["third? no, first", "second", "last"]

    response = client.post(
        "/v1/embeddings", json={"model": MODEL, "input": inputs}
    )

    assert response.status_code == 200
    assert_success_contract(response.json(), 3)
    assert fake_engine.embed_calls == [(MODEL, tuple(inputs))]
    returned = response.json()["vectors"]
    expected = [list(fake_engine._vector(value)) for value in inputs]
    assert [item["embedding"] for item in returned] == expected


@pytest.mark.parametrize(
    "text",
    [
        "Art. 5º: Todos são iguais perante a lei.",
        "Whisk eggs, fold in flour, and bake at 180°C.",
        "We estimate the learning rate using a held-out validation set.",
        "def greet(name: str) -> str:\n    return f'Olá, {name} 👋'",
        "教育は未来への架け橋です。 مرحباً بالعالم 🌍",
        "First paragraph.\n\nSecond paragraph with\ttabs and punctuation!",
    ],
    ids=["law", "recipe", "paper", "code", "unicode", "multiline"],
)
def test_arbitrary_text_domains_follow_the_same_contract(client, text: str) -> None:
    response = client.post(
        "/v1/embeddings", json={"model": MODEL, "input": text}
    )

    assert response.status_code == 200
    assert_success_contract(response.json(), 1)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"model": MODEL},
        {"input": "text"},
        {"model": MODEL, "input": None},
        {"model": MODEL, "input": 5},
        {"model": MODEL, "input": ["valid", 5]},
        {"model": MODEL, "input": {"text": "structured"}},
        {"model": MODEL, "input": "text", "input_type": "document"},
        {"model": MODEL, "input": "text", "metadata": {}},
        {"model": 5, "input": "text"},
        {"model": "   ", "input": "text"},
    ],
    ids=[
        "empty-object",
        "missing-input",
        "missing-model",
        "null-input",
        "numeric-input",
        "mixed-list",
        "structured-object",
        "input-type-not-supported",
        "metadata-not-supported",
        "numeric-model",
        "blank-model",
    ],
)
def test_invalid_request_shapes_return_safe_errors(client, payload) -> None:
    response = client.post("/v1/embeddings", json=payload)

    assert_structured_error(response, 422, "invalid_input")


@pytest.mark.parametrize("input_value", ["", "   ", [], ["ok", ""]])
def test_empty_inputs_are_rejected(client, input_value) -> None:
    response = client.post(
        "/v1/embeddings", json={"model": MODEL, "input": input_value}
    )

    assert_structured_error(response, 422, "invalid_input")


def test_batch_item_limit_is_reported() -> None:
    settings = make_settings(max_batch_items=2)
    app = create_app(settings=settings, engine=DeterministicFakeEngine(MODEL, 8))

    with TestClient(app) as client:
        response = client.post(
            "/v1/embeddings", json={"model": MODEL, "input": ["a", "b", "c"]}
        )

    body = assert_structured_error(response, 422, "invalid_input")
    assert body["details"] == {"max_batch_items": 2}


def test_per_item_utf8_byte_limit_is_reported() -> None:
    settings = make_settings(max_input_bytes=4, max_total_input_bytes=20)
    app = create_app(settings=settings, engine=DeterministicFakeEngine(MODEL, 8))

    with TestClient(app) as client:
        accepted = client.post(
            "/v1/embeddings", json={"model": MODEL, "input": "éé"}
        )
        response = client.post(
            "/v1/embeddings", json={"model": MODEL, "input": "ééa"}
        )

    assert accepted.status_code == 200
    body = assert_structured_error(response, 413, "input_too_large")
    assert body["details"] == {
        "index": 0,
        "actual_bytes": 5,
        "max_input_bytes": 4,
    }


def test_total_utf8_byte_limit_is_reported() -> None:
    settings = make_settings(max_input_bytes=4, max_total_input_bytes=5)
    app = create_app(settings=settings, engine=DeterministicFakeEngine(MODEL, 8))

    with TestClient(app) as client:
        response = client.post(
            "/v1/embeddings", json={"model": MODEL, "input": ["abc", "def"]}
        )

    body = assert_structured_error(response, 413, "input_too_large")
    assert body["details"] == {
        "actual_bytes": 6,
        "max_total_input_bytes": 5,
    }


def test_unsupported_model_has_structured_error_and_preserves_trace_id(client) -> None:
    trace_id = "rag-core-request-123"

    response = client.post(
        "/v1/embeddings",
        json={"model": "unknown", "input": "text"},
        headers={"x-trace-id": trace_id},
    )

    assert response.status_code == 404
    assert response.json() == {
        "code": "unsupported_model",
        "message": "The requested model is not supported.",
        "trace_id": trace_id,
        "details": {"model": "unknown"},
    }
    assert response.headers["x-trace-id"] == trace_id


@pytest.mark.parametrize("trace_id", ["", b"\xff", "x" * 129])
def test_invalid_incoming_trace_id_is_replaced(client, trace_id: str | bytes) -> None:
    response = client.get("/health/live", headers={"x-trace-id": trace_id})

    assert response.status_code == 200
    assert re.fullmatch(r"[0-9a-f]{32}", response.headers["x-trace-id"])


def test_health_endpoints_report_live_and_ready(client) -> None:
    live = client.get("/health/live")
    ready = client.get("/health/ready")

    assert live.status_code == 200
    assert live.json() == {"status": "live"}
    assert ready.status_code == 200
    assert ready.json() == {"status": "ready", "model": MODEL}
    assert "x-trace-id" in live.headers
    assert "x-trace-id" in ready.headers


def test_readiness_is_structured_503_before_lifespan_starts(settings, fake_engine) -> None:
    app = create_app(settings=settings, engine=fake_engine)
    client = TestClient(app)

    response = client.get("/health/ready")

    assert_structured_error(response, 503, "embedding_engine_unavailable")


class BrokenEngine:
    def __init__(self, batch: EmbeddingBatch):
        self.batch = batch

    def capabilities(self) -> tuple[ModelCapability, ...]:
        return (
            ModelCapability(
                model_id=MODEL,
                revision="broken",
                dimension=2,
                max_tokens_per_input=20,
                normalized=False,
            ),
        )

    def warmup(self, model_id: str) -> None:
        return None

    def embed(self, model_id: str, inputs: list[str]) -> EmbeddingBatch:
        return self.batch


@pytest.mark.parametrize(
    "batch",
    [
        EmbeddingBatch("wrong", ((0.1, 0.2),), 1),
        EmbeddingBatch(MODEL, (), 1),
        EmbeddingBatch(MODEL, ((0.1,),), 1),
        EmbeddingBatch(MODEL, ((0.1, math.nan),), 1),
        EmbeddingBatch(MODEL, ((0.1, "bad"),), 1),
    ],
    ids=["model", "count", "dimension", "nonfinite", "nonnumeric"],
)
def test_contract_violations_are_safe_structured_500_errors(batch) -> None:
    settings = make_settings(expected_dimension=2)
    app = create_app(settings=settings, engine=BrokenEngine(batch))

    with TestClient(app) as client:
        response = client.post(
            "/v1/embeddings", json={"model": MODEL, "input": "text"}
        )

    body = assert_structured_error(response, 500, "embedding_contract_violation")
    assert body["message"] == "The embedding engine returned an invalid result."
    assert "details" not in body


class RecordCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_request_log_has_safe_embedding_metadata(client) -> None:
    logger = logging.getLogger(REQUEST_LOGGER_NAME)
    collector = RecordCollector()
    logger.addHandler(collector)
    secret_text = "do-not-log-this-document"
    try:
        response = client.post(
            "/v1/embeddings",
            json={"model": MODEL, "input": [secret_text, "second"]},
            headers={"x-trace-id": "safe-trace-42"},
        )
    finally:
        logger.removeHandler(collector)

    assert response.status_code == 200
    event = json.loads(collector.records[-1].getMessage())
    assert event == {
        "device": "cpu",
        "duration_ms": event["duration_ms"],
        "event": "http_request_completed",
        "health_check": False,
        "input_bytes": len(secret_text.encode()) + len("second".encode()),
        "input_count": 2,
        "input_tokens": 2,
        "method": "POST",
        "model": MODEL,
        "model_revision": "test-revision",
        "path": "/v1/embeddings",
        "status_code": 200,
        "trace_id": "safe-trace-42",
    }
    serialized = collector.records[-1].getMessage()
    assert secret_text not in serialized
    assert "input" not in event
    assert "vectors" not in event


def test_health_and_error_requests_are_identifiable_in_logs(client) -> None:
    logger = logging.getLogger(REQUEST_LOGGER_NAME)
    collector = RecordCollector()
    logger.addHandler(collector)
    try:
        health_response = client.get("/health/live")
        error_response = client.post(
            "/v1/embeddings", json={"model": "unknown", "input": "text"}
        )
    finally:
        logger.removeHandler(collector)

    assert health_response.status_code == 200
    assert error_response.status_code == 404
    health_event, error_event = [
        json.loads(record.getMessage()) for record in collector.records[-2:]
    ]
    assert health_event["health_check"] is True
    assert health_event["status_code"] == 200
    assert error_event["health_check"] is False
    assert error_event["status_code"] == 404
    assert error_event["error_code"] == "unsupported_model"
    assert error_event["model"] == "unsupported"
    assert "unknown" not in collector.records[-1].getMessage()
