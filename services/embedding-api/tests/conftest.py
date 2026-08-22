from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from embedding_api.adapters.fake import DeterministicFakeEngine
from embedding_api.infrastructure.settings import Settings
from embedding_api.ports.engine import EmbeddingEngine
from embedding_api.transport.http import create_app


def make_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "model_id": "fake-general-purpose",
        "model_source": "unused-in-tests",
        "model_revision": "test-revision",
        "expected_dimension": 8,
        "inference_device": "cpu",
        "model_cache_dir": None,
        "normalize_embeddings": True,
        "engine_batch_size": 4,
        "max_batch_items": 8,
        "max_input_bytes": 1_024,
        "max_total_input_bytes": 4_096,
        "max_batch_tokens": 1_024,
        "log_level": "INFO",
    }
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def fake_engine() -> DeterministicFakeEngine:
    return DeterministicFakeEngine(model_id="fake-general-purpose", dimension=8)


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
def app(settings: Settings, fake_engine: DeterministicFakeEngine) -> FastAPI:
    return create_app(settings=settings, engine=fake_engine)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client
