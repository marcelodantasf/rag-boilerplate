import asyncio
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from fastapi.testclient import TestClient
import yaml

from rag_core.adapters.fakes import FakeEmbeddingGateway, InMemoryCollectionCatalog, InMemoryIdempotencyStore, InMemoryVectorStore
from rag_core.transport.http import create_app
from rag_core.domain.errors import (
    CollectionAlreadyExistsError,
    ConflictError,
    EmbeddingContractError,
    EmbeddingSchemaMismatchError,
    EmbeddingUnavailableError,
    GenerationNotReadyError,
    IdempotencyConflictError,
    InvalidRequestError,
    LimitExceededError,
    NotFoundError,
    PreconditionFailedError,
    VectorStoreUnavailableError,
)
from rag_core.domain.models import ReadinessResult


def client(settings):
    return TestClient(
        create_app(
            settings=settings,
            embedding=FakeEmbeddingGateway(),
            vector_store=InMemoryVectorStore(),
            catalog=InMemoryCollectionCatalog(),
            idempotency=InMemoryIdempotencyStore(),
        )
    )


def collection_request(settings):
    return {
        "collection_id": "employee-handbook",
        "embedding": {
            "model_id": settings.default_embedding_model,
            "revision": settings.embedding_revision,
            "dimension": settings.embedding_dimension,
            "normalized": settings.normalize_embeddings,
            "distance_metric": "cosine",
        },
        "index_schema_version": 1,
        "metadata_fields": [
            {"name": "department", "type": "keyword", "indexed": True},
            {"name": "year", "type": "integer", "indexed": True},
        ],
        "isolation_policy": "shared",
    }


def generation_request(settings):
    return {
        "embedding": {
            "model_id": settings.default_embedding_model,
            "revision": "generation-revision",
            "dimension": settings.embedding_dimension,
            "normalized": settings.normalize_embeddings,
            "distance_metric": "cosine",
        },
        "index_schema_version": 2,
        "metadata_fields": [
            {"name": "department", "type": "keyword", "indexed": True},
            {"name": "year", "type": "integer", "indexed": True},
        ],
    }


def assert_collection_wire(body):
    assert set(body) == {"collection_id", "active_generation_id", "retired", "generations"}
    for generation in body["generations"]:
        assert set(generation) == {
            "generation_id",
            "state",
            "embedding",
            "index_schema_version",
            "metadata_fields",
            "isolation_policy",
            "created_at",
            "activated_at",
            "source_generation_id",
        }
        assert set(generation["embedding"]) == {
            "model_id", "revision", "dimension", "normalized", "distance_metric"
        }


def test_product_level_vertical_slice_and_retirement_safeguards(settings) -> None:
    with client(settings) as api:
        created = api.post("/v1/vector-collections", json=collection_request(settings), headers={"x-trace-id": "trace-create"})
        assert created.status_code == 201
        assert created.headers["x-trace-id"] == "trace-create"
        collection = created.json()
        generation_id = collection["active_generation_id"]
        assert "physical_name" not in created.text

        document = {
            "collection_id": "employee-handbook",
            "document_id": "leave-policy-2026",
            "content": "Parental leave is available for sixteen weeks.",
            "metadata": {"department": "people", "year": 2026},
        }
        indexed = api.post("/v1/rag/documents", json=document, headers={"Idempotency-Key": "request-key-0001"})
        assert indexed.status_code == 201
        assert indexed.json()["generation_id"] == generation_id
        assert indexed.json()["document_version"].startswith("sha256:")

        replay = api.post("/v1/rag/documents", json=document, headers={"Idempotency-Key": "request-key-0001"})
        assert replay.status_code == 200
        assert replay.headers["Idempotency-Replayed"] == "true"
        assert replay.json() == indexed.json()

        retrieval = api.post(
            "/v1/rag/retrievals",
            json={
                "collection_id": "employee-handbook",
                "query": "How long is parental leave?",
                "top_k": 5,
                "filter": {"field": "department", "operator": "eq", "value": "people"},
            },
        )
        assert retrieval.status_code == 200
        assert retrieval.json()["results"][0]["document_id"] == "leave-policy-2026"
        assert "vector" not in retrieval.text
        assert "point_id" not in retrieval.text

        deleted = api.delete("/v1/rag/documents/leave-policy-2026?collection_id=employee-handbook")
        assert deleted.status_code == 200
        assert deleted.json()["chunks_deleted"] == indexed.json()["chunks_indexed"]

        bad_retire = api.delete(
            "/v1/vector-collections/employee-handbook",
            headers={"X-Confirm-Retirement": "wrong", "If-Match": generation_id},
        )
        assert bad_retire.status_code == 412
        assert bad_retire.json()["code"] == "precondition_failed"
        retired = api.delete(
            "/v1/vector-collections/employee-handbook",
            headers={"X-Confirm-Retirement": "employee-handbook", "If-Match": generation_id},
        )
        assert retired.status_code == 200
        assert retired.json()["retired"] is True


def test_openapi_has_only_product_and_collection_operations(settings) -> None:
    with client(settings) as api:
        paths = set(api.get("/openapi.json").json()["paths"])
    assert paths == {
        "/v1/rag/documents",
        "/v1/rag/documents/{document_id}",
        "/v1/rag/retrievals",
        "/v1/vector-collections",
        "/v1/vector-collections/{collection_id}",
        "/v1/vector-collections/{collection_id}/generations",
        "/v1/vector-collections/{collection_id}/activate",
        "/health/live",
        "/health/ready",
    }


def test_validation_uses_safe_error_envelope(settings) -> None:
    with client(settings) as api:
        response = api.post("/v1/rag/retrievals", json={"collection_id": "BAD", "query": ""})
    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"
    assert response.json()["trace_id"] == response.headers["x-trace-id"]
    assert "traceback" not in response.text.lower()


def test_unhandled_error_uses_safe_internal_error_envelope(settings) -> None:
    class FailingEmbedding(FakeEmbeddingGateway):
        async def embed(self, *args, **kwargs):
            raise RuntimeError("provider-secret-response")

    app = create_app(
        settings=settings,
        embedding=FailingEmbedding(),
        vector_store=InMemoryVectorStore(),
        catalog=InMemoryCollectionCatalog(),
        idempotency=InMemoryIdempotencyStore(),
    )
    with TestClient(app, raise_server_exceptions=False) as api:
        assert api.post(
            "/v1/vector-collections", json=collection_request(settings)
        ).status_code == 201
        response = api.post(
            "/v1/rag/retrievals",
            json={"collection_id": "employee-handbook", "query": "question"},
            headers={"x-trace-id": "safe-internal-trace"},
        )

    assert response.status_code == 500
    assert response.json() == {
        "code": "internal_error",
        "message": "An unexpected internal error occurred",
        "trace_id": "safe-internal-trace",
    }
    assert response.headers["x-trace-id"] == "safe-internal-trace"
    assert "provider-secret-response" not in response.text


def test_idempotency_key_reuse_with_different_payload_is_a_conflict(settings) -> None:
    with client(settings) as api:
        assert api.post("/v1/vector-collections", json=collection_request(settings)).status_code == 201
        headers = {"Idempotency-Key": "stable-request-key"}
        first = api.post(
            "/v1/rag/documents",
            headers=headers,
            json={
                "collection_id": "employee-handbook",
                "document_id": "policy",
                "content": "First content",
                "metadata": {},
            },
        )
        conflict = api.post(
            "/v1/rag/documents",
            headers=headers,
            json={
                "collection_id": "employee-handbook",
                "document_id": "policy",
                "content": "Different content",
                "metadata": {},
            },
        )
    assert first.status_code == 201
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "idempotency_conflict"


def test_readiness_dependency_failure_is_bounded_and_safe(settings) -> None:
    class SlowEmbedding(FakeEmbeddingGateway):
        async def ready(self, **kwargs):
            await asyncio.sleep(0.2)
            return ReadinessResult(True)

    class SlowStore(InMemoryVectorStore):
        async def ready(self, **kwargs):
            await asyncio.sleep(0.2)
            return ReadinessResult(True)

    class SlowCatalog(InMemoryCollectionCatalog):
        async def ready(self, **kwargs):
            await asyncio.sleep(0.2)
            return ReadinessResult(True)

    bounded = replace(
        settings,
        readiness_dependency_timeout_seconds=0.01,
        readiness_total_timeout_seconds=0.03,
    )
    app = create_app(
        settings=bounded,
        embedding=SlowEmbedding(),
        vector_store=SlowStore(),
        catalog=SlowCatalog(),
        idempotency=InMemoryIdempotencyStore(),
    )
    started = perf_counter()
    with TestClient(app) as api:
        response = api.get("/health/ready")
    elapsed = perf_counter() - started
    assert elapsed < 0.15
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert response.json()["checks"] == {
        "configuration": {"status": "ready", "code": None},
        "embedding": {"status": "not_ready", "code": "embedding_unavailable"},
        "vector_store": {"status": "not_ready", "code": "vector_store_unavailable"},
        "catalog": {"status": "not_ready", "code": "catalog_unavailable"},
    }


def test_collection_and_generation_idempotency_wire_contract(settings) -> None:
    with client(settings) as api:
        create_headers = {"Idempotency-Key": "create-collection-key"}
        first = api.post("/v1/vector-collections", json=collection_request(settings), headers=create_headers)
        replay = api.post("/v1/vector-collections", json=collection_request(settings), headers=create_headers)
        changed = collection_request(settings)
        changed["index_schema_version"] = 2
        conflict = api.post("/v1/vector-collections", json=changed, headers=create_headers)
        assert first.status_code == 201
        assert replay.status_code == 200
        assert first.json() == replay.json()
        assert first.headers["Idempotency-Replayed"] == "false"
        assert replay.headers["Idempotency-Replayed"] == "true"
        assert first.headers["Location"] == "/v1/vector-collections/employee-handbook"
        assert replay.headers["Location"] == first.headers["Location"]
        assert conflict.status_code == 409
        assert conflict.json()["code"] == "idempotency_conflict"
        assert_collection_wire(first.json())

        generation_headers = {"Idempotency-Key": "provision-generation-key"}
        provisioned = api.post(
            "/v1/vector-collections/employee-handbook/generations",
            json=generation_request(settings),
            headers=generation_headers,
        )
        generation_replay = api.post(
            "/v1/vector-collections/employee-handbook/generations",
            json=generation_request(settings),
            headers=generation_headers,
        )
        assert provisioned.status_code == 201
        assert generation_replay.status_code == 200
        assert provisioned.json() == generation_replay.json()
        assert set(provisioned.json()) == {"collection_id", "generation_id", "state"}
        assert provisioned.json()["state"] == "ready"
        assert provisioned.headers["Location"] == "/v1/vector-collections/employee-handbook"
        assert provisioned.headers["Idempotency-Replayed"] == "false"
        assert generation_replay.headers["Idempotency-Replayed"] == "true"

        invalid = generation_request(settings)
        invalid["validation_queries"] = ["silently ignored before fix"]
        rejected = api.post(
            "/v1/vector-collections/employee-handbook/generations",
            json=invalid,
        )
        assert rejected.status_code == 422
        assert rejected.json()["code"] == "invalid_request"


def test_preconditions_malformed_protocol_and_metadata_limits(settings) -> None:
    with client(settings) as api:
        created = api.post("/v1/vector-collections", json=collection_request(settings)).json()
        active = created["active_generation_id"]
        provisioned = api.post(
            "/v1/vector-collections/employee-handbook/generations",
            json=generation_request(settings),
        ).json()
        stale_activation = api.post(
            "/v1/vector-collections/employee-handbook/activate",
            json={
                "generation_id": provisioned["generation_id"],
                "expected_active_generation_id": "gen_00000000000000000000000000",
            },
        )
        wrong_confirmation = api.delete(
            "/v1/vector-collections/employee-handbook",
            headers={"X-Confirm-Retirement": "wrong", "If-Match": active},
        )
        stale_retirement = api.delete(
            "/v1/vector-collections/employee-handbook",
            headers={
                "X-Confirm-Retirement": "employee-handbook",
                "If-Match": "gen_00000000000000000000000000",
            },
        )
        malformed_timeout = api.post(
            "/v1/rag/retrievals",
            headers={"x-request-timeout-ms": "not-an-int", "x-trace-id": "bad-timeout-trace"},
            json={"collection_id": "employee-handbook", "query": "question"},
        )
        malformed_json = api.post(
            "/v1/rag/retrievals",
            content=b'{"collection_id":',
            headers={"content-type": "application/json"},
        )
        assert stale_activation.status_code == 412
        assert wrong_confirmation.status_code == 412
        assert stale_retirement.status_code == 412
        assert {stale_activation.json()["code"], wrong_confirmation.json()["code"], stale_retirement.json()["code"]} == {"precondition_failed"}
        assert malformed_timeout.status_code == 400
        assert malformed_timeout.json()["code"] == "invalid_request"
        assert malformed_timeout.headers["x-trace-id"] == "bad-timeout-trace"
        assert malformed_json.status_code == 400
        assert malformed_json.json()["code"] == "invalid_request"

        reserved = collection_request(settings)
        reserved["collection_id"] = "reserved-fields"
        reserved["metadata_fields"] = [{"name": "tenant_id", "type": "keyword"}]
        assert api.post("/v1/vector-collections", json=reserved).status_code == 422
        oversized_metadata = api.post(
            "/v1/rag/documents",
            json={
                "collection_id": "employee-handbook",
                "document_id": "doc:1",
                "content": "Content",
                "metadata": {"department": "x" * 600},
            },
        )
        assert oversized_metadata.status_code == 422
        colon_document = api.post(
            "/v1/rag/documents",
            json={
                "collection_id": "employee-handbook",
                "document_id": "doc:1",
                "content": "Content",
                "metadata": {"department": "people"},
            },
        )
        assert colon_document.status_code == 201


def test_collection_paths_schema_version_and_isolation_are_strict(settings) -> None:
    oversized = collection_request(settings)
    oversized["index_schema_version"] = 2147483648
    unsupported_isolation = collection_request(settings)
    unsupported_isolation["isolation_policy"] = "collection_per_tenant"

    with client(settings) as api:
        assert api.post("/v1/vector-collections", json=oversized).status_code == 422
        assert api.post(
            "/v1/vector-collections", json=unsupported_isolation
        ).status_code == 422
        for method, path, kwargs in (
            ("get", "/v1/vector-collections/BAD", {}),
            (
                "post",
                "/v1/vector-collections/BAD/generations",
                {"json": generation_request(settings)},
            ),
            (
                "post",
                "/v1/vector-collections/BAD/activate",
                {
                    "json": {
                        "generation_id": "gen_00000000000000000000000000",
                        "expected_active_generation_id": "gen_00000000000000000000000000",
                    }
                },
            ),
            (
                "delete",
                "/v1/vector-collections/BAD",
                {
                    "headers": {
                        "X-Confirm-Retirement": "BAD",
                        "If-Match": "gen_00000000000000000000000000",
                    }
                },
            ),
        ):
            response = getattr(api, method)(path, **kwargs)
            assert response.status_code == 422
            assert response.json()["code"] == "invalid_request"


def test_runtime_error_codes_and_statuses_are_declared_by_openapi() -> None:
    specification = yaml.safe_load(
        (Path(__file__).parents[3] / "docs/api/rag-core-openapi.yaml").read_text()
    )
    declared_codes = set(
        specification["components"]["schemas"]["Error"]["properties"]["code"]["enum"]
    )
    errors = (
        InvalidRequestError(),
        LimitExceededError(),
        NotFoundError(),
        ConflictError("conflict"),
        CollectionAlreadyExistsError(),
        GenerationNotReadyError(),
        EmbeddingSchemaMismatchError(),
        IdempotencyConflictError(),
        PreconditionFailedError(),
        EmbeddingUnavailableError(),
        EmbeddingUnavailableError(timeout=True),
        EmbeddingContractError(),
        VectorStoreUnavailableError(),
        VectorStoreUnavailableError(timeout=True),
    )
    assert {error.code for error in errors} <= declared_codes
    declared_statuses = {
        int(status)
        for path in specification["paths"].values()
        for operation in path.values()
        if isinstance(operation, dict)
        for status in operation.get("responses", {})
        if status != "default"
    }
    assert {error.status_code for error in errors} | {500} <= declared_statuses


def test_all_success_response_bodies_are_schema_exact(settings) -> None:
    with client(settings) as api:
        live = api.get("/health/live")
        ready = api.get("/health/ready")
        created = api.post("/v1/vector-collections", json=collection_request(settings))
        listed = api.get("/v1/vector-collections")
        inspected = api.get("/v1/vector-collections/employee-handbook")
        assert live.json() == {"status": "alive"}
        assert set(ready.json()) == {"status", "checks"}
        assert set(ready.json()["checks"]) == {"configuration", "embedding", "vector_store", "catalog"}
        assert_collection_wire(created.json())
        assert set(listed.json()) == {"items", "next_cursor"}
        assert_collection_wire(listed.json()["items"][0])
        assert_collection_wire(inspected.json())

        indexed = api.post(
            "/v1/rag/documents",
            json={
                "collection_id": "employee-handbook",
                "document_id": "policy",
                "content": "Parental leave is sixteen weeks.",
                "metadata": {"department": "people", "year": 2026},
            },
        )
        assert set(indexed.json()) == {"collection_id", "generation_id", "document_id", "document_version", "chunks_indexed", "status"}
        retrieved = api.post(
            "/v1/rag/retrievals",
            json={"collection_id": "employee-handbook", "query": "parental leave"},
        )
        assert set(retrieved.json()) == {"collection_id", "generation_id", "query", "top_k", "results"}
        assert set(retrieved.json()["results"][0]) == {"rank", "chunk_id", "document_id", "document_version", "text", "score", "metadata"}
        deleted = api.delete("/v1/rag/documents/policy?collection_id=employee-handbook")
        assert set(deleted.json()) == {"collection_id", "generation_id", "document_id", "chunks_deleted", "status"}

        provisioned = api.post(
            "/v1/vector-collections/employee-handbook/generations",
            json=generation_request(settings),
        )
        assert provisioned.status_code == 201
        assert set(provisioned.json()) == {"collection_id", "generation_id", "state"}
        activated = api.post(
            "/v1/vector-collections/employee-handbook/activate",
            json={
                "generation_id": provisioned.json()["generation_id"],
                "expected_active_generation_id": created.json()["active_generation_id"],
            },
        )
        assert activated.status_code == 200
        assert_collection_wire(activated.json())
        assert activated.json()["active_generation_id"] == provisioned.json()["generation_id"]
        retired = api.delete(
            "/v1/vector-collections/employee-handbook",
            headers={
                "X-Confirm-Retirement": "employee-handbook",
                "If-Match": provisioned.json()["generation_id"],
            },
        )
        assert set(retired.json()) == {"collection_id", "retired", "retained_until"}
        retired_inspection = api.get("/v1/vector-collections/employee-handbook")
        assert retired_inspection.status_code == 200
        assert_collection_wire(retired_inspection.json())
        assert retired_inspection.json()["retired"] is True
        assert {item["state"] for item in retired_inspection.json()["generations"]} == {"retired"}
