"""FastAPI controllers for product-level RAG and collection operations."""

import hashlib
import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Annotated, Any, Literal
from uuid import uuid4

from fastapi import FastAPI, Header, Path, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rag_core.adapters.catalog import QdrantCollectionCatalog
from rag_core.adapters.embedding.http import HttpEmbeddingGateway
from rag_core.adapters.fakes import InMemoryIdempotencyStore
from rag_core.adapters.vector_store import QdrantVectorStore
from rag_core.application.collections import CollectionService
from rag_core.application.rag import RagService
from rag_core.domain.errors import (
    CoreError,
    IdempotencyConflictError,
    InvalidRequestError,
    PreconditionFailedError,
)
from rag_core.domain.models import (
    CollectionContract,
    DistanceMetric,
    EmbeddingContract,
    FilterCondition,
    FilterExpression,
    FilterGroup,
    FilterGroupOperator,
    FilterOperator,
    MetadataField,
    MetadataFieldType,
    RESERVED_METADATA_FIELDS,
    TraceContext,
    safe_metadata,
)
from rag_core.infrastructure.settings import Settings
from rag_core.infrastructure.observability import configure_telemetry
from rag_core.infrastructure.instruments import (
    idempotency_operations,
    safe_attributes,
    validation_rejections,
)
from rag_core.ports.embedding import EmbeddingGateway
from rag_core.ports.idempotency import IdempotencyState, IdempotencyStore
from rag_core.ports.vector_store import CollectionCatalog, VectorStore


_TRACE_ID = re.compile(r"[A-Za-z0-9._:-]{1,64}")
_TRACEPARENT = re.compile(r"[\da-f]{2}-([\da-f]{32})-[\da-f]{16}-[\da-f]{2}")
_IDEMPOTENCY_KEY = re.compile(r"[!-~]{8,128}")
logger = logging.getLogger("rag_core.http")
CollectionIdPath = Annotated[str, Path(pattern=r"^[a-z][a-z0-9-]{2,62}$")]
DocumentIdPath = Annotated[
    str, Path(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class EmbeddingSpec(StrictModel):
    model_id: str = Field(min_length=1, max_length=128)
    revision: str = Field(min_length=7, max_length=128)
    dimension: int = Field(ge=1, le=65536)
    normalized: bool
    distance_metric: DistanceMetric

    @model_validator(mode="after")
    def dot_requires_normalized_embeddings(self):
        if self.distance_metric is DistanceMetric.DOT and not self.normalized:
            raise ValueError("dot distance requires normalized embeddings")
        return self


class MetadataFieldSpec(StrictModel):
    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
    type: MetadataFieldType
    indexed: bool = True

    @field_validator("name")
    @classmethod
    def name_is_not_reserved(cls, value: str) -> str:
        if value in RESERVED_METADATA_FIELDS or value.startswith("_"):
            raise ValueError("metadata field name is reserved")
        return value


class CreateCollectionRequest(StrictModel):
    collection_id: str = Field(pattern=r"^[a-z][a-z0-9-]{2,62}$")
    embedding: EmbeddingSpec
    index_schema_version: int = Field(ge=1, le=2147483647)
    metadata_fields: list[MetadataFieldSpec] = Field(max_length=32)
    isolation_policy: Literal["shared"]

    @field_validator("metadata_fields")
    @classmethod
    def metadata_fields_are_unique(cls, value: list[MetadataFieldSpec]):
        if len({item.name for item in value}) != len(value):
            raise ValueError("metadata field names must be unique")
        return value


class GenerationProvisionRequest(StrictModel):
    embedding: EmbeddingSpec
    index_schema_version: int = Field(ge=1, le=2147483647)
    metadata_fields: list[MetadataFieldSpec] = Field(max_length=32)

    @field_validator("metadata_fields")
    @classmethod
    def metadata_fields_are_unique(cls, value: list[MetadataFieldSpec]):
        if len({item.name for item in value}) != len(value):
            raise ValueError("metadata field names must be unique")
        return value


class ActivationRequest(StrictModel):
    generation_id: str = Field(pattern=r"^gen_[0-9A-HJKMNP-TV-Z]{26}$")
    expected_active_generation_id: str = Field(pattern=r"^gen_[0-9A-HJKMNP-TV-Z]{26}$")


class IngestRequest(StrictModel):
    collection_id: str = Field(pattern=r"^[a-z][a-z0-9-]{2,62}$")
    document_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
    content: str = Field(min_length=1, max_length=262144)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict, max_length=32)

    @field_validator("metadata")
    @classmethod
    def metadata_is_safe(cls, value: dict[str, Any]):
        try:
            return safe_metadata(value)
        except ValueError as error:
            raise ValueError(str(error)) from error


class FilterModel(StrictModel):
    all_: list["FilterModel"] | None = Field(default=None, alias="all", max_length=10)
    any_: list["FilterModel"] | None = Field(default=None, alias="any", max_length=10)
    not_: "FilterModel | None" = Field(default=None, alias="not")
    field: str | None = Field(default=None, pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
    operator: FilterOperator | None = None
    value: str | int | float | bool | list[str | int | float | bool] | None = None

    @model_validator(mode="after")
    def exactly_one_expression(self):
        groups = sum(item is not None for item in (self.all_, self.any_, self.not_))
        predicate = self.field is not None or self.operator is not None or self.value is not None
        if groups + int(predicate) != 1:
            raise ValueError("filter must contain exactly one all, any, not, or predicate expression")
        if predicate and (self.field is None or self.operator is None or self.value is None):
            raise ValueError("filter predicates require field, operator, and value")
        return self

    def domain(self) -> FilterExpression:
        if self.all_ is not None:
            return FilterGroup(FilterGroupOperator.ALL, tuple(item.domain() for item in self.all_))
        if self.any_ is not None:
            return FilterGroup(FilterGroupOperator.ANY, tuple(item.domain() for item in self.any_))
        if self.not_ is not None:
            return FilterGroup(FilterGroupOperator.NOT, (self.not_.domain(),))
        value = tuple(self.value) if isinstance(self.value, list) else self.value
        return FilterCondition(self.field or "", self.operator or FilterOperator.EQ, value)  # type: ignore[arg-type]


class RetrievalRequest(StrictModel):
    collection_id: str = Field(pattern=r"^[a-z][a-z0-9-]{2,62}$")
    query: str = Field(min_length=1, max_length=8192)
    top_k: int = Field(default=10, ge=1, le=100)
    minimum_score: float = Field(default=0, ge=0, le=1)
    filter: FilterModel | None = None


def create_app(
    *,
    settings: Settings | None = None,
    embedding: EmbeddingGateway | None = None,
    vector_store: VectorStore | None = None,
    catalog: CollectionCatalog | None = None,
    idempotency: IdempotencyStore | None = None,
) -> FastAPI:
    runtime = settings or Settings.from_env()
    runtime.validate()
    logging.basicConfig(level=runtime.log_level, format="%(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        embed = embedding or HttpEmbeddingGateway(
            base_url=runtime.embedding_base_url,
            model_id=runtime.default_embedding_model,
            revision=runtime.embedding_revision,
            expected_dimension=runtime.embedding_dimension,
            normalized=runtime.normalize_embeddings,
        )
        store = vector_store or QdrantVectorStore(runtime.vector_db_url, api_key=runtime.vector_db_api_key)
        cat = catalog or QdrantCollectionCatalog(runtime.vector_db_url, api_key=runtime.vector_db_api_key)
        idem = idempotency or InMemoryIdempotencyStore()
        app.state.embedding = embed
        app.state.vector_store = store
        app.state.catalog = cat
        app.state.idempotency = idem
        app.state.rag = RagService(catalog=cat, vector_store=store, embedding=embed, settings=runtime)
        app.state.collections = CollectionService(cat, store)
        yield
        await embed.close()
        await store.close()
        await cat.close()
        await idem.close()
        if app.state.telemetry is not None:
            app.state.telemetry.shutdown()

    app = FastAPI(title="RAG Core API", version="1.0.0", lifespan=lifespan)
    app.state.telemetry = configure_telemetry(app, runtime)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        started = perf_counter()
        trace_context = _trace_context(request)
        request.state.trace_context = trace_context
        status = 500
        try:
            try:
                request.state.deadline_seconds = _deadline(request)
            except CoreError as error:
                response = _error(
                    request,
                    400,
                    error.code,
                    error.message,
                    error.details,
                )
                status = response.status_code
                response.headers["x-trace-id"] = trace_context.trace_id
                return response
            response = await call_next(request)
            status = response.status_code
            response.headers["x-trace-id"] = trace_context.trace_id
            return response
        except Exception as error:
            status = 500
            logger.error(json.dumps({
                "event": "unhandled_http_error",
                "trace_id": trace_context.trace_id,
                "error_type": type(error).__name__,
            }, separators=(",", ":")))
            response = _error(
                request,
                500,
                "internal_error",
                "An unexpected internal error occurred",
            )
            response.headers["x-trace-id"] = trace_context.trace_id
            return response
        finally:
            route = request.scope.get("route")
            logger.info(json.dumps({
                "event": "http_request_completed",
                "trace_id": trace_context.trace_id,
                "method": request.method,
                "route": getattr(route, "path", "<unmatched>"),
                "status": status,
                "latency_ms": round((perf_counter() - started) * 1000, 3),
                "health_check": request.url.path.startswith("/health/"),
            }, separators=(",", ":")))
            if request.app.state.telemetry is not None:
                attributes = {
                    "http.request.method": request.method,
                    "http.route": getattr(route, "path", "<unmatched>"),
                    "http.response.status_code": status,
                }
                request.app.state.telemetry.request_count.add(1, attributes)
                request.app.state.telemetry.request_duration.record((perf_counter() - started) * 1000, attributes)

    @app.exception_handler(CoreError)
    async def core_error(request: Request, error: CoreError):
        if error.code in {"invalid_request", "limit_exceeded", "embedding_schema_mismatch"}:
            validation_rejections.add(
                1, safe_attributes(error_code=error.code, operation="http_request")
            )
        return _error(request, error.status_code, error.code, error.message, error.details)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError):
        validation_rejections.add(
            1, safe_attributes(error_code="invalid_request", operation="http_request")
        )
        details = {"fields": [{"path": ".".join(str(item) for item in issue["loc"] if item != "body"), "reason": issue["msg"]} for issue in error.errors()]}
        status_code = 400 if any(issue["type"] == "json_invalid" for issue in error.errors()) else 422
        return _error(request, status_code, "invalid_request", "The request is invalid.", details)

    @app.get("/health/live", tags=["Health"])
    async def live():
        return {"status": "alive"}

    @app.get("/health/ready", tags=["Health"])
    async def ready(request: Request):
        contract = _configured_embedding(runtime)
        context = request.state.trace_context
        dependency_timeout = runtime.readiness_dependency_timeout_seconds
        try:
            async with asyncio.timeout(runtime.readiness_total_timeout_seconds):
                outcomes = await asyncio.gather(
                    request.app.state.embedding.ready(
                        contract=contract,
                        deadline_seconds=dependency_timeout,
                        trace_context=context,
                    ),
                    request.app.state.vector_store.ready(
                        deadline_seconds=dependency_timeout
                    ),
                    request.app.state.catalog.ready(
                        deadline_seconds=dependency_timeout
                    ),
                    return_exceptions=True,
                )
        except TimeoutError:
            outcomes = (TimeoutError(), TimeoutError(), TimeoutError())
        checks = {
            "configuration": {"status": "ready", "code": None},
            "embedding": _safe_check(outcomes[0], "embedding_unavailable"),
            "vector_store": _safe_check(outcomes[1], "vector_store_unavailable"),
            "catalog": _safe_check(outcomes[2], "catalog_unavailable"),
        }
        is_ready = all(item["status"] == "ready" for item in checks.values())
        return JSONResponse(status_code=200 if is_ready else 503, content={"status": "ready" if is_ready else "not_ready", "checks": checks})

    @app.post("/v1/rag/documents", status_code=201, tags=["RAG operations"])
    async def ingest(request: Request, payload: IngestRequest, response: Response, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
        async def execute():
            result = await request.app.state.rag.ingest(
                collection_id=payload.collection_id,
                document_id=payload.document_id,
                text=payload.content,
                metadata=payload.metadata,
                trace_context=request.state.trace_context,
                deadline_seconds=request.state.deadline_seconds,
            )
            return result.__dict__ if hasattr(result, "__dict__") else {name: getattr(result, name) for name in result.__dataclass_fields__}
        body, replayed = await _idempotent(request, "ingestDocument", idempotency_key, payload.model_dump(mode="json"), execute, 201)
        response.headers["Idempotency-Replayed"] = str(replayed).lower()
        if replayed:
            response.status_code = 200
        return body

    @app.delete("/v1/rag/documents/{document_id}", tags=["RAG operations"])
    async def delete_document(request: Request, document_id: DocumentIdPath, collection_id: str = Query(pattern=r"^[a-z][a-z0-9-]{2,62}$")):
        result = await request.app.state.rag.delete(collection_id=collection_id, document_id=document_id, deadline_seconds=request.state.deadline_seconds)
        return {name: getattr(result, name) for name in result.__dataclass_fields__}

    @app.post("/v1/rag/retrievals", tags=["RAG operations"])
    async def retrieve(request: Request, payload: RetrievalRequest):
        result = await request.app.state.rag.retrieve(
            collection_id=payload.collection_id,
            query=payload.query,
            top_k=payload.top_k,
            filter=payload.filter.domain() if payload.filter else None,
            minimum_score=payload.minimum_score,
            trace_context=request.state.trace_context,
            deadline_seconds=request.state.deadline_seconds,
        )
        return _retrieval_response(result)

    @app.post("/v1/vector-collections", status_code=201, tags=["Vector collections"], operation_id="createVectorCollection")
    async def create_collection(request: Request, payload: CreateCollectionRequest, response: Response, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
        async def execute():
            contract = await request.app.state.collections.create(
                collection_id=payload.collection_id,
                embedding=_embedding(payload.embedding),
                index_schema_version=payload.index_schema_version,
                metadata_fields=_metadata_fields(payload.metadata_fields),
                isolation_policy=payload.isolation_policy,
                deadline_seconds=request.state.deadline_seconds,
            )
            return await _collection_response(request.app.state.collections, contract.collection_id)
        body, replayed = await _idempotent(
            request,
            "createVectorCollection",
            idempotency_key,
            payload.model_dump(mode="json"),
            execute,
            201,
        )
        response.headers["Location"] = f"/v1/vector-collections/{payload.collection_id}"
        response.headers["Idempotency-Replayed"] = str(replayed).lower()
        if replayed:
            response.status_code = 200
        return body

    @app.get("/v1/vector-collections", tags=["Vector collections"])
    async def list_collections(request: Request, limit: int = Query(default=50, ge=1, le=100), cursor: str | None = Query(default=None, max_length=512)):
        page = await request.app.state.collections.list(limit=limit, cursor=cursor)
        items = [await _collection_response(request.app.state.collections, item.collection_id) for item in page.items]
        return {"items": items, "next_cursor": page.next_cursor}

    @app.get("/v1/vector-collections/{collection_id}", tags=["Vector collections"])
    async def inspect_collection(request: Request, collection_id: CollectionIdPath):
        return await _collection_response(request.app.state.collections, collection_id)

    @app.post("/v1/vector-collections/{collection_id}/generations", status_code=201, tags=["Vector collections"], operation_id="provisionVectorCollectionGeneration")
    async def provision_generation(request: Request, response: Response, collection_id: CollectionIdPath, payload: GenerationProvisionRequest, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
        async def execute():
            contract = await request.app.state.collections.provision_generation(
                collection_id=collection_id,
                embedding=_embedding(payload.embedding),
                index_schema_version=payload.index_schema_version,
                metadata_fields=_metadata_fields(payload.metadata_fields),
                deadline_seconds=request.state.deadline_seconds,
            )
            return {
                "collection_id": collection_id,
                "generation_id": contract.generation_id,
                "state": contract.state,
            }
        identity = {"collection_id": collection_id, **payload.model_dump(mode="json")}
        body, replayed = await _idempotent(
            request,
            "provisionVectorCollectionGeneration",
            idempotency_key,
            identity,
            execute,
            201,
        )
        response.headers["Location"] = f"/v1/vector-collections/{collection_id}"
        response.headers["Idempotency-Replayed"] = str(replayed).lower()
        if replayed:
            response.status_code = 200
        return body

    @app.post("/v1/vector-collections/{collection_id}/activate", tags=["Vector collections"])
    async def activate(request: Request, collection_id: CollectionIdPath, payload: ActivationRequest):
        await request.app.state.collections.activate(collection_id, payload.generation_id, payload.expected_active_generation_id, request.state.deadline_seconds)
        return await _collection_response(request.app.state.collections, collection_id)

    @app.delete("/v1/vector-collections/{collection_id}", tags=["Vector collections"])
    async def retire(request: Request, collection_id: CollectionIdPath, x_confirm_retirement: str = Header(alias="X-Confirm-Retirement"), if_match: str = Header(alias="If-Match")):
        if x_confirm_retirement != collection_id:
            raise PreconditionFailedError("X-Confirm-Retirement does not match collection_id")
        retained_until = await request.app.state.collections.retire(collection_id, if_match, request.state.deadline_seconds)
        return {"collection_id": collection_id, "retired": True, "retained_until": retained_until.isoformat()}

    return app


def _trace_context(request: Request) -> TraceContext:
    parent = request.headers.get("traceparent")
    match = _TRACEPARENT.fullmatch(parent or "")
    legacy = request.headers.get("x-trace-id", "")
    trace_id = match.group(1) if match else legacy if _TRACE_ID.fullmatch(legacy) else uuid4().hex
    return TraceContext(trace_id, parent if match else None, request.headers.get("tracestate") if match else None)


def _deadline(request: Request) -> float:
    try:
        milliseconds = int(request.headers.get("x-request-timeout-ms", "10000"))
    except ValueError as error:
        raise InvalidRequestError("x-request-timeout-ms must be an integer") from error
    return max(100, min(30000, milliseconds)) / 1000


def _error(request: Request, status: int, code: str, message: str, details: dict | None = None):
    body = {"code": code, "message": message, "trace_id": request.state.trace_context.trace_id}
    if details:
        body["details"] = details
    return JSONResponse(status_code=status, content=body)


def _embedding(spec: EmbeddingSpec) -> EmbeddingContract:
    return EmbeddingContract(spec.model_id, spec.revision, spec.dimension, spec.normalized, spec.distance_metric)


def _metadata_fields(fields: list[MetadataFieldSpec]) -> tuple[MetadataField, ...]:
    return tuple(MetadataField(item.name, item.type, item.indexed) for item in fields)


def _configured_embedding(settings: Settings) -> EmbeddingContract:
    return EmbeddingContract(settings.default_embedding_model, settings.embedding_revision, settings.embedding_dimension, settings.normalize_embeddings, settings.distance_metric)


def _check(result):
    return {"status": "ready" if result.ready else "not_ready", "code": result.code}


def _safe_check(result: object, fallback_code: str):
    if isinstance(result, BaseException):
        return {"status": "not_ready", "code": fallback_code}
    return _check(result)


async def _collection_response(service: CollectionService, collection_id: str):
    generations = await service.inspect(collection_id)
    active = next((item for item in generations if item.state.value == "active"), None)
    return {
        "collection_id": collection_id,
        "active_generation_id": active.generation_id if active else None,
        "retired": active is None and all(item.state.value == "retired" for item in generations),
        "generations": [_generation_response(item) for item in generations],
    }


def _generation_response(item: CollectionContract):
    return {
        "generation_id": item.generation_id,
        "state": item.state,
        "embedding": {"model_id": item.embedding.model_id, "revision": item.embedding.revision, "dimension": item.embedding.dimension, "normalized": item.embedding.normalized, "distance_metric": item.embedding.distance_metric},
        "index_schema_version": item.index_schema_version,
        "metadata_fields": [{"name": field.name, "type": field.type, "indexed": field.indexed} for field in item.metadata_fields],
        "isolation_policy": item.isolation_policy,
        "created_at": item.created_at.isoformat(),
        "activated_at": item.activated_at.isoformat() if item.activated_at else None,
        "source_generation_id": item.source_generation_id,
    }


def _retrieval_response(result):
    return {
        "collection_id": result.collection_id,
        "generation_id": result.generation_id,
        "query": result.query,
        "top_k": result.top_k,
        "results": [{name: getattr(hit, name) for name in hit.__dataclass_fields__} for hit in result.results],
    }


async def _idempotent(request: Request, scope: str, key: str | None, payload: dict, execute, status: int):
    if key is None:
        return await execute(), False
    if _IDEMPOTENCY_KEY.fullmatch(key) is None:
        raise InvalidRequestError("Idempotency-Key must contain 8 to 128 printable characters")
    request_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    store = request.app.state.idempotency
    result = await store.begin(f"local:{scope}", key, request_hash, 86400)
    if result.state == IdempotencyState.REPLAY:
        idempotency_operations.add(1, safe_attributes(operation=scope, phase="replay"))
        return result.response or {}, True
    if result.state == IdempotencyState.CONFLICT:
        idempotency_operations.add(1, safe_attributes(operation=scope, phase="conflict"))
        raise IdempotencyConflictError()
    idempotency_operations.add(1, safe_attributes(operation=scope, phase="begin"))
    try:
        body = await execute()
    except Exception:
        await store.abandon(f"local:{scope}", key)
        idempotency_operations.add(1, safe_attributes(operation=scope, phase="abandon"))
        raise
    await store.complete(f"local:{scope}", key, status, body)
    idempotency_operations.add(1, safe_attributes(operation=scope, phase="complete"))
    return body, False
