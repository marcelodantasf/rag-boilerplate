"""FastAPI transport, request normalization, and safe error mapping."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import re
from time import perf_counter
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from embedding_api.adapters.sentence_transformer import SentenceTransformerEngine
from embedding_api.application.embed import EmbedTexts
from embedding_api.domain.errors import (
    EmbeddingEngineUnavailableError,
    EmbeddingError,
    InvalidInputError,
)
from embedding_api.infrastructure.settings import Settings
from embedding_api.infrastructure.observability import (
    configure_request_logging,
    log_request,
)
from embedding_api.ports.engine import EmbeddingEngine


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    model: StrictStr = Field(min_length=1)
    input: StrictStr | list[StrictStr]

    @field_validator("model")
    @classmethod
    def model_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model must not be blank")
        return value

    def normalized_inputs(self) -> list[str]:
        return [self.input] if isinstance(self.input, str) else self.input


class VectorResponse(BaseModel):
    index: int
    embedding: list[float]


class UsageResponse(BaseModel):
    input_count: int
    input_tokens: int


class EmbeddingResponse(BaseModel):
    model: str
    dimension: int
    vectors: list[VectorResponse]
    usage: UsageResponse


def _trace_id(request: Request) -> str:
    return getattr(request.state, "trace_id", uuid4().hex)


def _valid_trace_id(value: str) -> bool:
    return re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", value) is not None


def _error_response(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    body: dict[str, Any] = {
        "code": code,
        "message": message,
        "trace_id": _trace_id(request),
    }
    if details:
        body["details"] = details
    request.state.error_code = code
    return JSONResponse(status_code=status_code, content=body)


def create_app(
    *, settings: Settings | None = None, engine: EmbeddingEngine | None = None
) -> FastAPI:
    runtime_settings = settings or Settings.from_env()
    runtime_settings.validate()
    request_logger = configure_request_logging(runtime_settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        runtime_engine = engine or SentenceTransformerEngine(runtime_settings)
        runtime_engine.warmup(runtime_settings.model_id)
        app.state.embed_service = EmbedTexts(runtime_engine, runtime_settings)
        app.state.embedding_engine_ready = True
        yield
        app.state.embedding_engine_ready = False

    app = FastAPI(
        title="Embedding API",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def attach_trace_id(request: Request, call_next: Any):
        started_at = perf_counter()
        incoming = request.headers.get("x-trace-id", "")
        request.state.trace_id = incoming if _valid_trace_id(incoming) else uuid4().hex
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["x-trace-id"] = request.state.trace_id
            return response
        finally:
            route = request.scope.get("route")
            fields: dict[str, Any] = {
                "duration_ms": round((perf_counter() - started_at) * 1_000, 3),
                "event": "http_request_completed",
                "health_check": request.url.path.startswith("/health/"),
                "method": request.method,
                "path": getattr(route, "path", "<unmatched>"),
                "status_code": status_code,
                "trace_id": request.state.trace_id,
            }
            fields.update(getattr(request.state, "observability", {}))
            error_code = getattr(request.state, "error_code", None)
            if error_code is not None:
                fields["error_code"] = error_code
            log_request(request_logger, fields)

    @app.exception_handler(EmbeddingError)
    async def embedding_error_handler(request: Request, error: EmbeddingError):
        return _error_response(
            request,
            status_code=error.status_code,
            code=error.code,
            message=error.message,
            details=error.details,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, _: RequestValidationError):
        error = InvalidInputError()
        return _error_response(
            request,
            status_code=error.status_code,
            code=error.code,
            message=error.message,
        )

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "live"}

    @app.get("/health/ready", response_model=None)
    async def ready(request: Request) -> JSONResponse | dict[str, str]:
        if getattr(request.app.state, "embedding_engine_ready", False):
            return {"status": "ready", "model": runtime_settings.model_id}
        error = EmbeddingEngineUnavailableError()
        return _error_response(
            request,
            status_code=error.status_code,
            code=error.code,
            message=error.message,
        )

    @app.post("/v1/embeddings", response_model=EmbeddingResponse)
    def create_embeddings(request: Request, payload: EmbeddingRequest) -> EmbeddingResponse:
        service: EmbedTexts = request.app.state.embed_service
        inputs = payload.normalized_inputs()
        request.state.observability = {
            "device": runtime_settings.inference_device,
            "input_bytes": sum(
                len(value.encode("utf-8", errors="replace")) for value in inputs
            ),
            "input_count": len(inputs),
            "model": (
                payload.model
                if payload.model == runtime_settings.model_id
                else "unsupported"
            ),
            "model_revision": runtime_settings.model_revision,
        }
        result = service.execute(payload.model, inputs)
        request.state.observability["input_tokens"] = result.input_tokens
        return EmbeddingResponse(
            model=result.model_id,
            dimension=result.dimension,
            vectors=[
                VectorResponse(index=index, embedding=list(vector))
                for index, vector in enumerate(result.vectors)
            ],
            usage=UsageResponse(
                input_count=len(result.vectors), input_tokens=result.input_tokens
            ),
        )

    return app
