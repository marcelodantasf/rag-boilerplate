"""Bounded HTTP adapter for the private Embedding API."""

import asyncio
import math
from collections.abc import Sequence
from typing import Any

import httpx
from opentelemetry.propagate import inject

from rag_core.domain.errors import EmbeddingContractError, EmbeddingUnavailableError
from rag_core.domain.models import EmbeddingContract, EmbeddingResult, ReadinessResult, TraceContext
from rag_core.infrastructure.instruments import record_retry


class HttpEmbeddingGateway:
    def __init__(
        self,
        *,
        base_url: str,
        model_id: str,
        revision: str,
        expected_dimension: int,
        normalized: bool,
        client: httpx.AsyncClient | None = None,
        max_attempts: int = 2,
    ) -> None:
        self._revision = revision
        self._model_id = model_id
        self._dimension = expected_dimension
        self._normalized = normalized
        self._max_attempts = max_attempts
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(base_url=base_url.rstrip("/"), follow_redirects=False)

    async def embed(self, texts: Sequence[str], *, contract: EmbeddingContract, deadline_seconds: float, trace_context: TraceContext) -> EmbeddingResult:
        timeout = httpx.Timeout(deadline_seconds)
        response: httpx.Response | None = None
        for attempt in range(self._max_attempts):
            try:
                response = await self._client.post(
                    "/v1/embeddings",
                    json={"model": contract.model_id, "input": list(texts)},
                    headers=_trace_headers(trace_context, deadline_seconds),
                    timeout=timeout,
                )
            except httpx.TimeoutException as error:
                if attempt + 1 == self._max_attempts:
                    raise EmbeddingUnavailableError(timeout=True) from error
            except httpx.HTTPError as error:
                if attempt + 1 == self._max_attempts:
                    raise EmbeddingUnavailableError() from error
            else:
                if response.status_code < 500:
                    break
                if attempt + 1 == self._max_attempts:
                    raise EmbeddingUnavailableError(timeout=response.status_code == 504)
            await asyncio.sleep(min(0.05 * (2**attempt), deadline_seconds / 10))
            record_retry("embedding", "embed")
        if response is None or response.status_code != 200:
            raise EmbeddingUnavailableError(timeout=response is not None and response.status_code == 504)
        result = self._parse(response, model_id=contract.model_id, expected_count=len(texts))
        if contract != EmbeddingContract(result.model_id, result.revision, result.dimension, result.normalized, contract.distance_metric):
            raise EmbeddingContractError()
        return result

    def _parse(self, response: httpx.Response, *, model_id: str, expected_count: int) -> EmbeddingResult:
        try:
            body: dict[str, Any] = response.json()
            returned_model = body["model"]
            dimension = body["dimension"]
            raw_vectors = body["vectors"]
            if returned_model != model_id or type(dimension) is not int or dimension != self._dimension:
                raise ValueError
            if not isinstance(raw_vectors, list) or len(raw_vectors) != expected_count:
                raise ValueError
            ordered: list[tuple[float, ...] | None] = [None] * expected_count
            for item in raw_vectors:
                index = item["index"]
                values = item["embedding"]
                if type(index) is not int or not 0 <= index < expected_count or ordered[index] is not None:
                    raise ValueError
                if not isinstance(values, list) or len(values) != dimension:
                    raise ValueError
                vector = tuple(float(value) for value in values)
                if any(not math.isfinite(value) for value in vector):
                    raise ValueError
                ordered[index] = vector
            if any(vector is None for vector in ordered):
                raise ValueError
        except (KeyError, TypeError, ValueError) as error:
            raise EmbeddingContractError() from error
        return EmbeddingResult(
            model_id=returned_model,
            revision=self._revision,
            dimension=dimension,
            normalized=self._normalized,
            vectors=tuple(vector for vector in ordered if vector is not None),
        )

    async def ready(self, *, contract: EmbeddingContract, deadline_seconds: float, trace_context: TraceContext) -> ReadinessResult:
        try:
            result = await self.embed(
                ("RAG Core readiness capability probe",),
                contract=contract,
                deadline_seconds=deadline_seconds,
                trace_context=trace_context,
            )
        except (EmbeddingUnavailableError, EmbeddingContractError):
            return ReadinessResult(False, "embedding_unavailable")
        valid = result.dimension == self._dimension and result.revision == self._revision
        return ReadinessResult(valid, None if valid else "embedding_schema_mismatch")

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _trace_headers(context: TraceContext, deadline_seconds: float) -> dict[str, str]:
    headers = {
        "x-trace-id": context.trace_id,
        "x-request-deadline-ms": str(max(1, int(deadline_seconds * 1_000))),
    }
    if context.traceparent:
        headers["traceparent"] = context.traceparent
    if context.tracestate:
        headers["tracestate"] = context.tracestate
    inject(headers)
    return headers
