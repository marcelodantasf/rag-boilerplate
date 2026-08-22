import json

import httpx
import pytest

from rag_core.adapters.embedding.http import HttpEmbeddingGateway
from rag_core.domain.errors import EmbeddingContractError, EmbeddingUnavailableError
from rag_core.domain.models import DistanceMetric, EmbeddingContract, TraceContext


CONTRACT = EmbeddingContract("model", "rev", 3, True, DistanceMetric.COSINE)


def _gateway(handler, *, attempts: int = 1):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://embedding")
    return HttpEmbeddingGateway(
        base_url="http://embedding",
        model_id="model",
        revision="rev",
        expected_dimension=3,
        normalized=True,
        client=client,
        max_attempts=attempts,
    ), client


@pytest.mark.asyncio
async def test_gateway_preserves_order_and_trace_context() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-trace-id"] == "trace-1"
        assert json.loads(request.content)["input"] == ["a", "b"]
        return httpx.Response(
            200,
            json={
                "model": "model",
                "dimension": 3,
                "vectors": [
                    {"index": 1, "embedding": [0, 1, 0]},
                    {"index": 0, "embedding": [1, 0, 0]},
                ],
                "usage": {"input_count": 2, "input_tokens": 2},
            },
        )

    gateway, client = _gateway(handler)
    try:
        result = await gateway.embed(("a", "b"), contract=CONTRACT, deadline_seconds=1, trace_context=TraceContext("trace-1"))
    finally:
        await client.aclose()
    assert result.vectors == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    assert result.revision == "rev"


@pytest.mark.asyncio
async def test_gateway_rejects_malformed_dependency_response() -> None:
    gateway, client = _gateway(lambda _: httpx.Response(200, json={"model": "model", "dimension": 3, "vectors": []}))
    try:
        with pytest.raises(EmbeddingContractError):
            await gateway.embed(("a",), contract=CONTRACT, deadline_seconds=1, trace_context=TraceContext("trace"))
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_gateway_maps_dependency_timeout() -> None:
    def timeout(_: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("late")

    gateway, client = _gateway(timeout, attempts=2)
    try:
        with pytest.raises(EmbeddingUnavailableError) as captured:
            await gateway.embed(("a",), contract=CONTRACT, deadline_seconds=0.1, trace_context=TraceContext("trace"))
    finally:
        await client.aclose()
    assert captured.value.status_code == 504
