"""Embedding service port."""

from collections.abc import Sequence
from typing import Protocol

from rag_core.domain.models import EmbeddingContract, EmbeddingResult, ReadinessResult, TraceContext


class EmbeddingGateway(Protocol):
    async def embed(
        self,
        texts: Sequence[str],
        *,
        contract: EmbeddingContract,
        deadline_seconds: float,
        trace_context: TraceContext,
    ) -> EmbeddingResult: ...

    async def ready(
        self, *, contract: EmbeddingContract, deadline_seconds: float, trace_context: TraceContext
    ) -> ReadinessResult: ...

    async def close(self) -> None: ...
