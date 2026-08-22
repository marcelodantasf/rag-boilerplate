"""Durable idempotency reservation port."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol


class IdempotencyState(StrEnum):
    BEGUN = "begun"
    REPLAY = "replay"
    CONFLICT = "conflict"


@dataclass(frozen=True, slots=True)
class IdempotencyResult:
    state: IdempotencyState
    status: int | None = None
    response: dict[str, Any] | None = None


class IdempotencyStore(Protocol):
    async def begin(self, scope: str, key: str, request_hash: str, ttl: int) -> IdempotencyResult: ...

    async def complete(self, scope: str, key: str, status: int, response: dict[str, Any]) -> None: ...

    async def abandon(self, scope: str, key: str) -> None: ...

    async def close(self) -> None: ...
