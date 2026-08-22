"""Small, bounded Qdrant REST transport used by persistence adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote

import httpx

from ._errors import (
    QdrantContractError,
    QdrantResourceConflictError,
    QdrantResourceNotFoundError,
    QdrantTimeoutError,
    QdrantUnavailableError,
)

JsonObject = dict[str, Any]


class QdrantHttpClient:
    """Provider-private HTTP client with one timeout budget per operation."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 3.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        normalized_url = base_url.rstrip("/")
        if not normalized_url:
            raise ValueError("base_url must not be empty")

        headers = {"api-key": api_key} if api_key else None
        self._client = client or httpx.AsyncClient(
            base_url=normalized_url,
            headers=headers,
            timeout=httpx.Timeout(timeout_seconds),
        )
        self._owns_client = client is None

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def ready(self) -> bool:
        try:
            response = await self._client.get("/readyz")
            return response.status_code == 200
        except (httpx.TimeoutException, httpx.TransportError):
            return False

    async def get_collection(self, collection_name: str) -> JsonObject | None:
        response = await self._request(
            "GET", f"/collections/{_path_segment(collection_name)}", allow_not_found=True
        )
        if response is None:
            return None
        return _result_object(response)

    async def create_collection(
        self,
        collection_name: str,
        *,
        vector_size: int,
        distance: str,
        on_disk_payload: bool = True,
    ) -> None:
        await self._request(
            "PUT",
            f"/collections/{_path_segment(collection_name)}",
            json={
                "vectors": {"size": vector_size, "distance": distance},
                "on_disk_payload": on_disk_payload,
            },
        )

    async def delete_collection(self, collection_name: str) -> None:
        await self._request(
            "DELETE", f"/collections/{_path_segment(collection_name)}"
        )

    async def create_payload_index(
        self,
        collection_name: str,
        *,
        field_name: str,
        field_schema: str,
    ) -> None:
        await self._request(
            "PUT",
            f"/collections/{_path_segment(collection_name)}/index",
            params={"wait": "true"},
            json={"field_name": field_name, "field_schema": field_schema},
        )

    async def upsert_points(
        self, collection_name: str, points: Sequence[Mapping[str, Any]]
    ) -> None:
        if not points:
            return
        await self._request(
            "PUT",
            f"/collections/{_path_segment(collection_name)}/points",
            params={"wait": "true"},
            json={"points": [dict(point) for point in points]},
        )

    async def search_points(
        self,
        collection_name: str,
        *,
        vector: Sequence[float],
        limit: int,
        query_filter: Mapping[str, Any] | None,
    ) -> list[JsonObject]:
        body: JsonObject = {
            "vector": list(vector),
            "limit": limit,
            "with_payload": True,
            "with_vector": False,
        }
        if query_filter:
            body["filter"] = dict(query_filter)
        response = await self._request(
            "POST",
            f"/collections/{_path_segment(collection_name)}/points/search",
            json=body,
        )
        result = response.get("result")
        if not isinstance(result, list) or not all(
            isinstance(item, dict) for item in result
        ):
            raise QdrantContractError("Qdrant search result is malformed")
        return result

    async def delete_points_by_filter(
        self, collection_name: str, query_filter: Mapping[str, Any]
    ) -> None:
        await self._request(
            "POST",
            f"/collections/{_path_segment(collection_name)}/points/delete",
            params={"wait": "true"},
            json={"filter": dict(query_filter)},
        )

    async def delete_points_by_ids(
        self, collection_name: str, point_ids: Sequence[str]
    ) -> None:
        if not point_ids:
            return
        await self._request(
            "POST",
            f"/collections/{_path_segment(collection_name)}/points/delete",
            params={"wait": "true"},
            json={"points": list(point_ids)},
        )

    async def count_points(
        self, collection_name: str, query_filter: Mapping[str, Any]
    ) -> int:
        response = await self._request(
            "POST",
            f"/collections/{_path_segment(collection_name)}/points/count",
            json={"filter": dict(query_filter), "exact": True},
        )
        result = _result_object(response)
        count = result.get("count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise QdrantContractError("Qdrant count result is malformed")
        return count

    async def retrieve_points(
        self, collection_name: str, point_ids: Sequence[str]
    ) -> list[JsonObject]:
        if not point_ids:
            return []
        response = await self._request(
            "POST",
            f"/collections/{_path_segment(collection_name)}/points",
            json={"ids": list(point_ids), "with_payload": True, "with_vector": False},
        )
        result = response.get("result")
        if not isinstance(result, list) or not all(
            isinstance(item, dict) for item in result
        ):
            raise QdrantContractError("Qdrant retrieve result is malformed")
        return result

    async def scroll_points(
        self,
        collection_name: str,
        *,
        limit: int,
        offset: str | int | None = None,
        query_filter: Mapping[str, Any] | None = None,
    ) -> tuple[list[JsonObject], str | int | None]:
        body: JsonObject = {
            "limit": limit,
            "with_payload": True,
            "with_vector": False,
        }
        if offset is not None:
            body["offset"] = offset
        if query_filter:
            body["filter"] = dict(query_filter)
        response = await self._request(
            "POST",
            f"/collections/{_path_segment(collection_name)}/points/scroll",
            json=body,
        )
        result = _result_object(response)
        points = result.get("points")
        next_offset = result.get("next_page_offset")
        if not isinstance(points, list) or not all(
            isinstance(item, dict) for item in points
        ):
            raise QdrantContractError("Qdrant scroll result is malformed")
        if next_offset is not None and not isinstance(next_offset, (str, int)):
            raise QdrantContractError("Qdrant scroll offset is malformed")
        return points, next_offset

    async def update_aliases(self, actions: Sequence[Mapping[str, Any]]) -> None:
        if not actions:
            return
        await self._request(
            "POST",
            "/collections/aliases",
            params={"timeout": "5"},
            json={"actions": [dict(action) for action in actions]},
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        allow_not_found: bool = False,
        **kwargs: Any,
    ) -> JsonObject | None:
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise QdrantTimeoutError("Vector store request timed out") from exc
        except httpx.TransportError as exc:
            raise QdrantUnavailableError("Vector store request failed") from exc

        if allow_not_found and response.status_code == 404:
            return None
        if response.status_code == 404:
            raise QdrantResourceNotFoundError("Vector store resource was not found")
        if response.status_code in {409, 422}:
            raise QdrantResourceConflictError("Vector store mutation conflicted")
        if response.status_code >= 400:
            raise QdrantUnavailableError("Vector store request failed")

        try:
            body = response.json()
        except ValueError as exc:
            raise QdrantContractError("Vector store returned malformed JSON") from exc
        if not isinstance(body, dict):
            raise QdrantContractError("Vector store returned a malformed response")
        return body


def _path_segment(value: str) -> str:
    if not value:
        raise ValueError("Qdrant resource name must not be empty")
    return quote(value, safe="")


def _result_object(response: Mapping[str, Any]) -> JsonObject:
    result = response.get("result")
    if not isinstance(result, dict):
        raise QdrantContractError("Qdrant response result is malformed")
    return result
