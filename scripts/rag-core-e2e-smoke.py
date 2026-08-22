#!/usr/bin/env python3
"""Exercise the complete public create/ingest/retrieve/delete/retire workflow."""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request


BASE_URL = sys.argv[1].rstrip("/")
TRACE_ID = "delivery-gate-e2e"


def request(method: str, path: str, body: dict | None = None, headers: dict | None = None):
    request_headers = {"accept": "application/json", "x-trace-id": TRACE_ID}
    if body is not None:
        request_headers["content-type"] = "application/json"
    request_headers.update(headers or {})
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE_URL + path, data=data, headers=request_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            payload = json.loads(response.read() or b"{}")
            if response.headers.get("x-trace-id") != TRACE_ID:
                raise AssertionError(f"{path} did not return the propagated trace ID")
            return response.status, payload
    except urllib.error.HTTPError as error:
        payload = error.read().decode(errors="replace")
        raise AssertionError(f"{method} {path} failed with {error.code}: {payload}") from error


def main() -> int:
    collection_id = f"smoke-{int(time.time())}"
    create_status, collection = request(
        "POST",
        "/v1/vector-collections",
        {
            "collection_id": collection_id,
            "embedding": {
                "model_id": "all-MiniLM-L6-v2",
                "revision": "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
                "dimension": 384,
                "normalized": True,
                "distance_metric": "cosine",
            },
            "index_schema_version": 1,
            "metadata_fields": [
                {"name": "department", "type": "keyword", "indexed": True},
                {"name": "year", "type": "integer", "indexed": True},
            ],
            "isolation_policy": "shared",
        },
        {"Idempotency-Key": f"create-{collection_id}"},
    )
    if create_status != 201:
        raise AssertionError(f"expected collection creation 201, got {create_status}")
    generation_id = collection["active_generation_id"]

    ingest_status, ingested = request(
        "POST",
        "/v1/rag/documents",
        {
            "collection_id": collection_id,
            "document_id": "leave-policy",
            "content": "Parental leave lasts sixteen weeks. Requests go to People Operations.",
            "metadata": {"department": "people", "year": 2026},
        },
        {"Idempotency-Key": f"ingest-{collection_id}"},
    )
    if ingest_status != 201 or ingested["chunks_indexed"] < 1:
        raise AssertionError("document was not indexed")

    _, retrieved = request(
        "POST",
        "/v1/rag/retrievals",
        {
            "collection_id": collection_id,
            "query": "How long is parental leave?",
            "top_k": 5,
            "filter": {"all": [{"field": "department", "operator": "eq", "value": "people"}]},
        },
    )
    if not retrieved["results"] or retrieved["results"][0]["document_id"] != "leave-policy":
        raise AssertionError("expected leave-policy in retrieval results")
    if not 0 <= retrieved["results"][0]["score"] <= 1:
        raise AssertionError("public score is outside [0,1]")

    _, deleted = request(
        "DELETE", f"/v1/rag/documents/leave-policy?collection_id={collection_id}"
    )
    if deleted["chunks_deleted"] < 1:
        raise AssertionError("document deletion did not remove indexed chunks")

    _, empty = request(
        "POST",
        "/v1/rag/retrievals",
        {"collection_id": collection_id, "query": "parental leave", "top_k": 5},
    )
    if empty["results"]:
        raise AssertionError("deleted document remained retrievable")

    _, retired = request(
        "DELETE",
        f"/v1/vector-collections/{collection_id}",
        headers={"X-Confirm-Retirement": collection_id, "If-Match": generation_id},
    )
    if not retired["retired"]:
        raise AssertionError("collection retirement was not confirmed")
    print("RAG Core public E2E workflow passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
