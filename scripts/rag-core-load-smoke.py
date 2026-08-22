#!/usr/bin/env python3
"""Small repeatable warm-model retrieval load gate for the local reference stack."""

from __future__ import annotations

import concurrent.futures
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.request


BASE_URL = sys.argv[1].rstrip("/")
REQUESTS = int(os.getenv("LOAD_REQUESTS", "20"))
CONCURRENCY = int(os.getenv("LOAD_CONCURRENCY", "4"))
P95_LIMIT_MS = float(os.getenv("LOAD_P95_MS", "750"))


def request(method: str, path: str, body: dict | None = None, headers: dict | None = None):
    request_headers = {"accept": "application/json", "x-trace-id": "delivery-load-gate"}
    if body is not None:
        request_headers["content-type"] = "application/json"
    request_headers.update(headers or {})
    encoded = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE_URL + path, data=encoded, headers=request_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        raise AssertionError(f"{method} {path} failed with {error.code}: {error.read().decode(errors='replace')}") from error


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def main() -> int:
    if REQUESTS < 1 or CONCURRENCY < 1:
        raise SystemExit("LOAD_REQUESTS and LOAD_CONCURRENCY must be positive")
    collection_id = f"load-{int(time.time())}"
    _, collection = request(
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
            "metadata_fields": [{"name": "department", "type": "keyword", "indexed": True}],
            "isolation_policy": "shared",
        },
    )
    generation_id = collection["active_generation_id"]
    for index in range(5):
        request(
            "POST",
            "/v1/rag/documents",
            {
                "collection_id": collection_id,
                "document_id": f"policy-{index}",
                "content": f"Policy section {index}. Parental leave lasts sixteen weeks and requests go to People Operations.",
                "metadata": {"department": "people"},
            },
        )

    payload = {
        "collection_id": collection_id,
        "query": "How long is parental leave?",
        "top_k": 5,
        "filter": {"field": "department", "operator": "eq", "value": "people"},
    }
    request("POST", "/v1/rag/retrievals", payload)

    def retrieve(_: int) -> float:
        started = time.perf_counter()
        _, result = request("POST", "/v1/rag/retrievals", payload)
        if not result["results"]:
            raise AssertionError("load retrieval returned no results")
        return (time.perf_counter() - started) * 1_000

    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        durations = list(pool.map(retrieve, range(REQUESTS)))

    p95 = percentile(durations, 0.95)
    print(
        "retrieval load: "
        f"requests={REQUESTS} concurrency={CONCURRENCY} "
        f"p50_ms={statistics.median(durations):.2f} p95_ms={p95:.2f} max_ms={max(durations):.2f}"
    )
    request(
        "DELETE",
        f"/v1/vector-collections/{collection_id}",
        headers={"X-Confirm-Retirement": collection_id, "If-Match": generation_id},
    )
    if p95 > P95_LIMIT_MS:
        raise AssertionError(f"retrieval p95 {p95:.2f} ms exceeds {P95_LIMIT_MS:.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
