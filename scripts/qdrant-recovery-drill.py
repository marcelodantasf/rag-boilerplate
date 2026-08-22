#!/usr/bin/env python3
"""Create, restore, and verify a dedicated Qdrant collection snapshot."""

from __future__ import annotations

import sys
from uuid import uuid4

from qdrant_client import QdrantClient, models


def main() -> int:
    base_url = sys.argv[1]
    name = f"recovery-{uuid4().hex[:12]}"
    client = QdrantClient(url=base_url, check_compatibility=False)
    snapshot_name: str | None = None
    try:
        client.create_collection(
            name,
            vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE),
        )
        client.upsert(
            name,
            [models.PointStruct(id=1, vector=[1.0, 0.0, 0.0], payload={"probe": "safe"})],
            wait=True,
        )
        snapshot = client.create_snapshot(name, wait=True)
        if snapshot is None:
            raise AssertionError("Qdrant did not return a snapshot description")
        snapshot_name = snapshot.name
        if client.count(name, exact=True).count != 1:
            raise AssertionError("recovery fixture count was not durable before snapshot")
        if not client.delete_collection(name):
            raise AssertionError("recovery fixture collection could not be deleted")
        location = f"file:///qdrant/snapshots/{name}/{snapshot_name}"
        if not client.recover_snapshot(name, location, wait=True):
            raise AssertionError("snapshot recovery was rejected")
        points = client.retrieve(name, [1], with_payload=True)
        if len(points) != 1 or points[0].payload != {"probe": "safe"}:
            raise AssertionError("restored point count or payload did not match the snapshot")
        print("Qdrant snapshot restore drill passed: 1 point and safe payload recovered")
        return 0
    finally:
        try:
            if snapshot_name is not None:
                client.delete_snapshot(name, snapshot_name)
        except Exception:
            pass
        try:
            client.delete_collection(name)
        except Exception:
            pass
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
