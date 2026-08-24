# Vector Database Boundary — Infrastructure Architecture Guide

## Boundary and ownership

The vector database is infrastructure, not a public third microservice. RAG Core is its exclusive application client; neither consumer apps nor the Embedding API receives a database URL or credential.

```text
Client/public network → RAG Core → private data network → Vector DB
Embedding API ────────────────┘     (no vector DB access)
```

| Owner | Responsibility |
| --- | --- |
| Platform | Run, patch, secure, back up, and monitor the database. |
| RAG Core | Collections, point IDs, payload schema, search/write policy, migrations. |
| Embedding API | Compatible vectors only. |
| Consumers | RAG Core API only. |

This prevents provider details from becoming a client contract and centralizes filter/security policy.

## Collection and point contract

Treat a collection as a versioned index contract, not merely a name. Record or enforce:

```text
collection_id, index_schema_version, embedding model + revision,
vector dimension, distance metric, payload fields/indexes,
tenant isolation strategy, created_at, migration state
```

Each point represents a chunk. Typical payload: `document_id`, `chunk_id`, `chunk_index`, chunk text (or a content reference), safe metadata, embedding model, and schema version.

Use deterministic point IDs based on collection, document, document content/version hash, and chunk index. Upserts then become retry-safe, document deletion is precise, and ingestion is idempotent.

Never mix dimensions, distance metrics, or incompatible model revisions in one collection. A changed embedding schema normally means a new collection and a reindex.

## Filters and security

Design and index the metadata fields users truly need: tenant, source, type, department, language, or access scope. Keep payloads small. Chunk text is useful; huge original documents, secrets, and mutable authorization blobs are not. Keep a durable source-of-truth outside the vector index when auditing/reconstruction is required.

Filters are part of security. RAG Core must add tenant/access filters from authenticated server-side context to **every** search. Never trust an unverified tenant field from the request body. Use separate collections or instances when stronger isolation is worth the operational cost.

## Operations and migration

RAG Core's adapter needs only create/verify collection, upsert, search, filter, and delete-by-document. Snapshots, restore, compaction, replication, and cluster management are platform work, not request-path behavior.

Safe migration:

1. Create a new collection with explicit schema/version.
2. Re-embed and index source documents; optionally dual-write new ingestion.
3. Validate counts, sampling quality, filters, latency, and isolation.
4. Switch RAG Core reads through configuration or a logical alias.
5. Keep the old collection for rollback, then retire it deliberately.

Never change vector dimension in place or infer compatibility because a name already exists.

## Availability, recovery, and configuration

A persistent single instance in Docker Compose is excellent for local learning, but it is not high availability. Use a named volume locally; in shared environments define RPO (acceptable data loss), RTO (time to recover), backups, snapshot storage, and a reindex source.

A snapshot protects against data loss; a reproducible ingestion pipeline protects against bad index/schema changes. Test both a restore and a full rebuild.

Give RAG Core a bounded connection/request timeout and least-privilege credential. Keep administrator credentials outside app containers. Use private networking, secret storage and rotation, encryption across untrusted networks, and restricted dashboards/snapshot locations. A local published port is a development convenience only.

## Health, capacity, and testing

Monitor process health, storage/disk latency, memory, query/upsert latency and errors, backup age, and replication/snapshot state where relevant. RAG Core readiness should perform one cheap bounded dependency operation, not collection scans.

Capacity depends on vector count, dimension, datatype, index configuration, payload/filter cardinality, and concurrency. Measure representative chunks and filters—not merely document count.

| Test | Proves |
| --- | --- |
| Adapter integration | Real DB upsert/search/filter/delete/schema behavior |
| Migration | New collection/reindex/cutover retains retrieval properties |
| Restore drill | Snapshot or rebuild meets recovery objective |
| Security | Forbidden network paths cannot reach DB; secrets stay hidden |
| Performance | Representative workload meets latency/capacity assumptions |

Use a real database container for adapter tests. Mocks help RAG Core unit tests but cannot validate database filter/index semantics.

## Deployment shape, anti-patterns, and milestones

For the boilerplate, use a Compose service with named volume and private network. RAG Core receives its internal hostname through config. Process startup is not readiness; RAG Core must perform its own bounded readiness check. Create development collections with an explicit initialization task; production migrations are explicit deploy steps.

Review red flags:

- Consumers or Embedding API connect directly to the DB.
- Collection existence is mistaken for embedding compatibility.
- One unbounded collection has no tenant/access policy.
- Original documents exist only in the vector index.
- Random IDs make retries duplicate chunks.
- Production database ports/dashboard are public.
- No restore drill because data is “rebuildable.”
- Schema changes happen during ordinary search traffic without cutover.

1. Local Compose instance, named volume, private network, development health check.
2. Real RAG adapter for collection verification, upsert, search/filter, delete.
3. Schema metadata, deterministic IDs, payload indexes, integration tests.
4. Metrics, bounded readiness, snapshot and restore drill.
5. Production networking, secrets, capacity test, migration runbook.
