# RAG Core Service — Architecture Implementation Guide

## Purpose and boundary

RAG Core is the application-facing service for retrieval-augmented generation. It owns document ingestion, chunking, embedding orchestration, index operations, metadata filters, retrieval policy, and the public RAG contract.

It does **not** execute embedding models and does **not** expose the vector database. Its rule is simple: *Embedding API owns text-to-vector conversion; RAG Core owns RAG semantics; the vector database is private infrastructure.*

| RAG Core owns | It does not own |
| --- | --- |
| Document validation, normalization, chunking | Model loading, GPU/runtime management |
| Calling the Embedding API | Vector-database administration |
| Collection/tenant policy | A public vector-database API |
| Retrieval, filtering, result shape | Consumer UI or answer-generation policy |

A valuable review question: “Would changing this alter the application’s RAG behavior?” If yes, it probably belongs in RAG Core.

## Public API contract

Publish a versioned HTTP/JSON API such as \`/v1\`; write its OpenAPI specification before implementation. Consumers use stable application identifiers, never provider collection names or Qdrant syntax.

| Endpoint | Role |
| --- | --- |
| \`POST /v1/documents\` | Validate, chunk, embed, and index a document. |
| \`DELETE /v1/documents/{document_id}\` | Delete every chunk for a document. |
| \`POST /v1/search\` | Embed a query and return ranked, citable chunks. |
| \`POST /v1/collections\` | Create/logically register a collection. |
| \`GET /health/live\`, \`GET /health/ready\` | Liveness and dependency readiness. |

Example search request:

\`\`\`json
{"collection_id":"handbook","query":"How long is parental leave?","top_k":5,"filter":{"department":"people"}}
\`\`\`

Results should carry \`chunk_id\`, \`document_id\`, text, score, and safe metadata. Do not return database point IDs, raw provider scores without defining their meaning, or credentials/configuration.

## Internal design: ports and adapters

Keep dependencies pointing inward:

\`\`\`text
transport/http        routes, validation, HTTP error mapping
application           ingest, search, delete use cases
domain                chunk/document/retrieval policies
ports                 interfaces owned by RAG Core
adapters/embedding    HTTP EmbeddingGateway implementation
adapters/vector_store Qdrant implementation
infrastructure        settings, clients, telemetry bootstrap
\`\`\`

The application layer depends on ports, not on an HTTP client or Qdrant SDK:

\`\`\`text
EmbeddingGateway.embed(texts, model_hint?) -> EmbeddingBatch
VectorStore.ensure_collection(spec) -> void
VectorStore.upsert(collection, points) -> void
VectorStore.search(collection, vector, top_k, filter) -> RankedPoint[]
VectorStore.delete_by_document(collection, document_id) -> void
\`\`\`

The vector adapter translates IDs and filters to provider calls. Keep relevance policy, score thresholds, and public response shaping above that adapter.

## Request flows

### Ingestion

\`\`\`text
Client → RAG Core: document, collection, metadata
RAG Core: validate → normalize → deterministic document ID → chunk
RAG Core → Embedding API: bounded batch of chunk text
Embedding API → RAG Core: vectors + model metadata
RAG Core → Vector DB: upsert stable chunk points
RAG Core → Client: document ID, chunks indexed, status
\`\`\`

Make it idempotent. Accept an \`Idempotency-Key\` or derive point IDs from collection, document, version/content hash, and chunk index. A timeout retry must update the same points rather than duplicate them.

Start with synchronous ingestion for small documents. When measured document size or latency demands it, add a durable job queue, return \`202 Accepted\`, and offer an explicit status endpoint. Do not leave clients guessing which mode they received.

### Search

\`\`\`text
Client → RAG Core: query, top_k, metadata filter
RAG Core: validate limits and trusted filter policy
RAG Core → Embedding API: query text
RAG Core → Vector DB: vector plus translated filter
RAG Core: threshold/shape/cite safe chunks
RAG Core → Client: ranked results
\`\`\`

Document and query vectors must use a compatible model, revision, dimension, and normalization. Record these with the collection’s index schema and reject an incompatible request or require reindexing.

## Configuration

Use typed, startup-validated environment configuration. Commit an \`.env.example\`, never secrets.

| Setting | Purpose |
| --- | --- |
| \`RAG_PORT\` | HTTP bind port |
| \`EMBEDDING_BASE_URL\`, \`EMBEDDING_TIMEOUT_MS\` | Embedding dependency |
| \`VECTOR_DB_URL\`, \`VECTOR_DB_API_KEY\` | Vector-store connection |
| \`DEFAULT_EMBEDDING_MODEL\` | Default compatibility choice |
| \`DEFAULT_CHUNK_SIZE\`, \`DEFAULT_CHUNK_OVERLAP\` | Chunking policy |
| \`MAX_DOCUMENT_BYTES\`, \`MAX_SEARCH_TOP_K\` | Resource limits |
| \`LOG_LEVEL\`, \`OTEL_EXPORTER_OTLP_ENDPOINT\` | Telemetry |

Reject invalid settings at startup. Never log raw documents, queries, authorization headers, or secret values.

## Errors and resilience

Use a consistent envelope: stable \`code\`, safe \`message\`, \`trace_id\`, and optional field details—never stack traces.

| Condition | Status | Code |
| --- | --- | --- |
| Invalid input | 400/422 | \`invalid_request\` |
| Missing document/collection | 404 | \`not_found\` |
| Model/schema mismatch | 409 | \`embedding_schema_mismatch\` |
| Embedding dependency failure | 503/504 | \`embedding_unavailable\` |
| Vector database failure | 503/504 | \`vector_store_unavailable\` |
| Request/resource limit | 413/429 | \`limit_exceeded\` |

Set deadlines on all downstream calls and propagate the remaining deadline. Retry only safe, idempotent transient operations with capped exponential backoff and jitter. A retry must never create new IDs. Define recovery for partial ingestion: either use atomic/batched semantics supported by the provider or reliably retry the deterministic full upsert.

## Health, observability, testing

\`/health/live\` says only that the process can answer; it must not query dependencies. \`/health/ready\` confirms validated configuration plus bounded Embedding API and vector-store compatibility checks. On shutdown, stop new work, drain briefly, cancel by deadline, and close client pools.

Structured logs include trace/request IDs, route, collection ID when safe, duration, and dependency outcome—not content. Trace chunking, embedding, and vector operations as child spans and propagate W3C trace context. Measure request rate/latency/errors, dependency latency/errors, chunks per document, vectors indexed, search results, retries, and timeouts.

| Test | Proves |
| --- | --- |
| Unit | Chunking, IDs, filters, score/error policy |
| Adapter contract | Published Embedding API and real Qdrant mappings |
| Integration | Ingest/search/delete/idempotent re-ingest using a DB container |
| End-to-end | Compose stack works across both services |
| Failure | Timeout, malformed dependency, restart, and partial-write recovery |

For retrieval tests, use a small deterministic corpus and assert that an expected document appears near the top; avoid asserting exact floating-point scores.

## Deployment and review traps

Build a stateless image; scale RAG Core horizontally. It alone gets private-network access to Embedding API and the vector DB. Apply memory/request limits because chunking and batches are expensive. Roll out schema changes by creating a new collection, reindexing, validating, switching reads, then retiring the old index after rollback time.

Catch these anti-patterns in review:

- Route handlers call the Qdrant SDK directly.
- Consumers send vectors or provider-specific query objects.
- Liveness depends on the database.
- There are no document, chunk, \`top_k\`, timeout, or retry limits.
- Raw sensitive text appears in normal logs.
- A model/dimension change silently reuses an old collection.

## Implementation milestones

1. Skeleton: settings, OpenAPI, health endpoints, structured logs.
2. Ports plus in-memory fake adapters; unit-test use cases.
3. Search vertical slice with real Embedding API client and vector adapter.
4. Ingestion vertical slice with deterministic IDs and idempotency tests.
5. Hardening: deadlines, errors, tracing, metrics, graceful shutdown.
6. Evolve carefully: filters, collection lifecycle, then async ingestion if needed.
