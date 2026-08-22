# RAG Core API — Contract and Implementation Decisions

Status: **Accepted baseline v1.1**  
Contract: [`../api/rag-core-openapi.yaml`](../api/rag-core-openapi.yaml)  
Owners: Architect for the contract, Developer for application/test code, DBA for
vector/catalog adapters, and DevOps for runtime/telemetry/test enforcement.

This document resolves the cross-role decisions required to implement RAG Core.
The OpenAPI file is authoritative for the public HTTP shape. This document is
authoritative for dependency direction, internal port semantics, lifecycle,
operational requirements, and acceptance criteria. An implementation may add
private helpers, but may not silently weaken or reinterpret these contracts.

## 1. Product boundary and controllers

RAG Core owns application-facing RAG semantics and is the exclusive application
client of Qdrant. The Embedding API only converts text to vectors. Qdrant remains
private infrastructure.

Two controllers are required:

- `RagController` owns synchronous document ingestion, idempotent document
  deletion, and one-request top-k retrieval.
- `VectorCollectionController` owns logical collection creation, listing,
  inspection, empty-generation provisioning, generation activation, and deliberate
  retirement.

There are deliberately no public embed, raw vector search, point, payload,
physical collection, Qdrant filter, snapshot, or cluster-administration
operations. A retrieval call embeds, searches, enforces trusted policy, ranks,
and shapes citable results in one request.

Layer dependencies point inward:

```text
transport/http       OpenAPI models, controllers, middleware, HTTP errors
        ↓
application          ingest/retrieve/delete/collection use cases
        ↓
domain + ports       provider-neutral values, policies, interfaces
        ↑
adapters             Embedding HTTP, Qdrant, catalog, idempotency
        ↑
infrastructure       settings, clients, telemetry, lifecycle
```

HTTP request objects stop at the transport boundary. Qdrant SDK/JSON objects and
physical names stop at the vector/catalog adapters. Vectors never appear in the
public HTTP models.

## 2. Public resources and identifiers

The complete endpoint list and schemas are in OpenAPI. Public collection IDs are
logical and stable. Each immutable physical realization has a Core-generated
ULID-style `generation_id`. Provider physical names are catalog data and are
never serialized publicly.

Clients supply a stable `document_id`. Core computes:

```text
normalized_content = CRLF/CR → LF, Unicode NFC, outer whitespace removed
document_version   = "sha256:" + SHA-256(UTF-8(normalized_content))
chunk_id           = "chk_" + UUIDv5(core namespace,
                     collection_id + generation_id + document_id +
                     document_version + chunk_index)
point_id           = UUID portion of chunk_id
```

Normalization never performs domain rewriting and never silently truncates.
The default deterministic chunker uses characters: 1,000-character target with
200-character overlap, preferring the last paragraph, newline, sentence, then
whitespace boundary inside the target. A hard boundary is used only when none is
present. Empty chunks are forbidden and at most 512 chunks may be produced.
Chunk policy is configuration included in `index_schema_version`; changing it
requires a new generation.

`document_version` is content-based. Metadata changes with identical content
upsert the same deterministic points and update their payloads.

## 3. Domain contract

The following names and meanings are the versioned internal baseline. Python
representation may use frozen dataclasses and enums, but field meaning must not
change independently.

```text
MetadataScalar = str | int | finite float | bool

MetadataField:
  name: str
  type: keyword | integer | float | boolean
  indexed: bool

EmbeddingContract:
  model_id: str
  revision: str                 # immutable artifact revision
  dimension: int
  normalized: bool
  distance_metric: cosine | dot | euclid

CollectionContract:             # one immutable generation
  collection_id: str            # logical ID
  generation_id: str
  physical_name: str             # internal only
  embedding: EmbeddingContract
  index_schema_version: int
  metadata_fields: tuple[MetadataField, ...]
  isolation_policy: shared | collection_per_tenant
  state: building | ready | active | retired | failed
  created_at: datetime
  activated_at: datetime | None
  source_generation_id: str | None

VectorPoint:
  point_id, chunk_id, document_id, document_version: str
  chunk_index: int
  text: str
  metadata: mapping[str, MetadataScalar]
  vector: tuple[finite float, ...]
  embedding_model, embedding_revision: str
  index_schema_version: int

VectorMatch:
  chunk_id, document_id, document_version, text: str
  metric_value: float
  metadata: mapping[str, MetadataScalar]
```

The internal isolation value leaves room for a future
`collection_per_tenant` adapter, but the v1 HTTP contract accepts only `shared`.
Exposing another value before its provisioning and authenticated routing exist
would silently misrepresent the delivered isolation behavior.

For `VectorMatch.metric_value`, cosine/dot are similarities where higher is
better; Euclidean is a non-negative distance where lower is better. The Qdrant
adapter converts provider-native results to this semantic but does not calculate
the public score.

Filters use a typed AST, never a metadata dictionary or Qdrant object:

```text
FilterExpression = FilterAll | FilterAny | FilterNot | FilterPredicate
FilterAll(children)
FilterAny(children)
FilterNot(child)
FilterPredicate(field, operator, value)
operator = eq | in | gt | gte | lt | lte
```

The maximum tree depth is 3, the maximum predicate count is 10, and `in` accepts
1–20 same-typed values. Only contract fields with `indexed=true` may be filtered.
`eq` and `in` work for all metadata types. Ordering works only for integer and
float. Boolean is never treated as integer. Unknown fields, type mismatches,
NaN/infinity, and invalid operator/type combinations return `invalid_request`
before Qdrant is called.

## 4. Inward-owned ports

Signatures below are semantic; `Sequence` inputs must be consumed without
reordering and outputs should be immutable provider-neutral values. Deadlines are
remaining monotonic budgets, not fresh full timeouts.

```text
EmbeddingGateway.embed(
  texts: Sequence[str], *, contract: EmbeddingContract,
  deadline_seconds: float, trace_context: TraceContext
) -> EmbeddingBatch

EmbeddingGateway.ready(
  *, contract: EmbeddingContract, deadline_seconds: float,
  trace_context: TraceContext
) -> ReadinessResult

EmbeddingGateway.close() -> None
```

`EmbeddingBatch` carries model ID, immutable revision, dimension, normalization,
and ordered vectors. The currently implemented Embedding API returns model ID and
dimension but not revision or normalization. Therefore its adapter must attach the
deployment-configured immutable revision/normalization and validate the response
model, count, indices, finite values, and dimension. This limitation must remain
visible in configuration and readiness; changing configured capability without a
reindex is forbidden.

```text
VectorStore.create_collection(contract) -> None
VectorStore.verify_collection(contract, *, deadline_seconds) -> Verification
VectorStore.create_payload_indexes(contract) -> None
VectorStore.upsert(contract, points, *, deadline_seconds) -> None
VectorStore.search(
  contract, vector, *, top_k, filter: FilterExpression | None,
  trusted_filter: FilterExpression, deadline_seconds
) -> tuple[VectorMatch, ...]
VectorStore.delete_by_document(
  contract, document_id, *, deadline_seconds
) -> int
VectorStore.delete_older_document_versions(
  contract, document_id, keep_version, *, deadline_seconds
) -> int
VectorStore.activate_alias(
  logical_id, previous: CollectionContract | None,
  target: CollectionContract, *, deadline_seconds
) -> None
VectorStore.retire_alias(logical_id, *, deadline_seconds) -> None
VectorStore.delete_collection(contract, *, deadline_seconds) -> None
VectorStore.ready(*, deadline_seconds) -> ReadinessResult
VectorStore.close() -> None
```

`delete_collection` is an internal retention/operations capability and is not
invoked by the public retirement request. `search` may over-fetch by a small
bounded factor for transition de-duplication, but never exceeds a configured
provider limit.

```text
CollectionCatalog.add_generation(contract) -> None
CollectionCatalog.get_active(collection_id) -> CollectionContract | None
CollectionCatalog.get_generation(
  collection_id, generation_id
) -> CollectionContract | None
CollectionCatalog.list_logical(*, limit, cursor) -> CatalogPage
CollectionCatalog.list_generations(collection_id) -> tuple[CollectionContract, ...]
CollectionCatalog.update_state(
  collection_id, generation_id, expected_state, new_state,
  activated_at=None
) -> CollectionContract
CollectionCatalog.set_active(
  collection_id, target_generation_id, expected_active_generation_id
) -> None
CollectionCatalog.retire_logical(
  collection_id, expected_active_generation_id
) -> datetime                         # retained_until
CollectionCatalog.ready(*, deadline_seconds) -> ReadinessResult
CollectionCatalog.close() -> None
```

All state updates use compare-and-set semantics. There may be only one active
generation per logical collection.

Idempotency is a separate inward port so horizontal Core replicas behave the same:

```text
IdempotencyStore.begin(scope, key, request_hash, ttl) -> begun | replay | conflict
IdempotencyStore.complete(scope, key, status, response) -> None
IdempotencyStore.abandon(scope, key) -> None
```

The scope includes trusted tenant/principal and operation ID. The request hash is
SHA-256 over canonical JSON plus material path/query fields. Provider timeouts
abandon an uncommitted reservation so deterministic retry can repair work;
completed responses are replayed exactly. Do not cache 5xx errors. The shipped
adapter is process-local memory and is cleared on restart; a production adapter
must provide shared persistence and enforce its configured TTL.

## 5. Collection compatibility and catalog persistence

The compatibility tuple is immutable within a generation:

```text
(model_id, revision, dimension, normalized, distance_metric,
 index_schema_version, metadata_fields, isolation_policy)
```

Search/write must load the active catalog contract and verify model identity,
revision, dimension, normalization, and finite vector shape before contacting the
vector store. Qdrant collection verification must also compare vector size,
distance metric, required payload indexes, and an internal schema fingerprint.
Existence alone is never compatibility.

Cosine supports normalized or non-normalized vectors. Dot is accepted only when
`normalized=true`, which bounds its public transformation. Euclidean accepts either.
Changing any compatibility field creates a new physical generation and reindex;
no field is altered in place.

For this boilerplate, `CollectionCatalog` persists in a reserved private Qdrant
catalog collection named internally `__rag_core_catalog_v1`. It uses a
one-dimensional dummy vector and exact-match payload records; the adapter hides
this detail. Catalog records and physical collections are covered by the same
snapshot/restore policy. The port permits a transactional database replacement
later. Core must fail readiness if the catalog cannot be read; it must never
reconstruct contracts by guessing from physical names.

## 6. Collection lifecycle and generation provisioning

Generation transitions are:

```text
create:     building → ready → active
provision:  building → ready
failure:    building → failed
cutover:    previous active → retired, target ready → active
rollback:   retired target explicitly revalidated → ready → active
cleanup:    retired → physical deletion after retention (platform-authorized)
```

Initial creation may perform building/ready/active in one request after verifying
the empty provider collection and indexes. The generation-provisioning operation:

1. compares the target contract with the active contract and creates a generation
   record;
2. creates and verifies a new empty physical collection and payload indexes;
3. marks the target `ready` but never activates it automatically;
4. returns `201`, or `200` when replaying a completed idempotent request.

Provisioning does not reindex or validate document data. Before activation, an
authorized out-of-band workflow must rebuild the target from the external source
of truth and validate counts, schema, isolation, filters, latency, and retrieval
quality. The public ingestion route writes only to the active generation. The
vector index is not the durable source of original documents, and activating an
empty target is unsafe for a data-bearing collection. Platform owns snapshots,
restore, replication, patching, and final physical deletion.

Public retirement repeats the exact logical ID in `X-Confirm-Retirement` and the
active generation in `If-Match`. It removes request-path access immediately but
only schedules physical cleanup after the configured rollback retention (default
seven days).

## 7. Ingestion, deletion, retrieval, and score policy

### Ingestion

The use case validates the active collection and metadata schema, normalizes and
chunks, embeds in batches no larger than the Embedding API/configured limits,
validates every vector, performs a deterministic full upsert, then deletes older
versions of the same document. A changed-content replacement is not transactionally
atomic in Qdrant: new points are written before old points are removed so failure
does not erase the last complete version. Replay repairs partial new writes and
cleanup.

Identical request replay is safe with or without `Idempotency-Key`; the key adds a
stable stored response and detects accidental reuse with different input.

### Deletion

Deletion removes every version/chunk for the trusted tenant, active generation,
and document ID. Zero matches is successful. Collection absence or retirement is
not silently treated as document absence.

### Retrieval and scores

Trusted isolation filters are derived from authenticated server context and are
ANDed with the validated consumer filter. The client cannot supply reserved tenant
or access fields. Local unauthenticated development uses a fixed `local` tenant;
production configuration must provide authenticated identity before using shared
isolation.

The vector adapter returns `metric_value`; the application calculates:

```text
cosine score = (clamp(similarity, -1, 1) + 1) / 2
dot score    = (clamp(similarity, -1, 1) + 1) / 2  # normalized vectors required
euclid score = 1 / (1 + max(distance, 0))
```

Scores are clamped to `[0,1]`, thresholded, rounded to six decimals, and ordered
descending with `chunk_id` as the deterministic tie-breaker. The score is a
monotonic within-contract relevance indicator, not a probability and not comparable
across collections/models. Core returns at most requested `top_k`, and an empty list
is a successful no-match.

## 8. Errors, limits, deadlines, and retries

Every error is the OpenAPI `Error` envelope with a stable code, safe message,
trace ID, and optional safe details. Validation errors use `422`; malformed JSON or
headers use `400`; compatibility/lifecycle/idempotency conflicts use `409`; stale
compare-and-set/destructive preconditions use `412`; dependency unavailability uses
`503`; downstream deadline exhaustion uses `504`. Stack traces, provider bodies,
raw input, query text, vectors, credentials, authorization headers, and internal
physical names are never returned.

Initial configurable limits, also enforced by the OpenAPI-compatible validators:

| Limit | Default / maximum |
| --- | --- |
| Document UTF-8 bytes | 256 KiB |
| Chunks per document | 512 |
| Chunk target / overlap | 1,000 / 200 characters |
| Metadata fields / canonical bytes | 32 / 16 KiB |
| Metadata key / string value | 64 / 512 characters |
| Query UTF-8 bytes | 16 KiB |
| `top_k` | default 10, maximum 100 |
| Filter depth / predicates / `in` values | 3 / 10 / 20 |
| Caller timeout | default 10 s, clamped 100 ms–30 s |
| Idempotency key | 8–128 printable bytes; local retention is process lifetime |
| Collection metadata fields | 32 |

Every downstream call receives the remaining monotonic deadline. Use capped
exponential backoff with full jitter only for safe reads, deterministic upserts,
and compare-and-set operations whose outcome can be verified. Do not retry invalid
input, compatibility conflicts, or arbitrary destructive operations. Connection
pools, concurrent request work, embedding batches, migrations, and retry attempts
must all be bounded.

## 9. Trace, logs, metrics, and infrastructure observability

Core accepts and propagates W3C `traceparent`/`tracestate`. `x-trace-id` remains a
safe legacy correlation header: use it only when valid W3C context is absent and
always return a safe `x-trace-id`. Invalid trace input starts a new trace rather than
failing the product request. Child spans cover validation, chunking, each embedding
batch, catalog calls, vector verification/search/upsert/delete, migration phases,
and activation. Core → Embedding propagates trace context; Qdrant client spans are
children of the Core operation.

Structured JSON logs include timestamp, severity, service/version/environment,
trace/span/request IDs, route template, method, status, latency, error code,
dependency/operation outcome, collection state, safe aggregate counts, and health
marker. Do not log raw document/chunk/query content, vectors, metadata values,
headers, secrets, provider error bodies, point IDs, document IDs, idempotency keys,
or physical names. Hashing sensitive identifiers does not automatically make them
safe telemetry.

Required Core metrics:

- HTTP count, duration, and in-flight by route template/method/status class;
- retrieval duration, requested top-k, returned count, and no-match count;
- ingestion duration, documents/chunks indexed, batch sizes, and cleanup retries;
- dependency request count/duration/errors/timeouts/retries by dependency/operation;
- contract mismatch and validation rejection counts by low-cardinality code;
- idempotency begin/replay/conflict/abandon counts;
- migration phase/duration/failure and collection-generation state counts;
- readiness result and graceful-shutdown drain/cancel counts.

No collection, tenant, document, query, point, trace, or user ID is a metric label.

DevOps must collect Core and Embedding telemetry plus Qdrant process/query/upsert
latency/errors, vector count, storage/disk latency and utilization, memory, snapshot
age/status, and container CPU/memory/restarts. Dashboards cover golden signals,
dependency drill-down, ingestion/retrieval, migrations, and database capacity.
Alerts cover sustained error/latency SLO burn, dependency outage, mismatch spikes,
Qdrant disk pressure, failed/stale backup, restart loops, migration failure, and
ingestion cleanup backlog.

Initial objectives for a measured local reference workload (warm model, documents
within limits) are Core availability 99.9%, retrieval p95 under 750 ms, ingestion p95
under 3 s, and 5xx below 1%. These are starting measurement gates, not promises for
unknown hardware; load-test reports must state hardware, corpus, concurrency, and
whether model cache was warm.

## 10. Health, security, and runtime behavior

`/health/live` is process-only and calls no dependency. `/health/ready` uses an
independent short bound (default 1 s per dependency, 3 s total) to check validated
configuration, configured Embedding capability, a cheap Qdrant operation, and catalog
readability. It does not scan collections or mark a healthy saturated service
unready. Admission control represents saturation.

On shutdown, stop accepting new work, drain within a bounded grace period, cancel
remaining operations, flush telemetry within its own bound, and close HTTP/Qdrant
clients. The container runs non-root with read-only root filesystem, dropped
capabilities, bounded resources, private service networking, and secret injection.
Only Core is application-reachable; Embedding and Qdrant host ports are development
conveniences and absent in production.

Authentication is an extension point for this local boilerplate, not permission to
trust tenant metadata. Shared-isolation production mode must fail startup unless a
trusted principal/tenant provider is configured. Reserved payload fields are written
only from server context.

## 11. Documentation and verification acceptance criteria

Documentation is a standalone delivery gate. OpenAPI validation, examples, operation
IDs, response/error coverage, and implementation-route conformance must be tested.
A consumer must be able to create a collection, ingest, retrieve, and delete using
only the public specification without learning that Qdrant exists.

The Developer owns tests; DevOps independently enforces every applicable gate and
returns failures to Developer under `agents/AGENTS.md`. Completion requires:

1. unit tests for normalization/chunking, stable IDs, metadata/filter grammar,
   score transformations, compatibility, lifecycle, idempotency, and errors;
2. controller/OpenAPI contract tests for every status, header, limit, example, and
   absence of provider fields/micro-operation routes;
3. Embedding gateway contract tests for order, count, identity, finite dimensions,
   configured revision/normalization, timeouts, and malformed responses;
4. real-Qdrant integration tests for catalog durability, schema/index verification,
   upsert/search/filter/delete, typed filters, alias cutover, and restart behavior;
5. Compose end-to-end create → ingest → retrieve → re-ingest → delete plus trace
   propagation across Core and Embedding;
6. idempotency conflict/replay and partial-upsert/cleanup recovery tests;
7. empty generation provisioning/replay, explicit cutover/rollback, unsafe-field
   rejection, and stale activation tests;
8. failure tests for dependency timeout/unavailability/malformed output, catalog
   loss, Core/Qdrant restart, overload, cancellation, and graceful shutdown;
9. deterministic-corpus retrieval tests asserting expected-document presence near
   the top without exact floating-point expectations;
10. load tests with explicit environment/p95 results, security/network checks,
    OpenAPI validation, snapshot restore, and documented out-of-band rebuild drills
    when a source-of-truth workflow is supplied.

No required test may be waived merely because infrastructure is inconvenient. A
known unsupported production feature must be explicitly scoped, documented, and
approved by the Architect rather than represented as complete.

## 12. Cross-role change protocol

This baseline is version 1.1. A proposed change to a public schema, error code, port
semantic, compatibility field, score formula, lifecycle transition, telemetry field,
or acceptance gate requires an Architect decision. The proposer must identify
affected implementation files, adapters, tests, OpenAPI examples, guides, Compose/CI,
dashboards, and alerts. DBA and Developer may refine private implementation details
without approval when all observable and port semantics remain intact.
