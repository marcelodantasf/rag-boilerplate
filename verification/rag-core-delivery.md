# RAG Core delivery verification

Status: **passed for the single-replica boilerplate deployment**  
Verification owner: DevOps  
Completed: `2026-08-22T02:31:14Z`

## Tested tree and environment

- Baseline revision: `1a92d7e6029a531d72f9f295839730dbb2e856cf`
  on branch `develop`, plus the implementation worktree listed by `git status` at
  verification time. The implementation had not yet been committed, so the exact
  delivered files—not the baseline revision alone—are the tested artifact.
- macOS ARM64 host; Python `3.12.13` from the service `uv` environment.
- `uv 0.12.1`, Docker Engine `29.4.3`, Docker Compose `5.1.3`.
- Qdrant `1.12.5`, OpenTelemetry Collector `0.111.0`, Prometheus `2.55.1`,
  Tempo `2.6.1`, Loki `3.2.1`, and Grafana `11.3.0`.
- Pinned embedding model revision
  `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`, warm after the first image/model
  download, normalized 384-dimensional vectors, CPU execution.

## Final independent gate

The final command was:

```sh
./scripts/verify-rag-core.sh all
```

It completed successfully after all repair callbacks. No required test selected
by the gate was skipped.

| Gate | Final result |
| --- | --- |
| Embedding unit/HTTP/adapter/settings | 84 passed |
| Core unit, API contract, failures, telemetry | 31 passed; 2 integration tests deliberately selected by the later integration gate |
| OpenAPI 3.1 validation and runtime-boundary check | Passed; exactly 9 matching product/health paths and no public micro-operation endpoint |
| Pinned real-model integration | 6 passed |
| Real-Qdrant adapter/catalog, generation lifecycle, rollback, retirement, tenant isolation | 2 passed |
| Locked Core image and security metadata | Passed; pinned base digest, non-root user, health check, SIGTERM, installed console entrypoint |
| Compose product workflow | Passed: create → ingest → retrieve → delete → retire, with both dependencies and Core healthy |
| Idempotency and partial failure | Passed: document/create/generation first execution and replay, canonical conflict `409/idempotency_conflict`, deterministic cleanup repair |
| Compatibility and failure mapping | Passed: malformed embeddings/JSON/timeout headers, schema mismatch before write, safe 500 containment, dependency timeout, bounded readiness |
| Generation provisioning/cutover/rollback | Passed against real Qdrant: empty target provisioning, explicit activation, stale-precondition `412`, predecessor retention, rollback, and complete logical retirement |
| Trusted tenant isolation | Passed against real Qdrant for search, deletion, and old-version cleanup |
| Reference retrieval load | Passed: 20 warm requests, concurrency 4, p50 `149.73 ms`, p95 `157.28 ms`, max `202.75 ms`; p95 limit `750 ms` |
| Snapshot restore | Passed: dedicated collection snapshot, deletion, restore, exact point/payload verification, cleanup |
| Observability configuration | Collector config valid; Prometheus config and all 10 alert rules valid; dashboard JSON valid |
| Live observability pipeline | All seven observability services healthy; Core OTLP product metrics, Qdrant and collector targets, and Grafana database health verified |
| Whitespace/configuration | `git diff --check` and full Compose configuration validation passed |

Live Prometheus queries returned all required samples checked by the gate:

```text
rag_core_http_requests_total
rag_core_ingestion_duration_milliseconds_count
rag_core_retrieval_duration_milliseconds_count
rag_core_dependency_requests_total
rag_core_idempotency_operations_total
rag_core_generation_operations_total
```

The exported samples used route/status plus allowlisted low-cardinality product
dimensions. Tests reject query, content, vector, collection, document, and tenant
telemetry attributes. Core explicit spans cover chunking, embedding, vector, and
generation operations and propagate active trace context to Embedding.

## Failure and callback record

1. The pinned Qdrant image did not include `curl`, making the original Compose
   health check permanently fail. DevOps replaced it with a bounded HTTP `/readyz`
   probe using Bash TCP support and independently verified health.
2. Docker suppressed the localhost Qdrant mapping when the container belonged
   only to an internal network. DevOps added a Qdrant-only development bridge;
   Core continues to use the private data network and Embedding cannot reach the
   database. Integration gates use isolated ports `16333`–`16337` to avoid local
   collisions.
3. The first Core E2E start failed deterministically with
   `Could not import module "app"`. DevOps called Developer back with the exact
   command/log and wheel invariant. Developer moved the ASGI target into the
   installed package and added a regression test. DevOps reran E2E and the full
   gate successfully.
4. The first coverage audit found missing idempotency-conflict, partial-recovery,
   real migration/rollback, tenant-isolation, readiness-failure, and product
   telemetry checks. DevOps called Developer back. Developer added the product
   behavior, safe instrumentation, and tests; DevOps independently reran the
   focused and complete gates successfully.
5. The final documentation audit found that the controller did not honor several
   published wire semantics. DevOps called Developer back with deterministic
   evidence for collection/generation idempotency, stale preconditions, ignored
   fields, malformed protocol input, response fields, metadata limits, and error
   taxonomy. Architect replaced the misleading migration route with explicit
   empty generation provisioning and restricted v1 isolation to `shared`; DBA
   made logical retirement cover every nonfailed generation. Developer added
   schema-exact contract and lifecycle tests. DevOps independently reran the
   original focused gates, real Qdrant, Compose E2E, observability validation,
   and the complete gate successfully.
6. The first ad-hoc live telemetry rerun encountered observability containers
   attached to a network removed by an earlier application-only teardown.
   DevOps recreated the complete application/observability profile, reran the
   public E2E workflow, and queried Prometheus successfully for the renamed
   `rag_core_generation_operations_total` samples before clean teardown.

## Deployment scope and known limitation

The reserved Qdrant catalog uses expected-state checks, a process lock, batched
writes, and post-write verification. Qdrant does not provide a transactional
compare-and-set spanning these catalog records, so strict concurrent activation
across multiple Core replicas is not guaranteed. The verified Compose deployment
runs one Core replica and fails closed. A horizontally scaled production deployment
must first move `CollectionCatalog` to a transactional store or add equivalent
distributed serialization. This limitation does not weaken the tested
single-replica boilerplate result.

The default idempotency adapter is process-local memory and the unauthenticated
development tenant is fixed to `local`. A production deployment needs a shared
TTL-enforcing idempotency store plus authenticated tenant/access context.
Generation provisioning intentionally creates an empty target. An authorized
out-of-band source-of-truth rebuild and validation workflow must populate it
before activation; activating an empty target is unsafe for a data-bearing
collection.

Qdrant cluster administration, scheduled backup execution, remote snapshot
storage, and production restore objectives remain platform responsibilities. The
local gate proves snapshot recoverability. It does not certify a production
out-of-band rebuild workflow or production RPO/RTO.
