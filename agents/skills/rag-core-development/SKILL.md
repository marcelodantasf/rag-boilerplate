---
name: rag-core-development
description: Implement the RAG Core API and own its complete automated test suite, including resolving every defect returned by the DevOps verification gate.
---

# RAG Core Development and Testing

Use this skill for application implementation, controller/use-case work,
Embedding API integration, test creation, or any test failure returned by
DevOps.

## Implementation ownership

- Build the typed Core package, validated settings, application factory,
  middleware, error mapping, health endpoints, and graceful shutdown.
- Implement the Architect's ports and provider-neutral domain models.
- Implement `RagController` product operations for document ingestion, document
  deletion, and top-k retrieval.
- Implement `VectorCollectionController` application orchestration while DBA's
  adapter owns Qdrant translation and lifecycle mechanics.
- Implement the Embedding API gateway with deadlines, trace propagation,
  stable error mapping, and strict response/model/dimension validation.
- Implement deterministic normalization, chunking, IDs, bounded batches,
  idempotent upserts, partial-write recovery, approved filters, score policy,
  and citable result shaping.
- Keep OpenAPI and examples synchronized with actual runtime behavior; request
  Architect review for semantic changes.

## Test ownership

Developer owns all automated product tests and their fixtures, including:

- domain and application unit tests;
- HTTP/OpenAPI contract and error-envelope tests;
- Embedding API and vector-store adapter contract tests;
- real-Qdrant integration tests for schema, filter, upsert, search, and delete;
- Compose end-to-end ingest/retrieve/delete and deterministic-corpus quality;
- idempotency, retry, timeout, malformed dependency, restart, partial-write,
  migration/cutover/rollback, isolation, and graceful-shutdown scenarios;
- performance/load checks and documentation-contract checks agreed with
  Architect and DevOps.

Test meaningful behavior and invariants. Do not assert exact floating-point
vectors or scores across hardware. Never weaken or delete a valid assertion to
make a gate pass without an Architect-approved contract change.

## DevOps callback protocol

When DevOps returns a failure:

1. Reproduce it using the supplied command and environment.
2. Determine whether implementation, fixture, assertion, or an approved
   contract changed.
3. Consult DBA for persistence semantics or Architect for contract semantics
   when necessary; remain responsible for resolution.
4. Apply the narrow fix, add a regression test when behavior was missing, and
   run the focused test plus every affected suite.
5. Report root cause, changed files, commands, and results to DevOps and request
   an independent full-gate rerun.

Do not declare success based only on a local focused test. Completion occurs
only after DevOps independently reruns every required gate successfully.

