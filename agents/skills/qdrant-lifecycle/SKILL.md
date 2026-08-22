---
name: qdrant-lifecycle
description: Design and implement RAG Core's Qdrant collection, payload, filtering, migration, and recovery boundary without exposing database internals publicly.
---

# Qdrant Lifecycle

Use this skill for the DBA role and for reviews of collection schemas, Qdrant
adapters, filters, migrations, aliases, capacity, or recovery behavior.

## Scope

Own the persistence behavior behind `VectorStore` and `CollectionCatalog`.
Coordinate public and port-level changes with Architect; coordinate automated
test implementation with Developer.

## Collection contract

For every logical collection, enforce and make inspectable:

- logical ID and provider-safe physical name;
- index schema version and migration state;
- immutable embedding model identity and revision;
- vector dimension, normalization, and distance metric;
- allowed payload fields and required payload indexes;
- tenant/access isolation policy, active alias, and timestamps.

One point represents one chunk. Use deterministic IDs derived from the logical
collection, document identity/version or content hash, and chunk index. Payloads
contain only fields required for retrieval, citation, filtering, compatibility,
and safe metadata.

## Required behavior

- Implement create/verify collection, create payload indexes, batch upsert,
  bounded search, trusted-filter translation, delete by document, alias
  activation, and deliberate collection retirement.
- Reject dimension, metric, normalization, model-revision, or schema mismatch
  before reads or writes.
- Treat tenant/access filters as security controls supplied by trusted server
  context, not as unverified request metadata.
- Use create, reindex, validate, alias cutover, rollback window, then deliberate
  retirement for migrations. Never mutate incompatible vector schemas in place.
- Keep Qdrant SDK models and raw point IDs inside the adapter.
- Specify bounded timeouts, least-privilege credentials, private networking,
  capacity signals, snapshot expectations, restore drills, and full-rebuild
  behavior.

## Collaboration and verification

Supply Developer with database invariants, fixtures, and realistic edge cases
for adapter and migration tests. Review the resulting automated tests, but
Developer owns their implementation and repair. Supply DevOps with Qdrant
health, latency, error, storage, backup-age, and restore signals to monitor.

Do not expose snapshot, compaction, replication, cluster administration, or raw
Qdrant query operations through `VectorCollectionController`; those are
platform workflows.

