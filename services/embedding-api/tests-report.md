# Embedding API Test Report

## Result

**PASS — 90 passed, 0 failed, 0 skipped.**

The implementation satisfies the requested single-document, true request-level
batch, and general-purpose text behavior. The final run included the real
pinned Sentence Transformers model and an end-to-end HTTP request using that
model.

## Environment

| Component | Value |
| --- | --- |
| Operating system | macOS 26.5.2 (arm64) |
| Python | 3.12.13 |
| FastAPI | 0.116.1 |
| HTTPX | 0.28.1 |
| pytest | 8.4.1 |
| sentence-transformers | 5.1.0 |
| PyTorch | 2.13.0 |
| Inference device | CPU |
| Real model | `sentence-transformers/all-MiniLM-L6-v2` |
| Pinned revision | `c9745ed1d9f207416be6d2e6f8de32d1f16199bf` |

Dependencies were installed in the service-local virtual environment with
`uv`. The real model artifact was downloaded on the first integration run and
read from the local Hugging Face cache on the final run.

## Commands

From `services/embedding-api`:

```sh
uv sync --extra test
uv run pytest -q --ignore=tests/test_real_model_integration.py
uv run pytest -q
uv run pytest --collect-only -q
```

The isolated unit/HTTP suite passed before the real model was exercised. The
authoritative final command completed with:

```text
90 passed in 3.57s
```

The first real-model run, including artifact download, completed with 81 tests
in 23.26 seconds and emitted one third-party Hugging Face Xet deprecation
warning. The warning was not produced by service code and did not affect model
loading or inference. Two additional readiness/end-to-end cases were then
added. Seven DevOps-phase configuration and safe-observability cases brought
the final total to 90.

## Coverage

### Request and response contract

- A single arbitrary string is normalized to a one-item batch and returns one
  vector.
- A list is passed to the engine in one call, not a loop of single calls.
- Batch indexes and vectors preserve input order.
- Single and batch requests share the same `model`, `dimension`, `vectors`, and
  `usage` response shape.
- Law, recipe, educational/research, source code, Unicode/multilingual, and
  multiline documents all use the same structure-free input path.
- Domain-specific fields such as `input_type` and `metadata`, structured input
  objects, and other extra fields are rejected rather than interpreted.

### Validation and routing

- Missing, null, numeric, object, mixed-type, blank, and empty inputs are
  rejected with safe structured errors.
- Empty arrays and empty array items are rejected.
- The maximum batch item count is enforced before inference.
- Per-item and whole-request UTF-8 byte limits are enforced, including
  multibyte characters.
- Unsupported model IDs are rejected before inference.
- Environment defaults, aliases, overrides, positive integer requirements,
  normalization parsing, model identity, immutable revision, and byte-limit
  relationships are validated.

### Engine contract and errors

- Wrong output model, vector count, vector dimension, non-finite components,
  Boolean components, and nonnumeric components are rejected as
  `embedding_contract_violation`.
- Model load, tokenizer, and encode failures are normalized without exposing
  provider exception text.
- Validation, oversized input, unsupported model, unavailable engine, and
  contract errors include a safe `code`, `message`, and trace ID.
- Valid incoming trace IDs are preserved; absent, non-ASCII, or overlong trace
  IDs are replaced and returned in both body/header where applicable.
- Liveness, ready state, and the pre-startup not-ready response are covered.

### Tokenization and real model

- The adapter tokenizes with `truncation=False` before inference.
- Per-item model token limits and total batch token limits reject work before
  `encode`, proving there is no silent truncation path.
- Adapter configuration pins source/revision, disables remote code, and passes
  device/cache settings explicitly.
- The real pinned model reports dimension 384 and produces one finite,
  L2-normalized vector per input.
- Real-model batches preserve count/order across reversed inputs and are
  deterministic within a small floating-point tolerance.
- A tolerant semantic check confirms that two paraphrased bread/oven sentences
  are substantially closer than an unrelated quantum-physics sentence.
- An overlong real input is explicitly rejected instead of truncated.
- The real engine successfully serves a mixed-domain batch through
  `POST /v1/embeddings`, including usage, indexes, dimensions, and trace
  propagation.

## Failures and repair loop

No service implementation defect was found. Three failures in the first local
test-only run were caused by test fixture construction and an invalid HTTPX
header fixture; those test files were corrected locally. Because the service
was not at fault, the developer repair loop was not invoked.

## Remaining risks and intentionally untested scope

- Results were exercised on CPU/arm64 only; GPU/MPS numerical behavior was not
  run, though assertions intentionally avoid exact float snapshots.
- No load, latency percentile, bounded-concurrency, cancellation, or saturation
  testing was performed. Those controls are outside the two required product
  behaviors and remain an operational concern.
- The Docker image could not be built or run because the local Docker daemon
  socket was unavailable. Compose rendered successfully, the exact base-image
  manifest was available, and dependency-lock validation passed, but a real
  image build remains required on a Docker-enabled host.
- Offline model-cache seeding was not exercised. A first container start needs
  network access to the pinned Hugging Face artifact unless the named cache
  volume is pre-populated.
- The semantic smoke check is deliberately narrow. It validates that the
  configured model is doing meaningful inference, not broad embedding quality
  across every language or subject domain.

## DevOps verification addendum

The production console entry point was started locally against the real cached
model. Readiness returned `200`; one single-document request returned one
384-dimensional vector; one three-document batch returned three ordered
384-dimensional vectors. All three responses propagated their supplied trace
IDs. The emitted JSON events contained status, latency, route, trace, safe
model/revision/device data, aggregate byte counts, item counts, and token counts
without raw text or vectors.

Additional checks completed successfully:

```sh
uv lock --check
docker compose config --quiet
docker manifest inspect python:3.12.13-slim-bookworm
uv run pytest -q
```

The attempted `docker compose build embedding-api` stopped before reading the
Dockerfile because `/Users/marcelo/.docker/run/docker.sock` did not exist.
