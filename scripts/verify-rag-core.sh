#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
embedding_dir="$repo_dir/services/embedding-api"
core_dir="$repo_dir/services/core-rag-api"
gate="${1:-all}"

run_unit() {
  (
    cd "$embedding_dir"
    uv sync --frozen --extra test
    uv run pytest --ignore=tests/test_real_model_integration.py
  )
  (
    cd "$core_dir"
    uv sync --frozen --extra test
    uv run pytest -m "not integration and not e2e"
  )
}

run_contract() {
  cd "$repo_dir"
  uvx --from openapi-spec-validator==0.7.2 openapi-spec-validator docs/api/rag-core-openapi.yaml
  python3 scripts/verify-openapi-boundary.py docs/api/rag-core-openapi.yaml
  docker compose --profile application --profile observability config --quiet
}

run_model() {
  (
    cd "$embedding_dir"
    uv sync --frozen --extra test
    uv run pytest tests/test_real_model_integration.py
  )
}

run_integration() {
  cd "$repo_dir"
  qdrant_test_port="${QDRANT_TEST_PORT:-16333}"
  QDRANT_PORT="$qdrant_test_port" docker compose up --detach --wait qdrant
  (
    cd "$core_dir"
    QDRANT_TEST_URL="${QDRANT_TEST_URL:-http://127.0.0.1:$qdrant_test_port}" \
      uv run pytest -m integration
  )
}

run_image() {
  cd "$repo_dir"
  docker build --tag rag-boilerplate-core-rag-api:verify services/core-rag-api
  image_id="$(docker image inspect --format '{{.Id}}' rag-boilerplate-core-rag-api:verify)"
  python3 scripts/verify-container-security.py "$image_id"
}

run_recovery() {
  cd "$repo_dir"
  qdrant_recovery_port="${QDRANT_RECOVERY_PORT:-16337}"
  QDRANT_PORT="$qdrant_recovery_port" docker compose up --detach --wait qdrant
  (
    cd "$core_dir"
    uv run python ../../scripts/qdrant-recovery-drill.py "http://127.0.0.1:$qdrant_recovery_port"
  )
}

run_e2e() {
  cd "$repo_dir"
  export QDRANT_PORT="${E2E_QDRANT_PORT:-16334}"
  cleanup() {
    docker compose --profile application down --remove-orphans
  }
  trap cleanup EXIT
  docker compose --profile application up --detach --build --wait
  python3 scripts/rag-core-e2e-smoke.py "${RAG_CORE_BASE_URL:-http://127.0.0.1:8000}"
}

run_observability() {
  cd "$repo_dir"
  docker compose --profile observability config --quiet
  docker run --rm \
    --volume "$repo_dir/observability/otel-collector.yaml:/etc/otelcol-contrib/config.yaml:ro" \
    otel/opentelemetry-collector-contrib:0.111.0 \
    validate --config=/etc/otelcol-contrib/config.yaml
  docker run --rm \
    --volume "$repo_dir/observability/prometheus.yaml:/etc/prometheus/prometheus.yaml:ro" \
    --volume "$repo_dir/observability/alerts.yaml:/etc/prometheus/alerts.yaml:ro" \
    --entrypoint promtool \
    prom/prometheus:v2.55.1 \
    check config /etc/prometheus/prometheus.yaml
  python3 -m json.tool observability/grafana/dashboards/rag-overview.json >/dev/null
}

run_load() {
  cd "$repo_dir"
  export QDRANT_PORT="${LOAD_QDRANT_PORT:-16336}"
  cleanup() {
    docker compose --profile application down --remove-orphans
  }
  trap cleanup EXIT
  docker compose --profile application up --detach --build --wait
  python3 scripts/rag-core-load-smoke.py "${RAG_CORE_BASE_URL:-http://127.0.0.1:8000}"
}

case "$gate" in
  unit) run_unit ;;
  contract) run_contract ;;
  model) run_model ;;
  integration) run_unit; run_integration ;;
  image) run_image ;;
  recovery) run_recovery ;;
  e2e) run_e2e ;;
  load) run_load ;;
  observability) run_observability ;;
  all)
    run_unit
    run_contract
    run_model
    run_integration
    run_image
    run_recovery
    run_e2e
    run_load
    run_observability
    ;;
  *)
    echo "usage: $0 {unit|contract|model|integration|image|recovery|e2e|load|observability|all}" >&2
    exit 2
    ;;
esac
