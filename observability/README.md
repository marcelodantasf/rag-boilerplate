# Local observability profile

The optional Compose `observability` profile receives OTLP telemetry, scrapes
Core, Qdrant, and container metrics, and provisions a Grafana overview plus alert
rules.

```sh
docker compose --profile application --profile observability up --build
```

The local defaults expose Grafana at `http://localhost:3000` and Prometheus at
`http://localhost:9090`. Change the credentials in `.env` before using this on a
shared machine. Prometheus retains seven days, Tempo retains traces for one day,
and Loki retains logs for seven days. These local single-node services are not a
production high-availability observability platform.

Core and Embedding send OTLP to `otel-collector:4318`. The collector exports
metrics to Prometheus, traces to Tempo, and OTLP logs to Loki. Services must emit
OTLP telemetry themselves; the collector intentionally does not mount Docker log
files. Standard output therefore remains available through `docker compose logs`
without granting the collector access to the Docker daemon.

The cAdvisor container is optional local infrastructure and requires privileged
host inspection to report container CPU, memory, restart, and filesystem signals.
Do not copy that privilege into an application container or a shared deployment;
use the orchestrator's native metrics integration there.

Qdrant snapshot age is an external platform signal. The provided stale-snapshot
alert activates only when a snapshot job publishes
`qdrant_snapshot_last_success_timestamp_seconds`; Core never creates or manages
database backups.
