"""Production/local process entry point."""

from __future__ import annotations

import os

import uvicorn


def _server_port() -> int:
    raw = os.getenv("EMBEDDING_PORT", "8001")
    try:
        port = int(raw)
    except ValueError as error:
        raise ValueError("EMBEDDING_PORT must be an integer") from error
    if not 1 <= port <= 65_535:
        raise ValueError("EMBEDDING_PORT must be between 1 and 65535")
    return port


def main() -> None:
    """Run one worker; model replicas should be managed by the orchestrator."""

    uvicorn.run(
        "embedding_api.transport.http:create_app",
        factory=True,
        host="0.0.0.0",
        port=_server_port(),
        log_level=os.getenv("LOG_LEVEL", "INFO").lower(),
        access_log=False,
        workers=1,
    )


if __name__ == "__main__":
    main()
