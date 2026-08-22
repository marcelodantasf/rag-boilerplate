"""RAG Core API console entry point."""

import uvicorn

from rag_core.infrastructure.settings import Settings


def main() -> None:
    settings = Settings.from_env()
    uvicorn.run("rag_core.app:app", host="0.0.0.0", port=settings.rag_port, workers=1)


if __name__ == "__main__":
    main()
