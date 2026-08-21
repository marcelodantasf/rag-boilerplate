"""ASGI entry point kept at ``app:app`` for local runners."""

from embedding_api.transport.http import create_app

app = create_app()
