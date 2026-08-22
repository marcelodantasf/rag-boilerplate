"""Installed ASGI application target."""

from rag_core.transport.http import create_app

app = create_app()
