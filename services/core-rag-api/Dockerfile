# Keep this image intentionally small and dependency-free until this service has
# an implementation and its own dependency manifest.
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --system app && useradd --system --gid app --create-home app

COPY --chown=app:app src/ ./src/

USER app

# Add this service's dependency installation and ASGI/WSGI command only when the
# implementation exists. Its container is opt-in through the "application"
# Compose profile meanwhile.
