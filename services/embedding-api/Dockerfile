# Keep this image intentionally small and dependency-free until this service has
# an implementation and its own dependency manifest.
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --system app && useradd --system --gid app --create-home app

COPY --chown=app:app src/ ./src/

USER app

# Add the pinned inference runtime, model strategy, and application command only
# when implemented. This container is opt-in through the "application" profile.
