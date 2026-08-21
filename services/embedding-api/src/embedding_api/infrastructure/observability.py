"""Lightweight structured request logging with a deliberately safe field set."""

from __future__ import annotations

import json
import logging
from typing import Any


REQUEST_LOGGER_NAME = "embedding_api.requests"
_HANDLER_MARKER = "_embedding_api_json_handler"


def configure_request_logging(level: str) -> logging.Logger:
    """Configure one JSON line per request without changing global logging."""

    logger = logging.getLogger(REQUEST_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    if not any(getattr(handler, _HANDLER_MARKER, False) for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        setattr(handler, _HANDLER_MARKER, True)
        logger.addHandler(handler)
    return logger


def log_request(logger: logging.Logger, fields: dict[str, Any]) -> None:
    """Serialize only caller-selected, aggregate request metadata."""

    logger.info(json.dumps(fields, separators=(",", ":"), sort_keys=True))
