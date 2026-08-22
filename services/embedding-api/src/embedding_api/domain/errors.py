"""Safe errors that may cross the HTTP boundary."""

from typing import Any


class EmbeddingError(Exception):
    code = "embedding_error"
    status_code = 500
    safe_message = "The embedding request could not be completed."

    def __init__(self, message: str | None = None, *, details: dict[str, Any] | None = None):
        super().__init__(message or self.safe_message)
        self.message = message or self.safe_message
        self.details = details


class InvalidInputError(EmbeddingError):
    code = "invalid_input"
    status_code = 422
    safe_message = "Input must contain one or more non-empty text strings."


class InputTooLargeError(EmbeddingError):
    code = "input_too_large"
    status_code = 413
    safe_message = "Input exceeds the configured embedding limits."


class UnsupportedModelError(EmbeddingError):
    code = "unsupported_model"
    status_code = 404
    safe_message = "The requested model is not supported."


class EmbeddingEngineUnavailableError(EmbeddingError):
    code = "embedding_engine_unavailable"
    status_code = 503
    safe_message = "The embedding engine is temporarily unavailable."


class EmbeddingContractViolationError(EmbeddingError):
    code = "embedding_contract_violation"
    status_code = 500
    safe_message = "The embedding engine returned an invalid result."
