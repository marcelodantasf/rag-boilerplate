"""Private failures raised by the Qdrant transport.

These exceptions intentionally carry no response body. Qdrant error payloads may
contain provider details and must never cross the adapter boundary.
"""

from __future__ import annotations


class QdrantAdapterError(RuntimeError):
    """Base class for failures private to the Qdrant adapter."""


class QdrantUnavailableError(QdrantAdapterError):
    """Qdrant could not complete a bounded request."""


class QdrantTimeoutError(QdrantUnavailableError):
    """Qdrant exceeded the adapter's operation timeout."""


class QdrantResourceNotFoundError(QdrantAdapterError):
    """The requested provider resource does not exist."""


class QdrantResourceConflictError(QdrantAdapterError):
    """The requested provider mutation conflicts with current state."""


class QdrantContractError(QdrantAdapterError):
    """Qdrant returned a response that violates the adapter contract."""
