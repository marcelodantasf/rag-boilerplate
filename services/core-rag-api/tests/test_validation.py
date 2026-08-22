import pytest

from rag_core.application.validation import ensure_compatible, validate_filter, validate_metadata
from rag_core.domain.errors import EmbeddingSchemaMismatchError, InvalidRequestError
from rag_core.domain.models import (
    EmbeddingResult,
    FilterCondition,
    FilterGroup,
    FilterGroupOperator,
    FilterOperator,
)


def test_embedding_contract_rejects_mismatch(contract) -> None:
    result = EmbeddingResult(
        contract.embedding.model_id,
        "different-revision",
        contract.embedding.dimension,
        contract.embedding.normalized,
        ((0.0,) * contract.embedding.dimension,),
    )
    with pytest.raises(EmbeddingSchemaMismatchError) as captured:
        ensure_compatible(contract, result)
    assert captured.value.details == {"fields": ["revision"]}


def test_metadata_requires_declared_typed_fields(contract) -> None:
    validate_metadata({"department": "people", "year": 2026, "confidence": 1, "published": True}, contract)
    with pytest.raises(InvalidRequestError):
        validate_metadata({"unknown": "value"}, contract)
    with pytest.raises(InvalidRequestError):
        validate_metadata({"year": "2026"}, contract)


def test_filter_tree_is_provider_neutral_and_typed(contract) -> None:
    expression = FilterGroup(
        FilterGroupOperator.ALL,
        (
            FilterCondition("department", FilterOperator.IN, ("people", "legal")),
            FilterCondition("year", FilterOperator.GTE, 2024),
        ),
    )
    validate_filter(expression, contract)
    with pytest.raises(InvalidRequestError):
        validate_filter(FilterCondition("department", FilterOperator.GT, "people"), contract)
    with pytest.raises(InvalidRequestError):
        validate_filter(FilterGroup(FilterGroupOperator.NOT, (expression, expression)), contract)
