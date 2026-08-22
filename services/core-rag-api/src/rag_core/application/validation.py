"""Collection compatibility, metadata, and filter validation."""

import re

from rag_core.domain.errors import EmbeddingSchemaMismatchError, InvalidRequestError
from rag_core.domain.models import (
    CollectionContract,
    EmbeddingResult,
    FilterCondition,
    FilterExpression,
    FilterGroup,
    FilterGroupOperator,
    FilterOperator,
    Metadata,
    MetadataField,
    MetadataFieldType,
)


RESOURCE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")


def validate_resource_id(value: str, field: str) -> str:
    if RESOURCE_ID.fullmatch(value) is None:
        raise InvalidRequestError(f"{field} has an invalid format", field=field)
    return value


def ensure_compatible(contract: CollectionContract, result: EmbeddingResult) -> None:
    differences: list[str] = []
    if result.model_id != contract.embedding.model_id:
        differences.append("model")
    if result.revision != contract.embedding.revision:
        differences.append("revision")
    if result.dimension != contract.embedding.dimension:
        differences.append("dimension")
    if result.normalized != contract.embedding.normalized:
        differences.append("normalization")
    if differences:
        raise EmbeddingSchemaMismatchError(fields=differences)


def validate_metadata(metadata: Metadata, contract: CollectionContract) -> None:
    schema = {field.name: field for field in contract.metadata_fields}
    for key, value in metadata.items():
        field = schema.get(key)
        if field is None:
            raise InvalidRequestError("Metadata field is not declared by the collection", field=key)
        if not _matches_type(value, field):
            raise InvalidRequestError("Metadata value has the wrong type", field=key, expected=field.type.value)


def validate_filter(expression: FilterExpression | None, contract: CollectionContract, *, max_depth: int = 3, max_conditions: int = 10) -> None:
    if expression is None:
        return
    schema = {field.name: field for field in contract.metadata_fields}
    count = 0

    def visit(node: FilterExpression, depth: int) -> None:
        nonlocal count
        if depth > max_depth:
            raise InvalidRequestError("Filter nesting is too deep")
        if isinstance(node, FilterGroup):
            if not node.clauses:
                raise InvalidRequestError("Filter groups may not be empty")
            if node.operator == FilterGroupOperator.NOT and len(node.clauses) != 1:
                raise InvalidRequestError("A not filter must contain exactly one clause")
            for clause in node.clauses:
                visit(clause, depth + 1)
            return
        count += 1
        if count > max_conditions:
            raise InvalidRequestError("The filter has too many conditions")
        field = schema.get(node.field)
        if field is None or not field.indexed:
            raise InvalidRequestError("Filter field is not indexed", field=node.field)
        if node.operator == FilterOperator.IN:
            if not isinstance(node.value, tuple) or not 1 <= len(node.value) <= 20:
                raise InvalidRequestError("An in filter requires 1 to 20 values", field=node.field)
            if any(not _matches_type(value, field) for value in node.value):
                raise InvalidRequestError("Filter value has the wrong type", field=node.field)
        else:
            if isinstance(node.value, tuple) or not _matches_type(node.value, field):
                raise InvalidRequestError("Filter value has the wrong type", field=node.field)
        if node.operator in {FilterOperator.GT, FilterOperator.GTE, FilterOperator.LT, FilterOperator.LTE} and field.type not in {MetadataFieldType.INTEGER, MetadataFieldType.FLOAT}:
            raise InvalidRequestError("Range operators require a numeric field", field=node.field)

    visit(expression, 1)


def _matches_type(value: object, field: MetadataField) -> bool:
    if field.type == MetadataFieldType.KEYWORD:
        return isinstance(value, str)
    if field.type == MetadataFieldType.INTEGER:
        return type(value) is int
    if field.type == MetadataFieldType.FLOAT:
        return type(value) in {int, float}
    return type(value) is bool
