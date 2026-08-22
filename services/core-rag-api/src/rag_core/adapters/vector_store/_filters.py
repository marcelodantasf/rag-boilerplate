"""Translate the public, provider-neutral filter grammar to Qdrant filters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from rag_core.domain.errors import InvalidRequestError
from rag_core.domain.models import (
    CollectionContract,
    FilterCondition,
    FilterExpression,
    FilterGroup,
    FilterGroupOperator,
    FilterOperator,
    JsonScalar,
    MetadataField,
    MetadataFieldType,
)

MAX_FILTER_DEPTH = 3
MAX_FILTER_PREDICATES = 10
MAX_IN_VALUES = 20


@dataclass(slots=True)
class _FilterBudget:
    predicates: int = 0


def translate_filter(
    expression: FilterExpression | None, contract: CollectionContract
) -> dict[str, Any] | None:
    if expression is None:
        return None
    schema = {field.name: field for field in contract.metadata_fields}
    return _translate(expression, schema, depth=1, budget=_FilterBudget())


def validate_metadata(
    metadata: dict[str, JsonScalar], contract: CollectionContract
) -> dict[str, JsonScalar]:
    schema = {field.name: field for field in contract.metadata_fields}
    unknown = sorted(set(metadata).difference(schema))
    if unknown:
        raise InvalidRequestError(
            "Metadata contains fields outside the collection schema",
            fields=unknown,
        )
    for name, value in metadata.items():
        _validate_scalar(value, schema[name], field_name=name)
    return dict(metadata)


def _translate(
    expression: FilterExpression,
    schema: dict[str, MetadataField],
    *,
    depth: int,
    budget: _FilterBudget,
) -> dict[str, Any]:
    if depth > MAX_FILTER_DEPTH:
        raise InvalidRequestError(
            "Filter nesting is too deep", max_depth=MAX_FILTER_DEPTH
        )
    if isinstance(expression, FilterCondition):
        budget.predicates += 1
        if budget.predicates > MAX_FILTER_PREDICATES:
            raise InvalidRequestError(
                "Filter has too many predicates",
                max_predicates=MAX_FILTER_PREDICATES,
            )
        return _condition(expression, schema)
    if not isinstance(expression, FilterGroup):
        raise InvalidRequestError("Filter expression is invalid")
    if not expression.clauses:
        raise InvalidRequestError("Filter groups must not be empty")
    if expression.operator is FilterGroupOperator.NOT and len(expression.clauses) != 1:
        raise InvalidRequestError("A not filter must contain exactly one clause")

    translated = [
        _translate(clause, schema, depth=depth + 1, budget=budget)
        for clause in expression.clauses
    ]
    if expression.operator is FilterGroupOperator.ALL:
        return {"must": translated}
    if expression.operator is FilterGroupOperator.ANY:
        return {"should": translated}
    if expression.operator is FilterGroupOperator.NOT:
        return {"must_not": translated}
    raise InvalidRequestError("Filter group operator is invalid")


def _condition(
    condition: FilterCondition, schema: dict[str, MetadataField]
) -> dict[str, Any]:
    field = schema.get(condition.field)
    if field is None:
        raise InvalidRequestError(
            "Filter field is not part of the collection schema",
            field=condition.field,
        )
    if not field.indexed:
        raise InvalidRequestError(
            "Filter field is not indexed", field=condition.field
        )
    key = f"metadata.{field.name}"

    if condition.operator is FilterOperator.IN:
        values = condition.value
        if not isinstance(values, tuple) or not values or len(values) > MAX_IN_VALUES:
            raise InvalidRequestError(
                "The in operator requires a non-empty bounded value list",
                field=field.name,
                max_values=MAX_IN_VALUES,
            )
        for value in values:
            _validate_scalar(value, field, field_name=field.name)
        return {"key": key, "match": {"any": list(values)}}

    if isinstance(condition.value, tuple):
        raise InvalidRequestError(
            "This filter operator requires one scalar value", field=field.name
        )
    _validate_scalar(condition.value, field, field_name=field.name)
    if condition.operator is FilterOperator.EQ:
        return {"key": key, "match": {"value": condition.value}}
    if condition.operator not in {
        FilterOperator.GT,
        FilterOperator.GTE,
        FilterOperator.LT,
        FilterOperator.LTE,
    }:
        raise InvalidRequestError("Filter operator is invalid", field=field.name)
    if field.type not in {MetadataFieldType.INTEGER, MetadataFieldType.FLOAT}:
        raise InvalidRequestError(
            "Range filters require an integer or float field", field=field.name
        )
    return {"key": key, "range": {condition.operator.value: condition.value}}


def _validate_scalar(
    value: JsonScalar, field: MetadataField, *, field_name: str
) -> None:
    expected: type[object]
    if field.type is MetadataFieldType.KEYWORD:
        expected = str
    elif field.type is MetadataFieldType.INTEGER:
        expected = int
    elif field.type is MetadataFieldType.FLOAT:
        expected = float
    elif field.type is MetadataFieldType.BOOLEAN:
        expected = bool
    else:
        raise InvalidRequestError("Metadata field type is invalid", field=field_name)
    if type(value) is not expected:
        raise InvalidRequestError(
            "Metadata value does not match the collection schema", field=field_name
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise InvalidRequestError(
            "Metadata values must be finite", field=field_name
        )
