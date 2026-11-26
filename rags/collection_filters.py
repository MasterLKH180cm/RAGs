"""Bounded collection predicates with eager validation and shared regex deadlines."""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import regex

from rags.collection_models import canonical_metadata, valid_text
from rags.native import equal_value
from rags.registry import CapabilityError


MAX_FILTER_DEPTH = 8
MAX_FILTER_NODES = 64
MAX_FILTER_LIST = 100
MAX_PATTERN_CHARS = 256
_MISSING = object()
_FIELD_OPERATORS = frozenset({
    "$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin",
    "$contains", "$not_contains",
})
_DOCUMENT_OPERATORS = frozenset({"$contains", "$not_contains", "$regex", "$not_regex"})
_LOGICAL_OPERATORS = frozenset({"$and", "$or"})


def _invalid_filter() -> CapabilityError:
    return CapabilityError(
        "invalid_collection_filter", "Collection filter is invalid or exceeds its supported limits", 422,
    )


def filter_remaining_seconds(deadline: float) -> float:
    """Use the caller's single monotonic deadline; never renew per record or branch."""
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise _invalid_filter()
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise CapabilityError("collection_filter_timeout", "Collection filtering exceeded its time limit", 422)
    return remaining


@dataclass(frozen=True)
class _Predicate:
    kind: str
    operator: str
    field: str | None = None
    expected: Any = None
    children: tuple[_Predicate, ...] = ()

    def matches(self, metadata: dict, document: str | None, deadline: float) -> bool:
        remaining = filter_remaining_seconds(deadline)
        if self.kind == "logical":
            if self.operator == "$and":
                return all(child.matches(metadata, document, deadline) for child in self.children)
            return any(child.matches(metadata, document, deadline) for child in self.children)
        if self.kind == "document":
            if document is None:
                return False
            if self.operator in {"$regex", "$not_regex"}:
                try:
                    found = self.expected.search(document, timeout=remaining) is not None
                except TimeoutError as exc:
                    raise CapabilityError(
                        "collection_filter_timeout", "Collection filtering exceeded its time limit", 422,
                    ) from exc
                filter_remaining_seconds(deadline)
            else:
                found = self.expected in document
            return not found if self.operator in {"$not_contains", "$not_regex"} else found
        actual = metadata.get(self.field, _MISSING)
        if actual is _MISSING:
            return False
        if self.operator in {"$contains", "$not_contains"}:
            if not isinstance(actual, list):
                return False
            found = any(equal_value(value, self.expected) for value in actual)
            return not found if self.operator == "$not_contains" else found
        if type(actual) not in (str, int, float, bool, type(None)):
            return False
        if self.operator in {"$gt", "$gte", "$lt", "$lte"}:
            if type(actual) not in (int, float):
                return False
            return {
                "$gt": actual > self.expected, "$gte": actual >= self.expected,
                "$lt": actual < self.expected, "$lte": actual <= self.expected,
            }[self.operator]
        if self.operator in {"$in", "$nin"}:
            found = any(equal_value(actual, value) for value in self.expected)
            return not found if self.operator == "$nin" else found
        found = equal_value(actual, self.expected)
        return not found if self.operator == "$ne" else found


@dataclass(frozen=True)
class CompiledFilter:
    """Immutable, already-validated predicates shared by every record in a scan."""

    metadata: _Predicate | None
    document: _Predicate | None
    has_regex: bool = False

    def matches(self, metadata: dict, document: str | None, *, deadline: float) -> bool:
        filter_remaining_seconds(deadline)
        if not isinstance(metadata, dict) or (document is not None and not isinstance(document, str)):
            raise CapabilityError("invalid_collection_data", "Stored collection data is invalid", 500)
        matched = (
            (self.metadata is None or self.metadata.matches(metadata, document, deadline))
            and (self.document is None or self.document.matches(metadata, document, deadline))
        )
        filter_remaining_seconds(deadline)
        return matched


class _Compiler:
    def __init__(self, allow_regex: bool):
        self.allow_regex = allow_regex
        self.nodes = 0
        self.has_regex = False

    def _visit(self, depth: int) -> None:
        self.nodes += 1
        if depth > MAX_FILTER_DEPTH or self.nodes > MAX_FILTER_NODES:
            raise _invalid_filter()

    @staticmethod
    def _scalar(value: Any) -> Any:
        if type(value) not in (str, int, float, bool, type(None)):
            raise _invalid_filter()
        try:
            return canonical_metadata({"value": value})["value"]
        except (ValueError, TypeError, UnicodeError) as exc:
            raise _invalid_filter() from exc

    def _field(self, field: str, operator: str, value: Any) -> _Predicate:
        if operator not in _FIELD_OPERATORS:
            raise _invalid_filter()
        if operator in {"$in", "$nin"}:
            if not isinstance(value, list) or len(value) > MAX_FILTER_LIST:
                raise _invalid_filter()
            try:
                expected = tuple(canonical_metadata({"values": value})["values"])
            except (ValueError, TypeError, UnicodeError) as exc:
                raise _invalid_filter() from exc
        else:
            expected = self._scalar(value)
            if operator in {"$gt", "$gte", "$lt", "$lte"} and type(expected) not in (int, float):
                raise _invalid_filter()
            if operator in {"$contains", "$not_contains"} and expected is None:
                raise _invalid_filter()
        return _Predicate("field", operator, field=field, expected=expected)

    def parse(self, expression: dict | None, *, kind: str, depth: int = 1) -> _Predicate | None:
        if expression is None and depth == 1:
            return None
        self._visit(depth)
        if not isinstance(expression, dict) or not expression:
            raise _invalid_filter()
        keys = list(expression)
        if any(not isinstance(key, str) for key in keys):
            raise _invalid_filter()
        logical = set(keys) & _LOGICAL_OPERATORS
        if logical:
            if len(keys) != 1:
                raise _invalid_filter()
            operator = keys[0]
            values = expression[operator]
            if not isinstance(values, list) or not 1 <= len(values) <= MAX_FILTER_LIST:
                raise _invalid_filter()
            # Eager construction validates every branch, including short-circuited ones.
            children = tuple(self.parse(value, kind=kind, depth=depth + 1) for value in values)
            return _Predicate("logical", operator, children=children)
        if kind == "document":
            if len(keys) != 1 or keys[0] not in _DOCUMENT_OPERATORS:
                raise _invalid_filter()
            operator = keys[0]
            value = expression[operator]
            try:
                valid_text(value, MAX_PATTERN_CHARS, allow_empty=True)
                if not value:
                    raise ValueError("Empty document predicate")
            except (ValueError, TypeError, UnicodeError) as exc:
                raise _invalid_filter() from exc
            if operator in {"$regex", "$not_regex"}:
                self.has_regex = True
                if not self.allow_regex:
                    raise CapabilityError("feature_disabled", "Feature is disabled: collections.regex", 403)
                try:
                    value = regex.compile(value, regex.VERSION1)
                except (regex.error, ValueError, OverflowError, RecursionError) as exc:
                    raise _invalid_filter() from exc
            return _Predicate("document", operator, expected=value)
        predicates = []
        for field, value in expression.items():
            try:
                valid_text(field, 128)
            except (ValueError, TypeError, UnicodeError) as exc:
                raise _invalid_filter() from exc
            if field.startswith(("$", "_rags_")):
                raise _invalid_filter()
            operations = value if isinstance(value, dict) else {"$eq": value}
            if not operations:
                raise _invalid_filter()
            for operator, expected in operations.items():
                if predicates:
                    self._visit(depth)
                if not isinstance(operator, str):
                    raise _invalid_filter()
                predicates.append(self._field(field, operator, expected))
        return predicates[0] if len(predicates) == 1 else _Predicate(
            "logical", "$and", children=tuple(predicates),
        )


def compile_filters(
    where: dict | None, where_document: dict | None, *, allow_regex: bool = True,
) -> CompiledFilter:
    """Compile both trees under one node budget, before any model/network operation."""
    compiler = _Compiler(allow_regex)
    metadata = compiler.parse(where, kind="metadata")
    document = compiler.parse(where_document, kind="document")
    return CompiledFilter(metadata, document, compiler.has_regex)
