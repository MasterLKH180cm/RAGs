"""Actual predicate semantics, eager validation and bounded regex execution."""
import time

import pytest

from rags import collection_filters as filters
from rags.collection_filters import compile_filters
from rags.registry import CapabilityError


pytestmark = pytest.mark.unit


def matches(where=None, where_document=None, *, metadata=None, document=None):
    return compile_filters(where, where_document).matches(
        metadata or {}, document, deadline=time.monotonic() + 1,
    )


@pytest.mark.parametrize(("where", "metadata", "expected"), [
    ({"x": 1}, {"x": 1.0}, True),
    ({"x": True}, {"x": 1}, False),
    ({"x": {"$eq": False}}, {"x": 0.0}, False),
    ({"x": {"$eq": None}}, {"x": None}, True),
    ({"x": {"$eq": None}}, {}, False),
    ({"x": {"$ne": 1}}, {"x": 2}, True),
    ({"x": {"$ne": None}}, {}, False),
    ({"x": {"$ne": 1}}, {"x": [2]}, False),
    ({"x": {"$gt": 2}}, {"x": 3}, True),
    ({"x": {"$gte": 2}}, {"x": 2.0}, True),
    ({"x": {"$lt": 2}}, {"x": 1.0}, True),
    ({"x": {"$lte": 2}}, {"x": 2}, True),
    ({"x": {"$gte": 1}}, {"x": True}, False),
    ({"x": {"$gt": 2}}, {"x": "3"}, False),
    ({"x": {"$in": [1, 2.0]}}, {"x": 2}, True),
    ({"x": {"$in": [1, 2]}}, {"x": True}, False),
    ({"x": {"$nin": [1, 2]}}, {"x": 3}, True),
    ({"x": {"$nin": [1, 2]}}, {}, False),
    ({"x": {"$in": []}}, {"x": 1}, False),
    ({"x": {"$nin": []}}, {"x": 1}, True),
    ({"x": {"$contains": "policy"}}, {"x": ["policy", "news"]}, True),
    ({"x": {"$contains": "policy"}}, {"x": "policy"}, False),
    ({"x": {"$contains": 1.0}}, {"x": [1, 2]}, True),
    ({"x": {"$contains": True}}, {"x": [1, 2]}, False),
    ({"x": {"$not_contains": "policy"}}, {"x": ["news"]}, True),
    ({"x": {"$not_contains": "policy"}}, {"x": []}, True),
    ({"x": {"$not_contains": "policy"}}, {}, False),
    ({"x": {"$not_contains": "policy"}}, {"x": "news"}, False),
])
def test_metadata_operators_preserve_type_missing_and_array_semantics(where, metadata, expected):
    assert matches(where, metadata=metadata) is expected


def test_logical_trees_and_field_ranges_are_conjunctive_and_keep_original_values():
    where = {
        "$and": [
            {"year": {"$gte": 2024, "$lt": 2027}, "status": "published"},
            {"$or": [{"tags": {"$contains": "policy"}}, {"public": True}]},
        ],
    }
    metadata = {"year": 2026, "status": "published", "tags": ["policy"], "public": False}
    compiled = compile_filters(where, {"$and": [{"$contains": "refund"}, {"$not_contains": "draft"}]})
    where["$and"][0]["status"] = "private"
    assert compiled.matches(metadata, "refund in 14 days", deadline=time.monotonic() + 1)
    assert not compiled.matches(metadata, "refund draft", deadline=time.monotonic() + 1)
    assert not compiled.matches({**metadata, "year": 2027}, "refund", deadline=time.monotonic() + 1)


@pytest.mark.parametrize(("predicate", "document", "expected"), [
    ({"$contains": "Refund"}, "Refund in 14 days", True),
    ({"$contains": "Refund"}, "refund in 14 days", False),
    ({"$contains": " "}, "two words", True),
    ({"$not_contains": "draft"}, "", True),
    ({"$not_contains": "draft"}, None, False),
    ({"$regex": r"(?m)^refund: \d+ days$"}, "title\nrefund: 14 days\nend", True),
    ({"$not_regex": r"\d+"}, "no numeric term", True),
    ({"$not_regex": r"\d+"}, None, False),
    ({"$or": [{"$contains": "refund"}, {"$regex": "退款"}]}, "退款期限", True),
    ({"$and": [{"$contains": "refund"}, {"$not_regex": "draft"}]}, "refund policy", True),
])
def test_full_document_predicates_are_case_sensitive_and_null_is_never_matched(predicate, document, expected):
    assert matches(where_document=predicate, document=document) is expected


def test_absent_filters_allow_vector_only_record():
    assert matches(metadata={}, document=None)
    assert not compile_filters(None, None).has_regex


@pytest.mark.parametrize(("where", "where_document"), [
    ({}, None),
    ({"$and": []}, None),
    ({"$or": "private-expression"}, None),
    ({"$and": [{"x": 1}], "y": 2}, None),
    ({"$unknown": []}, None),
    ({1: "invalid field"}, None),
    ({"_rags_private": "secret"}, None),
    ({"x": {}}, None),
    ({"x": {"$unknown": "private-expression"}}, None),
    ({"x": {"$eq": {"private": "nested"}}}, None),
    ({"x": {"$eq": []}}, None),
    ({"x": {"$gt": True}}, None),
    ({"x": {"$gt": "2"}}, None),
    ({"x": {"$eq": float("nan")}}, None),
    ({"x": {"$eq": float("inf")}}, None),
    ({"x": {"$eq": 2 ** 63}}, None),
    ({"x": {"$in": "wrong"}}, None),
    ({"x": {"$in": [1, "2"]}}, None),
    ({"x": {"$in": [1, True]}}, None),
    ({"x": {"$in": [None]}}, None),
    ({"x": {"$in": list(range(101))}}, None),
    ({"x": {"$contains": None}}, None),
    ({"x" * 129: 1}, None),
    ({"x": "\ud800"}, None),
    ({"x": "x" * 4097}, None),
    ({"x": "\x00"}, None),
    (None, {}),
    (None, {"$contains": ""}),
    (None, {"$contains": "x" * 257}),
    (None, {"$contains": 1}),
    (None, {"$unknown": "private-expression"}),
    (None, {"$contains": "x", "$regex": "x"}),
    (None, {"$regex": "["}),
    (None, {"$regex": "\udfff"}),
    (None, {"$and": [None]}),
])
def test_invalid_filters_have_fixed_private_safe_errors(where, where_document):
    with pytest.raises(CapabilityError) as exc:
        compile_filters(where, where_document)
    assert exc.value.code == "invalid_collection_filter"
    assert exc.value.status_code == 422
    assert "private-expression" not in str(exc.value)


def test_all_regex_branches_compile_before_short_circuit_or_any_record_scan():
    with pytest.raises(CapabilityError, match="Collection filter"):
        compile_filters(None, {"$or": [{"$contains": "safe"}, {"$regex": "("}]})
    with pytest.raises(CapabilityError) as exc:
        compile_filters(None, {"$or": [{"$contains": "safe"}, {"$regex": "valid"}]}, allow_regex=False)
    assert exc.value.code == "feature_disabled" and exc.value.status_code == 403


def test_node_and_depth_limits_count_both_trees():
    where = {"x": 1}
    for _ in range(7):
        where = {"$and": [where]}
    assert matches(where, metadata={"x": 1})
    with pytest.raises(CapabilityError):
        compile_filters({"$and": [where]}, None)
    assert matches({"$or": [{"x": n} for n in range(63)]}, metadata={"x": 20})
    with pytest.raises(CapabilityError):
        compile_filters({"$or": [{"x": n} for n in range(64)]}, None)
    with pytest.raises(CapabilityError):
        compile_filters({f"field{n}": n for n in range(64)}, {"$contains": "x"})
    assert matches({"x": {"$in": list(range(100))}}, metadata={"x": 99})


def test_shared_deadline_is_not_renewed_between_regex_records(monkeypatch):
    clock = [10.0]
    received = []

    class Pattern:
        def search(self, document, *, timeout):
            received.append(timeout)
            clock[0] += 0.4
            return object()

    monkeypatch.setattr(filters.regex, "compile", lambda *args: Pattern())
    monkeypatch.setattr(filters.time, "monotonic", lambda: clock[0])
    compiled = compile_filters(None, {"$regex": "term"})
    assert compiled.matches({}, "term", deadline=11.0)
    assert compiled.matches({}, "term", deadline=11.0)
    with pytest.raises(CapabilityError) as exc:
        compiled.matches({}, "term", deadline=11.0)
    assert exc.value.code == "collection_filter_timeout"
    assert received == pytest.approx([1.0, 0.6, 0.2])


def test_actual_catastrophic_regex_is_interrupted_by_native_timeout():
    compiled = compile_filters(None, {"$regex": r"(a|aa)+$"})
    started = time.monotonic()
    with pytest.raises(CapabilityError) as exc:
        compiled.matches({}, "a" * 20_000 + "!", deadline=started + 0.02)
    assert exc.value.code == "collection_filter_timeout" and exc.value.status_code == 422
    assert time.monotonic() - started < 2


@pytest.mark.parametrize("deadline", [0, float("nan"), float("inf"), True, None])
def test_invalid_or_expired_deadlines_fail_without_partial_match(deadline):
    with pytest.raises(CapabilityError):
        compile_filters(None, None).matches({}, None, deadline=deadline)


def test_corrupted_snapshot_fields_fail_without_echoing_data():
    with pytest.raises(CapabilityError) as exc:
        compile_filters(None, None).matches(["private-data"], "text", deadline=time.monotonic() + 1)
    assert exc.value.code == "invalid_collection_data"
    assert "private-data" not in str(exc.value)
