"""Pure ranking, graph source identity and structured result contracts."""
import math

import pytest

from rags.native import aggregate_rows, cluster_vectors, cosine, equal_value, graph_search, row_matches, rrf
from rags.storage import Store, timestamp

pytestmark = pytest.mark.unit


def test_rrf_deduplicates_each_ranking_and_resolves_equal_scores_by_source_id():
    result = rrf([["b", "a", "a", "b"], ["a", "b"]], 5)
    assert result == [("a", 1 / 61 + 1 / 62), ("b", 1 / 61 + 1 / 62)]
    assert rrf([["b", "a"], ["a", "b"]], 1) == result[:1]
    assert rrf([], 5) == []


def test_cosine_preserves_direction_scale_and_zero_vector_semantics():
    assert cosine([3.0, 4.0], [6.0, 8.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine([1.0, 0.0], [-1.0, 0.0]) == -1.0
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


@pytest.mark.parametrize(("left", "right"), [([], []), ([1.0], [1.0, 2.0]),
                                              ([math.nan], [1.0]), ([1.0], [math.inf])])
def test_cosine_rejects_mismatched_empty_and_nonfinite_vectors(left, right):
    with pytest.raises(ValueError):
        cosine(left, right)


def test_clustering_has_bounded_fan_in_stable_ties_and_exactly_once_source_membership():
    nodes = [{"id": "d", "embedding": [0.0, -1.0]}, {"id": "c", "embedding": [0.0, 1.0]},
             {"id": "b", "embedding": [1.0, 0.0]}, {"id": "a", "embedding": [1.0, 0.0]}]
    groups = cluster_vectors(nodes, group_size=2)
    assert [[node["id"] for node in group] for group in groups] == [["a", "b"], ["c", "d"]]
    assert groups == cluster_vectors(list(reversed(nodes)), group_size=2)
    assert sorted(node["id"] for group in groups for node in group) == ["a", "b", "c", "d"]
    assert all(len(group) <= 2 for group in groups)


def test_graph_cycle_and_hop_limit_keep_only_reachable_source_evidence():
    edges = [{"source": left, "target": right, "evidence_id": source}
             for left, right, source in [("Alpha", "Beta", "ab"), ("Beta", "Gamma", "bc"),
                                        ("Gamma", "Alpha", "ca"), ("Gamma", "Delta", "cd"),
                                        ("Delta", "Epsilon", "de"), ("Private", "Other", "private")]]
    edges.append(dict(edges[0]))
    hits, trace = graph_search(edges, ["alpha"], max_hops=2)
    assert hits == [("ab", 1.0), ("ca", 1.0), ("bc", 0.5), ("cd", 0.5)]
    assert trace["visited"] == 4 and trace["seeds"] == ["alpha"]
    assert trace["max_hops"] == 2
    assert graph_search(edges, ["missing"])[0] == []
    no_hits, no_seeds = graph_search([], ["alpha"])
    assert no_hits == [] and no_seeds["visited"] == 0
    assert graph_search(edges, ["alpha"], max_hops=0)[0] == []


@pytest.mark.parametrize(("left", "right", "expected"), [
    (True, 1, False), (False, 0.0, False), (True, True, True), (1, 1.0, True),
    ("1", 1, False), (None, None, True), ("North", "north", False),
])
def test_structured_equality_distinguishes_boolean_string_and_numeric_domains(left, right, expected):
    assert equal_value(left, right) is expected


@pytest.mark.parametrize(("operator", "actual", "expected", "matched"), [
    ("eq", 10, 10.0, True), ("ne", True, 1, True),
    ("in", 2, [False, 2.0], True), ("in", True, [1], False), ("in", 2, "2", False),
    ("contains", "north-east", "east", True), ("contains", ["east"], "east", False),
    ("gt", 5, 4, True), ("gte", 5, 5, True), ("lt", 4, 5, True), ("lte", 5, 5, True),
    ("gt", True, 0, False), ("gte", "5", 5, False), ("lt", 5, 4, False),
])
def test_structured_predicate_operator_contracts(operator, actual, expected, matched):
    assert row_matches({"value": actual}, [{"field": "value", "op": operator, "value": expected}]) is matched


def test_structured_predicates_require_all_fields_and_all_conditions():
    assert row_matches({"amount": 5}, [])
    assert not row_matches({"amount": 5}, [{"field": "missing", "op": "ne", "value": None}])
    assert not row_matches({"amount": 5}, [{"field": "amount", "op": "gte", "value": 2},
                                         {"field": "amount", "op": "lt", "value": 4}])
    with pytest.raises(ValueError, match="Unsupported structured predicate"):
        row_matches({"amount": 5}, [{"field": "amount", "op": "raw-sql", "value": 2}])


@pytest.mark.parametrize(("operation", "expected"), [("sum", 9.0), ("mean", 3.0), ("min", -2), ("max", 7)])
def test_aggregations_use_every_matching_numeric_row(operation, expected):
    rows = [{"amount": -2}, {"amount": 4.0}, {"amount": 7}]
    assert aggregate_rows(rows, {"operation": operation, "field": "amount"}) == {
        "operation": operation, "field": "amount", "value": expected, "matched_rows": 3,
    }
    empty = aggregate_rows([], {"operation": operation, "field": "amount"})
    assert empty["matched_rows"] == 0
    assert empty["value"] == (0 if operation == "sum" else None)


def test_exact_zero_count_is_a_result_and_absent_aggregation_stays_absent():
    assert aggregate_rows([], {"operation": "count"}) == {"operation": "count", "value": 0, "matched_rows": 0}
    assert aggregate_rows([{"amount": 0}], {"operation": "sum", "field": "amount"})["value"] == 0
    assert aggregate_rows([], None) is None


@pytest.mark.parametrize("row", [{}, {"amount": None}, {"amount": True}, {"amount": "5"},
                                  {"amount": math.nan}, {"amount": math.inf}])
def test_numeric_aggregations_reject_missing_nonnumeric_and_nonfinite_values(row):
    with pytest.raises(ValueError, match="present and numeric"):
        aggregate_rows([row], {"operation": "mean", "field": "amount"})


def test_aggregation_requires_field_and_supported_operation():
    with pytest.raises(ValueError, match="requires a field"):
        aggregate_rows([{"amount": 3}], {"operation": "sum"})
    with pytest.raises(ValueError, match="Unsupported structured aggregation"):
        aggregate_rows([{"amount": 3}], {"operation": "execute", "field": "amount"})


def test_timestamp_normalizes_offsets_for_exact_storage_comparison():
    assert timestamp("2024-01-02T03:04:05+08:00") == "2024-01-01T19:04:05.000000+00:00"
    assert timestamp("2024-01-01T19:04:05Z") == timestamp("2024-01-02T03:04:05+08:00")
    for value in ("not-a-date", "2024-01-01T19:04:05"):
        with pytest.raises(ValueError, match="timezone"):
            timestamp(value)


def test_store_rejects_an_unshared_memory_database_before_opening_a_connection():
    with pytest.raises(ValueError, match="independent SQLite connections"):
        Store(":memory:")
