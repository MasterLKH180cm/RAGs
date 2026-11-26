"""Numerical ranking, snapshot callbacks, typed grouping and bounded projections."""
from copy import deepcopy
import math
import time

import pytest

from rags import collection_search as search_module
from rags.collection_models import GetRecords, SearchQuery, canonical_dense, canonical_sparse
from rags.collection_search import (
    bm25_rank, execute_get, execute_search, select_matching_records, sparse_rank, weighted_rrf,
)
from rags.native import cosine
from rags.registry import CapabilityError


pytestmark = pytest.mark.unit


def record(identifier, document=None, *, metadata=None, vector=None, sparse=None):
    return {
        "id": identifier, "version": "version-" + identifier, "document": document,
        "metadata": deepcopy(metadata or {}), "embedding": canonical_dense(vector or [1.0, 0.0]),
        "sparse_embedding": canonical_sparse(sparse),
    }


class Snapshot:
    def __init__(self, records):
        self.records = [{key: deepcopy(value) for key, value in row.items() if key != "embedding"}
                        for row in records]
        self.vectors = {row["id"]: row["embedding"] for row in records}
        self.dense_calls = []
        self.embedding_calls = []

    def dense(self, vector, allowed_record_ids, limit):
        self.dense_calls.append((vector, allowed_record_ids, limit))
        scores = [(identifier, 1.0 - cosine(vector, embedding))
                  for identifier, embedding in self.vectors.items() if identifier in allowed_record_ids]
        return sorted(scores, key=lambda item: (item[1], item[0]))[:limit]

    def embeddings(self, ids):
        self.embedding_calls.append(list(ids))
        return {identifier: list(self.vectors[identifier]) for identifier in ids}


def run(snapshot, *, vector=None, **parameters):
    return execute_search(snapshot, SearchQuery(**parameters), query_embedding=vector,
                          deadline=time.monotonic() + 1)


def test_bm25_known_numerics_include_whole_filtered_corpus_and_repeated_terms_once():
    rows = [record("a", "apple apple orange"), record("b", "apple pear pear pear"),
            record("c", "apple"), record("d")]
    ranked = bm25_rank(rows, "apple", 100)
    assert [identifier for identifier, _score in ranked] == ["c", "a", "b"]
    # Four documents, df=3, average length=2. The three denominators are hand-derived.
    idf = math.log(10 / 7)
    assert dict(ranked) == pytest.approx({"a": idf * 4.4 / 3.65, "b": idf * 2.2 / 3.1,
                                        "c": idf * 2.2 / 1.75})
    assert bm25_rank(rows, "apple APPLE apple", 100) == ranked
    assert bm25_rank(rows, "absent", 100) == []
    assert bm25_rank(rows, "???", 100) == []
    assert bm25_rank([record("empty", ""), record("null")], "apple", 100) == []
    assert bm25_rank([], "apple", 100) == []


def test_bm25_uses_unicode_words_casefold_and_stable_record_ties():
    rows = [record("b", "Straße CAFÉ 退款"), record("a", "strasse café 退款")]
    ranked = bm25_rank(rows, "STRASSE café 退款", 100)
    assert [identifier for identifier, _score in ranked] == ["a", "b"]
    assert ranked[0][1] == ranked[1][1]


def test_lexical_ranking_recomputes_statistics_after_authorized_filters():
    rows = [record("a", "apple apple orange", metadata={"status": "public"}),
            record("b", "apple pear pear pear", metadata={"status": "public"}),
            record("c", "apple", metadata={"status": "private"}), record("d")]
    snapshot = Snapshot(rows)
    result = run(snapshot, query_text="apple", mode="lexical", where={"status": "public"})
    assert [item["id"] for item in result["items"]] == ["a", "b"]
    assert result["matched_count"] == result["candidate_count"] == 2
    assert result["items"][0]["score"] == pytest.approx(math.log(1.2) * 4.4 / (2 + 1.2 * (0.25 + 0.75 * 3 / 3.5)))
    assert result["score_kind"] == "bm25"
    assert snapshot.dense_calls == snapshot.embedding_calls == []


def test_dense_search_calls_native_snapshot_only_with_filtered_ids_and_preserves_actual_distance():
    snapshot = Snapshot([record("a", "text", metadata={"allowed": True}),
                         record("b", "private", metadata={"allowed": False}),
                         record("c", "text", metadata={"allowed": True})])
    native_calls = []

    def native(vector, allowed, limit):
        native_calls.append((vector, allowed, limit))
        return [("c", 0.7), ("a", 1.2)]

    snapshot.dense = native
    result = run(snapshot, vector=[1.0, 0.0], mode="dense", query_text="text", where={"allowed": True})
    assert native_calls == [([1.0, 0.0], {"a", "c"}, 100)]
    assert [item["id"] for item in result["items"]] == ["c", "a"]
    assert [item["distance"] for item in result["items"]] == [0.7, 1.2]
    assert [item["score"] for item in result["items"]] == pytest.approx([0.3, -0.2])
    assert result["score_kind"] == "cosine_similarity"
    assert snapshot.embedding_calls == []


@pytest.mark.parametrize("native", [
    [("other-tenant", 0.0)], [("a", 0.0), ("a", 0.1)], [("a", float("nan"))],
    [("a", float("inf"))], [("a", True)], [("a", -0.1)], [("a", 2.1)],
])
def test_dense_callback_cannot_return_unknown_duplicate_or_invalid_results(native):
    snapshot = Snapshot([record("a", "text")])
    snapshot.dense = lambda *args: native
    with pytest.raises(CapabilityError) as exc:
        run(snapshot, vector=[1.0, 0.0], mode="dense", query_text="text")
    assert exc.value.code == "invalid_collection_data"


def test_native_dense_roundoff_is_bounded_and_missing_prepared_vector_is_rejected():
    snapshot = Snapshot([record("a", "text")])
    snapshot.dense = lambda *args: [("a", -1e-8)]
    assert run(snapshot, vector=[1.0, 0.0], mode="dense", query_text="text")["items"][0]["distance"] == 0
    with pytest.raises(CapabilityError) as exc:
        run(snapshot, mode="dense", query_text="text")
    assert exc.value.code == "invalid_collection_search"


@pytest.mark.parametrize("mode", ["dense", "hybrid"])
def test_empty_dense_snapshot_needs_no_provider_vector_or_native_distance(mode):
    snapshot = Snapshot([])
    result = run(snapshot, mode=mode, query_text="text")
    assert result["items"] == []
    assert result["matched_count"] == result["candidate_count"] == 0
    assert result["has_more"] is False
    assert snapshot.dense_calls == snapshot.embedding_calls == []


def test_empty_dense_filter_skips_native_distance_for_prepared_vector():
    snapshot = Snapshot([record("a", "text", metadata={"visible": False})])
    result = run(snapshot, vector=[1.0, 0.0], mode="dense", query_text="text", where={"visible": True})
    assert result["items"] == []
    assert result["matched_count"] == result["candidate_count"] == 0
    assert snapshot.dense_calls == snapshot.embedding_calls == []


def test_sparse_dot_product_uses_coordinates_and_immutable_space_and_excludes_zero_matches():
    query = {"space_id": "vocab-v1", "indices": [1, 5], "values": [2.0, 3.0]}
    rows = [
        record("a", sparse={"space_id": "vocab-v1", "indices": [1, 5], "values": [4.0, 2.0]}),
        record("b", sparse={"space_id": "vocab-v1", "indices": [5], "values": [5.0]}),
        record("c", sparse={"space_id": "vocab-v1", "indices": [7], "values": [10.0]}),
        record("d"),
    ]
    assert sparse_rank(rows, query, 100) == [("b", 15.0), ("a", 14.0)]
    snapshot = Snapshot(rows)
    result = run(snapshot, mode="sparse", query_sparse=query)
    assert [item["id"] for item in result["items"]] == ["b", "a"]
    assert result["score_kind"] == "sparse_dot_product" and result["matched_count"] == 4
    assert snapshot.dense_calls == snapshot.embedding_calls == []
    rows[0]["sparse_embedding"]["space_id"] = "other-space"
    with pytest.raises(CapabilityError) as exc:
        sparse_rank(rows, query, 100)
    assert exc.value.code == "invalid_collection_data"


def test_positive_weighted_rrf_matches_manual_rank_arithmetic_and_deduplicates():
    ranked = weighted_rrf([["a", "b", "a"], ["b", "c"]], [0.75, 0.25])
    assert [identifier for identifier, _score in ranked] == ["b", "a", "c"]
    assert dict(ranked) == pytest.approx({"a": 0.75 / 61, "b": 0.75 / 62 + 0.25 / 61, "c": 0.25 / 62})
    assert weighted_rrf([["b", "a"], ["a", "b"]], [1, 1])[0][0] == "a"
    assert weighted_rrf([["a"], ["b"]], [0, 1]) == [("b", 1 / 61)]
    assert weighted_rrf([[], []], [1, 1]) == []
    assert weighted_rrf([["a", "b"]], [1], k=1, limit=1) == [("a", 0.5)]


@pytest.mark.parametrize(("weights", "k", "limit"), [
    ([0, 0], 60, 100), ([-1, 1], 60, 100), ([True, 1], 60, 100),
    ([float("nan"), 1], 60, 100), ([float("inf"), 1], 60, 100),
    ([2 ** 1024, 1], 60, 100), ([1], 60, 100), ([1, 1], 0, 100),
    ([1, 1], True, 100), ([1, 1], 60, 101),
])
def test_invalid_weighted_rrf_configuration_is_rejected(weights, k, limit):
    with pytest.raises(CapabilityError):
        weighted_rrf([["a"], ["b"]], weights, k=k, limit=limit)


def test_hybrid_zero_weight_legs_are_not_executed(monkeypatch):
    snapshot = Snapshot([record("a", "apple"), record("b", "apple apple", vector=[0.0, 1.0])])
    lexical_only = run(snapshot, mode="hybrid", query_text="apple", dense_weight=0, sparse_weight=1)
    assert lexical_only["score_kind"] == "weighted_rrf" and snapshot.dense_calls == []
    monkeypatch.setattr(search_module, "bm25_rank", lambda *args: pytest.fail("Disabled lexical leg ran"))
    dense_only = run(snapshot, vector=[1.0, 0.0], mode="hybrid", query_text="apple",
                     dense_weight=1, sparse_weight=0)
    assert [item["id"] for item in dense_only["items"]] == ["a", "b"]
    assert len(snapshot.dense_calls) == 1


def test_hybrid_uses_supplied_sparse_ranking_instead_of_lexical(monkeypatch):
    snapshot = Snapshot([
        record("a", "apple", sparse={"space_id": "s", "indices": [2], "values": [2]}),
        record("b", "not apple", vector=[0, 1], sparse={"space_id": "s", "indices": [1], "values": [3]}),
    ])
    monkeypatch.setattr(search_module, "bm25_rank", lambda *args: pytest.fail("Unselected BM25 leg ran"))
    result = run(snapshot, vector=[1, 0], mode="hybrid", query_text="apple",
                 query_sparse={"space_id": "s", "indices": [1], "values": [1]})
    assert [item["id"] for item in result["items"]] == ["b", "a"]


def test_grouping_distinguishes_missing_null_boolean_number_string_and_array_before_pagination():
    rows = [
        record("a", "policy"),
        record("b", "policy", metadata={"category": None}),
        record("c", "policy", metadata={"category": True}),
        record("d", "policy", metadata={"category": 1}),
        record("e", "policy", metadata={"category": 1.0}),
        record("f", "policy", metadata={"category": "1"}),
        record("g", "policy", metadata={"category": [1]}),
        record("h", "policy", metadata={"category": [1.0]}),
        record("i", "policy", metadata={"category": [True]}),
        record("j", "policy", metadata={"category": False}),
    ]
    result = run(Snapshot(rows), mode="lexical", query_text="policy",
                 group_by={"field": "category", "max_per_group": 1}, offset=2, limit=4)
    assert [item["id"] for item in result["items"]] == ["c", "d", "f", "g"]
    assert result["matched_count"] == result["candidate_count"] == 10
    assert result["has_more"] is True


def test_pagination_describes_only_the_bounded_candidate_window():
    snapshot = Snapshot([record(str(i), "policy") for i in range(10)])
    result = run(snapshot, vector=[1, 0], mode="dense", query_text="policy", candidate_limit=3, limit=3)
    assert result["matched_count"] == 10 and result["candidate_count"] == 3
    assert result["has_more"] is False
    page = run(snapshot, vector=[1, 0], mode="dense", query_text="policy", candidate_limit=3, limit=1)
    assert page["has_more"] is True


def test_id_selection_and_document_filters_apply_before_search_and_empty_result_is_explicit():
    snapshot = Snapshot([record("a", "refund: 14 days"), record("b", "refund: 30 days")])
    result = run(snapshot, mode="lexical", query_text="refund", ids=["a"],
                 where_document={"$regex": r"30 days$"})
    assert result["items"] == [] and result["matched_count"] == result["candidate_count"] == 0
    assert result["has_more"] is False
    assert snapshot.dense_calls == []
    with pytest.raises(CapabilityError):
        run(snapshot, vector=[1, 0], mode="dense", query_text="refund",
            where_document={"$or": [{"$contains": "refund"}, {"$regex": "("}]})
    assert snapshot.dense_calls == []


def test_get_projection_preserves_identity_fetches_only_page_vectors_and_does_not_mutate_metadata():
    snapshot = Snapshot([record("c", "third"), record("a", "first"),
                         record("b", "second", metadata={"tags": ["test"]})])
    result = execute_get(snapshot, GetRecords(offset=1, limit=1, include=["metadata", "embedding"]),
                         deadline=time.monotonic() + 1)
    assert result["total"] == 3
    assert result["items"] == [{"id": "b", "version": "version-b", "metadata": {"tags": ["test"]},
                               "embedding": [1.0, 0.0]}]
    assert snapshot.embedding_calls == [["b"]]
    result["items"][0]["metadata"]["tags"].append("changed")
    assert snapshot.records[2]["metadata"] == {"tags": ["test"]}
    minimal = execute_get(snapshot, GetRecords(ids=["a"], include=[]), deadline=time.monotonic() + 1)
    assert minimal["items"] == [{"id": "a", "version": "version-a"}]
    selected = select_matching_records(snapshot, ids=["a", "unknown"], deadline=time.monotonic() + 1)
    assert [item["id"] for item in selected] == ["a"]


@pytest.mark.parametrize("vectors", [
    {}, {"a": [1.0], "foreign": [1.0]}, {"a": []}, {"a": [0.0]},
    {"a": [True]}, {"a": [float("nan")]}, {"a": [1.0] * 4097},
])
def test_embedding_projection_rejects_unknown_missing_and_invalid_snapshot_vectors(vectors):
    snapshot = Snapshot([record("a", "text")])
    snapshot.embeddings = lambda ids: vectors
    with pytest.raises(CapabilityError) as exc:
        execute_get(snapshot, GetRecords(include=["embedding"]), deadline=time.monotonic() + 1)
    assert exc.value.code == "invalid_collection_data"


def test_payload_limit_is_enforced_and_projection_can_remove_large_documents():
    snapshot = Snapshot([record(str(index), "文" * 20_000) for index in range(100)])
    with pytest.raises(CapabilityError) as exc:
        execute_get(snapshot, GetRecords(), deadline=time.monotonic() + 1)
    assert exc.value.code == "collection_response_limit" and exc.value.status_code == 422
    result = execute_get(snapshot, GetRecords(include=[]), deadline=time.monotonic() + 1)
    assert len(result["items"]) == 100 and all(set(item) == {"id", "version"} for item in result["items"])


def test_expired_deadline_rejects_even_empty_snapshot():
    with pytest.raises(CapabilityError) as exc:
        execute_get(Snapshot([]), GetRecords(), deadline=time.monotonic() - 1)
    assert exc.value.code == "collection_filter_timeout"
