"""Collection contracts reject ambiguous vectors, input coercion and unsafe JSON."""
import json
import math

import pytest
from pydantic import ValidationError

from rags.collection_models import (
    CollectionAsk, CollectionConfig, CollectionCreate, CollectionModify, DeleteRecords,
    GetRecords, RecordBatch, SearchBatch, SearchQuery, canonical_dense,
    canonical_metadata, canonical_sparse,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("value", [
    [], [0, 0], [True, 1], ["1", 2], [float("nan"), 1], [float("inf")],
    [1e39], [10 ** 1000], [1e-50], [1] * 4097, None, {},
])
def test_dense_rejects_nonfinite_zero_coerced_or_unrepresentable_values(value):
    with pytest.raises(ValueError):
        canonical_dense(value)


@pytest.mark.parametrize("value", [[3, 4], [3e38, -3e38], [1e-40, 2e-40], [1], [-1, 0, 0]])
def test_dense_float32_normalization_is_stable_and_direction_preserving(value):
    result = canonical_dense(value)
    assert math.sqrt(math.fsum(x * x for x in result)) == pytest.approx(1, abs=1e-6)
    assert canonical_dense(result) == result
    assert [x >= 0 for x in result] == [x >= 0 for x in value]


@pytest.mark.parametrize("value", [
    {"x": [True, 1]}, {"x": [None]}, {"x": {"nested": 1}}, {"x": float("nan")},
    {"x": 2 ** 63}, {"x": -(2 ** 63) - 1}, {"x": "\x00"}, {"x": "\ud800"},
    {"$reserved": 1}, {"_rags_secret": 1}, {"": 1}, {"x": "a" * 4097},
    {"x": list(range(101))}, {str(i): 0 for i in range(65)},
    {str(i): "é" * 4096 for i in range(3)},
])
def test_metadata_fails_closed_on_ambiguous_types_and_portability_boundaries(value):
    with pytest.raises(ValueError):
        canonical_metadata(value)


def test_metadata_types_remain_distinct_and_input_is_detached():
    original = {"numbers": [1, 2.5], "booleans": [True, False], "empty": [], "null": None,
                "unicode": "臺灣", "large": 2 ** 63 - 1}
    copied = canonical_metadata(original)
    assert copied == original
    original["numbers"].append(4)
    assert copied["numbers"] == [1, 2.5]


@pytest.mark.parametrize("change", [
    {"indices": [1, 1]}, {"indices": [2, 1]}, {"indices": [-1, 1]},
    {"indices": [True, 1]}, {"indices": [1, 2 ** 31]}, {"indices": []},
    {"values": [0, 1]}, {"values": [True, 1]}, {"values": [1e-50, 1]},
    {"values": [float("nan"), 1]}, {"values": [1e7, 1]}, {"values": [1]},
    {"space_id": ""}, {"space_id": "bad\x00"}, {"unknown": 1},
])
def test_sparse_requires_an_explicit_well_formed_space(change):
    vector = {"indices": [1, 3], "values": [0.5, 2], "space_id": "v1", **change}
    with pytest.raises(ValueError):
        canonical_sparse(vector)


@pytest.mark.parametrize("model,data", [
    (CollectionCreate, {"name": "127.0.0.1"}),
    (CollectionCreate, {"name": "notes", "tenant": "other"}),
    (CollectionCreate, {"name": "notes", "embedding": {"model": "x", "space_id": "x", "dimensions": True}}),
    (CollectionModify, {}), (DeleteRecords, {}), (DeleteRecords, {"where": {}}),
    (GetRecords, {"ids": []}), (GetRecords, {"include": ["unknown"]}),
    (GetRecords, {"include": ["metadata", "metadata"]}),
    (RecordBatch, {"ids": ["a", "a"], "documents": ["one", "two"]}),
    (RecordBatch, {"ids": ["a"], "documents": ["a", "b"]}),
    (RecordBatch, {"ids": ["a"]}),
    (RecordBatch, {"ids": ["a"], "embeddings": [[1]]}),
    (RecordBatch, {"ids": ["a"], "documents": ["ok"], "embedding_space_id": "x"}),
    (RecordBatch, {"ids": ["a"], "documents": ["x" * 20001]}),
    (SearchQuery, {"mode": "dense", "query_text": "x", "query_embedding": [1], "embedding_space_id": "x"}),
    (SearchQuery, {"mode": "lexical", "query_text": "x", "dense_weight": True}),
    (SearchQuery, {"query_text": "x", "dense_weight": 0, "sparse_weight": 0}),
    (SearchQuery, {"query_embedding": [1], "embedding_space_id": "x"}),
    (SearchQuery, {"query_text": "x", "offset": 95, "limit": 10}),
    (SearchQuery, {"query_text": "x", "limit": "10"}),
    (SearchQuery, {"query_text": "x", "where": {"x": float("inf")}}),
    (SearchBatch, {"searches": []}),
    (CollectionAsk, {"question": "what?", "mode": "research_graphrag"}),
    (CollectionAsk, {"question": "what?", "dense_weight": 0, "sparse_weight": 0}),
])
def test_public_contracts_reject_ambiguous_or_unqualified_operations(model, data):
    with pytest.raises(ValidationError):
        model.model_validate(data)


def test_contract_byte_limit_applies_before_provider_work():
    payload = {"ids": [str(i) for i in range(100)], "documents": ["界" * 20000] * 100}
    with pytest.raises(ValidationError, match="byte limit"):
        RecordBatch.model_validate(payload)


@pytest.mark.parametrize("raw", [
    "[]", "null", '{"unknown":1}', '{"max_batch":1,"max_batch":2}',
    '{"max_batch":true}', '{"max_records":10001}', '{"filter_timeout_seconds":NaN}',
    '{"max_response_bytes":32767}', '{"max_scan_bytes":0}', "x" * 2049,
])
def test_operator_limits_fail_closed(monkeypatch, raw):
    monkeypatch.setenv("RAGS_COLLECTION_LIMITS_JSON", raw)
    with pytest.raises(ValueError):
        CollectionConfig.from_env()


def test_operator_can_reduce_limits_without_unbounding_control_responses(monkeypatch):
    monkeypatch.setenv("RAGS_COLLECTION_LIMITS_JSON", json.dumps({
        "max_batch": 2, "max_records": 25, "max_queries": 1, "max_response_bytes": 32768,
        "max_scan_bytes": 8192, "filter_timeout_seconds": 0.05,
    }))
    actual = CollectionConfig.from_env()
    assert actual.max_batch == 2 and actual.filter_timeout_seconds == 0.05


def test_empty_compose_limit_override_retains_defaults(monkeypatch):
    monkeypatch.setenv("RAGS_COLLECTION_LIMITS_JSON", "")
    assert CollectionConfig.from_env() == CollectionConfig()
