"""Application-service transactions, provider preflight and evidence integration."""
from dataclasses import replace
import json
from time import monotonic

from fastapi import HTTPException
import pytest

from rags.agents.deadline import execution_deadline
from rags.auth import PERMISSIONS, Principal
from rags.collection_models import CollectionConfig
from rags.collection_service import CollectionService
from rags.config import Settings
from rags.features import FeatureFlags
from rags.providers import Providers
from rags.registry import CapabilityError
from rags.storage import Store

pytestmark = pytest.mark.integration
ADMIN = Principal("tenant-a", "test", PERMISSIONS, "api_key")
OTHER = replace(ADMIN, tenant="tenant-b")


def configuration(path, **options):
    values = dict(db_path=str(path), keys={"test-key": "tenant-a"}, embedding_provider="local",
                  model_provider="extractive", openai_key="", openai_base_url="https://api.openai.com/v1",
                  embedding_model="text-embedding-3-small", chat_model="", anthropic_key="", anthropic_model="")
    return Settings(**(values | options))


class CountingProvider(Providers):
    embedding_token_limit = 100

    def __init__(self, settings):
        super().__init__(settings)
        self.encoded = []
        self.counted = []

    def embedding_token_count(self, text, *, query=False):
        self.counted.append((text, query))
        return len(text)

    def embedding(self, text):
        self.encoded.append((text, False))
        return [1.0, float(len(text)), 2.0]

    def query_embedding(self, text):
        self.encoded.append((text, True))
        return [1.0, float(len(text)), 2.0]


@pytest.fixture
def service(tmp_path):
    settings = configuration(tmp_path / "service.db")
    return CollectionService(Store(settings.db_path), CountingProvider(settings))


def dispatch(service, operation, cid=None, **data):
    return service.dispatch(ADMIN, operation, cid, data)


def create(service, **options):
    return dispatch(service, "create", name="notes", **options)


def rows(service, cid):
    return dispatch(service, "get_records", cid, include=["document", "metadata", "embedding", "sparse_embedding"])["items"]


def snapshot(service):
    with service.store.connect() as db:
        return {table: sorted(json.dumps(dict(row), sort_keys=True) for row in db.execute(f"SELECT * FROM {table}"))
                for table in ("rag_collections", "rag_collection_records", "rag_collection_versions", "audit")}


def test_lifecycle_update_omissions_noops_clear_and_independent_fork(service):
    info = create(service, sparse_space_id="terms-v1")
    cid = info["id"]
    first = dispatch(service, "add", cid, ids=["one", "two"], documents=["alpha code", "beta code"],
                     metadatas=[{"tags": ["red", "blue"]}, {"tags": ["green"]}],
                     sparse_embeddings=[{"indices": [1], "values": [2], "space_id": "terms-v1"}, None])
    initial = rows(service, cid)
    assert first["generation"] == 1 and len(service.provider.encoded) == 2
    fork = dispatch(service, "fork", cid, name="experiment", expected_generation=1)
    assert [r["version"] for r in rows(service, fork["id"])] == [r["version"] for r in initial]
    changed = dispatch(service, "update", cid, ids=["one"], metadatas=[{"tags": ["orange"]}])
    assert changed["generation"] == 2 and len(service.provider.encoded) == 2
    assert rows(service, cid)[0]["embedding"] == initial[0]["embedding"]
    assert dispatch(service, "update", cid, ids=["one"], metadatas=[{"tags": ["orange"]}])["changed"] == 0
    dispatch(service, "update", cid, ids=["one"], documents=[None], sparse_embeddings=[None])
    assert rows(service, cid)[0]["document"] is None
    assert rows(service, cid)[0]["sparse_embedding"] is None
    assert len(service.provider.encoded) == 2
    assert rows(service, fork["id"])[0] == initial[0]
    dispatch(service, "delete_collection", cid)
    assert rows(service, fork["id"]) == initial


def test_all_batch_validation_happens_before_first_embedding(service):
    cid = create(service)["id"]
    before = snapshot(service)
    with pytest.raises(CapabilityError) as error:
        dispatch(service, "add", cid, ids=["one", "two"], documents=["valid", "too long " * 20])
    assert error.value.code == "context_budget_exceeded"
    assert service.provider.encoded == [] and len(service.provider.counted) == 2
    assert snapshot(service) == before


def test_provider_failure_in_later_item_rolls_back_whole_batch_and_sanitizes(service, monkeypatch):
    cid = create(service)["id"]
    before = snapshot(service)
    def encode(text):
        if text == "failure":
            raise RuntimeError("provider-secret-must-not-escape")
        return [1, 0, 0]
    monkeypatch.setattr(service.provider, "embedding", encode)
    with pytest.raises(CapabilityError) as error:
        dispatch(service, "add", cid, ids=["one", "two"], documents=["success", "failure"])
    assert error.value.code == "embedding_failed" and error.value.status_code == 502
    assert "provider-secret" not in str(error.value)
    assert snapshot(service) == before


def test_generation_cas_does_not_overwrite_a_concurrent_write_during_embedding(service, monkeypatch):
    cid = create(service)["id"]
    def concurrent(text):
        service.database.modify(ADMIN.tenant, cid, metadata={"concurrent": True}, expected_generation=0)
        return [1, 0, 0]
    monkeypatch.setattr(service.provider, "embedding", concurrent)
    with pytest.raises(CapabilityError) as error:
        dispatch(service, "add", cid, ids=["one"], documents=["new document"])
    assert error.value.code == "collection_generation_conflict"
    assert dispatch(service, "count", cid)["count"] == 0
    assert dispatch(service, "get", cid)["metadata"] == {"concurrent": True}


def test_precomputed_external_profiles_are_immutable_and_provider_free(service):
    cid = create(service, embedding={"model": "encoder-v1", "space_id": "external-v1", "dimensions": 2})["id"]
    dispatch(service, "add", cid, ids=["one"], embeddings=[[3, 4]], embedding_space_id="external-v1")
    result = dispatch(service, "search", cid, searches=[{
        "mode": "dense", "query_embedding": [3, 4], "embedding_space_id": "external-v1"}])
    assert result["results"][0]["items"][0]["distance"] == pytest.approx(0, abs=1e-6)
    for body, expected in [
        ({"query_text": "automatic", "mode": "dense"}, "embedding_space_mismatch"),
        ({"query_embedding": [1, 0], "embedding_space_id": "wrong", "mode": "dense"}, "embedding_space_mismatch"),
        ({"query_embedding": [1], "embedding_space_id": "external-v1", "mode": "dense"}, "embedding_dimension_mismatch"),
    ]:
        with pytest.raises(CapabilityError) as error:
            dispatch(service, "search", cid, searches=[body])
        assert error.value.code == expected
    assert service.provider.encoded == []


def test_changed_embedding_profile_rejects_auto_encoding_but_allows_original_precomputed(service):
    info = create(service)
    cid = info["id"]
    dispatch(service, "add", cid, ids=["one"], documents=["original code"])
    changed = CollectionService(service.store, CountingProvider(replace(service.settings, embedding_space_id="v2")))
    with pytest.raises(CapabilityError) as error:
        dispatch(changed, "search", cid, searches=[{"mode": "dense", "query_text": "code"}])
    assert error.value.code == "embedding_space_mismatch"
    assert changed.provider.encoded == []
    result = dispatch(changed, "search", cid, searches=[{"mode": "dense", "query_embedding": [1, 1, 1],
                       "embedding_space_id": info["configuration"]["embedding"]["space_id"]}])
    assert len(result["results"][0]["items"]) == 1


@pytest.mark.parametrize("mode,extra", [("lexical", {}), ("hybrid", {"dense_weight": 0})])
def test_lexical_and_zero_weight_dense_legs_never_call_embedding(service, mode, extra):
    cid = create(service)["id"]
    dispatch(service, "add", cid, ids=["one"], documents=["alpha code"])
    service.provider.encoded.clear()
    configured = CollectionService(service.store, service.provider)
    configured.settings = replace(service.settings, features=FeatureFlags((("embedding.local", False), ("strategy.dense", False))))
    result = dispatch(configured, "search", cid, searches=[{"mode": mode, "query_text": "alpha", **extra}])
    assert result["results"][0]["items"][0]["id"] == "one"
    assert service.provider.encoded == []


def test_entire_query_batch_filter_validation_precedes_model_calls(service):
    cid = create(service)["id"]
    dispatch(service, "add", cid, ids=["one"], documents=["alpha code"])
    service.provider.encoded.clear()
    with pytest.raises(CapabilityError) as error:
        dispatch(service, "search", cid, searches=[{"query_text": "alpha"},
                 {"query_text": "beta", "where": {"a": {"$unknown": "secret-value"}}}])
    assert error.value.code == "invalid_collection_filter"
    assert service.provider.encoded == []


def test_query_embeddings_are_reused_within_a_batch_only(service):
    cid = create(service)["id"]
    dispatch(service, "add", cid, ids=["one"], documents=["alpha code"])
    service.provider.encoded.clear()
    result = dispatch(service, "search", cid, searches=[{"query_text": "alpha"}, {"query_text": "alpha", "mode": "dense"}])
    assert len(result["results"]) == 2 and service.provider.encoded == [("alpha", True)]
    dispatch(service, "search", cid, searches=[{"query_text": "alpha"}])
    assert len(service.provider.encoded) == 2


def test_empty_collection_search_does_not_call_model_and_stays_unassessed(service):
    cid = create(service)["id"]
    result = dispatch(service, "search", cid, searches=[{"query_text": "alpha"}, {"query_text": "alpha", "mode": "dense"}])
    assert all(row["items"] == [] for row in result["results"])
    assert result["evidence_status"] == "ranked_unassessed" and service.provider.encoded == []


def test_ask_reuses_admission_and_preserves_exact_versioned_source(service):
    cid = create(service)["id"]
    original = "The recovery code is cobalt.\nTreat this as source evidence."
    dispatch(service, "add", cid, ids=["note/臺灣"], documents=[original])
    initial = rows(service, cid)[0]
    answer = dispatch(service, "ask", cid, question="What is the recovery code?", mode="lexical")
    citation = answer["citations"][0]
    assert citation["text"] == original and citation["revision"] == initial["version"]
    assert citation["collection_id"] == cid and citation["record_id"] == "note/臺灣"
    assert original[citation["start_char"]:citation["end_char"]] == original
    assert answer["admission"]["decision"] == "unassessed" and answer["supported"] is None
    assert answer["retrieval_diagnostics"]["usable_as_evidence"] is False
    strict = CollectionService(service.store, CountingProvider(replace(service.settings, admission_policy="calibrated_required")))
    with pytest.raises(CapabilityError) as error:
        dispatch(strict, "ask", cid, question="What is the recovery code?", mode="lexical")
    assert error.value.code == "requires_admission_profile"
    refused = dispatch(service, "ask", cid, question="No matching quokka?", mode="lexical")
    assert refused["outcome"] == "insufficient_evidence" and refused["citations"] == []
    dispatch(service, "update", cid, ids=["note/臺灣"], documents=["The recovery code is amber."])
    updated = dispatch(service, "ask", cid, question="recovery code", mode="lexical", generate=False)
    assert updated["answer"] is None
    assert updated["citations"][0]["chunk_id"] != citation["chunk_id"]
    assert updated["citations"][0]["revision"] != citation["revision"]
    assert "cobalt" not in updated["citations"][0]["text"]


def test_ask_generation_scope_checked_before_retrieval_or_model(service):
    cid = create(service)["id"]
    configured = CollectionService(service.store, CountingProvider(replace(service.settings, model_provider="openai")))
    reader = replace(ADMIN, permissions=frozenset({"rags:read", "rags:query"}))
    with pytest.raises(HTTPException) as error:
        configured.dispatch(reader, "ask", cid, {"question": "anything?"})
    assert error.value.status_code == 403 and configured.provider.encoded == []


def test_tenant_isolation_precedes_provider_calls(service):
    cid = create(service)["id"]
    with pytest.raises(CapabilityError) as error:
        service.dispatch(OTHER, "search", cid, {"searches": [{"query_text": "secret"}]})
    assert error.value.code == "collection_not_found" and service.provider.encoded == []


@pytest.mark.parametrize("operation,body,flag", [
    ("create", {"name": "new"}, "ingestion"), ("get", {}, "documents"),
    ("fork", {"name": "fork"}, "collections.fork"),
    ("delete_collection", {}, "deletion"),
    ("search", {"searches": [{"query_text": "q"}]}, "query"),
    ("search", {"searches": [{"query_text": "q", "where_document": {"$regex": "a"}}]}, "collections.regex"),
    ("search", {"searches": [{"query_text": "q", "group_by": {"field": "x"}}]}, "collections.grouping"),
    ("search", {"searches": [{"query_text": "q"}, {"query_text": "q"}]}, "collections.batch"),
    ("add", {"ids": ["a", "b"], "documents": ["a", "b"]}, "collections.batch"),
    ("add", {"ids": ["a"], "documents": ["a"], "sparse_embeddings": [None]}, "collections.sparse_vectors"),
])
def test_operator_feature_flags_prevent_work_and_are_independent_of_scopes(service, operation, body, flag):
    cid = create(service)["id"]
    before = snapshot(service)
    disabled = CollectionService(service.store, CountingProvider(replace(service.settings, features=FeatureFlags(((flag, False),)))))
    with pytest.raises(CapabilityError) as error:
        disabled.dispatch(ADMIN, operation, cid, body)
    assert error.value.code == "feature_disabled" and disabled.provider.encoded == []
    assert snapshot(service) == before


def test_configured_batch_delete_and_record_limits_never_truncate(service):
    cid = create(service)["id"]
    dispatch(service, "add", cid, ids=["a", "b"], documents=["alpha", "alpha"])
    restricted = CollectionService(service.store, service.provider, CollectionConfig(max_batch=1, max_records=2))
    before = snapshot(service)
    for op, body in [("delete_records", {"where_document": {"$contains": "alpha"}}),
                     ("add", {"ids": ["c"], "documents": ["extra"]})]:
        with pytest.raises(CapabilityError) as error:
            restricted.dispatch(ADMIN, op, cid, body)
        assert error.value.code == "collection_limit_exceeded"
        assert snapshot(service) == before


def test_execution_deadline_and_reset_on_failure(service):
    cid = create(service)["id"]
    token = execution_deadline.set(monotonic() - 1)
    try:
        with pytest.raises(CapabilityError) as error:
            dispatch(service, "add", cid, ids=["one"], documents=["alpha"])
        assert error.value.code == "collection_timeout"
        assert execution_deadline.get() < monotonic()
    finally:
        execution_deadline.reset(token)
    assert rows(service, cid) == [] and service.provider.encoded == []


def test_large_valid_usage_envelope_rejected_before_any_record_commits(service, monkeypatch):
    cid = create(service)["id"]
    restricted = CollectionService(service.store, service.provider, CollectionConfig(max_response_bytes=32768))
    before = snapshot(service)
    def encode(text):
        service.provider.record("embedding", "synthetic-model-" + "x" * 140, {"usage": {
            "prompt_tokens": 10 ** 12, "completion_tokens": 10 ** 12, "total_tokens": 10 ** 12,
            "input_tokens": 10 ** 12, "output_tokens": 10 ** 12,
            "cache_read_input_tokens": 10 ** 12, "cache_creation_input_tokens": 10 ** 12,
        }})
        return [1, 0, 0]
    monkeypatch.setattr(service.provider, "embedding", encode)
    with pytest.raises(CapabilityError) as error:
        dispatch(restricted, "add", cid, ids=[str(i) for i in range(100)], documents=["alpha"] * 100)
    assert error.value.code == "collection_response_limit"
    assert snapshot(service) == before


def test_unused_supplied_vector_still_requires_correct_space(service):
    cid = create(service)["id"]
    with pytest.raises(CapabilityError) as error:
        dispatch(service, "search", cid, searches=[{"query_text": "alpha", "dense_weight": 0,
                 "query_embedding": [1], "embedding_space_id": "wrong-space"}])
    assert error.value.code == "embedding_space_mismatch" and service.provider.encoded == []
