"""Backend contract tests; the actual pretrained encoder smoke is opt-in."""
import json
import math
import os
from types import SimpleNamespace

import httpx
import pytest

import rags.backends as backends
from rags.backends import BackendManager, BackendUnavailable, maxsim

HTTPX_CLIENT = httpx.Client

@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("RAGS_ENV", "test")
    monkeypatch.delenv("RAGS_LATE_INTERACTION_MODEL", raising=False)
    monkeypatch.setenv("RAGS_RESEARCH_BACKENDS_JSON", json.dumps({"full_graphrag": {
        "url": "https://research.example/retrieve", "model": "operator-engine-v1",
        "tenant_tokens": {"a": "key-for-a", "b": "key-for-b"}}}))
    return BackendManager()


def transport(monkeypatch, handler):
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: HTTPX_CLIENT(transport=httpx.MockTransport(handler), **kwargs))


def search(manager, tenant="a"):
    return manager.search("full_graphrag", tenant=tenant, question="What is the policy?",
                          chunks=[{"id": "authorized", "content": "Authorized policy text."}], top_k=3,
                          options={"url": "https://attacker.example", "tenant": "other", "max_iterations": 2})


def test_maxsim_exact_masked_and_large_finite_vectors():
    assert maxsim([[2, 0], [0, 4]], [[10, 0], [0, 20]]) == pytest.approx(2)
    assert maxsim([[2, 0], [0, 4]], [[10, 0], [0, 20]], [True, False]) == pytest.approx(1)
    assert maxsim([[2, 0]], [[0, 0], [10, 0]], document_mask=[False, True]) == pytest.approx(1)
    assert maxsim([[1e308, 1e308]], [[1e308, 1e308]]) == pytest.approx(1)
    assert maxsim([[1e-308, 0]], [[3e-308, 0]]) == pytest.approx(1)


@pytest.mark.parametrize("query,document,query_mask,document_mask", [
    ([[float("nan"), 0]], [[1, 0]], None, None),
    ([[float("inf"), 0]], [[1, 0]], None, None),
    ([[True, 0]], [[1, 0]], None, None),
    ([["1", 0]], [[1, 0]], None, None),
    ([[0, 0]], [[1, 0]], None, None),
    ([[1, 0]], [[1, 0, 0]], None, None),
    ([[1, 0], [1]], [[1, 0]], None, None),
    ([[1, 0]], [[1, 0]], [2], None),
    ([[1, 0]], [[1, 0]], [0.5], None),
    ([[1, 0]], [[1, 0]], [], None),
    ([[1, 0]], [[1, 0]], [False], None),
])
def test_maxsim_rejects_invalid_matrices_and_masks(query, document, query_mask, document_mask):
    with pytest.raises(ValueError):
        maxsim(query, document, query_mask, document_mask)


def test_catalog_is_tenant_aware_and_unknown_encoder_is_unavailable(configured, monkeypatch):
    available = {entry["id"]: entry for entry in configured.catalog(tenant="a")}
    hidden = {entry["id"]: entry for entry in configured.catalog(tenant="foreign")}
    assert available["full_graphrag"]["status"] == "available"
    assert available["full_graphrag"]["evidence_level"] == "external_unverified"
    assert hidden["full_graphrag"]["status"] == "requires_backend"
    monkeypatch.setenv("RAGS_LATE_INTERACTION_MODEL", "nonexistent/model")
    catalog = {entry["id"]: entry for entry in BackendManager().catalog(tenant="a")}
    assert catalog["colbert"]["status"] == "requires_backend"


def test_http_passes_only_authorized_sources_and_tenant_credential(configured, monkeypatch):
    requests = []
    def handler(request):
        data = json.loads(request.content)
        requests.append((str(request.url), request.headers["authorization"], data))
        return httpx.Response(200, json={"hits": [{"chunk_id": "authorized", "score": 0.5}],
                                         "usage": {"input_tokens": 20, "output_tokens": 5}})
    transport(monkeypatch, handler)
    assert search(configured, "a")["hits"] == [{"chunk_id": "authorized", "score": 0.5}]
    search(configured, "b")
    for url, token, data in requests:
        assert url == "https://research.example/retrieve"
        assert token == "Bearer key-for-" + data["namespace"]
        assert data["passages"] == [{"id": "authorized", "text": "Authorized policy text."}]
        assert data["options"] == {"max_iterations": 2}
    before = len(requests)
    with pytest.raises(BackendUnavailable, match="tenant credential"):
        search(configured, "foreign")
    assert len(requests) == before


@pytest.mark.parametrize("payload", [
    {"hits": [{"chunk_id": "foreign", "score": 1}]},
    {"hits": [{"chunk_id": "authorized", "score": True}]},
    {"hits": [{"chunk_id": "authorized", "score": "1"}]},
    {"hits": [{"chunk_id": "authorized", "score": float("nan")}]},
    {"hits": [{"chunk_id": "authorized", "score": float("inf")}]},
    {"hits": [], "usage": {"input_tokens": -1}},
    {"hits": [], "usage": {"secret": "unexpected provider text"}},
])
def test_http_rejects_forged_evidence_scores_and_usage(configured, monkeypatch, payload):
    transport(monkeypatch, lambda request: httpx.Response(200, content=json.dumps(payload)))
    with pytest.raises(BackendUnavailable, match="validation"):
        search(configured)


def test_http_response_byte_deadline_and_compression_limits(configured, monkeypatch):
    state = {"parts": 0, "closed": False, "seconds": 0}
    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            for _ in range(100):
                state["parts"] += 1
                yield b"x" * 100_000
        def close(self):
            state["closed"] = True
    transport(monkeypatch, lambda request: httpx.Response(200, stream=Stream()))
    with pytest.raises(BackendUnavailable, match="exceeds limit"):
        search(configured)
    assert state["parts"] <= 11 and state["closed"]

    class Trickle(httpx.SyncByteStream):
        def __iter__(self):
            for _ in range(100):
                state["parts"] += 1
                yield b"x"
    def clock():
        state["seconds"] += 10
        return state["seconds"]
    state["parts"] = 0
    monkeypatch.setattr(backends, "time", SimpleNamespace(monotonic=clock))
    transport(monkeypatch, lambda request: httpx.Response(200, stream=Trickle()))
    with pytest.raises(BackendUnavailable, match="deadline"):
        search(configured)
    assert state["parts"] <= 4
    transport(monkeypatch, lambda request: httpx.Response(200, headers={"content-encoding": "gzip"}, content=b""))
    with pytest.raises(BackendUnavailable, match="Compressed"):
        search(configured)


@pytest.mark.parametrize("config", [
    {"url": "https://:secret@research.example", "tenant_tokens": {"a": "a"}},
    {"url": "https://research.example", "tenant_tokens": {"a": "shared", "b": "shared"}},
    {"url": "https://research.example", "tenant_tokens": {"": "empty-tenant"}},
])
def test_configuration_rejects_ambiguous_identity(monkeypatch, config):
    monkeypatch.setenv("RAGS_RESEARCH_BACKENDS_JSON", json.dumps({"raptor": config}))
    with pytest.raises(ValueError):
        BackendManager()


@pytest.mark.skipif(os.getenv("RAGS_TEST_COLBERT") != "1", reason="Requires an actual downloaded ColBERT checkpoint")
def test_real_colbert_checkpoint_query_document_encoding_and_maxsim(monkeypatch):
    monkeypatch.setenv("RAGS_LATE_INTERACTION_MODEL", "answerdotai/answerai-colbert-small-v1")
    monkeypatch.setenv("RAGS_MODEL_CACHE", os.getenv("RAGS_MODEL_CACHE", "/tmp/rags-model-cache"))
    monkeypatch.delenv("RAGS_RESEARCH_BACKENDS_JSON", raising=False)
    manager = BackendManager()
    question = "How long does a login session last before it expires?"
    matrix = manager.encode_tokens(question, is_query=True)
    assert len(matrix) == 32 and {len(row) for row in matrix} == {96}
    assert all(math.isfinite(value) for row in matrix for value in row)
    chunks = [
        {"id": "session", "content": "Login sessions expire after 30 minutes of inactivity. Refresh tokens rotate at login."},
        {"id": "invoice", "content": "Invoices are issued in US dollars and paid at the end of each month."},
        {"id": "garden", "content": "Tomatoes grow best in bright sunlight with regular watering."},
    ]
    result = manager.search("colbert", tenant="synthetic-fixture", question=question, chunks=chunks, top_k=3)
    assert result["hits"][0]["chunk_id"] == "session"
    assert all(math.isfinite(hit["score"]) for hit in result["hits"])
    assert result["engine"] == "fastembed:answerdotai/answerai-colbert-small-v1"
    assert {item["id"]: item for item in manager.catalog(tenant="synthetic-fixture")}["colbert"]["evidence_level"] == "runtime_encoder_loaded"
