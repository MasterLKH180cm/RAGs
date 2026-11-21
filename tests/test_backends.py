"""Backend contract tests; the actual pretrained encoder smoke is opt-in."""
import json
import math
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread
import time
from types import SimpleNamespace

import httpx
import pytest

import rags.backends as backends
from rags.backends import BackendManager, BackendUnavailable, maxsim

HTTPX_CLIENT = httpx.AsyncClient

@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("RAGS_ENV", "test")
    monkeypatch.delenv("RAGS_LATE_INTERACTION_MODEL", raising=False)
    monkeypatch.setenv("RAGS_RESEARCH_BACKENDS_JSON", json.dumps({"full_graphrag": {
        "url": "https://research.example/retrieve", "model": "operator-engine-v1",
        "tenant_tokens": {"a": "key-for-a", "b": "key-for-b"}}}))
    return BackendManager()


def transport(monkeypatch, handler):
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: HTTPX_CLIENT(transport=httpx.MockTransport(handler), **kwargs))


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
    {"hits": [{"chunk_id": "authorized", "score": 10**1000}]},
    {"hits": [], "usage": {"input_tokens": -1}},
    {"hits": [], "usage": {"secret": "unexpected provider text"}},
])
def test_http_rejects_forged_evidence_scores_and_usage(configured, monkeypatch, payload):
    transport(monkeypatch, lambda request: httpx.Response(200, content=json.dumps(payload)))
    with pytest.raises(BackendUnavailable, match="validation"):
        search(configured)


def test_http_response_byte_deadline_and_compression_limits(configured, monkeypatch):
    state = {"parts": 0, "closed": False, "seconds": 0}
    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for _ in range(100):
                state["parts"] += 1
                yield b"x" * 100_000
        async def aclose(self):
            state["closed"] = True
    transport(monkeypatch, lambda request: httpx.Response(200, stream=Stream()))
    with pytest.raises(BackendUnavailable, match="exceeds limit"):
        search(configured)
    assert state["parts"] <= 11 and state["closed"]

    class Trickle(httpx.AsyncByteStream):
        async def __aiter__(self):
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


@pytest.mark.parametrize("timeout", [True, 0, -1, 361, "120", float("nan"), float("inf")])
def test_research_timeout_is_a_bounded_operator_setting(monkeypatch, timeout):
    monkeypatch.setenv("RAGS_RESEARCH_BACKENDS_JSON", json.dumps({"raptor": {
        "url": "https://research.example/retrieve", "tenant_tokens": {"a": "key"}, "timeout_seconds": timeout}}))
    with pytest.raises(ValueError, match="timeout_seconds"):
        BackendManager()


def test_pinned_upstream_provenance_and_deadline_are_tenant_scoped(monkeypatch):
    from rags.research_protocol import UPSTREAMS
    monkeypatch.setenv("RAGS_RESEARCH_BACKENDS_JSON", json.dumps({"raptor": {
        "url": "https://research.example/retrieve", "tenant_tokens": {"a": "key-a", "b": "key-b"},
        "timeout_seconds": 150, "require_pinned_upstream": True}}))
    manager = BackendManager()
    observed = []
    response = {"hits": [{"chunk_id": "authorized", "score": 0.5}],
                "implementation": dict(UPSTREAMS["raptor"]),
                "diagnostics": {"source_count": 1, "engine_executed": True, "summary_layers": 2}}

    def handler(request):
        observed.append(request)
        return httpx.Response(200, json=response)

    transport(monkeypatch, handler)
    arguments = {"tenant": "a", "question": "What is the policy?", "chunks": [{"id": "authorized", "content": "policy"}], "top_k": 1}
    result = manager.search("raptor", **arguments)
    assert set(observed[0].extensions["timeout"].values()) == {150}
    assert result["trace"][0]["implementation"] == UPSTREAMS["raptor"]
    assert result["trace"][0]["diagnostics"]["summary_layers"] == 2
    own = {item["id"]: item for item in manager.catalog("a")}["raptor"]
    other = {item["id"]: item for item in manager.catalog("b")}["raptor"]
    global_catalog = {item["id"]: item for item in manager.catalog()}["raptor"]
    assert own["evidence_level"] == "pinned_upstream_reported"
    assert other["evidence_level"] == global_catalog["evidence_level"] == "external_unverified"
    assert other["upstream"] is global_catalog["upstream"] is None

    response["diagnostics"]["private_trace"] = "source-sentinel"
    with pytest.raises(BackendUnavailable, match="validation"):
        manager.search("raptor", **arguments)
    response.clear()
    response["hits"] = [{"chunk_id": "authorized", "score": 1}]
    with pytest.raises(BackendUnavailable, match="validation"):
        manager.search("raptor", **arguments)


def test_empty_upstream_probe_does_not_claim_algorithm_execution(monkeypatch):
    from rags.research_protocol import UPSTREAMS
    monkeypatch.setenv("RAGS_RESEARCH_BACKENDS_JSON", json.dumps({"lightrag": {
        "url": "https://research.example/retrieve", "tenant_tokens": {"a": "key-a"}, "require_pinned_upstream": True}}))
    manager = BackendManager()
    transport(monkeypatch, lambda request: httpx.Response(200, json={
        "hits": [], "implementation": UPSTREAMS["lightrag"],
        "diagnostics": {"source_count": 0, "engine_executed": False}}))
    manager.search("lightrag", tenant="a", question="Any evidence?", chunks=[], top_k=1)
    assert {item["id"]: item for item in manager.catalog("a")}["lightrag"]["evidence_level"] == "external_unverified"


def test_actual_http_deadline_covers_headers_and_stalled_body(monkeypatch):
    """Per-read timeouts alone would wait another full second after headers."""
    release = Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            release.wait(0.65)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "100")
            self.end_headers()
            self.wfile.write(b'{"hits":[')
            self.wfile.flush()
            release.wait(5)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("RAGS_ENV", "test")
    monkeypatch.setenv("RAGS_RESEARCH_BACKENDS_JSON", json.dumps({"full_graphrag": {
        "url": f"http://127.0.0.1:{server.server_port}/retrieve", "tenant_tokens": {"a": "key-a"}, "timeout_seconds": 1}}))
    try:
        started = time.monotonic()
        with pytest.raises(BackendUnavailable, match="deadline exceeded"):
            search(BackendManager())
        assert time.monotonic() - started < 1.5
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(2)


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
