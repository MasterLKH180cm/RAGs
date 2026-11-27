"""Behavioral fixtures exercise different mechanisms and their trust boundaries."""
import asyncio
from dataclasses import replace
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from rags.api import RowPredicate, create_app
from rags.config import Settings
from rags.providers import Providers
from rags.registry import CapabilityError
from rags.retrieval import RAGService
from rags.storage import Store
import rags.retrieval as retrieval_module

A = {"Authorization": "Bearer a-key"}
B = {"Authorization": "Bearer b-key"}


@pytest.fixture
def system(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGS_ENV", "test")
    monkeypatch.setenv("RAGS_DB_PATH", str(tmp_path / "native.db"))
    monkeypatch.setenv("RAGS_KEYS_JSON", '{"a-key":"a","b-key":"b"}')
    monkeypatch.setenv("RAGS_EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("RAGS_MODEL_PROVIDER", "extractive")
    settings = Settings.from_env()
    app = create_app(settings)
    with TestClient(app) as client:
        yield client, app.state.service, settings


def ingest(client, *, token=A, **data):
    response = client.post("/documents", headers=token, json=data)
    assert response.status_code == 201, response.text
    return response.json()


def query(client, *, token=A, **data):
    response = client.post("/query", headers=token, json={"generate": False, **data})
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.integration
@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_sqlite_dense_ranks_extreme_finite_vectors_by_direction(system, monkeypatch, scale):
    client, service, _ = system
    monkeypatch.setattr(service.provider, "embedding", lambda text: [scale, 0.0] if "aligned" in text else [0.0, scale])
    monkeypatch.setattr(service.provider, "query_embedding", lambda text: [scale, 0.0] if "aligned" in text else [0.0, scale])
    # The lower source ID is deliberately orthogonal: zero/NaN scores must not
    # reduce the ranking to insertion order or the stable identifier tie-break.
    monkeypatch.setattr(retrieval_module, "_identifier", lambda *parts: parts[1] + "-" + parts[-1])
    ingest(client, document_id="a-orthogonal", title="Orthogonal", text="orthogonal evidence")
    ingest(client, document_id="z-aligned", title="Aligned", text="aligned evidence")
    result = query(client, question="aligned", mode="dense", top_k=2)
    assert [citation["document_id"] for citation in result["citations"]] == ["z-aligned", "a-orthogonal"]
    assert all(citation["revision"] for citation in result["citations"])


def test_temporal_versions_metadata_selection_and_purge(system):
    client, service, _ = system
    first = ingest(client, document_id="policy", title="Policy", text="The lunar policy costs 10 credits.",
                   metadata={"region": "north"}, valid_from="2024-01-01T00:00:00Z")
    second = ingest(client, document_id="policy", title="Policy", text="The solar policy costs 20 credits.",
                    metadata={"region": "south"}, valid_from="2024-06-01T00:00:00Z", expected_revision=first["revision"])
    old = query(client, question="policy costs", mode="temporal", as_of="2024-03-01T00:00:00Z")
    new = query(client, question="policy costs", mode="temporal", as_of="2024-07-01T00:00:00Z")
    assert {c["revision"] for c in old["citations"]} == {first["revision"]}
    assert {c["revision"] for c in new["citations"]} == {second["revision"]}
    assert "10 credits" in old["citations"][0]["text"]
    assert "20 credits" in new["citations"][0]["text"]
    assert query(client, question="lunar", mode="sparse")["citations"] == []
    assert query(client, question="policy", filters={"metadata": {"region": "north"}})["citations"] == []
    assert query(client, token=B, question="policy", mode="temporal", as_of="2024-03-01T00:00:00Z")["citations"] == []
    assert client.get("/documents/policy/revisions", headers=B).status_code == 404
    assert len(client.get("/documents/policy/revisions", headers=A).json()) == 2
    assert client.delete("/documents/policy", headers=A).status_code == 200
    assert service.store.revisions("a", "policy") == []
    assert query(client, question="policy", mode="temporal", as_of="2024-03-01T00:00:00Z")["citations"] == []


def test_parent_expansion_hierarchy_and_source_offsets(system):
    client, service, _ = system
    source = "# Asteroids\n" + "Vesta has a bright basaltic surface. Ceres contains water ice. " * 10
    indexed = ingest(client, document_id="space", title="Space", text=source,
                     chunking={"strategy": "parent_child", "max_tokens": 65, "overlap_tokens": 8, "parent_max_tokens": 200},
                     index_features=["hierarchy"])
    assert indexed["derived_nodes"] > 1
    parent = query(client, question="Vesta basaltic surface", mode="parent_child", top_k=5)
    assert len({c["chunk_id"] for c in parent["citations"]}) == len(parent["citations"])
    assert all(c["text"] == source[c["start_char"]:c["end_char"]] for c in parent["citations"])
    assert any(len(c["text"]) > 65 for c in parent["citations"])
    hierarchy = query(client, question="Vesta Ceres surface ice", mode="hierarchical", top_k=5)
    assert hierarchy["citations"]
    assert all(c["text"] == source[c["start_char"]:c["end_char"]] for c in hierarchy["citations"])
    nodes = service.store.scope("a")
    assert len({c["level"] for c in nodes if c["role"] == "summary"}) >= 2
    assert client.post("/query", headers=B, json={"question": "Vesta", "mode": "hierarchical"}).status_code == 409


def test_graph_traverses_two_hops_without_foreign_edges(system):
    client, _, _ = system
    ingest(client, document_id="first", title="First relation", text="Alpha service -> Beta service")
    ingest(client, token=B, document_id="secret", title="Private relation", text="Beta service -> Secret service")
    before = query(client, question="Alpha service", mode="graph")
    assert {c["document_id"] for c in before["citations"]} == {"first"}
    ingest(client, document_id="second", title="Second relation", text="Beta service -> Gamma service")
    after = query(client, question="Alpha", mode="graph")
    assert {c["document_id"] for c in after["citations"]} == {"first", "second"}
    assert after["trace"][1]["visited"] == 3
    assert "Secret" not in json.dumps(after)
    ingest(client, document_id="second", title="Changed relation", text="No relation remains.")
    updated = query(client, question="Alpha", mode="graph")
    assert {c["document_id"] for c in updated["citations"]} == {"first"}


def test_structured_predicates_aggregate_and_scalar_types(system):
    client, _, _ = system
    rows = [{"region": "north", "amount": 10, "active": True},
            {"region": "north", "amount": 15, "active": False},
            {"region": "south", "amount": 99, "active": True}]
    ingest(client, document_id="sales", title="Sales", rows=rows)
    result = query(client, question="sales amount", mode="structured",
                   row_filters=[{"field": "region", "op": "eq", "value": "north"},
                                {"field": "amount", "op": "gte", "value": 10}],
                   aggregate={"operation": "sum", "field": "amount"})
    assert result["structured_result"]["aggregate"]["value"] == 25
    assert result["structured_result"]["matched_rows"] == 2
    boolean = query(client, question="sales", mode="structured", row_filters=[{"field": "active", "op": "eq", "value": 1}])
    assert boolean["structured_result"]["matched_rows"] == 0
    other = query(client, token=B, question="sales", mode="structured", aggregate={"operation": "count"})
    assert other["structured_result"]["aggregate"]["value"] == 0
    assert client.post("/query", headers=A, json={"question": "sales", "mode": "structured", "sql": "SELECT *"}).status_code == 422


def test_reindex_atomicity_profile_and_conflicting_revision(system, monkeypatch):
    client, service, _ = system
    first = ingest(client, document_id="guide", title="Guide", text="A long bright constellation. " * 20)
    original = service.provider.embedding
    def failure(text):
        raise RuntimeError("synthetic provider outage")
    monkeypatch.setattr(service.provider, "embedding", failure)
    with pytest.raises(RuntimeError):
        service.reindex("a", "guide", first["revision"], chunking={"strategy": "fixed", "max_tokens": 50, "overlap_tokens": 5})
    assert service.store.get_document("a", "guide")["revision"] == first["revision"]
    monkeypatch.setattr(service.provider, "embedding", original)
    rebuilt = client.post("/documents/guide/reindex", headers=A, json={
        "expected_revision": first["revision"], "chunking": {"strategy": "fixed", "max_tokens": 50, "overlap_tokens": 5}})
    assert rebuilt.status_code == 200, rebuilt.text
    assert rebuilt.json()["profile_id"] != first["profile_id"]
    assert rebuilt.json()["chunks"] > first["chunks"]
    stale = client.post("/documents/guide/reindex", headers=A, json={"expected_revision": first["revision"]})
    assert stale.status_code == 409
    assert client.post("/documents/guide/reindex", headers=B, json={"expected_revision": first["revision"]}).status_code == 404


def test_federation_uses_distinct_tenant_credentials_and_rejects_unknown_sources(system, monkeypatch):
    _, _, settings = system
    sources = {"archive": {"url": "https://archive.example", "tenant_keys": {"a": "remote-a", "b": "remote-b"}},
               "catalog": {"url": "https://catalog.example", "tenant_keys": {"a": "catalog-a"}}}
    configured = replace(settings, federated_sources=sources)
    calls = []
    original_client = httpx.AsyncClient
    def handler(request):
        calls.append((str(request.url), request.headers["Authorization"]))
        data = json.loads(request.content)
        assert data["generate"] is False and "tenant" not in data
        return httpx.Response(200, request=request, json={"citations": [{"chunk_id": "same-id", "document_id": "doc",
              "title": "Authorized", "source": "archive", "section": "Policy", "text": request.headers["Authorization"] + " policy"}]})
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs))
    service = RAGService(Store(settings.db_path), Providers(configured))
    result = service.query("a", "policy", "federated", generate=False, sources=["archive", "catalog"])
    assert {c["chunk_id"] for c in result["citations"]} == {"archive:same-id", "catalog:same-id"}
    result_b = service.query("b", "policy", "federated", generate=False, sources=["archive"])
    assert "remote-b" in result_b["citations"][0]["text"]
    assert ("https://archive.example/query", "Bearer remote-a") in calls
    assert ("https://archive.example/query", "Bearer remote-b") in calls
    before = len(calls)
    with pytest.raises(CapabilityError, match="not authorized"):
        service.query("b", "policy", "federated", sources=["catalog"])
    with pytest.raises(CapabilityError, match="not authorized"):
        service.query("a", "policy", "federated", sources=["https://attacker.example"])
    assert len(calls) == before


def test_model_modes_have_distinct_queries_and_bounded_control_flow(system, monkeypatch):
    client, service, settings = system
    ingest(client, document_id="one", title="Mars", text="Mars is the red planet with two moons.")
    provider = Providers(replace(settings, model_provider="openai", openai_key="fixture-key"))
    upgraded = RAGService(service.store, provider)
    calls = []
    def model(task, data):
        calls.append((task, data))
        if task.startswith("Generate2"):
            return {"queries": ["Mars red planet", "planet two moons"]}
        if task.startswith("Generate1"):
            return {"documents": ["Mars is a planet with moons.", "The red planet is Mars."]}
        if task.startswith("Choose"):
            return {"mode": "sparse", "reason": "Exact astronomical terms"}
        if task.startswith("Plan"):
            return {"queries": ["Mars moons", "red planet"]}
        return {"supported": False, "keep_ids": [], "next_query": "Try another planet", "reason": "Insufficient evidence"}
    monkeypatch.setattr(provider, "json_task", model)
    fusion = upgraded.query("a", "Mars?", "fusion", generate=False)
    assert fusion["trace"][1]["queries"] == ["Mars red planet", "planet two moons"]
    hyde = upgraded.query("a", "Mars?", "hyde", generate=False)
    assert hyde["trace"][1]["hypotheses"] == 2
    routed = upgraded.query("a", "Mars?", "adaptive", generate=False)
    assert routed["mode_used"] == "sparse" and routed["trace"][0]["action"] == "route"
    corrective = upgraded.query("a", "Mars?", "corrective", generate=False, max_iterations=2)
    assert corrective["supported"] is False
    assert corrective["trace"][-1]["iterations"] <= 2
    assert corrective["trace"][-1]["reason"] == "iteration_limit"
    # A proposed source ID is never accepted merely because a model emitted it.
    monkeypatch.setattr(provider, "json_task", lambda task, data: {"supported": True, "keep_ids": ["foreign-source"], "reason": "yes"})
    with pytest.raises(CapabilityError, match="invalid source"):
        upgraded.query("a", "Mars?", "corrective", generate=False)


def test_context_budget_includes_question_and_source_headers(system):
    client, service, settings = system
    ingest(client, document_id="long", title="Long title " * 10, text="Zephyr evidence with words. " * 50,
           chunking={"strategy": "fixed", "max_tokens": 800, "overlap_tokens": 50})
    limited = RAGService(service.store, Providers(replace(settings, context_budget=700, output_budget=100, envelope_budget=60)))
    response = limited.query("a", "What is the Zephyr evidence?", "hybrid", top_k=5)
    assert response["budget"]["input_units"] <= 540
    assert response["budget"]["unit"] == "unicode_codepoints"
    assert response["budget"]["truncated_ids"] or response["budget"]["dropped_ids"]
    with pytest.raises(ValueError):
        limited.query("a", "Q" * 1000, "hybrid")


def test_generated_context_is_indexed_but_never_substitutes_source_text(system, monkeypatch):
    _, base, settings = system
    provider = Providers(replace(settings, model_provider="openai", openai_key="fixture-key"))
    called = []
    def contextualizer(document, chunk, title, *, max_units, tokenizer):
        called.append((max_units, tokenizer.unit))
        return "Orion telescope calibration"
    monkeypatch.setattr(provider, "contextualize", contextualizer)
    service = RAGService(base.store, provider)
    raw = "The mirror alignment requires three adjustment screws."
    service.ingest("a", "Manual", raw, "manual", document_id="contextual",
                   chunking={"strategy": "contextual", "max_tokens": 80, "overlap_tokens": 10, "context_max_tokens": 40})
    assert called == [(40, "unicode_codepoints")]
    found = service.query("a", "Orion", "contextual", generate=False)
    assert found["citations"][0]["text"] == raw
    sparse = service.query("a", "Orion", "sparse", generate=False)
    assert sparse["citations"][0]["document_id"] == "contextual"
    stored = service.store.scope("a")[0]
    assert stored["processing_metadata"]["context_token_count"] > 0
    assert stored["index_profile"]["context_model"].startswith("openai:")


def test_backend_cannot_promote_generated_summary_to_source_evidence(system, monkeypatch):
    client, service, _ = system
    ingest(client, title="Hierarchy", document_id="hierarchy", text="Source fact about cedar trees. " * 20,
           chunking={"strategy": "fixed", "max_tokens": 80, "overlap_tokens": 5}, index_features=["hierarchy"])
    summary = next(c for c in service.store.scope("a") if c["role"] == "summary")
    class UntrustedBackend:
        def search(self, *args, **kwargs):
            return {"hits": [{"chunk_id": summary["id"], "score": 1.0}]}
    monkeypatch.setattr(service, "_backend", lambda: UntrustedBackend())
    with pytest.raises(CapabilityError, match="outside the authorized corpus"):
        service.query("a", "cedar", "full_graphrag", generate=False)


def test_self_reflective_and_agentic_success_require_valid_evidence(system, monkeypatch):
    client, base, settings = system
    ingest(client, title="Orbital facts", document_id="orbit", text="Phobos and Deimos are the two moons of Mars.")
    provider = Providers(replace(settings, model_provider="openai", openai_key="fixture-key"))
    service = RAGService(base.store, provider)
    source_id = service.store.scope("a")[0]["id"]
    reviews = []
    def model(task, data):
        if task.startswith("Draft"):
            return {"answer": f"Mars has two moons. [{source_id}]", "cited_ids": [source_id]}
        if task.startswith("Plan"):
            return {"queries": ["Phobos", "Deimos"]}
        reviews.append(task)
        return {"supported": True, "keep_ids": [source_id], "next_query": "", "reason": "Both moons appear in the source"}
    monkeypatch.setattr(provider, "json_task", model)
    reflected = service.query("a", "How many moons does Mars have?", "self_reflective")
    assert reflected["answer"] == f"Mars has two moons. [{source_id}]"
    assert reflected["trace"][-1]["reason"] == "supported_evidence"
    agentic = service.query("a", "Mars moons", "agentic", generate=False)
    assert agentic["supported"] is True
    assert agentic["trace"][0]["query"] == "Phobos"
    assert len(reviews) == 2


def test_migration_marks_legacy_processing_and_requires_real_reindex(tmp_path, monkeypatch):
    import sqlite3
    path = str(tmp_path / "legacy.db")
    with sqlite3.connect(path) as connection:
        connection.executescript('''
        CREATE TABLE documents(tenant TEXT,id TEXT,title TEXT,source TEXT,body TEXT,metadata TEXT,checksum TEXT,updated_at TEXT,PRIMARY KEY(tenant,id));
        CREATE TABLE chunks(id TEXT PRIMARY KEY,tenant TEXT,document_id TEXT,ordinal INTEGER,section TEXT,content TEXT,embedding TEXT,embedding_model TEXT);
        INSERT INTO documents VALUES('a','old','Old','manual','Legacy fact.','{}','old-checksum','2020-01-01T00:00:00+00:00');
        INSERT INTO chunks VALUES('chunk-old','a','old',0,'Document','Legacy fact.','[1,0]','offline-hash-v1');
        ''')
    monkeypatch.setenv("RAGS_ENV", "test")
    settings = replace(Settings.from_env(), db_path=path)
    service = RAGService(Store(path), Providers(settings))
    row = service.store.scope("a")[0]
    assert row["profile_id"] == "legacy"
    assert row["start_char"] is None and row["token_count"] is None
    with pytest.raises(CapabilityError, match="legacy"):
        service.query("a", "Legacy fact", "dense", generate=False)
    rebuilt = service.reindex("a", "old", "legacy")
    assert rebuilt["profile_id"] != "legacy"
    assert service.query("a", "Legacy", "dense", generate=False)["citations"]


def test_federation_stream_limit_closes_oversized_response(system, monkeypatch):
    _, base, settings = system
    state = {"parts": 0, "closed": False}
    class LargeStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for _ in range(100):
                state["parts"] += 1
                yield b"x" * 100_000
        async def aclose(self):
            state["closed"] = True
    original_client = httpx.AsyncClient
    def handler(request):
        return httpx.Response(200, request=request, stream=LargeStream())
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs))
    configured = replace(settings, federated_sources={"archive": {"url": "https://archive.example", "tenant_keys": {"a": "credential-a"}}})
    service = RAGService(base.store, Providers(configured))
    with pytest.raises(CapabilityError, match="no partial result"):
        service.query("a", "policy", "federated", generate=False)
    assert state["parts"] <= 21
    assert state["closed"] is True


def test_predicate_lists_reject_nonfinite_values():
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            RowPredicate(field="amount", op="in", value=[value])


def test_reflective_draft_validates_inline_citations_even_without_answer_generation(system, monkeypatch):
    client, base, settings = system
    ingest(client, title="Moons", document_id="moons", text="Mars has two moons.")
    provider = Providers(replace(settings, model_provider="openai", openai_key="fixture-key"))
    source_id = base.store.scope("a")[0]["id"]
    monkeypatch.setattr(provider, "json_task", lambda task, data: {
        "answer": "Mars has two moons. [foreign-source]", "cited_ids": [source_id]})
    with pytest.raises(CapabilityError, match="unavailable source"):
        RAGService(base.store, provider).query("a", "Mars moons", "self_reflective", generate=False)


def test_non_ascii_bearer_is_rejected_without_server_error(system):
    client, _, _ = system
    response = client.get("/documents", headers={b"Authorization": "Bearer 測試錯誤憑證".encode()})
    assert response.status_code == 401


def test_federation_checks_deadline_for_each_small_transport_chunk(system, monkeypatch):
    _, base, settings = system
    state = {"parts": 0, "closed": False}
    class Trickle(httpx.AsyncByteStream):
        async def __aiter__(self):
            for _ in range(100):
                await asyncio.sleep(0.01)
                state["parts"] += 1
                yield b"x"
        async def aclose(self):
            state["closed"] = True
    original_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original_client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=Trickle())), **kwargs))
    monkeypatch.setattr(retrieval_module, "FEDERATED_TIMEOUT_SECONDS", 0.05)
    settings = replace(settings, federated_sources={"archive": {"url": "https://archive.example", "tenant_keys": {"a": "key-a"}}})
    with pytest.raises(CapabilityError, match="no partial result"):
        RAGService(base.store, Providers(settings)).query("a", "policy", "federated", generate=False)
    assert state["parts"] <= 8
    assert state["closed"] is True
