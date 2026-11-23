"""Versioned research answers, external provenance and scoped worker settings."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from rags.backends import BackendManager, BackendUnavailable
from rags.config import Settings
from rags.providers import Providers
from rags.registry import CapabilityError
from rags.research.common import ProviderBridge
from rags.research_protocol import ResearchRequest, UPSTREAMS, external_source_id, validate_result
from rags.research_service import ResearchSettings, authorize_external, worker_environment
from rags.retrieval import RAGService
from rags.storage import Store


def request(engine="self_rag", **updates):
    return ResearchRequest.model_validate({
        "protocol": "authorized-evidence-v2" if engine in {"self_rag", "crag"} else "authorized-evidence-v1",
        "engine": engine, "namespace": "tenant-a", "question": "When does the policy expire?", "top_k": 3,
        "passages": [{"id": "a", "text": "The policy expires after fourteen days."}], **updates,
    })


def response(req):
    value = {"implementation": dict(UPSTREAMS[req.engine]),
             "hits": [{"chunk_id": p.id, "score": 1.0} for p in req.passages[:req.top_k]],
             "diagnostics": {"source_count": len(req.passages), "engine_executed": bool(req.passages or req.external_hosts)},
             "usage": {"input_tokens": 10, "output_tokens": 5}}
    if req.protocol.endswith("v2"):
        value["grounded_answer"] = {"text": "Fourteen days.", "cited_ids": [req.passages[0].id]} if req.passages else None
    return value


@pytest.mark.unit
@pytest.mark.parametrize("engine", list(UPSTREAMS))
def test_each_engine_has_explicit_protocol_and_pinned_identity(engine):
    req = request(engine)
    assert validate_result(response(req), req)["implementation"] == UPSTREAMS[engine]
    changed = response(req)
    changed["implementation"]["source_commit"] = "0" * 40
    with pytest.raises(ValueError):
        validate_result(changed, req)


@pytest.mark.unit
@pytest.mark.parametrize("change", [
    {"grounded_answer": None}, {"grounded_answer": {"text": "x", "cited_ids": ["foreign"]}},
    {"grounded_answer": {"text": "x", "cited_ids": ["a", "a"]}},
    {"grounded_answer": {"text": "x", "cited_ids": ["a"], "secret": "hidden"}},
    {"external_sources": [{"id": "foreign"}]}, {"unknown": 1},
])
def test_invalid_answer_envelope_never_falls_back_to_ordinary_generation(change):
    req = request()
    with pytest.raises((ValueError, TypeError)):
        validate_result({**response(req), **change}, req)


@pytest.mark.unit
def test_v1_cannot_smuggle_an_upstream_answer_or_new_external_sources():
    req = request("raptor")
    for field, value in (("grounded_answer", {"text": "x", "cited_ids": ["a"]}), ("external_sources", [])):
        with pytest.raises(ValueError):
            validate_result({**response(req), field: value}, req)


def external(req):
    text, url = "The policy expires after fourteen days.", "https://example.org/policy"
    digest = hashlib.sha256(text.encode()).hexdigest()
    return {"id": external_source_id(req.namespace, url, digest), "url": url, "title": "Policy",
            "text": text, "content_sha256": digest, "fetched_at": "2026-11-23T00:00:00+00:00"}


@pytest.mark.unit
@pytest.mark.parametrize("mutation", ["tenant", "text", "host", "http", "port", "userinfo", "unselected", "naive-time"])
def test_external_snapshot_rejects_broken_authorization_and_provenance(mutation):
    req = request("crag", external_hosts=["example.org"])
    item = external(req)
    if mutation == "tenant":
        item["id"] = external_source_id("tenant-b", item["url"], item["content_sha256"])
    elif mutation == "text":
        item["text"] = "Tampered source"
    elif mutation == "host":
        item["url"] = "https://evil.example/policy"
    elif mutation == "http":
        item["url"] = "http://example.org/policy"
    elif mutation == "port":
        item["url"] = "https://example.org:8443/policy"
    elif mutation == "userinfo":
        item["url"] = "https://user@example.org/policy"
    elif mutation == "naive-time":
        item["fetched_at"] = "2026-11-23T00:00:00"
    result = response(req)
    result["external_sources"] = [item]
    if mutation != "unselected":
        result["hits"].append({"chunk_id": item["id"], "score": 1.0})
    with pytest.raises(ValueError):
        validate_result(result, req)


@pytest.mark.unit
def test_external_acquisition_needs_both_host_and_tenant_permissions(monkeypatch):
    monkeypatch.setenv("RAGS_ENV", "test")
    config = ResearchSettings("crag", {"a": "tenant-a", "b": "tenant-b"},
                              external_hosts=("example.org",), external_tenants=("tenant-a",))
    authorize_external(config, request("crag", external_hosts=["example.org"]))
    for updates in ({"namespace": "tenant-b"}, {"external_hosts": ["elsewhere.org"]}):
        req = request("crag", **{"external_hosts": ["example.org"], **updates})
        with pytest.raises(CapabilityError) as error:
            authorize_external(config, req)
        assert error.value.status_code == 403


@pytest.mark.unit
def test_worker_gateway_configuration_forwards_only_selected_credentials(monkeypatch, tmp_path):
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "RAGS_GENERATION_API_KEY", "RAGS_EMBEDDING_API_KEY",
                 "RAGS_KEYS_JSON", "RAGS_RESEARCH_KEYS_JSON"):
        monkeypatch.setenv(name, "unrelated-secret-sentinel")
    monkeypatch.setenv("RAGS_ENV", "test")
    monkeypatch.setenv("RAGS_MODEL_PROVIDER", "router")
    monkeypatch.setenv("RAGS_EMBEDDING_PROVIDER", "fastembed")
    monkeypatch.setenv("RAGS_ROUTER_GATEWAY_URL", "https://gateway.example/v1")
    monkeypatch.setenv("RAGS_ROUTER_MODELS_JSON", '{"main":"openai/gpt-4.1-mini"}')
    monkeypatch.setenv("RAGS_ROUTER_API_KEY_FILE", "gateway.key")
    monkeypatch.chdir(tmp_path)
    environment = worker_environment(ResearchSettings("hipporag", {"key": "tenant-a"}))
    assert "unrelated-secret-sentinel" not in json.dumps(environment)
    assert environment["RAGS_ROUTER_GATEWAY_URL"] == "https://gateway.example/v1"
    assert environment["RAGS_ROUTER_API_KEY_FILE"] == str(tmp_path / "gateway.key")
    assert environment["RAGS_AGENT_ENABLED"] == "false"
    native = worker_environment(ResearchSettings("self_rag", {"key": "tenant-a"}))
    assert not any("API_KEY" in name for name in native)


@pytest.mark.unit
@pytest.mark.parametrize("resolved,accepted", [("openai/text-embedding-3-small", True),
                                             ("gateway/embedding-alias", False), ("openai/arbitrary", False)])
def test_router_embedding_alias_requires_a_known_exact_token_window(resolved, accepted):
    candidate = SimpleNamespace(
        settings=SimpleNamespace(embedding_provider="router", embedding_model="embed"),
        has_generation=True, embedding_model="router:" + resolved, model_identity="fixture:generator",
        model_router=SimpleNamespace(config=SimpleNamespace(models={"embed": resolved})),
        embedding=lambda text: [1., 0.], query_embedding=lambda text: [0., 1.],
    )
    bridge = ProviderBridge(candidate)
    if accepted:
        assert bridge.embed("An actual counted input") == [1., 0.]
        assert bridge.embedding_token_limit == 8191
    else:
        with pytest.raises(CapabilityError, match="exact token counter"):
            bridge.embed("Do not send this before validating the counter")


@pytest.fixture
def grounded_service(monkeypatch, tmp_path):
    monkeypatch.setenv("RAGS_ENV", "test")
    monkeypatch.setenv("RAGS_KEYS_JSON", '{"test":"tenant-a"}')
    monkeypatch.setenv("RAGS_MODEL_PROVIDER", "extractive")
    monkeypatch.setenv("RAGS_EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("RAGS_RESEARCH_BACKENDS_JSON", json.dumps({
        "self_rag": {"url": "https://engine.example", "tenant_tokens": {"tenant-a": "key-a", "tenant-b": "key-b"},
                     "require_pinned_upstream": True}}))
    settings = replace(Settings.from_env(), db_path=str(tmp_path / "data.sqlite"))
    service = RAGService(Store(settings.db_path), Providers(settings))
    def forbidden(*args):
        pytest.fail("The ordinary generator must never overwrite the trained answer")
    monkeypatch.setattr(service.provider, "complete", forbidden)
    async def fake_http(config, key, encoded):
        return json.dumps(response(ResearchRequest.model_validate_json(encoded))).encode()
    monkeypatch.setattr(BackendManager, "_request", staticmethod(fake_http))
    service.ingest("tenant-a", "Policy", "The policy expires after fourteen days.", "manual", document_id="policy")
    yield service


@pytest.mark.integration
def test_selected_trained_answer_and_backend_usage_survive_actual_retrieval_pipeline(grounded_service):
    result = grounded_service.query("tenant-a", "When does it expire?", mode="self_rag")
    assert result["answer"].startswith("Fourteen days.")
    assert result["citations"][0]["chunk_id"] in result["answer"]
    assert result["backend_usage"]["reported"] == {"input_tokens": 10, "output_tokens": 5}
    assert "draft_answer" not in result and "grounded_ids" not in result
    assert grounded_service.query("tenant-b", "When does it expire?", mode="self_rag")["citations"] == []
    assert grounded_service.query("tenant-a", "When does it expire?", mode="self_rag", generate=False)["answer"] is None


@pytest.mark.integration
def test_final_context_budget_cannot_reuse_answer_from_truncated_source(grounded_service):
    grounded_service.ingest("tenant-a", "Policy", "The policy expires after fourteen days. " + "source detail " * 50,
                            "manual", document_id="policy")
    grounded_service.settings = replace(grounded_service.settings, context_budget=800, output_budget=256, envelope_budget=64)
    result = grounded_service.query("tenant-a", "When does the policy expire?", mode="self_rag")
    assert result["outcome"] == "insufficient_evidence"
    assert result["refusal_reason"] == "context_budget_exhausted"
    assert result["citations"] == [] and "Fourteen days" not in result["answer"]


@pytest.mark.integration
def test_backend_rejects_unselected_grounded_answer_before_exposure(grounded_service, monkeypatch):
    async def corrupt(config, key, encoded):
        req = ResearchRequest.model_validate_json(encoded)
        result = deepcopy(response(req))
        result["grounded_answer"]["cited_ids"] = ["tenant-b-source"]
        return json.dumps(result).encode()
    monkeypatch.setattr(BackendManager, "_request", staticmethod(corrupt))
    with pytest.raises(BackendUnavailable):
        grounded_service._backend().search("self_rag", tenant="tenant-a", question="When does it expire?",
                                          chunks=grounded_service.store.scope("tenant-a"), top_k=1)
