"""Feature permissions cross actual API, service, index and MCP boundaries."""
import asyncio
from dataclasses import replace
import json
import os

from fastapi.testclient import TestClient
import httpx
import pytest

from rags.api import create_app
from rags.chunking import STRATEGIES
from rags.config import Settings
from rags.features import FeatureFlags
from rags.providers import Providers
from rags.registry import CapabilityError, MODES
from rags.research_protocol import ResearchRequest
from rags.research_service import ResearchSettings, ProcessRunner, create_app as research_app, worker_environment
from rags.retrieval import RAGService
from rags.storage import Store

pytestmark = pytest.mark.integration
AUTH = {"Authorization": "Bearer feature-test-key"}


@pytest.fixture
def settings(tmp_path, monkeypatch):
    for name in tuple(os.environ):
        if name.startswith(("RAGS_", "OPENAI_", "ANTHROPIC_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("RAGS_ENV", "test")
    return replace(Settings.from_env(), db_path=str(tmp_path / "features.db"),
                   keys={"feature-test-key": "tenant-a", "other-key": "tenant-b"})


def flags(*names):
    return FeatureFlags(tuple((name, False) for name in names))


def ingest(settings):
    service = RAGService(Store(settings.db_path), Providers(settings))
    result = service.ingest("tenant-a", "Policy", "Alpha service -> Beta service\nTelemetry is retained for thirty days.",
                            "spec://policy", document_id="policy")
    return service, result


@pytest.mark.parametrize("mode", MODES)
def test_each_strategy_is_disabled_before_reading_sources_or_calling_provider(settings, monkeypatch, mode):
    service = RAGService(Store(settings.db_path), Providers(replace(settings, features=flags("strategy." + mode))))
    monkeypatch.setattr(service.store, "scope", lambda *a, **kw: pytest.fail("disabled strategy read source data"))
    monkeypatch.setattr(service.provider, "json_task", lambda *a, **kw: pytest.fail("disabled strategy invoked generation"))
    with pytest.raises(CapabilityError) as error:
        service.query("tenant-a", "Where is the policy?", mode, generate=False)
    assert error.value.code == "feature_disabled" and error.value.status_code == 403


@pytest.mark.parametrize("mode,feature", [
    ("fusion", "strategy.dense"), ("hyde", "embedding.local"), ("hybrid", "strategy.sparse"),
    ("corrective", "strategy.hybrid"), ("federated", "federation"), ("raptor", "research"),
    ("lightrag", "model_generation"), ("multimodal_native", "multimodal"),
])
def test_transitive_permissions_precede_generation_network_or_scope(settings, monkeypatch, mode, feature):
    configured = replace(settings, model_provider="openai", openai_key="private-key", features=flags(feature))
    service = RAGService(Store(settings.db_path), Providers(configured))
    monkeypatch.setattr(service.store, "scope", lambda *a, **kw: pytest.fail("dependent feature read data"))
    monkeypatch.setattr(httpx, "post", lambda *a, **kw: pytest.fail("dependent feature called HTTP"))
    with pytest.raises(CapabilityError, match="Feature is disabled"):
        service.query("tenant-a", "Where is the policy?", mode, generate=False)


@pytest.mark.parametrize("chunker", STRATEGIES)
def test_each_chunker_is_disabled_before_embedding_or_source_mutation(settings, monkeypatch, chunker):
    service = RAGService(Store(settings.db_path), Providers(replace(settings, features=flags("chunking." + chunker))))
    monkeypatch.setattr(service.provider, "embedding", lambda *a: pytest.fail("disabled chunker called encoder"))
    with pytest.raises(CapabilityError) as error:
        service.ingest("tenant-a", "Title", "Source text", "manual", chunking={"strategy": chunker})
    assert error.value.code == "feature_disabled"
    assert service.store.list_documents("tenant-a") == []


@pytest.mark.parametrize("feature,options", [
    ("ingestion", {}), ("tokenizer.unicode", {}), ("embedding.local", {}),
    ("index.hierarchy", {"index_features": ["hierarchy"]}),
    ("index.graph", {"index_features": ["graph"]}),
])
def test_ingestion_permissions_preserve_existing_revision(settings, feature, options):
    service, first = ingest(settings)
    restricted = RAGService(service.store, Providers(replace(settings, features=flags(feature))))
    with pytest.raises(CapabilityError) as error:
        restricted.ingest("tenant-a", "New title", "New text", "manual", document_id="policy", **options)
    assert error.value.code == "feature_disabled"
    assert service.store.get_document("tenant-a", "policy")["revision"] == first["revision"]
    assert len(service.store.revisions("tenant-a", "policy")) == 1


@pytest.mark.parametrize("feature,method,path,body", [
    ("documents", "GET", "/documents", None),
    ("documents", "GET", "/documents/policy", None),
    ("documents", "GET", "/documents/policy/revisions", None),
    ("deletion", "DELETE", "/documents/policy", None),
    ("ingestion", "POST", "/documents", {"title": "New", "text": "new content"}),
    ("reindex", "POST", "/documents/policy/reindex", {"expected_revision": "unchanged"}),
    ("query", "POST", "/query", {"question": "policy", "generate": False}),
])
def test_http_permissions_preserve_authentication_and_sources(settings, feature, method, path, body):
    service, first = ingest(settings)
    app = create_app(replace(settings, features=flags(feature)))
    with TestClient(app) as client:
        request = {"json": body} if body is not None else {}
        assert client.request(method, path, **request).status_code == 401
        response = client.request(method, path, headers=AUTH, **request)
        assert response.status_code == 403, response.text
        assert response.json()["detail"]["code"] == "feature_disabled"
    assert service.store.get_document("tenant-a", "policy")["revision"] == first["revision"]


@pytest.mark.parametrize("path,feature", [("/documents/upload", "uploads"), ("/images", "image_captioning"),
                                            ("/images", "model_generation")])
def test_upload_permissions_fail_before_parsers_and_vision(settings, path, feature, monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **kw: pytest.fail("disabled image called provider"))
    with TestClient(create_app(replace(settings, features=flags(feature)))) as client:
        response = client.post(path, headers=AUTH, files={"file": ("bad.pdf", b"invalid file", "application/pdf")})
        assert response.status_code == 403 and response.json()["detail"]["code"] == "feature_disabled"
        assert client.app.state.service.store.list_documents("tenant-a") == []


def test_capabilities_report_effective_permissions_and_storage_without_secrets(settings):
    selected = replace(settings, features=flags("api_docs", "strategy.dense", "mcp"),
                       openai_key="secret-key-sentinel", embedding_base_url="https://private-endpoint.invalid/v1")
    with TestClient(create_app(selected)) as client:
        assert client.get("/capabilities").status_code == 401
        response = client.get("/capabilities", headers=AUTH)
        assert response.status_code == 200
        result = response.json()
        assert result["storage"] == {"backend": "sqlite", "vector_search": "python_exact"}
        assert result["features"]["strategy.dense"] is False and result["features"]["ingestion"] is True
        catalog = {item["id"]: item for item in result["capabilities"]}
        assert catalog["hybrid"]["status"] == "feature_disabled"
        assert "strategy.dense" in catalog["hybrid"]["disabled_by"]
        assert catalog["sparse"]["status"] == "available"
        strategies = client.get("/strategies", headers=AUTH).json()
        assert "dense" not in strategies["strategies"] and "hybrid" not in strategies["strategies"]
        assert "sparse" in strategies["strategies"]
        for path in ("/docs", "/redoc", "/openapi.json"):
            assert client.get(path).status_code == 404
        assert client.get("/live").status_code == 200
        assert client.get("/capabilities", headers={**AUTH, "X-RAGS-Interface": "mcp"}).status_code == 403
        assert "secret-key-sentinel" not in response.text and "private-endpoint" not in response.text
        assert "features.db" not in response.text


def test_disabled_graph_builder_does_not_create_implicit_edges(settings):
    service = RAGService(Store(settings.db_path), Providers(replace(settings, features=flags("index.graph"))))
    service.ingest("tenant-a", "Relations", "Alpha service -> Beta service", "manual")
    assert service.store.relations("tenant-a", {chunk["id"] for chunk in service.store.scope("tenant-a")}) == []


def test_configured_model_is_not_silently_replaced_after_generation_disabled(settings, monkeypatch):
    ingest(settings)
    configured = replace(settings, model_provider="openai", openai_key="configured-key", features=flags("model_generation"))
    monkeypatch.setattr(httpx, "post", lambda *a, **kw: pytest.fail("disabled model called HTTP"))
    with TestClient(create_app(configured)) as client:
        response = client.post("/query", headers=AUTH, json={"question": "Telemetry", "mode": "sparse"})
        assert response.status_code == 403
        response = client.post("/query", headers=AUTH, json={"question": "Telemetry", "mode": "sparse", "generate": False})
        assert response.status_code == 200 and response.json()["citations"]
        response = client.post("/documents", headers=AUTH, json={"title": "Graph", "text": "Alpha -> Beta", "index_features": ["graph"]})
        assert response.status_code == 403


def test_admission_scoring_permission_does_not_disable_required_policy(settings):
    selected = replace(settings, features=flags("admission_scoring"), admission_policy="calibrated_required")
    with pytest.raises(ValueError, match="Required evidence"):
        RAGService(Store(settings.db_path), Providers(selected))
    selected = replace(selected, admission_policy="compatibility", admission_profile="not-read-when-disabled.json")
    service = RAGService(Store(settings.db_path), Providers(selected))
    assert service.query("tenant-a", "nothing indexed", generate=False)["outcome"] == "insufficient_evidence"


def test_mcp_registration_and_calls_obey_both_local_and_server_permissions(settings, monkeypatch):
    import rags.mcp_server as bridge
    from mcp.server.fastmcp import FastMCP
    app = create_app(replace(settings, features=flags("query")))
    monkeypatch.setattr(bridge, "BASE_URL", "http://testserver")
    monkeypatch.setattr(bridge, "HEADERS", {**AUTH, "X-RAGS-Interface": "mcp"})
    monkeypatch.setattr(bridge, "FEATURES", FeatureFlags())
    monkeypatch.setattr(bridge.httpx, "Client", lambda **kwargs: TestClient(app))
    tools = FastMCP("features-test")
    tools.add_tool(bridge.search_knowledge)
    tools.add_tool(bridge.list_knowledge_documents)
    monkeypatch.setattr(bridge, "mcp", tools)
    bridge.configure_tools()
    assert [tool.name for tool in asyncio.run(tools.list_tools())] == ["list_knowledge_documents"]
    assert bridge.list_knowledge_documents() == []
    with pytest.raises(CapabilityError, match="query"):
        bridge.search_knowledge("retention policy")
    monkeypatch.setattr(bridge, "FEATURES", flags("mcp"))
    monkeypatch.setattr(bridge.httpx, "Client", lambda **kwargs: pytest.fail("disabled MCP constructed HTTP client"))
    with pytest.raises(CapabilityError, match="mcp"):
        bridge.configure_tools()
    with pytest.raises(CapabilityError, match="mcp"):
        bridge.search_knowledge("retention policy")


def test_research_flag_propagates_to_worker_and_denies_before_spawn(settings, monkeypatch):
    selected = ResearchSettings("lightrag", {"feature-test-key": "tenant-a"}, features=flags("research"))
    environment = worker_environment(selected)
    assert json.loads(environment["RAGS_FEATURES_JSON"])["research"] is False
    body = {"protocol": "authorized-evidence-v1", "engine": "lightrag", "namespace": "tenant-a",
            "question": "policy", "top_k": 1, "passages": []}
    monkeypatch.setattr(asyncio, "create_subprocess_exec", lambda *a, **kw: pytest.fail("disabled research spawned worker"))
    with pytest.raises(CapabilityError, match="research"):
        asyncio.run(ProcessRunner(selected).run(ResearchRequest.model_validate(body)))
    with TestClient(research_app(selected)) as client:
        response = client.post("/retrieve", headers=AUTH, json=body)
        assert response.status_code == 403 and response.json()["detail"]["code"] == "feature_disabled"
