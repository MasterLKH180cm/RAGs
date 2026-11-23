"""External research acquisition cannot broaden an existing query scope."""
from dataclasses import replace
import hashlib
import json

import pytest

from rags.backends import BackendManager
from rags.config import Settings
from rags.providers import Providers
from rags.registry import CapabilityError
from rags.research_protocol import ResearchRequest, UPSTREAMS, external_source_id
from rags.retrieval import RAGService
from rags.storage import Store


pytestmark = pytest.mark.integration


@pytest.fixture
def external_service(monkeypatch, tmp_path):
    monkeypatch.setenv("RAGS_ENV", "test")
    monkeypatch.setenv("RAGS_KEYS_JSON", '{"test":"tenant-a"}')
    monkeypatch.setenv("RAGS_MODEL_PROVIDER", "extractive")
    monkeypatch.setenv("RAGS_EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("RAGS_RESEARCH_BACKENDS_JSON", json.dumps({
        "crag": {"url": "https://engine.example", "tenant_tokens": {"tenant-a": "key-a"},
                 "require_pinned_upstream": True, "external_hosts": ["example.org"]},
    }))
    settings = replace(Settings.from_env(), db_path=str(tmp_path / "data.sqlite"))
    service = RAGService(Store(settings.db_path), Providers(settings))
    service.ingest("tenant-a", "Office", "The office is in Taipei.", "manual",
                   {"region": "east"}, document_id="office", valid_from="2025-01-01T00:00:00Z")
    calls = []

    async def external_response(config, credential, encoded):
        request = ResearchRequest.model_validate_json(encoded)
        calls.append(request)
        text, url = "The office is now in Taipei.", "https://example.org/office"
        digest = hashlib.sha256(text.encode()).hexdigest()
        source = {"id": external_source_id(request.namespace, url, digest), "url": url,
                  "title": "Office", "text": text, "content_sha256": digest,
                  "fetched_at": "2026-11-23T00:00:00+00:00"}
        return json.dumps({
            "implementation": UPSTREAMS["crag"],
            "hits": [{"chunk_id": source["id"], "score": 1.0}],
            "diagnostics": {"source_count": len(request.passages), "engine_executed": True},
            "grounded_answer": {"text": "Taipei.", "cited_ids": [source["id"]]},
            "external_sources": [source],
        }).encode()

    monkeypatch.setattr(BackendManager, "_request", staticmethod(external_response))
    return service, calls


@pytest.mark.parametrize("options", [
    {"filters": {"source": "manual"}},
    {"filters": {"source": "not-a-source"}},
    {"filters": {"metadata": {"region": "east"}}},
    {"filters": {"metadata": {"region": "west"}}},
    {"filters": {"valid_before": "2024-01-01T00:00:00Z"}},
    {"as_of": "2001-01-01T00:00:00Z"},
    {"as_of": "2026-11-23T00:00:00Z"},
])
def test_crag_web_cannot_expand_filtered_or_historical_scope_before_network(external_service, options):
    service, calls = external_service
    with pytest.raises(CapabilityError) as error:
        service.query("tenant-a", "Where is the office?", "crag", **options)
    assert error.value.code == "external_scope_unsupported"
    assert error.value.status_code == 422
    assert calls == []


@pytest.mark.parametrize("options", [{}, {"filters": None}, {"filters": {}}])
def test_explicit_web_authorization_still_allows_an_unconstrained_query(external_service, options):
    service, calls = external_service
    result = service.query("tenant-a", "Where is the office?", "crag", **options)
    assert len(calls) == 1
    assert result["outcome"] == "success"
    assert result["answer"].startswith("Taipei.")
    assert [citation["source"] for citation in result["citations"]] == ["https://example.org/office"]
    assert result["citations"][0]["origin_source"] == "crag_web"
