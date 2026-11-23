"""Provider failures cross the real ASGI/storage boundary as safe execution errors."""
import json
from dataclasses import replace

import httpx
import pytest
from fastapi.testclient import TestClient

import rags.providers as provider_module
from rags.api import create_app
from rags.config import Settings

pytestmark = pytest.mark.integration
AUTH = {"Authorization": "Bearer provider-key"}


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGS_ENV", "test")
    return replace(Settings.from_env(), db_path=str(tmp_path / "provider-api.db"),
                   keys={"provider-key": "tenant"}, embedding_provider="local", model_provider="extractive",
                   openai_key="fixture-secret-openai", anthropic_key="fixture-secret-anthropic",
                   openai_base_url="https://provider.invalid/v1", generation_tokenizer_config={"backend": "unicode"})


def invalid_response(monkeypatch, failure):
    def post(url, **_):
        request = httpx.Request("POST", url)
        if failure == "http":
            return httpx.Response(503, request=request, text="private upstream credential fixture-secret-openai")
        if failure == "timeout":
            raise httpx.ReadTimeout("private upstream credential fixture-secret-openai", request=request)
        if failure == "json":
            return httpx.Response(200, request=request, text="private upstream credential fixture-secret-openai")
        return httpx.Response(200, request=request, json={"private": "fixture-secret-openai"})

    monkeypatch.setattr(provider_module.provider_transport, "post", post)


@pytest.mark.parametrize("failure", ["envelope", "json", "http", "timeout"])
def test_failed_provider_reindex_returns_502_and_preserves_old_source_revision_and_indexes(settings, monkeypatch, failure):
    with TestClient(create_app(settings)) as client:
        first = client.post("/documents", headers=AUTH, json={
            "document_id": "policy", "title": "Network", "text": "Alpha -> Beta", "source": "v1",
        })
        assert first.status_code == 201
        before = client.get("/documents/policy", headers=AUTH).json()
    invalid_response(monkeypatch, failure)
    upgraded = replace(settings, embedding_provider="openai")
    with TestClient(create_app(upgraded), raise_server_exceptions=False) as client:
        response = client.post("/documents/policy/reindex", headers=AUTH, json={"expected_revision": first.json()["revision"]})
        assert response.status_code == 502, response.text
        detail = response.json()["detail"]
        assert detail["code"] == ("invalid_embedding" if failure in {"envelope", "json"} else "provider_failed")
        assert "private" not in response.text and "fixture-secret" not in response.text
        assert client.get("/documents/policy", headers=AUTH).json() == before
        assert len(client.get("/documents/policy/revisions", headers=AUTH).json()) == 1
        with client.app.state.service.store.connect() as connection:
            assert connection.execute("SELECT count(*) FROM audit WHERE action='upsert'").fetchone()[0] == 1
            assert connection.execute("SELECT count(*) FROM chunks_fts").fetchone()[0] == 1
            assert connection.execute("SELECT count(*) FROM graph_relations").fetchone()[0] == 1
    with TestClient(create_app(settings)) as client:
        result = client.post("/query", headers=AUTH, json={"question": "Beta", "mode": "sparse", "generate": False})
        assert result.status_code == 200
        assert {citation["revision"] for citation in result.json()["citations"]} == {first.json()["revision"]}


@pytest.mark.parametrize("model_provider", ["openai", "anthropic"])
@pytest.mark.parametrize("failure", ["envelope", "json"])
def test_malformed_generation_is_execution_failure_not_evidence_refusal(settings, monkeypatch, model_provider, failure):
    invalid_response(monkeypatch, failure)
    with TestClient(create_app(replace(settings, model_provider=model_provider)), raise_server_exceptions=False) as client:
        first = client.post("/documents", headers=AUTH, json={
            "title": "Retention", "text": "Satellite telemetry is retained for 30 days.",
        })
        assert first.status_code == 201
        response = client.post("/query", headers=AUTH, json={"question": "telemetry retention", "mode": "sparse"})
        assert response.status_code == 502, response.text
        assert response.json()["detail"]["code"] == "invalid_model_output"
        assert "evidence_outcome" not in response.json()
        assert "private" not in response.text and "fixture-secret" not in response.text
        assert "answer" not in json.dumps(response.json())
