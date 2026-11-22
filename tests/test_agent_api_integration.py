"""Full ASGI Agent Platform lifecycle with authenticated tenant isolation."""
from dataclasses import replace
import os

from fastapi.testclient import TestClient
import pytest

from rags.api import create_app
from rags.agents.router import ModelRouter, ModelReply
from rags.config import Settings

pytestmark = pytest.mark.integration

ROUTER_ENV = {
    "RAGS_ROUTER_ENABLED": "true",
    "RAGS_AGENT_ENABLED": "true",
    "RAGS_MEMORY_ENABLED": "true",
    "RAGS_ROUTER_GATEWAY_URL": "https://gateway.invalid/v1",
    "RAGS_ROUTER_API_KEY": "fake-test-secret",
    "RAGS_ROUTER_MODELS_JSON": '{"main":"openai/mock"}',
}
AGENT = {"Authorization": "Bearer agent-a"}
FOREIGN = {"Authorization": "Bearer agent-b"}
READER = {"Authorization": "Bearer reader-a"}
ADMIN = {"Authorization": "Bearer admin-a"}


@pytest.fixture
def settings(tmp_path, monkeypatch):
    for name in tuple(os.environ):
        if name.startswith(("RAGS_", "OPENAI_", "ANTHROPIC_")):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RAGS_ENV", "test")
    for key, value in ROUTER_ENV.items():
        monkeypatch.setenv(key, value)
    settings = Settings.from_env()
    return replace(settings, db_path=str(tmp_path / "agent-api.db"),
                   keys={"agent-a": "tenant-a", "agent-b": "tenant-b",
                         "reader-a": "tenant-a", "admin-a": "tenant-a"},
                   key_roles={"agent-a": ("agent_runner",),
                              "agent-b": ("agent_runner",),
                              "reader-a": ("reader",),
                              "admin-a": ("admin",)})


def test_agent_api_returns_tenant_scoped_checkpoint_and_history(settings, monkeypatch):
    calls = []
    def response(self, *, model, messages, tools):
        calls.append((model, messages, tools))
        return ModelReply(content="Confirmed answer", model="openai/mock", finish_reason="stop")
    monkeypatch.setattr(ModelRouter, "complete", response)
    with TestClient(create_app(settings)) as client:
        assert client.get("/agents/capabilities").status_code == 401
        assert client.post("/agents/runs", headers=READER, json={
            "goal": "Find relevant documentation", "model": "main", "session_id": "one"}).status_code == 403
        cap = client.get("/agents/capabilities", headers=AGENT)
        assert cap.status_code == 200
        assert cap.json()["models"] == ["main"] and cap.json()["memory_enabled"] is True
        assert "remember_observation" in cap.json()["tools"]
        result = client.post("/agents/runs", headers=AGENT, json={
            "goal": "What is the support policy?", "model": "main", "session_id": "one"})
        assert result.status_code == 201, result.text
        body = result.json()
        assert body["status"] == "completed" and body["answer"] == "Confirmed answer"
        identifier = body["run_id"]
        assert client.get("/agents/runs/"+identifier, headers=AGENT).status_code == 200
        assert client.get("/agents/runs/"+identifier, headers=FOREIGN).status_code == 404
        assert client.get("/agents/runs/"+identifier, headers=READER).status_code == 403
        assert client.get("/agents/sessions/one/messages", headers=FOREIGN).json()["messages"] == []
        memory = client.get("/agents/sessions/one/messages", headers=AGENT).json()
        assert [row["role"] for row in memory["messages"]] == ["user", "assistant"]
        assert client.post("/agents/runs/"+identifier+"/resume", headers=AGENT, json={
            "expected_revision": body["revision"], "approved": True}).status_code == 409
        assert "tenant-a" not in repr(calls)


def test_agent_api_memory_observations_are_explicit_and_tenant_scoped(settings, monkeypatch):
    monkeypatch.setattr(ModelRouter, "complete", lambda *_args, **_kw:
                        ModelReply(content="Done", model="openai/mock", finish_reason="stop"))
    with TestClient(create_app(settings)) as client:
        made = client.post("/agents/sessions/session1/observations", headers=AGENT, json={
            "fact": "Prefer concise, technical explanations",
            "source_id": "explicit-review", "confidence": .95})
        assert made.status_code == 201, made.text
        digest = made.json()["observation_id"]
        assert len(digest) == 64
        mine = client.get("/agents/sessions/session1/observations", headers=AGENT)
        assert mine.json()["observations"][0]["fact"] == "Prefer concise, technical explanations"
        assert client.get("/agents/sessions/session1/observations", headers=FOREIGN).json() == {
            "observations": []}
        assert client.post("/agents/sessions/session1/observations", headers=READER, json={
            "fact": "Not approved", "source_id": "unknown"}).status_code == 403
        invalid = client.post("/agents/sessions/session1/observations", headers=AGENT, json={
            "fact": "Hello", "source_id": "../../bad"})
        assert invalid.status_code == 422


def test_agent_api_malformed_or_unapproved_models_do_not_call_gateway(settings, monkeypatch):
    monkeypatch.setattr(ModelRouter, "complete", lambda *_a, **_kw:
                        pytest.fail("A forbidden model started a gateway request"))
    with TestClient(create_app(settings)) as client:
        response = client.post("/agents/runs", headers=AGENT, json={
            "goal": "Please analyze this", "model": "unlisted", "session_id": "sess"})
        assert response.status_code == 403
        assert client.post("/agents/runs", headers=AGENT, json={
            "goal": "valid", "model": "main", "session_id": "../../bad"}).status_code == 422


def test_agent_api_routes_are_unmounted_when_disabled(settings, monkeypatch):
    monkeypatch.setenv("RAGS_AGENT_ENABLED", "false")
    monkeypatch.setenv("RAGS_ROUTER_ENABLED", "false")
    with TestClient(create_app(settings)) as client:
        assert client.get("/agents/capabilities", headers=AGENT).status_code == 404
        assert client.post("/agents/runs", headers=AGENT, json={
            "goal": "Try to start", "model": "main", "session_id": "one"}).status_code == 404
        assert client.get("/live").status_code == 200


def test_agent_api_feature_flags_are_independent_of_identity(settings, monkeypatch):
    from rags.features import FeatureFlags
    limited = replace(settings, features=FeatureFlags((("agents", False),)))
    monkeypatch.setattr(ModelRouter, "complete", lambda *_a, **_kw:
                        pytest.fail("Disabled agent should never call model"))
    with TestClient(create_app(limited)) as client:
        assert client.get("/agents/capabilities", headers=AGENT).status_code == 403
        assert client.post("/agents/runs", headers=AGENT, json={
            "goal": "Find the policy", "model": "main", "session_id": "sess"}).status_code == 403


def test_default_on_is_discoverable_but_unconfigured_gateway_never_calls_network(tmp_path, monkeypatch):
    for name in list(os.environ):
        if name.startswith(("RAGS_ROUTER_", "RAGS_AGENT_", "RAGS_MEMORY_")):
            monkeypatch.delenv(name, raising=False)
    from rags.agents.config import AgentPlatformConfig
    cfg = AgentPlatformConfig.from_env()
    assert cfg.agent_enabled and cfg.router_enabled and cfg.memory_enabled
    assert not cfg.router_ready
    settings = replace(Settings.from_env(), db_path=str(tmp_path / "unconfigured.db"),
                       keys={"admin-token": "tenant"}, key_roles={"admin-token": ("admin",)})
    client = TestClient(create_app(settings))
    try:
        headers = {"Authorization": "Bearer admin-token"}
        capabilities = client.get("/agents/capabilities", headers=headers)
        assert capabilities.status_code == 200
        assert capabilities.json()["configuration_required"] is True
        response = client.post("/agents/runs", headers=headers,
                               json={"goal": "Please find details", "model": "main", "session_id": "sess"})
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "router_unconfigured"
    finally:
        client.close()
