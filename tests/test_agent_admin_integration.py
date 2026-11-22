"""Secure admin BFF integration for agent run/approval and tenant switching."""
from dataclasses import replace
import os

from fastapi.testclient import TestClient
import pytest

from rags.api import create_app
from rags.agents.router import ModelRouter, ModelReply, ToolCall
from rags.config import Settings

pytestmark = pytest.mark.integration
ORIGIN = "https://testserver"


@pytest.fixture
def settings(tmp_path, monkeypatch):
    for name in tuple(os.environ):
        if name.startswith(("RAGS_", "OPENAI_", "ANTHROPIC_")):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RAGS_ENV", "test")
    for key, value in {
        "RAGS_ROUTER_ENABLED": "true",
        "RAGS_AGENT_ENABLED": "true",
        "RAGS_MEMORY_ENABLED": "true",
        "RAGS_ROUTER_GATEWAY_URL": "https://gateway.example/v1",
        "RAGS_ROUTER_API_KEY": "private-router-token",
        "RAGS_ROUTER_MODELS_JSON": '{"main":"openai/mock"}',
    }.items():
        monkeypatch.setenv(key, value)
    return replace(Settings.from_env(), db_path=str(tmp_path / "admin-agents.db"),
                   admin_enabled=True,
                   keys={"admin-a": "tenant-a", "admin-b": "tenant-b",
                         "reader": "tenant-a"},
                   key_roles={"admin-a": ("admin",), "admin-b": ("admin",),
                              "reader": ("reader",)})


def login(client, key):
    response = client.post("/admin/api/session",
                           headers={"Authorization": "Bearer " + key, "Origin": ORIGIN})
    assert response.status_code == 200, response.text
    return client.get("/admin/api/session").json()["csrf"]


def test_agent_admin_requires_csrf_and_correct_scopes(settings, monkeypatch):
    monkeypatch.setattr(ModelRouter, "complete", lambda *_a, **_k:
                        ModelReply(content="Done safely", finish_reason="stop", model="openai/mock"))
    with TestClient(create_app(settings), base_url=ORIGIN) as client:
        assert client.get("/admin/api/agents/capabilities").status_code == 401
        assert client.post("/admin/api/session", headers={
            "Authorization": "Bearer reader", "Origin": ORIGIN}).status_code == 403
        token = login(client, "admin-a")
        cap = client.get("/admin/api/agents/capabilities")
        assert cap.status_code == 200 and cap.json()["models"] == ["main"]
        assert client.post("/admin/api/agents/runs", json={
            "goal": "Find the policy", "model": "main", "session_id": "one"}).status_code == 403
        assert client.post("/admin/api/agents/runs", headers={
            "Origin": "https://attacker.example", "X-CSRF-Token": token}, json={
            "goal": "Find the policy", "model": "main", "session_id": "one"}).status_code == 403
        created = client.post("/admin/api/agents/runs", headers={
            "Origin": ORIGIN, "X-CSRF-Token": token}, json={
            "goal": "Find the policy", "model": "main", "session_id": "one"})
        assert created.status_code == 201, created.text
        run_id = created.json()["run_id"]
        assert client.get("/admin/api/agents/runs/" + run_id).json()["answer"] == "Done safely"
        assert client.get("/agents/runs/" + run_id).status_code == 401


def test_agent_admin_human_approval_waits_without_side_effect(settings, monkeypatch):
    responses = iter([
        ModelReply(content="", finish_reason="tool_calls", model="openai/mock",
                   tool_calls=[ToolCall(id="approval_1", name="remember_observation",
                                        arguments={"fact": "Prefer short answers",
                                                   "source_id": "explicit-consent"})]),
        ModelReply(content="Observation saved after approval", finish_reason="stop", model="openai/mock"),
    ])
    monkeypatch.setattr(ModelRouter, "complete", lambda *_a, **_k: next(responses))
    with TestClient(create_app(settings), base_url=ORIGIN) as client:
        csrf = login(client, "admin-a")
        headers = {"Origin": ORIGIN, "X-CSRF-Token": csrf}
        started = client.post("/admin/api/agents/runs", headers=headers, json={
            "goal": "Record only after confirmation", "model": "main", "session_id": "session"})
        assert started.status_code == 201, started.text
        pending = started.json()
        assert pending["status"] == "waiting_for_approval"
        assert pending["pending_approval"]["name"] == "remember_observation"
        before = client.get("/agents/sessions/session/observations",
                            headers={"Authorization": "Bearer admin-a"})
        assert before.status_code == 200 and before.json()["observations"] == []
        assert client.post("/admin/api/agents/runs/" + pending["run_id"] + "/resume",
                           json={"expected_revision": pending["revision"], "approved": True}).status_code == 403
        approved = client.post("/admin/api/agents/runs/" + pending["run_id"] + "/resume",
                               headers=headers, json={"expected_revision": pending["revision"],
                                                      "approved": True})
        assert approved.status_code == 200 and approved.json()["status"] == "completed"
        after = client.get("/agents/sessions/session/observations",
                           headers={"Authorization": "Bearer admin-a"})
        assert after.json()["observations"][0]["fact"] == "Prefer short answers"
        assert client.post("/admin/api/agents/runs/" + pending["run_id"] + "/resume",
                           headers=headers, json={"expected_revision": pending["revision"],
                                                  "approved": True}).status_code == 409


def test_agent_admin_sessions_do_not_cross_tenant_run_boundary(settings, monkeypatch):
    monkeypatch.setattr(ModelRouter, "complete", lambda *_a, **_kw:
                        ModelReply(content="Tenant private answer", finish_reason="stop", model="openai/mock"))
    app = create_app(settings)
    with TestClient(app, base_url=ORIGIN) as first, TestClient(app, base_url=ORIGIN) as second:
        one = login(first, "admin-a")
        response = first.post("/admin/api/agents/runs",
                              headers={"Origin": ORIGIN, "X-CSRF-Token": one},
                              json={"goal": "Summarize private tenant", "model": "main",
                                    "session_id": "session"})
        run_id = response.json()["run_id"]
        login(second, "admin-b")
        assert second.get("/admin/api/agents/runs/" + run_id).status_code == 404
        assert first.get("/admin/api/agents/runs/" + run_id).status_code == 200
        static = first.get("/admin/")
        assert static.status_code == 200
        assert 'id="view-agents"' in static.text and 'id="agent-approval"' in static.text
        script = first.get("/admin/assets/app.js")
        assert "function resetTenantUI" in script.text
        assert "agent-pending-args" in script.text
        assert "eval(" not in script.text and "innerHTML" not in script.text
