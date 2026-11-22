"""Real ASGI request authorization: no role, tenant or token scope bypass."""
from dataclasses import replace
import os
import time

from fastapi.testclient import TestClient
import httpx
import pytest

from rags import auth
from rags.api import create_app
from rags.config import Settings
from rags.features import FeatureFlags

pytestmark = pytest.mark.integration


@pytest.fixture
def base_settings(tmp_path, monkeypatch):
    for name in tuple(os.environ):
        if name.startswith(("RAGS_", "OPENAI_", "ANTHROPIC_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("RAGS_ENV", "test")
    return replace(Settings.from_env(), db_path=str(tmp_path/"auth-integration.db"))


def headers(token: str, **extra) -> dict:
    return {"Authorization": "Bearer " + token, **extra}


def add(client, token="admin-a", doc_id="shared-id"):
    return client.post("/documents", headers=headers(token),
                       json={"title": "Private", "document_id": doc_id,
                             "text": "Secret alpha policy retained for seven days."})


def test_static_roles_prevent_writes_deletes_reindex_originals_and_mcp(base_settings):
    cfg = replace(base_settings,
                  keys={"reader-a": "tenant-a", "editor-a": "tenant-a", "admin-a": "tenant-a"},
                  key_roles={"reader-a": ("reader",), "editor-a": ("editor",), "admin-a": ("admin",)})
    with TestClient(create_app(cfg)) as client:
        assert add(client).status_code == 201
        assert client.get("/documents/shared-id", headers=headers("reader-a")).status_code == 200
        assert client.post("/query", headers=headers("reader-a"),
                           json={"question": "retention", "mode": "sparse", "generate": False}).status_code == 200
        for verb, url, body in (
            ("POST", "/documents", {"title": "Invalid", "text": "No permission."}),
            ("DELETE", "/documents/shared-id", None),
            ("POST", "/documents/shared-id/reindex", {"expected_revision": "bad"}),
            ("POST", "/documents/upload", None),
            ("POST", "/images", None),
            ("POST", "/assets/images", None),
            ("DELETE", "/assets/images/missing", None),
            ("POST", "/assets/images/missing/reindex", None),
        ):
            kwargs = {"json": body} if body is not None else {}
            reply = client.request(verb, url, headers=headers("reader-a"), **kwargs)
            assert reply.status_code == 403, (verb, url, reply.text)
            assert reply.json()["detail"]["code"] == "insufficient_scope"
        assert client.get("/assets/images/missing", headers=headers("reader-a")).status_code == 403
        assert client.get("/assets/images/missing", headers=headers("admin-a")).status_code == 404
        assert client.get("/capabilities", headers=headers("reader-a")).json()["features"]["ingestion"] is False
        assert client.get("/capabilities", headers=headers("reader-a")).json()["features"]["model_generation"] is False
        assert client.get("/documents", headers=headers("reader-a", **{"X-RAGS-Interface": "mcp"})).status_code == 403
        assert client.get("/documents", headers=headers("admin-a", **{"X-RAGS-Interface": "mcp"})).status_code == 200
        assert client.post("/documents", headers=headers("editor-a"),
                           json={"title": "Editor source", "text": "valid"}).status_code == 201
        assert client.delete("/documents/shared-id", headers=headers("editor-a")).status_code == 403
        assert client.delete("/documents/shared-id", headers=headers("admin-a")).status_code == 200
        assert client.get("/documents/shared-id", headers=headers("admin-a")).status_code == 404


def test_oauth2_tenant_claims_enforce_data_isolation_and_authentication(base_settings, monkeypatch):
    raw = {
        "admin-a": ("tenant-a", "rags:read rags:query rags:write rags:delete rags:reindex rags:generate rags:mcp"),
        "reader-a": ("tenant-a", "rags:read rags:query"),
        "reader-b": ("tenant-b", "rags:read rags:query"),
        "no-scope": ("tenant-b", ""),
        "revoked": ("tenant-a", ""),
    }
    calls = []
    def handler(request: httpx.Request) -> httpx.Response:
        from urllib.parse import parse_qs
        form = parse_qs(request.content.decode("utf-8"))
        token = form["token"][0]
        calls.append(token)
        if token == "revoked":
            return httpx.Response(200, json={"active": False})
        if token not in raw:
            return httpx.Response(200, json={"active": False})
        tenant, scope = raw[token]
        return httpx.Response(200, json={
            "active": True, "iss": "https://idp.example/", "aud": "rags-api",
            "tenant_id": tenant, "sub": token, "exp": time.time()+60, "scope": scope,
        })
    real_client = httpx.Client
    monkeypatch.setattr(auth, "_client", lambda _config: real_client(transport=httpx.MockTransport(handler)))
    cfg = replace(base_settings, auth_mode="oauth2_introspection", keys={}, key_roles={},
                  oauth2=auth.OAuth2Config(
                      "https://idp.example/introspect", "https://idp.example/",
                      "rags-api", "resource-app", "test-secret"))
    with TestClient(create_app(cfg)) as client:
        assert add(client, "admin-a").status_code == 201
        assert client.get("/documents/shared-id", headers=headers("reader-a")).status_code == 200
        assert client.get("/documents/shared-id", headers=headers("reader-b")).status_code == 404
        assert client.get("/documents", headers=headers("reader-b", **{"X-RAGS-Tenant": "tenant-a"})).json() == []
        assert client.post("/query", headers=headers("reader-b"),
                           json={"question": "Secret alpha policy", "mode": "sparse", "generate": False}).json()["citations"] == []
        assert client.post("/documents", headers=headers("reader-b"),
                           json={"title": "Forbidden", "text": "Private"}).status_code == 403
        assert client.get("/documents", headers=headers("no-scope")).status_code == 403
        assert client.get("/documents", headers=headers("revoked")).status_code == 401
        assert client.get("/documents").status_code == 401
        assert client.get("/documents", headers=headers("any-static-key")).status_code == 401
    assert "admin-a" in calls and "reader-b" in calls and "revoked" in calls


def test_oauth2_model_invocation_requires_separate_scope(base_settings, monkeypatch):
    # Authorization precedes a provider call even when generation is enabled.
    cfg = replace(base_settings, model_provider="openai", openai_key="private-model-key",
                  keys={"reader": "tenant"}, key_roles={"reader": ("reader",)})
    app = create_app(cfg)
    monkeypatch.setattr(app.state.service.provider, "json_task",
                        lambda *_a, **_kw: pytest.fail("Unauthorized generation invoked provider"))
    with TestClient(app) as client:
        result = client.post("/query", headers=headers("reader"),
                             json={"question": "retention", "mode": "sparse", "generate": True})
        assert result.status_code == 403
        result = client.post("/query", headers=headers("reader"),
                             json={"question": "retention", "mode": "fusion", "generate": False})
        assert result.status_code == 403
        result = client.post("/query", headers=headers("reader"),
                             json={"question": "retention", "mode": "sparse", "generate": False})
        assert result.status_code == 200


def test_global_feature_flag_remains_additional_requirement(base_settings):
    cfg = replace(base_settings, keys={"reader": "tenant"}, key_roles={"reader": ("reader",)},
                  features=FeatureFlags((("query", False),)))
    with TestClient(create_app(cfg)) as client:
        r = client.post("/query", headers=headers("reader"), json={
            "question": "anything?", "mode": "sparse", "generate": False
        })
        assert r.status_code == 403 and r.json()["detail"]["code"] == "feature_disabled"
