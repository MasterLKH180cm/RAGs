"""Admin BFF integration: cookie, CSRF, scopes, tenant isolation and real actions."""
from dataclasses import replace
import json
import os

from fastapi.testclient import TestClient
import pytest

from rags.admin import SessionStore
from rags.api import create_app
from rags.auth import Principal
from rags.config import Settings

pytestmark = pytest.mark.integration
ORIGIN = "https://testserver"
ADMIN_A = {"Authorization": "Bearer admin-a", "Origin": ORIGIN}
ADMIN_B = {"Authorization": "Bearer admin-b", "Origin": ORIGIN}
READER = {"Authorization": "Bearer reader-a", "Origin": ORIGIN}


@pytest.fixture
def config(tmp_path, monkeypatch):
    for name in tuple(os.environ):
        if name.startswith(("RAGS_", "OPENAI_", "ANTHROPIC_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("RAGS_ENV", "test")
    return replace(Settings.from_env(), db_path=str(tmp_path / "admin.db"),
                   keys={"admin-a": "tenant-a", "reader-a": "tenant-a", "admin-b": "tenant-b"},
                   key_roles={"admin-a": ("admin",), "reader-a": ("reader",), "admin-b": ("admin",)},
                   admin_enabled=True)


def sign_in(client, headers=ADMIN_A):
    result = client.post("/admin/api/session", headers=headers)
    return result


def csrf(client):
    response = client.get("/admin/api/session")
    assert response.status_code == 200, response.text
    return response.json()["csrf"]


def secured(csrf_value, *, origin=ORIGIN, extra=None):
    return {"Origin": origin, "X-CSRF-Token": csrf_value, **(extra or {})}


def test_optin_admin_not_mounted_by_default(config):
    app = create_app(replace(config, admin_enabled=False))
    with TestClient(app, base_url=ORIGIN) as client:
        assert client.get("/admin/").status_code == 404
        assert client.get("/admin/api/overview").status_code == 404
        assert client.get("/live").status_code == 200


def test_static_dashboard_is_bundled_accessible_and_csp_protected(config):
    app = create_app(config)
    with TestClient(app, base_url=ORIGIN) as client:
        page = client.get("/admin/")
        assert page.status_code == 200
        assert "Content-Security-Policy" in page.headers
        assert "frame-ancestors 'none'" in page.headers["content-security-policy"]
        assert page.headers["strict-transport-security"] == "max-age=31536000"
        assert page.headers["cache-control"] == "no-store"
        assert "管理知識" in page.text
        assert "<html lang=\"zh-Hant\">" in page.text
        for target in ("overview", "documents", "explorer", "observability", "settings"):
            assert 'id="view-' + target + '"' in page.text
        for element in ("login-form", "query-form", "add-dialog", "delete-dialog", "global-notice"):
            assert 'id="' + element + '"' in page.text
        for name in ("style.css", "app.js"):
            response = client.get("/admin/assets/" + name)
            assert response.status_code == 200
            assert response.headers["X-Content-Type-Options"] == "nosniff"
            assert not any(origin in response.text for origin in (
                "cdn.jsdelivr.net", "unpkg.com", "fonts.googleapis.com",
            ))
        assert client.get("/admin/assets/../auth.py").status_code != 200
        assert client.get("/admin/assets/missing.js").status_code == 404
        js = client.get("/admin/assets/app.js").text
        assert "innerHTML" not in js and "eval(" not in js
        assert "localStorage.setItem('rags-admin-theme'" in js
        assert "localStorage.setItem('rags-admin-token'" not in js
        css = client.get("/admin/assets/style.css").text
        assert "@media(max-width:650px)" in css
        assert "prefers-reduced-motion" in css
        assert "focus-visible" in css
        assert "Mobile touch targets follow 44px" in css


def test_login_requires_origin_https_and_admin_scope(config):
    app = create_app(config)
    with TestClient(app, base_url=ORIGIN) as client:
        assert client.post("/admin/api/session", headers={"Authorization": "Bearer admin-a"}).status_code == 403
        assert client.post("/admin/api/session", headers={"Authorization": "Bearer admin-a",
                                                        "Origin": "https://attacker.invalid"}).status_code == 403
        assert sign_in(client, READER).status_code == 403
        assert client.get("/admin/api/overview").status_code == 401
        login = sign_in(client)
        assert login.status_code == 200
        cookies = login.headers["set-cookie"]
        assert "__Host-rags_admin=" in cookies and "httponly" in cookies.lower()
        assert "samesite=strict" in cookies.lower() and "secure" in cookies.lower()
        assert "admin-a" not in login.text and "admin-a" not in cookies
        session = client.get("/admin/api/session")
        assert session.status_code == 200
        data = session.json()
        assert data["tenant"] == "tenant-a" and data["csrf"]
        assert "rags:admin" in data["permissions"]
        assert "admin-a" not in json.dumps(data)
        assert client.get("/documents").status_code == 401  # No cookie as Bearer fallback.


def test_unsafe_http_requires_explicit_local_dev_mode(config):
    with TestClient(create_app(config), base_url="http://testserver") as client:
        assert sign_in(client, {"Authorization": "Bearer admin-a", "Origin": "http://testserver"}).status_code == 403
    with TestClient(create_app(replace(config, admin_dev_http=True)),
                    base_url="http://testserver") as client:
        result = sign_in(client, {"Authorization": "Bearer admin-a", "Origin": "http://testserver"})
        assert result.status_code == 200
        assert "rags_admin_dev=" in result.headers["set-cookie"]
        assert "secure" not in result.headers["set-cookie"].lower()
        assert client.get("/admin/api/session").status_code == 200


def test_sessions_expire_at_absolute_deadline_and_capacity_is_bounded():
    now = [0.]
    store = SessionStore(clock=lambda: now[0], capacity=1)
    principal = Principal("tenant-a", "alice", frozenset(("rags:admin", "rags:read")), "api_key")
    cookie, session = store.create("private-token", principal)
    assert "private-token" not in repr(session)
    assert store.get(cookie).tenant == "tenant-a"
    with pytest.raises(Exception) as full:
        store.create("another", principal)
    assert full.value.status_code == 429
    now[0] = 1800.
    with pytest.raises(Exception) as expired:
        store.get(cookie)
    assert expired.value.status_code == 401
    new_cookie, _ = store.create("another", principal)
    store.revoke(new_cookie)
    with pytest.raises(Exception):
        store.get(new_cookie)


def test_admin_document_management_is_tenant_scoped_and_cas_protected(config):
    with TestClient(create_app(config), base_url=ORIGIN) as client:
        assert sign_in(client).status_code == 200
        nonce = csrf(client)
        payload = {"title": "Retention policies", "text": "Retention is 30 days. Version 1.",
                   "source": "spec", "document_id": "document-a"}
        assert client.post("/admin/api/documents", json=payload).status_code == 403  # Missing CSRF.
        assert client.post("/admin/api/documents", json=payload,
                           headers=secured(nonce, origin="https://evil.invalid")).status_code == 403
        made = client.post("/admin/api/documents", json=payload, headers=secured(nonce))
        assert made.status_code == 201, made.text
        revision = made.json()["revision"]
        overview = client.get("/admin/api/overview")
        assert overview.status_code == 200
        assert overview.json()["counts"]["documents"] == 1
        assert overview.json()["counts"]["chunks"] >= 1
        assert overview.json()["counts"]["revisions"] == 1
        rows = client.get("/admin/api/documents?search=Retention&limit=5").json()
        assert rows["total"] == 1 and rows["items"][0]["revision"] == revision
        assert "text" not in rows["items"][0] and "body" not in rows["items"][0]
        # A literal LIKE wildcard cannot match arbitrary document names.
        assert client.get("/admin/api/documents?search=%25").json()["total"] == 0
        result = client.post("/admin/api/query", headers=secured(nonce),
                             json={"question": "Retention period", "mode": "sparse", "generate": False})
        assert result.status_code == 200 and result.json()["citations"]
        assert client.delete("/admin/api/documents/document-a", headers=secured(nonce)).status_code == 422
        mismatch = client.delete("/admin/api/documents/document-a", headers=secured(
            nonce, extra={"If-Match": "wrong-revision"}))
        assert mismatch.status_code == 409 and mismatch.json()["detail"]["code"] == "revision_conflict"
        assert client.get("/admin/api/documents").json()["total"] == 1
        removed = client.delete("/admin/api/documents/document-a", headers=secured(
            nonce, extra={"If-Match": revision}))
        assert removed.status_code == 200
        assert client.get("/admin/api/documents").json()["total"] == 0
        assert client.get("/admin/api/overview").json()["counts"]["documents"] == 0


def test_admin_session_isolation_across_users_tenants_and_cookie_context(config):
    app = create_app(config)
    with TestClient(app, base_url=ORIGIN) as first, TestClient(app, base_url=ORIGIN) as second:
        assert sign_in(first, ADMIN_A).status_code == 200
        a = csrf(first)
        assert first.post("/admin/api/documents", json={
            "title": "Private tenant A", "text": "private details", "document_id": "shared",
        }, headers=secured(a)).status_code == 201
        assert sign_in(second, ADMIN_B).status_code == 200
        b = csrf(second)
        assert second.get("/admin/api/overview").json()["counts"]["documents"] == 0
        assert second.get("/admin/api/documents").json()["items"] == []
        response = second.delete("/admin/api/documents/shared", headers=secured(
            b, extra={"If-Match": "arbitrary"}))
        assert response.status_code == 404
        assert second.post("/admin/api/query", headers=secured(b),
                           json={"question": "private details", "mode": "sparse",
                                 "generate": False}).json()["citations"] == []
        assert first.get("/admin/api/overview").json()["counts"]["documents"] == 1
        assert first.get("/admin/api/documents").json()["items"][0]["id"] == "shared"
        assert first.delete("/admin/api/session", headers=secured(a)).status_code == 204
        assert first.get("/admin/api/session").status_code == 401
        assert second.get("/admin/api/session").status_code == 200


def test_admin_permissions_still_require_original_operation_feature_flags(config):
    from rags.features import FeatureFlags
    flags = FeatureFlags((("deletion", False), ("query", False)))
    with TestClient(create_app(replace(config, features=flags)), base_url=ORIGIN) as client:
        assert sign_in(client).status_code == 200
        token = csrf(client)
        assert client.post("/admin/api/query", headers=secured(token),
                           json={"question": "what is policy?", "generate": False}).status_code == 403
        assert client.delete("/admin/api/documents/any", headers=secured(
            token, extra={"If-Match": "revision"})).status_code == 403
        ops = client.get("/admin/api/operations")
        assert ops.status_code == 200
        assert ops.json()["features"]["deletion"] is False
        assert ops.json()["features"]["query"] is False
        assert "private" not in ops.text.lower()


def test_dashboard_requires_explicit_admin_scope_even_for_legacy_compatible_key(config):
    custom = replace(config, keys={"editor": "tenant-a"}, key_roles={"editor": ("editor",)})
    with TestClient(create_app(custom), base_url=ORIGIN) as client:
        assert sign_in(client, {"Authorization": "Bearer editor", "Origin": ORIGIN}).status_code == 403


def test_disabling_document_reads_hides_admin_overview_and_inventory(config):
    from rags.features import FeatureFlags

    app = create_app(replace(config, features=FeatureFlags((("documents", False),))))
    with TestClient(app, base_url=ORIGIN) as client:
        assert sign_in(client).status_code == 200
        for route in ("/admin/api/overview", "/admin/api/documents"):
            response = client.get(route)
            assert response.status_code == 403
            assert response.json()["detail"]["code"] == "feature_disabled"
        operations = client.get("/admin/api/operations")
        assert operations.status_code == 200
        assert operations.json()["features"]["documents"] is False


def test_oauth2_admin_session_revalidates_revocation_on_every_request(config, monkeypatch):
    """The BFF cannot turn an expired/revoked OAuth access token into a durable cookie."""
    from rags import auth
    import httpx
    import time

    live = {"active": True}
    requests = []

    def introspect(request):
        requests.append(request)
        assert request.method == "POST"
        assert request.url == "https://idp.example/introspect"
        assert request.headers["authorization"].startswith("Basic ")
        return httpx.Response(200, json={
            "active": live["active"], "iss": "https://idp.example/",
            "aud": "rags-api", "sub": "operator-1", "tenant_id": "tenant-a",
            "exp": time.time() + 120, "scope": "rags:admin rags:read rags:query",
        })

    actual_client = httpx.Client
    monkeypatch.setattr(auth, "_client",
                        lambda _cfg: actual_client(transport=httpx.MockTransport(introspect)))
    oauth = auth.OAuth2Config("https://idp.example/introspect", "https://idp.example/",
                              "rags-api", "rags-dashboard", "test-secret")
    configured = replace(config, auth_mode="oauth2_introspection",
                         keys={}, key_roles={}, oauth2=oauth)
    with TestClient(create_app(configured), base_url=ORIGIN) as client:
        login = sign_in(client, {"Authorization": "Bearer opaque-credential",
                                 "Origin": ORIGIN})
        assert login.status_code == 200, login.text
        assert "opaque-credential" not in login.text
        assert client.get("/admin/api/overview").status_code == 200
        before = len(requests)
        assert before >= 2
        live["active"] = False
        assert client.get("/admin/api/overview").status_code == 401
        assert client.get("/admin/api/session").status_code == 401
        assert len(requests) == before + 1  # Revoked session removed after first failure.


def test_light_theme_foreground_tokens_retain_well_above_aa_contrast():
    """Readable small labels on white, tinted surfaces, and light canvas."""
    import re
    from importlib import resources

    css = resources.files("rags").joinpath("admin_static", "style.css").read_text()
    light = css.split('body[data-theme="light"]{', 1)[1].split("}", 1)[0]
    swatches = dict(re.findall(r"--([a-z-]+):(#(?:[0-9a-f]{6}))", light))

    def luminance(color: str) -> float:
        rgb = [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        channels = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in rgb]
        return sum(a * b for a, b in zip(channels, (0.2126, 0.7152, 0.0722)))

    for name in ("text", "muted", "subtle", "accent"):
        foreground = luminance(swatches[name])
        for background in ("#ffffff", "#f7f9fd", "#ecf1f8"):
            back = luminance(background)
            ratio = (max(foreground, back) + .05) / (min(foreground, back) + .05)
            assert ratio >= 4.5, (name, background, ratio)


def test_dashboard_workflow_graph_lifecycle_and_csrf_scope(config):
    """The visual workflow uses the same tenant model and an exact CSRF boundary."""
    from rags.agents.config import AgentPlatformConfig
    import os

    for name in ("RAGS_ROUTER_ENABLED", "RAGS_AGENT_ENABLED", "RAGS_MEMORY_ENABLED"):
        os.environ.pop(name, None)
    app = create_app(config)
    with TestClient(app, base_url=ORIGIN) as client:
        assert sign_in(client).status_code == 200
        nonce = csrf(client)
        templates = client.get("/admin/api/agents/workflows")
        assert templates.status_code == 200
        graph = templates.json()["templates"][0]["nodes"]
        assert [node["kind"] for node in graph] == [
            "then", "branch", "parallel", "then", "then"]
        payload = {"question": "Which documents define retention policy?",
                   "mode": "sparse", "top_k": 3}
        endpoint = "/admin/api/agents/workflows/evidence_review/runs"
        assert client.post(endpoint, json=payload).status_code == 403
        assert client.post(endpoint, json=payload,
                           headers=secured(nonce, origin="https://evil.invalid")).status_code == 403
        started = client.post(endpoint, json=payload, headers=secured(nonce))
        assert started.status_code == 201, started.text
        run = started.json()
        assert run["status"] == "waiting_for_approval"
        path = "/admin/api/agents/workflows/runs/" + run["run_id"]
        assert client.get(path).status_code == 200
        assert client.post(path + "/resume",
                           json={"expected_revision": run["revision"], "approved": True}).status_code == 403
        done = client.post(path + "/resume", headers=secured(nonce),
                           json={"expected_revision": run["revision"], "approved": True})
        assert done.status_code == 200 and done.json()["status"] == "completed"
        assert client.post(path + "/resume", headers=secured(nonce),
                           json={"expected_revision": run["revision"], "approved": True}).status_code == 409


def test_admin_workflow_isolation_between_tenants_and_subjects(config):
    app = create_app(config)
    with TestClient(app, base_url=ORIGIN) as a, TestClient(app, base_url=ORIGIN) as b:
        assert sign_in(a, ADMIN_A).status_code == 200
        key_a = csrf(a)
        created = a.post("/admin/api/agents/workflows/evidence_review/runs",
                         headers=secured(key_a), json={"question": "Find the source retention policy"})
        assert created.status_code == 201, created.text
        run = created.json()
        assert sign_in(b, ADMIN_B).status_code == 200
        key_b = csrf(b)
        path = "/admin/api/agents/workflows/runs/" + run["run_id"]
        assert b.get(path).status_code == 404
        assert b.post(path + "/resume", headers=secured(key_b),
                      json={"expected_revision": run["revision"], "approved": True}).status_code == 404
        assert a.get(path).status_code == 200
