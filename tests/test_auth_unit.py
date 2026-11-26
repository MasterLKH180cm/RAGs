"""Unit contracts for tenant principals, API key roles and RFC 7662 boundaries."""
import asyncio
import json
import os
import time
from dataclasses import replace

from fastapi import HTTPException
import httpx
import pytest

from rags import auth
from rags.config import Settings

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    for name in tuple(os.environ):
        if name.startswith(("RAGS_", "OPENAI_", "ANTHROPIC_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("RAGS_ENV", "test")


def config() -> auth.OAuth2Config:
    return auth.OAuth2Config(
        "https://idp.example/oauth2/introspect", "https://idp.example/", "rags-api",
        "rags-service", "secret-do-not-leak",
    )


def claims(**changes):
    data = {"active": True, "iss": "https://idp.example/", "aud": ["rags-api"],
            "sub": "account-1", "tenant_id": "tenant-a", "exp": time.time() + 60,
            "scope": "openid profile rags:read rags:query", "token_type": "Bearer"}
    data.update(changes)
    return data


@pytest.mark.parametrize(("updates", "reason"), [
    ({"introspection_url": "http://idp.example/introspect"}, "HTTPS"),
    ({"introspection_url": "https://user:pass@idp.example"}, "HTTPS"),
    ({"introspection_url": "https://idp.example/introspect?token=secret"}, "HTTPS"),
    ({"introspection_url": "https://idp.example/introspect#fragment"}, "HTTPS"),
    ({"issuer": ""}, "issuer"),
    ({"audience": ""}, "audience"),
    ({"client_secret": ""}, "credentials"),
    ({"tenant_claim": "scope"}, "tenant claim"),
    ({"tenant_claim": "bad-path.name"}, "tenant claim"),
    ({"timeout_seconds": 0}, "timeout"),
    ({"timeout_seconds": float("nan")}, "timeout"),
    ({"timeout_seconds": 11}, "timeout"),
])
def test_oauth2_configuration_fails_closed(updates, reason):
    data = {"introspection_url": "https://idp.example/introspect", "issuer": "https://idp.example/",
            "audience": "rags-api", "client_id": "rags-app", "client_secret": "secret"}
    data.update(updates)
    with pytest.raises(ValueError, match=reason):
        auth.OAuth2Config(**data)


@pytest.mark.parametrize(("options", "message"), [
    ({"RAGS_AUTH_MODE": "custom"}, "RAGS_AUTH_MODE"),
    ({"RAGS_KEY_ROLES_JSON": '{"unmapped":["reader"]}'}, "RAGS_KEY_ROLES_JSON"),
    ({"RAGS_KEY_ROLES_JSON": '{"reader-key":["root"]}'}, "RAGS_KEY_ROLES_JSON"),
    ({"RAGS_KEY_ROLES_JSON": '{"reader-key":["reader","reader"]}'}, "RAGS_KEY_ROLES_JSON"),
    ({"RAGS_KEY_ROLES_JSON": '{"reader-key":[]}'}, "RAGS_KEY_ROLES_JSON"),
    ({"RAGS_KEY_ROLES_JSON": '{"reader-key":"admin"}'}, "RAGS_KEY_ROLES_JSON"),
    ({"RAGS_KEYS_JSON": '{"reader-key":"tenant invalid"}'}, "tenant"),
])
def test_invalid_api_key_roles_or_tenants_rejected(monkeypatch, options, message):
    monkeypatch.setenv("RAGS_KEYS_JSON", '{"reader-key":"tenant-a"}')
    for name, value in options.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=message):
        Settings.from_env()


def test_all_api_keys_require_explicit_roles_once_role_map_is_configured(monkeypatch):
    monkeypatch.setenv("RAGS_KEYS_JSON", '{"reader-key":"tenant-a","editor-key":"tenant-b"}')
    monkeypatch.setenv("RAGS_KEY_ROLES_JSON", '{"reader-key":["reader"]}')
    with pytest.raises(ValueError, match="cover all"):
        Settings.from_env()


def test_key_roles_and_legacy_compatibility(monkeypatch):
    monkeypatch.setenv("RAGS_KEYS_JSON", '{"reader-key":"tenant-a","editor-key":"tenant-a","admin-key":"tenant-b"}')
    monkeypatch.setenv(
        "RAGS_KEY_ROLES_JSON",
        '{"reader-key":["reader"],"editor-key":["editor"],"admin-key":["admin"]}',
    )
    settings = Settings.from_env()
    service = auth.Authenticator(mode=settings.auth_mode, keys=settings.keys,
                                 key_roles=settings.key_roles, oauth2=None)
    reader = service.authenticate("Bearer reader-key")
    editor = service.authenticate("Bearer editor-key")
    administrator = service.authenticate("Bearer admin-key")
    assert reader.permissions == auth.ROLES["reader"]
    assert editor.permissions == auth.ROLES["editor"]
    assert administrator.permissions == auth.PERMISSIONS
    assert reader.tenant == editor.tenant == "tenant-a" and administrator.tenant == "tenant-b"
    assert "reader-key" not in reader.subject
    with pytest.raises(HTTPException) as error:
        reader.require("rags:delete")
    assert error.value.status_code == 403
    assert error.value.headers["WWW-Authenticate"].startswith('Bearer error="insufficient_scope"')
    assert service.authenticate("bearer editor-key").subject.startswith("api-key:")
    assert all(len(p.subject) < 40 for p in (reader, editor, administrator))
    assert not settings.oauth2

    legacy = replace(settings, key_roles={})
    principal = auth.Authenticator(mode=legacy.auth_mode, keys=legacy.keys, key_roles={},
                                   oauth2=None).authenticate("Bearer reader-key")
    assert principal.permissions == auth.PERMISSIONS


@pytest.mark.parametrize("header", [None, "", "Basic secret", "Bearer", "Bearer ",
                                     "Bearer abc def", "Bearer \nsecret", "Bearer "+("x"*4097)])
def test_invalid_bearer_syntax_is_401(header):
    checker = auth.Authenticator(mode="api_key", keys={"good": "tenant-a"}, key_roles={}, oauth2=None)
    with pytest.raises(HTTPException) as error:
        checker.authenticate(header)
    assert error.value.status_code == 401 and error.value.headers["WWW-Authenticate"].startswith("Bearer")


@pytest.mark.parametrize(("path","method","allowed"), [
    ("/health", "GET", ("rags:read",)),
    ("/documents", "GET", ("rags:read",)),
    ("/documents", "POST", ("rags:write",)),
    ("/documents/doc123", "GET", ("rags:read",)),
    ("/documents/doc123/revisions", "GET", ("rags:read",)),
    ("/documents/doc123/reindex", "POST", ("rags:reindex",)),
    ("/documents/doc123", "DELETE", ("rags:delete",)),
    ("/documents/upload", "POST", ("rags:write",)),
    ("/images", "POST", ("rags:write", "rags:generate")),
    ("/query", "POST", ("rags:query",)),
    ("/assets/images", "POST", ("rags:write",)),
    ("/assets/images", "GET", ("rags:read",)),
    ("/assets/images/file", "GET", ("rags:read", "rags:original_image")),
    ("/assets/images/file/reindex", "POST", ("rags:reindex",)),
    ("/assets/images/file", "DELETE", ("rags:delete",)),
    ("/assets/search", "POST", ("rags:query",)),
    ("/assets/search/image", "POST", ("rags:query",)),
    ("/documents/other/delete", "POST", ()),
    ("/assets/malicious", "GET", ()),
    ("/documents", "DELETE", ()),
])
def test_all_routes_explicitly_map_to_privileges(path, method, allowed):
    assert auth.route_permissions(path, method) == allowed


def test_oauth2_mode_does_not_fallback_to_static_secrets(monkeypatch):
    values = {"RAGS_AUTH_MODE": "oauth2_introspection",
              "RAGS_OAUTH2_INTROSPECTION_URL": "https://idp.example/oauth/introspect",
              "RAGS_OAUTH2_ISSUER": "https://idp.example/",
              "RAGS_OAUTH2_AUDIENCE": "rags-api",
              "RAGS_OAUTH2_CLIENT_ID": "resource-server",
              "RAGS_OAUTH2_CLIENT_SECRET": "secret"}
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    config = Settings.from_env()
    assert config.auth_mode == "oauth2_introspection" and config.keys == {}
    assert config.oauth2.client_id == "resource-server"
    monkeypatch.setenv("RAGS_KEYS_JSON", '{"legacy-key":"tenant-a"}')
    with pytest.raises(ValueError, match="cannot use static"):
        Settings.from_env()


@pytest.mark.parametrize("changes", [
    {"active": False}, {"active": 1}, {"iss": "https://evil.invalid/"}, {"aud": "another-api"},
    {"aud": []}, {"aud": 12}, {"aud": ["rags-api", 1]}, {"exp": 0},
    {"exp": True}, {"exp": float("nan")}, {"nbf": time.time()+300},
    {"exp": 10**400}, {"nbf": 10**400},
    {"nbf": "not-a-time"}, {"sub": ""}, {"sub": 42}, {"tenant_id": "tenant with spaces"},
    {"tenant_id": ""}, {"tenant_id": None}, {"tenant_id": "../../parent"},
    {"scope": ["rags:read"]}, {"scope": 42}, {"scope": "rags:read\nrags:delete"},
    {"token_type": "refresh_token"}, {"cnf": {"jkt": "token-proof"}},
])
def test_claim_validation_rejects_untrusted_or_wrongly_audienced_tokens(changes):
    with pytest.raises(HTTPException) as error:
        auth._validate_introspection(claims(**changes), config(), time.time())
    assert error.value.status_code == 401


def test_rfc7662_introspection_has_basic_auth_form_post_and_no_cross_tenant_user_header(monkeypatch):
    received = []
    def provider(request):
        received.append(request)
        assert request.method == "POST"
        assert request.url == "https://idp.example/oauth2/introspect"
        assert request.headers["authorization"].startswith("Basic ")
        assert b"token=opaque-token" in request.content
        assert b"token_type_hint=access_token" in request.content
        return httpx.Response(200, json=claims())

    real_client = httpx.AsyncClient
    monkeypatch.setattr(auth, "_client", lambda _cfg: real_client(transport=httpx.MockTransport(provider)))
    principal = auth.Authenticator(mode="oauth2_introspection", keys={}, key_roles={},
                                   oauth2=config()).authenticate("Bearer opaque-token")
    assert principal.tenant == "tenant-a" and principal.subject == "account-1"
    assert principal.permissions == frozenset(("rags:read", "rags:query"))
    assert len(received) == 1
    with pytest.raises(HTTPException) as error:
        principal.require("rags:write")
    assert error.value.status_code == 403


@pytest.mark.parametrize("failure", [
    lambda: httpx.Response(302, headers={"Location": "https://untrusted.invalid"}),
    lambda: httpx.Response(503, text="private-provider-error-secret"),
    lambda: httpx.Response(200, content=b"x"*17_000),
    lambda: httpx.Response(200, content=b"<not-json>"),
])
def test_provider_failures_fail_closed_without_leaking_secret(monkeypatch, failure):
    real_client = httpx.AsyncClient
    monkeypatch.setattr(auth, "_client",
                        lambda _cfg: real_client(transport=httpx.MockTransport(lambda _: failure())))
    with pytest.raises(HTTPException) as error:
        auth._introspect("opaque-private-token", config())
    assert error.value.status_code == 503
    assert "private" not in json.dumps(error.value.detail)
    assert "idp.example" not in str(error.value.detail)


@pytest.mark.parametrize("body", [
    b'{"active":false,"active":true}',
    b'{"active":false,"unused":NaN}',
    b'{"active":false,"unused":1e999}',
    b"\xff",
])
def test_ambiguous_or_unparseable_claims_are_sanitized_unavailable(monkeypatch, body):
    real_client = httpx.AsyncClient
    monkeypatch.setattr(auth, "_client", lambda _cfg: real_client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=body)),
    ))
    with pytest.raises(HTTPException) as error:
        auth._introspect("private-token", config())
    assert error.value.status_code == 503
    assert error.value.detail == {"code": "identity_provider_unavailable"}


def test_nested_nonobject_claims_fail_closed_across_json_decoders(monkeypatch):
    # Some supported Python decoders accept deeply nested valid JSON; others
    # reject it with RecursionError. Neither may turn it into a principal or 500.
    body = b"[" * 7000 + b"0" + b"]" * 7000
    real_client = httpx.AsyncClient
    monkeypatch.setattr(auth, "_client", lambda _cfg: real_client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=body)),
    ))
    with pytest.raises(HTTPException) as error:
        auth._introspect("private-token", config())
    assert error.value.status_code in {401, 503}
    assert error.value.detail["code"] in {"invalid_token", "identity_provider_unavailable"}


@pytest.mark.parametrize("response_headers", [
    {"Content-Encoding": "gzip"},
    {"Content-Length": str(auth.MAX_INTROSPECTION_BYTES + 1)},
    {"Content-Length": "not-an-integer"},
])
def test_introspection_rejects_unsafe_envelopes_before_reading(monkeypatch, response_headers):
    class Unreadable(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            pytest.fail("Rejected envelope must never read or decompress the body")
            yield b""

        async def aclose(self):
            self.closed = True

    stream = Unreadable()
    clients = []
    real_client = httpx.AsyncClient

    def factory(_cfg):
        client = real_client(transport=httpx.MockTransport(lambda _: httpx.Response(
            200, headers=response_headers, stream=stream,
        )))
        clients.append(client)
        return client

    monkeypatch.setattr(auth, "_client", factory)
    with pytest.raises(HTTPException) as error:
        auth._introspect("private-token", config())
    assert error.value.status_code == 503
    assert stream.closed and clients[0].is_closed


@pytest.mark.parametrize("phase", ["headers", "slow_body"])
def test_introspection_wall_deadline_cancels_and_closes_owned_request(monkeypatch, phase):
    class SlowClaims(httpx.AsyncByteStream):
        closed = False
        reads = 0

        async def __aiter__(self):
            for _ in range(100):
                await asyncio.sleep(0.02)
                self.reads += 1
                yield b" "
            yield json.dumps(claims()).encode()

        async def aclose(self):
            self.closed = True

    stream = SlowClaims()
    clients, cancelled = [], []
    real_client = httpx.AsyncClient

    async def provider(request):
        if phase == "headers":
            try:
                await asyncio.sleep(10)
            finally:
                cancelled.append(True)
        return httpx.Response(200, stream=stream)

    def factory(_cfg):
        client = real_client(transport=httpx.MockTransport(provider))
        clients.append(client)
        return client

    monkeypatch.setattr(auth, "_client", factory)
    started = time.monotonic()
    with pytest.raises(HTTPException) as error:
        auth._introspect("private-token", replace(config(), timeout_seconds=1))
    assert time.monotonic() - started < 2.5
    assert error.value.status_code == 503
    assert error.value.detail == {"code": "identity_provider_unavailable"}
    assert clients[0].is_closed
    if phase == "headers":
        assert cancelled == [True]
    else:
        assert stream.closed and 0 < stream.reads < 100
