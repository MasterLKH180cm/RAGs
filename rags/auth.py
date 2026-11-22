"""Tenant-scoped principals, least-privilege roles, and RFC 7662 resource auth.

The application is an OAuth2 resource server, *not* a token issuer, OIDC
client, or JWT decoder. Token metadata is accepted only from the configured,
authenticated HTTPS introspection endpoint.
"""
from dataclasses import dataclass, field
import hashlib
import hmac
import json
import math
import re
import time
from urllib.parse import urlsplit

import httpx
from fastapi import HTTPException


PERMISSIONS = frozenset({
    "rags:read", "rags:query", "rags:write", "rags:reindex",
    "rags:delete", "rags:generate", "rags:original_image", "rags:mcp", "rags:admin",
})
ROLES = {
    "reader": frozenset(("rags:read", "rags:query")),
    "editor": frozenset(("rags:read", "rags:query", "rags:write", "rags:reindex")),
    "admin": PERMISSIONS,
}
TENANT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
CLAIM = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}\Z")
MAX_INTROSPECTION_BYTES = 16_384
MAX_TOKEN_LENGTH = 4_096


def _invalid_token() -> HTTPException:
    return HTTPException(
        status_code=401,
        detail={"code": "invalid_token"},
        headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
    )


def _missing_bearer() -> HTTPException:
    return HTTPException(401, detail={"code": "authentication_required"},
                         headers={"WWW-Authenticate": 'Bearer realm="rags"'})


def _unavailable() -> HTTPException:
    return HTTPException(503, detail={"code": "identity_provider_unavailable"})


@dataclass(frozen=True)
class Principal:
    tenant: str
    subject: str
    permissions: frozenset[str]
    method: str

    def require(self, *required: str) -> None:
        for permission in required:
            if permission not in PERMISSIONS:
                raise ValueError("Unknown application permission")
            if permission not in self.permissions:
                raise HTTPException(
                    403, detail={"code": "insufficient_scope"},
                    headers={"WWW-Authenticate":
                             f'Bearer error="insufficient_scope", scope="{permission}"'},
                )


@dataclass(frozen=True)
class OAuth2Config:
    introspection_url: str
    issuer: str
    audience: str
    client_id: str
    client_secret: str = field(repr=False)
    tenant_claim: str = "tenant_id"
    timeout_seconds: float = 3.0

    def __post_init__(self):
        parsed = urlsplit(self.introspection_url)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username is not None
                or parsed.password is not None or parsed.query or parsed.fragment):
            raise ValueError("OAuth2 introspection URL must be a clean HTTPS endpoint")
        if (not self.issuer or len(self.issuer) > 512 or not self.audience or len(self.audience) > 256
                or not self.client_id or len(self.client_id) > 256 or
                not self.client_secret or len(self.client_secret) > 4096):
            raise ValueError("OAuth2 issuer, audience and client credentials are required")
        if not CLAIM.fullmatch(self.tenant_claim) or self.tenant_claim in {
            "scope", "active", "aud", "iss", "exp", "nbf", "cnf", "sub", "client_id"
        }:
            raise ValueError("OAuth2 tenant claim must be a dedicated top-level name")
        if (type(self.timeout_seconds) not in (float, int) or
                not math.isfinite(self.timeout_seconds) or not 1 <= self.timeout_seconds <= 10):
            raise ValueError("OAuth2 introspection timeout must be 1..10 seconds")


def _validate_introspection(raw: object, config: OAuth2Config, now: float) -> Principal:
    if not isinstance(raw, dict) or raw.get("active") is not True:
        raise _invalid_token()
    token_type = raw.get("token_type", "Bearer")
    if (raw.get("iss") != config.issuer or "cnf" in raw or
            not isinstance(token_type, str) or token_type.lower() != "bearer"):
        raise _invalid_token()
    audience = raw.get("aud")
    if (not isinstance(audience, (str, list)) or
            (isinstance(audience, list) and
             (not audience or any(not isinstance(item, str) for item in audience)))):
        raise _invalid_token()
    if config.audience not in ([audience] if isinstance(audience, str) else audience):
        raise _invalid_token()
    def timestamp_is_valid(value: object) -> bool:
        # Bounds also reject hostile arbitrary-precision ints that overflow float conversion.
        return type(value) in (int, float) and 0 <= value < 1_000_000_000_000

    exp = raw.get("exp")
    if not timestamp_is_valid(exp) or exp <= now:
        raise _invalid_token()
    nbf = raw.get("nbf")
    if nbf is not None and (not timestamp_is_valid(nbf) or nbf > now):
        raise _invalid_token()
    subject, tenant, scopes = raw.get("sub"), raw.get(config.tenant_claim), raw.get("scope")
    if not isinstance(subject, str) or not 1 <= len(subject) <= 256:
        raise _invalid_token()
    if not isinstance(tenant, str) or not TENANT.fullmatch(tenant):
        raise _invalid_token()
    if not isinstance(scopes, str) or len(scopes) > 2048 or any(
        not 33 <= ord(char) <= 126 and char != " " for char in scopes
    ):
        raise _invalid_token()
    selected = frozenset(scopes.split()) & PERMISSIONS
    return Principal(tenant, subject, selected, "oauth2_introspection")


def _client(config: OAuth2Config) -> httpx.Client:
    """Construct a fresh, non-proxying client per request; no revoked-token cache."""
    return httpx.Client(timeout=config.timeout_seconds, follow_redirects=False, trust_env=False)


def _introspect(token: str, config: OAuth2Config) -> Principal:
    try:
        # No redirects, ambient proxies, token logging or introspection caching.
        with _client(config) as client:
            with client.stream(
                "POST", config.introspection_url,
                data={"token": token, "token_type_hint": "access_token"},
                auth=httpx.BasicAuth(config.client_id, config.client_secret),
                headers={"Accept": "application/json"},
            ) as response:
                if response.status_code != 200:
                    raise _unavailable()
                chunks, length = [], 0
                for chunk in response.iter_bytes():
                    length += len(chunk)
                    if length > MAX_INTROSPECTION_BYTES:
                        raise _unavailable()
                    chunks.append(chunk)
        payload = json.loads(b"".join(chunks))
    except HTTPException:
        raise
    except (httpx.HTTPError, ValueError, UnicodeError, TypeError) as exc:
        # Provider error bodies and URLs may contain credentials. Never echo them.
        raise _unavailable() from exc
    return _validate_introspection(payload, config, time.time())


class Authenticator:
    def __init__(self, *, mode: str, keys: dict[str, str],
                 key_roles: dict[str, tuple[str, ...]], oauth2: OAuth2Config | None):
        if mode not in {"api_key", "oauth2_introspection"}:
            raise ValueError("Unknown auth mode")
        if mode == "oauth2_introspection" and oauth2 is None:
            raise ValueError("OAuth2 mode requires introspection settings")
        self.mode, self.keys, self.key_roles, self.oauth2 = mode, keys, key_roles, oauth2

    def authenticate(self, authorization: str | None) -> Principal:
        if not isinstance(authorization, str):
            raise _missing_bearer()
        scheme, separator, token = authorization.partition(" ")
        if (not separator or scheme.lower() != "bearer" or not 1 <= len(token) <= MAX_TOKEN_LENGTH
                or any(not 33 <= ord(char) <= 126 for char in token)):
            raise _invalid_token()
        if self.mode == "oauth2_introspection":
            return _introspect(token, self.oauth2)
        # Scan each key without revealing the matched raw secret in responses.
        matched = None
        for key, tenant in self.keys.items():
            if hmac.compare_digest(token.encode("utf-8"), key.encode("utf-8")):
                matched = (key, tenant)
        if matched is None:
            raise _invalid_token()
        key, tenant = matched
        roles = self.key_roles.get(key, ("admin",))
        permissions = frozenset().union(*(ROLES[role] for role in roles))
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        return Principal(tenant, "api-key:" + digest, permissions, "api_key")


def route_permissions(path: str, method: str) -> tuple[str, ...]:
    """Deny by default; never derive privileges from user-supplied payload."""
    if method == "GET" and path in {
        "/health", "/capabilities", "/strategies", "/assets/capabilities",
        "/documents", "/assets/images",
    }:
        return ("rags:read",)
    if path == "/documents" and method == "POST":
        return ("rags:write",)
    if path == "/documents/upload" and method == "POST":
        return ("rags:write",)
    if path == "/images" and method == "POST":
        return ("rags:write", "rags:generate")
    if path == "/query" and method == "POST":
        return ("rags:query",)
    if path == "/assets/images" and method == "POST":
        return ("rags:write",)
    if path in {"/assets/search", "/assets/search/image"} and method == "POST":
        return ("rags:query",)
    if path.startswith("/documents/"):
        suffix = path[len("/documents/"):]
        if "/" not in suffix:
            return { "GET": ("rags:read",), "DELETE": ("rags:delete",) }.get(method, ())
        doc_id, separator, action = suffix.partition("/")
        if doc_id and action == "revisions" and method == "GET":
            return ("rags:read",)
        if doc_id and action == "reindex" and method == "POST":
            return ("rags:reindex",)
    if path.startswith("/assets/images/"):
        suffix = path[len("/assets/images/"):]
        if "/" not in suffix:
            return {"GET": ("rags:read", "rags:original_image"),
                    "DELETE": ("rags:delete",)}.get(method, ())
        asset_id, separator, action = suffix.partition("/")
        if asset_id and action == "reindex" and method == "POST":
            return ("rags:reindex",)
    return ()


def authorization_for_path(principal: Principal, *, path: str, method: str,
                           is_mcp: bool = False) -> None:
    permissions = route_permissions(path, method)
    if not permissions:
        raise HTTPException(403, detail={"code": "permission_denied"})
    principal.require(*permissions)
    if is_mcp:
        principal.require("rags:mcp")
