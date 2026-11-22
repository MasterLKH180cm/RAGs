# SPEC V06 — Tenant-aware Resource Authorization and OAuth2 Introspection

Date: 2026-11-23. Status: implementation specification, written before code.

## Audit baseline

- The original service maps each static bearer API key to one tenant string. All routes can be used by that key unless a global feature flag disables them. Global feature flags are not principal authorization.
- Store, retrieval, revision, vector search, graph and images carry tenant predicates; dedicated research sidecar checks body namespace against a static key's tenant. Cross-tenant isolation is application enforced, with no PostgreSQL RLS or per-document ACL.
- No OAuth2/OIDC access token verification, no roles, and no per-resource OAuth scopes exist. MCP uses the same static API key as the HTTP service; declaring `X-RAGS-Interface` is not client authentication.
- Federated source routing binds tenant to operator-selected downstream credentials. No cross-tenant query parameter must influence the authenticated tenant.

## Threats and invariants

1. A user with read-only permission must be unable to ingest, reindex, delete, call a model-generation workflow, or retrieve original image bytes.
2. A tenant's token cannot identify or access another tenant through URL paths, request bodies, metadata, uploaded file names, image search, MCP or federation. Not-found must be 404 rather than 403 when tenant ownership is unknown.
3. OAuth2 response fields are untrusted until received over a configured HTTPS RFC 7662 introspection endpoint authenticated by the resource server. The endpoint is configured by the operator and cannot be supplied by a request. Never accept `X-Tenant`, roles, `scope` or an unsigned JWT payload as a tenant proof.
4. Fail closed on missing, expired, revoked, malformed, wrong-issuer, wrong-audience, wrong-tenant or insufficient-scope tokens. Introspection errors must not fall back to API keys.
5. Preserve existing static API key mappings and old tests. A new optional explicit `RAGS_KEY_ROLES_JSON` provides least privilege; absent this map, legacy keys retain the existing full-access compatibility behavior. Document that this is a migration mode, not tenant-user RBAC by itself.
6. Keep `FeatureFlags` globally authoritative for operation enablement, and apply per-principal permissions *as an additional mandatory check* before any DB access or model/provider work.
7. Keep OAuth2 Resource Server distinct from an OAuth2 authorization server/OIDC login client; this version uses mandatory online RFC 7662 token introspection (opaque or JWT tokens if the IdP accepts them). OIDC discovery/JWKS, Authorization Code+PKCE, token issuance, and mTLS/DPoP proof validation are not claimed to be implemented.

## Permission vocabulary

A bounded exact allowlist of permissions:
- `rags:read` — list, view and revision read; asset metadata, capabilities.
- `rags:query` — text and image retrieval.
- `rags:write` — ingest documents/asset bytes or AI captions.
- `rags:reindex` — reindex a document or asset.
- `rags:delete` — delete a document or asset.
- `rags:generate` — call model generation, including internal model-backed query strategies and image captioning; subordinate to query/write scopes.
- `rags:original_image` — download original image bytes; requires `rags:read` as well.
- `rags:mcp` — use shipped MCP bridge, as asserted by the bridge's request header; not an authentication credential in its own right.

Static API key roles: `reader` (read/query), `editor` (reader + write/reindex), `admin` (all permissions including model generation, deletion, originals, MCP). `RAGS_KEY_ROLES_JSON` must be a complete key-to-roles map when supplied (no unlisted implicit admin), with strict role validation. Subject identity is a stable key digest, never the raw key.

OAuth2 scopes are read from the introspection response's exact, space-delimited `scope` string (only allowlisted scopes grant rights). No `roles`, `groups`, headers, or raw claims confer additional privileges. Issuer, audience, non-empty subject, non-empty tenant ID and finite `exp` are mandatory, even though RFC 7662 makes some fields optional. This is a documented integration contract for participating authorization servers.

## Configuration, transport, and safety

- `RAGS_AUTH_MODE=api_key|oauth2_introspection`; default `api_key` for backward compatibility.
- OAuth2 mode requires `RAGS_OAUTH2_INTROSPECTION_URL` (HTTPS; no query, fragment or userinfo), `RAGS_OAUTH2_ISSUER`, `RAGS_OAUTH2_AUDIENCE`, `RAGS_OAUTH2_CLIENT_ID`, `RAGS_OAUTH2_CLIENT_SECRET` or `RAGS_OAUTH2_CLIENT_SECRET_FILE`; reject configured API keys in this mode to prevent dual-auth fallback.
- `RAGS_OAUTH2_TENANT_CLAIM` may choose a single top-level field name; default `tenant_id`; never authorize a tenant from the HTTP request. Configure a short outbound timeout; TLS verification required; redirect handling disabled; trust_env=False to avoid implicit outbound proxy access.
- Per request: POST application/x-www-form-urlencoded `token` and `token_type_hint=access_token` to fixed HTTPS endpoint using HTTP Basic client authentication. Limit response bytes, parse strict JSON, require `active is True`, `iss`, `aud`, `exp` and `sub`, enforce `nbf` if present, validate bounded tenant ID and scope string. Reject `cnf` tokens (sender-constrained proof not implemented). Never cache introspection results to preserve revocation behavior.
- Return RFC 6750 `WWW-Authenticate` challenge: 401 invalid token, 403 insufficient scope; 503 sanitized unavailable introspection (no secret or URL leakage).
- Avoid any external model/DB operation when authorization is denied.

## Route mapping

| Paths | Method | Permission |
|---|---|---|
| /health, /capabilities, /strategies, /assets/capabilities | GET | rags:read |
| /documents, /documents/{id}, /documents/{id}/revisions, /assets/images | GET | rags:read |
| /documents, /documents/upload, /images, /assets/images | POST | rags:write |
| /documents/{id}/reindex, /assets/images/{id}/reindex | POST | rags:reindex |
| /documents/{id}, /assets/images/{id} | DELETE | rags:delete |
| /query, /assets/search, /assets/search/image | POST | rags:query |
| /assets/images/{id} | GET | rags:read + rags:original_image |
| /images | POST | rags:write + rags:generate |
| /query model-backed modes or configured generation | POST | rags:query + rags:generate |

Unknown authenticated route/method combinations deny by default. Preserve existing feature flag checks at the request/service layers.

## Acceptance

- Unit tests: role config and key mapping, invalid values, default legacy fallback, exact role permissions, denied operations and missing/wrong scopes, malformed bearer headers, token-size and character limits, claim strictness, revoked token, expired/nbf, wrong issuer/audience, tenant format, mTLS/DPoP cnf rejection, parsing limits, sanitized errors, unsafe redirect/outage, no calls on early rejection.
- Integration tests: real FastAPI endpoints cover read/write/reindex/delete/image original, generate, MCP, missing/foreign tenant 404, source provenance and reindex, query without generation, and proof of actual tenant isolation with two simulated introspection tokens and a MockTransport.
- PostgreSQL existing actual pgvector tests must pass unchanged. No RLS is claimed or silently enabled.
- CI: complete static analysis, unit, integration, actual PostgreSQL tests, Docker Compose, research/upstream and admission checks, installed wheel; coverage floors maintained. New functionality is opt-in and does not alter Compose default authorization mode.
- A security review document shall distinguish confirmed fixed controls vs remaining risks (RLS, issuer service configuration, OAuth client login, token sender proof and production IAM hardening).
