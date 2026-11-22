# V06 Security Audit — Multitenancy, RBAC and OAuth2

## Before

- The API keyed a Bearer secret to exactly one tenant, with tenant filters across documents, revisions, sparse/dense search, graph evidence, original image assets and the research sidecar. The design has **application-layer tenant isolation**; PostgreSQL RLS is not installed.
- Each key could read/ingest/reindex/delete resources for its entire tenant. Global `FeatureFlags` control features, not human/service identity privileges. No roles or per-resource scopes.
- Despite the Bearer header, the original service performed static API key comparison, not OAuth2 validation. No OAuth2/OIDC, introspection, issuer or audience checks.

## Delivered changes

1. Exact `rags:read`, `rags:query`, `rags:write`, `rags:reindex`, `rags:delete`, `rags:generate`, `rags:original_image`, `rags:mcp` scopes; explicit read-only, editor and admin key roles.
2. `RAGS_KEY_ROLES_JSON` is an optional complete key-to-roles map: partial/unknown mappings fail startup. Omission retains existing full access per API key **for compatibility**, and is not a least-privilege production default.
3. Independent opt-in `RAGS_AUTH_MODE=oauth2_introspection`: static credentials are rejected; every request uses authenticated HTTPS RFC 7662 POST introspection and validates active, issuer, audience, subject, strict tenant ID, expiration, nbf, scope and unsupported proof-of-possession rejection.
4. Deny-by-default HTTP permissions for documents, uploads, image originals, delete/reindex, queries, internal generation, metadata and MCP; global FeatureFlags are still required.
5. Error hygiene: 401 Bearer challenge for missing/invalid token, 403 for insufficient scope, 503 generic for IdP outages, no credential echo, no positive-token cache (revocation observed per request).

## Explicit residual risks

- **No PostgreSQL RLS:** one service DB role remains the enforcement point. Do not claim database-enforced multitenancy. Separate tenant-aware RLS policies, transaction-local tenant context, non-bypass roles, migration isolation and hostile SQL tests are needed before enabling.
- **No document-level ACL/ABAC:** the scopes grant tenant-wide data access. Use separate tenants for finer classification until an ACL design is implemented.
- **OAuth2 Resource Server only:** no Authorization Code + PKCE, login screen, OIDC Discovery/JWKS, JWT local verification, ID token handling, refresh token issuance or authorization server. Use an external IdP supporting RFC 7662 and the required claims.
- **No mTLS/DPoP sender-constrained proof support:** introspection replies with `cnf` are rejected, not erroneously accepted as bearer. Compromised bearer tokens are still replayable until revoked or expired.
- **No production IdP interoperability guarantee:** mock end-to-end tests prove our HTTP contract, but live identity-provider, key rotation, organization membership lifecycle, quotas/rate limiting and abuse protections require staging validation.
- **MCP header is not an identity credential.** It signals an additional permission requirement; the actual MCP bridge must be provisioned with a correctly scoped key/access token.

## Deployment / operational review

Choose one mode. Bind the service behind TLS/gateway, restrict DB and sidecar network reachability, keep client credentials in secret stores or read-only secret files, issue least-privilege short-TTL audience-restricted access tokens, rotate keys, and audit revocation/IdP outage handling. Do not put token-to-role mappings into public source control. Use the two-tenant regression suite before confidential-data deployment.

See [SPEC_V06](SPEC_V06.md) for explicit route mapping, requirements and acceptance criteria.
