# SPEC V07 — Secure Admin Dashboard, OTLP/HTTP observability and UX

Status: specification first; implementation follows on this branch.

## Goals and boundaries

Deliver a responsive, accessible, self-hosted administration experience at `/admin/` that works from the installed `rags-platform` wheel and Docker deployment without a Node runtime or CDN. Preserve the v0.6 OAuth2 resource server, API-key RBAC, feature flags, tenant filters and public API contracts. Add optional vendor-neutral OpenTelemetry Protocol (OTLP/HTTP JSON) traces and metrics export, plus a bounded per-tenant in-memory operational view. Operational metrics must never contain patient text, prompts, raw tenant identifiers in OTLP attributes, API keys, access tokens, cookies, request bodies, URL paths with IDs, or raw SQL.

## UX product requirements

- Desktop sidebar + responsive mobile navigation, concise overview cards (documents, revisions, indexed chunks, recent activity, 5-minute request/error/latency pulse), document inventory with debounce search and pagination, knowledge-query playground with citation/evidence, service capability/settings view, observability view showing API health and telemetry delivery state.
- Accessible semantics: correct headings/labels, keyboard focus, skip link, ARIA announcements for loading/errors, `prefers-reduced-motion`, >=44px mobile click targets, dark/light theme (persist only theme, never credentials), color-independent statuses, empty/loading/error states.
- No externally loaded JavaScript, fonts, analytics or CDN, no `innerHTML` with user-supplied values; use `textContent`/DOM nodes. Strict CSP. Usable while OTel is disabled or collector offline.
- No destructive bulk operations. A single document delete requires typed matching ID and current revision, CSRF token and server-side compare-and-swap.

## Authentication and authorization

- Dashboard is explicit opt-in `RAGS_ADMIN_ENABLED=false` by default, independent of global feature flags. Require a new `rags:admin` permission; add only to `admin` API-key role. OAuth2 tokens require exact `rags:admin` scope in introspection response (and `rags:read` for overview; query/write/delete require their existing scopes).
- `/admin/api/session` login POST accepts a manually supplied Bearer access token over TLS (or explicit loopback-only development mode). No token is stored in browser localStorage, sessionStorage or cookies. The server creates a cryptographically random opaque cookie session with 30-minute absolute lifetime in bounded memory, stores the bearer only server-side, re-authenticates on every request (so revoked OAuth2 tokens fail), and provides a per-session CSRF token.
- Production cookie: `__Host-rags_admin`, Secure, HttpOnly, SameSite=Strict, Path=/, no Domain; local development HTTP only with `RAGS_ENV=development|test` plus `RAGS_ADMIN_DEV_HTTP=true`, otherwise reject insecure login. Session storage is process-local (explicitly document single worker / sticky routing and restarts logging out).
- `GET /admin/api/session` verifies cookie and returns CSRF token; `DELETE /admin/api/session` is CSRF-protected. Mutation APIs require CSRF token and same-origin checks. All admin API responses are `no-store`; don't emit secrets, trace context, tenant IDs in OTLP, or unscoped resource details.
- Every admin endpoint checks authenticated principal + `rags:admin` + operation scope (`rags:read/query/write/delete`) before any DB / model call. Tenant comes only from authenticated principal. The standard public API never accepts the admin cookie as a Bearer replacement.
- CSP/Security headers for dashboard HTML and assets, no third-party scripts; auth errors are generic. No public metrics endpoint or cross-tenant aggregation.

## Admin backend API

- `GET /admin/` and `/admin/assets/{filename}` serve only static, bundled, no-CDN UI while enabled.
- `POST /admin/api/session`, `GET/DELETE /admin/api/session` for auth; for API-key or OAuth2 token mode, explicit `rags:admin` permission mandatory.
- `GET /admin/api/overview`: tenant-scoped counts for documents, current chunks and revisions, recent audit events; bounded in-memory request telemetry. Counts must be SQL-scoped, not full scans and cannot return individual patient text.
- `GET /admin/api/documents?search=&limit=&offset=`: server-side paged, tenant-scoped, escaped LIKE filter; upper limits enforced, deterministic ordering; never leak foreign tenant IDs. Return ID/title/source/revision/updated_at (not raw source body).
- `POST /admin/api/documents`: validated text source ingestion (same typed `DocumentIn` and size limits as API). Respect generation permission when configured source processing uses model generation. No bypass of feature flags.
- `DELETE /admin/api/documents/{id}`: CSRF + delete scope + `If-Match` revision + atomic CAS in Store.delete; 404 foreign/absent, 409 concurrent version mismatch.
- `POST /admin/api/query`: validated `QueryIn`; default `generate=false`, check generation/strategy features and rights, return bounded citations and answer. No stored query history.
- `GET /admin/api/operations`: effective features, auth mode, storage name, embedding/generation provider names, OTel status (enabled/disabled, dropped, dispatched, errors) and bounded 5-minute metrics; no endpoint credentials or URLs.

## OpenTelemetry

- `RAGS_OTEL_ENABLED=false` default; `RAGS_OTEL_ENDPOINT` is operator-supplied, HTTPS-only by default; local/private HTTP requires explicit `RAGS_OTEL_ALLOW_INSECURE=true`, never allowed from arbitrary request input.
- Export **actual OTLP/HTTP JSON protobuf-mapped** traces to `/v1/traces`, metrics to `/v1/metrics` using already installed `httpx`, with explicit schema tests against Collector protocol examples. This is a minimal native OTLP exporter rather than an OpenTelemetry Python SDK integration; document that distinction. No added package-lock changes.
- Queue bounded at 1024 and background export worker with short timeout, batching, backpressure-drop counter and bounded shutdown. Network export must never block HTTP responses or crash API. `service.name` and low-cardinality `http.route`, method, status class only. For unknown routes use `unmatched`, never raw path. Trace IDs and span IDs randomly generated. Metrics monotonic cumulative process counters/histogram, not client/proxy/patient identities. Disable tracing for admin asset requests where appropriate; still record safe route metrics. Never send Authorization/cookies/query strings or exception messages.
- Bounded in-process tenant metrics for dashboard: last 60min, bounded per-tenant records, locked and max tenants, no bodies or document IDs. Use 5-minute buckets (count, errors, p50/p95). Explicitly indicate process-local scope; not global distributed analytics.

## Testing and release

- Unit+integration: session cookie flags, scope isolation, OAuth2 revocation on next request, CSRF, origin check, session expiry, invalid credentials, no token in browser response, no bearer/cookie fallback, concurrency CAS, SQL tenant filters, query/ingest/generation, static asset CSP/cache policy, mobile accessibility selectors/markup, zero dependencies on external CDN.
- OTLP: real schema shape, trace ID lengths, metric temporality/aggregation, no dynamic URL cardinality, privacy/no secret leakage, overflow behavior, disabled/no-network, failed-collector non-blocking, endpoint validation, graceful stop. Explicit test with fake HTTP collector.
- Package installed wheel must include HTML/CSS/JS under `rags/admin_static`; CI exercises isolated wheel dashboard with HTTPS TestClient.
- Existing 516+ unit, 305+ integration, 23 PostgreSQL, research, encoders and admission tests; all coverage floors remain green. PR Codex Review via only `@codex review` comments; merge only after tests and review.
- Exclusions: no OIDC interactive login redirects, SQL-backed durable sessions, distributed query metrics aggregation, public Prometheus scraping, Grafana/Jaeger provisioning, full OTel SDK auto-instrumentation, RLS or resource ACL; explicitly document production hardening path.
