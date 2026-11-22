# V07 Design, Security and Operations Review

## Scope and confirmed implementation

- The service ships a self-hosted admin UI under `/admin/` (only when `RAGS_ADMIN_ENABLED=true`), bundled into the Hatch wheel and Docker image. The UI uses static HTML/CSS/vanilla JS, no CDNs or externally loaded fonts/scripts.
- An explicit `rags:admin` permission restricts the dashboard. API-key admin roles gain this scope; OAuth2 bearer tokens must receive it from the trusted introspection service. Public API authentication remains bearer-only.
- Dashboard requests use an opaque, short-lived, HttpOnly browser session with server-side credential revalidation on every request. Session responses never include the original bearer credential; all state-mutating calls need Origin and CSRF checks. JSON is no-store. HTTPS cookies use `__Host-` + Secure/SameSite=Strict.
- All summaries, queries and documents are scoped to the authenticated tenant. Stored documents are never preloaded in full for the document list; documents support paged SQL search with bound parameters and escaped LIKE literals. Deletion needs a matching revision in the same transaction.
- The service emits wire-compatible OTLP/HTTP JSON traces and cumulative request/histogram metrics with bounded queue and nonblocking export, without request bodies, query strings, raw paths, headers, credentials or tenant identifiers as OpenTelemetry attributes.

## What has NOT been claimed or built

- **No OpenTelemetry Python SDK instrumentation**; this release uses a small OTLP wire exporter. It does not automatically trace SQL spans, outbound LLM network requests or create distributed context propagation. Use the official SDK/auto-instrumentation for a future full tracing rollout.
- **No complete multi-replica metrics aggregation or durable sessions**. UI metrics and sessions are per process; use one worker or sticky routing. Collector export is a separate telemetry data plane, not a durable dashboard datastore.
- **No OIDC browser login**, user provisioning UI, RBAC role editor, full RLS, document-level ACL, or production IdP self-service onboarding. Use existing OAuth2 introspection and scopes; further IAM work requires a separate audited design.
- **No live end-user usability study or guaranteed WCAG certification**. Markup provides accessible labels, keyboard focus states, reduced-motion support, responsive CSS and high-contrast themes; screen reader and device-specific manual acceptance remains necessary.
- **No public monitoring endpoint, built-in Grafana, Jaeger, alert routing, cost billing or model content observability**. Example Collector's debug exporter is for local testing and must be replaced for production.

## Production checklist

1. Keep `RAGS_ADMIN_ENABLED=false` unless operator intends a dashboard; require external TLS, trusted proxy scheme/Host and appropriate rate limiting.
2. Grant `rags:admin` only to tenant operators. Avoid reusing admin bearer tokens between personnel; rotate tokens, use short expiry and monitor sign-in attempts.
3. Protect application server memory and database credentials. Deploy a dedicated encrypted session store for horizontal scale; preserve CSRF and OAuth2 revalidation semantics.
4. Restrict Collector access to the private service network, enable TLS/auth for non-local export, set telemetry retention/alert rules, and ensure zero PHI/prompt content in telemetry pipelines.
5. Keep database tenant predicates and OAuth2 scopes authoritative; dashboard must never switch tenant by URL/body.
6. Run all automated tests and manual desktop/mobile keyboard, assistive-tech and UX acceptance in the actual deployment environment.
