# V08 Agent Platform Implementation and Safety Review

## Delivered contracts

- The operator-controlled **LiteLLM Gateway HTTP adapter** accepts an exact alias whitelist and validated Pydantic chat/tool output. Public upstream LiteLLM documentation advertises more than 100 model providers; the RAG code performs only one bounded standard Gateway request and is tested using a mock HTTP Gateway, not 100 direct vendor endpoints.
- Tools are code-registered, typed by Pydantic, scoped by authenticated tenant and identity permissions, and bounded by explicit token, tool count, time, and iteration controls. No model-supplied arbitrary code or external network destinations.
- Workflow graph DSL supports ordered `.then`, conditional `.branch`, and idempotent concurrent `.parallel` branches. Approval gate persists before any gated operation. Checkpoints use optimistic CAS and fingerprint to prevent resume against changed workflow definitions.
- Tenant+session scoped user/assistant history and explicit provenance/TTL observations. Observation writes from the agent's model require human approval; external explicit API observation writes require a `rags:memory` scope. Separate memory enable flag.
- Existing OAuth2 introspection, `rags:agent`/ `rags:generate` scopes, FeatureFlags, PostgreSQL+pgvector and frontend/admin paths retain their authority.

## Known limitations

1. No LiteLLM Gateway deployment is bundled; operator must operate its TLS-secured LiteLLM Gateway with upstream provider credentials, routing policy, budgets, retries and cost controls. No proof that every of its 100+ advertised upstream providers works with tool calling in the selected account.
2. Does not expose dynamically defined Python tools through the HTTP API. Operator must register any new functions and Pydantic schemas in code. No sandbox for running generated arbitrary Python.
3. Workflow callbacks and tool handlers must be idempotent when called after crash/retry; this release is not a distributed exactly-once transaction coordinator. Parallel branches can be replayed as a unit on crash.
4. Stored conversation, observations and workflow state are plaintext data within the existing database; encrypt storage, restrict DB access, configure tenant ACL, retention and privacy settings appropriate to real content. No automatic sensitive data/PHI extraction or model-driven user profiling.
5. No automatic JWT/OIDC browser login, Cross-tenant RLS, background job scheduler, Agent Desktop UX workflow editor, auto-evaluation billing or combined OTel distributed traces beyond the existing telemetry.
6. The model router and tool handling are tested against fake HTTP Gateway responses and deterministic agent scripts. No live OpenAI/Anthropic/Gemini/xAI credentialed test was performed.
7. Run/tool limits are per request and are not distributed abuse prevention. Configure network egress policy, per-tenant rate limits, quotas and incident monitoring for deployment.

## Acceptance

- Existing unit/integration, real PostgreSQL+pgvector, model training/research sidecars, wheel installation, static analysis and coverage gates stay in CI.
- Added HTTP mock provider normalization, denied alias/status/oversized/malformed responses, Pydantic tool schema, tenant isolation, human gate approval/denial, workflow restore/version conflict, conversation and observation TTL, and real PostgreSQL state read/write tests.
- All features remain opt-in, retain a single Gateway egress endpoint controlled by the operator, and never silently downgrade OAuth2/feature/scoped authorizations.

## Default-on readiness, operator workflow and nonclaims

The v0.8 agent, router and observation memory interfaces default enabled; this never
grants model credentials or auto-selects an external endpoint. Without a valid
gateway, the API reports router_ready=false, agent inference returns 503, and
the registered evidence-review workflow still operates using local authorized RAG.
The Admin Dashboard renders a DAG for sequential, conditional, parallel and approval
steps. Workflow API/CLI can start, inspect, approve, deny and recover checkpoints.

Upstream LiteLLM provider coverage does not establish live interoperability or
feature parity for every upstream account. A user-generated Python workflow editor
is not exposed; only approved operator-registered steps can run.

## Cross-RAG model routing and prompt trust boundary

Existing RAG generation and embeddings can explicitly select the same Gateway
using the router provider. Only operator-approved aliases and valid HTTPS
credentials permit network calls. Gateway embeddings are strictly checked
for numeric finite vector dimensions before indexing. No vendor-specific
tool-call guarantees are inferred from Gateway support.

Untrusted observations and RAG citations are placed in user-role data blocks
in the Agent prompt, never as privileged system messages. Existing tenant and
evidence admission checks continue to apply.
