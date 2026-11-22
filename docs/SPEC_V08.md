# SPEC V08 — Unified model router, agents, workflow DAG, durable HITL and observation memory

Date: 2026-11-23. Status: contract-first. Implementation follows on a feature branch from v0.7.

## Public product claims and nonclaims

- Expose one typed \`ModelRouter.complete\` request/response format with normalized \`tool_calls\`, regardless of the configured provider. Transport is an **operator-controlled LiteLLM Gateway** speaking OpenAI-compatible Chat Completions; its upstream [provider catalog](https://docs.litellm.ai/docs/providers) exceeds forty providers, including OpenAI (\`openai/\`), Anthropic (\`anthropic/\`), Google Gemini (\`gemini/\`) and xAI (\`xai/\`). The user never supplies a URL, token, model path, or provider credentials. All model names are exact pre-approved operator aliases.
- This is a **LiteLLM Gateway adapter**, not a bundled direct implementation and not a claim that this repo has tested >40 upstream credentials/models. Integration tests with a fake gateway verify normalization and safety; real-provider interoperability remains environment dependent. Do not add an unpinned Python SDK dependency or break \`uv sync --locked\`.
- An opt-in agent uses bounded execution, explicit tool permissions, typed arguments, timeouts, stop conditions and human approval. Workflow DSL composes Python-registered steps using \`.then()\`, \`.branch()\`, and \`.parallel()\`, without evaluating dynamically supplied user code.
- Persist resumable agent/workflow checkpoints, conversation history, and attributable *observation records* in the same tenant-scoped SQLite/PostgreSQL storage. Observation memory is a bounded, provenance-aware compaction, not unsupported human-like understanding and not automatic storage of PHI, secrets, or entire prompts.
- Preserve v0.7 Dashboard, OTel, API auth modes, tenant filters, provider defaults, existing model-generated RAG and all prior tests.

## Activation and isolation

- \`RAGS_AGENT_ENABLED=false\`, \`RAGS_ROUTER_ENABLED=false\`, \`RAGS_MEMORY_ENABLED=false\` by default (three separate operator gates).
- Router requires configured HTTPS gateway endpoint, token loaded via env or \`_FILE\`, and exact \`RAGS_ROUTER_MODELS_JSON\` alias-to-model mapping; names restricted to operator-defined aliases and model values (provider/model). Optional loopback/private HTTP allowed only with explicitly opted-in development flag; no implicit fallback.
- Config includes bounded \`max_steps <= 8\`, \`max_tool_calls <= 16\`, \`max_runtime_seconds <= 90\`, \`max_output_tokens <= 4096\`, \`max_history_turns <= 20\`, and typed input sizes. No model fees or completion budget are guessed.
- New \`rags:agent\` permission (admin & agent-runner roles) on start/resume; \`rags:memory\` only on direct memory management. Every \`knowledge_search\` tool additionally requires \`rags:query\`, \`query\` Feature Flag and tenant from Principal, never a token/model supplied tenant.
- HTTP endpoints require both model/agent flag and OAuth2 scopes (or API role); no arbitrary tool registration from request; no SSRF URL/query to gateway, no unbounded messages, no dynamic function evaluation.

## Typed router API

\`ModelRouter.complete(model_alias, messages, tool_definitions=[], max_output_tokens=...)\` returns a \`ModelReply\` (content, tool_calls, finish_reason, usage). Messages and tool metadata use Pydantic validation, strict bounds, and JSON serialization. Accept only \`tool_calls\` with valid IDs, known functions, bounded JSON arguments and unique call IDs. Handle provider 429/5xx/timeouts via sanitized typed errors, never log raw prompt/response, never silently substitute another provider/model. User-supplied model aliases must be in the configuration allowlist. No cache of credentials.

## Agent run state machine and tools

States \`running\`, \`waiting_for_approval\`, \`completed\`, \`failed\`, \`stopped\` (plus \`needs_resume\` when step budget reached). Every step appends a normalized model/tool message, with a strict cap and truncation policies. Tool specs register one Pydantic BaseModel argument schema, a callable implementation, required permission, approval requirement, and side-effect/idempotency metadata. One loop step may request multiple tool calls. Tool results must be JSON-safe and bounded. The engine rejects unknown/unauthorized tools and malformed JSON before execution. If a tool requires approval, **persist the pending request before invoking it** and return a checkpoint for human intervention. Resume verifies same tenant/principal, expected revision, approval decision and revalidated tool call before executing; rejected actions yield a structured tool rejection. Repeated or stale resume never reexecutes side effects.
Runtime stop conditions: step budget, tool budget, wall-time budget, repeated tool call loops, and no-tool final answer. Returning a stop status is better than an unbounded loop or hidden retry.

## Durable checkpoint / HITL

A \`StateStore\` uses a tenant/run composite primary key, optimistic revision CAS, immutable workflow-definition fingerprint, serialized bounded state, and parameterized SQL for SQLite and PostgreSQL. No code or secrets are serialized. Checkpoints are written after every completed step, before human gates, and after approval. Restarts use original workflow step registry; mismatch fails closed. Concurrency-safe in a transaction (SQLite \`BEGIN IMMEDIATE\`, pg advisory lock or \`UPDATE WHERE revision=...\`). Parallel nodes must be explicitly idempotent/read-only because a crash may replay an unfinished branch; checkpoint commits happen only at a complete barrier.

## Workflow DSL

\`\`\`python
flow = (
    Workflow("review")
      .then("retrieve", search_step)
      .branch("quality", quality_predicate, if_true=accept_step, if_false=review_step)
      .parallel("cross_checks", {"a": check_a, "b": check_b}, idempotent=True)
      .then("publish", persist_result)
)
\`\`\`

A workflow is a deterministic, bounded graph plan compiled from a Python registry. `branch` invokes exactly one of two operator-registered callbacks and continues to the next node; it does not jump to arbitrary node labels. \`.parallel\` returns named results with deterministic ordering using bounded workers, with no unsafe concurrent checkpoint mutation. An approval-gated `.then(approval=True,idempotent=True)` persists intent; on resume, approve/deny with expected revision and continue from the same node without replaying earlier nodes. Branches and parallel steps are explicit, not implicit LLM-generated code.

## Conversation and observational memory

- \`ConversationStore\` persists tenant+session+turn+role+short text with append and read limits. No messages shared across tenants, even for equal session ID. History assembly uses last-N turn + character budgets and never an unbounded prompt.
- \`ObservationStore\` saves short, explicitly supplied factual observations with source ID, timestamp, evidence digest, confidence and an expiry (TTL). A deterministic consolidation step deduplicates by normalized fact hash; expired observations excluded. **Do not automatically infer or store secret/health attributes**, nor send raw conversation contents to telemetry. Only observations submitted through permitted workflow/API are persisted, and they are never elevated above retrieved source citations.
- \`AgentContext\` joins conversation snippets, authorized RAG citations, relevant recent observations as separately marked untrusted evidence and original goal; no tools bypass tenant or feature controls.

## API, CLI, UX, tests

- Agent HTTP: \`POST /agents/runs\`, \`GET /agents/runs/{id}\`, \`POST /agents/runs/{id}/resume\` ; memory endpoints \`GET /agents/sessions/{id}/messages\` and \`GET /agents/sessions/{id}/observations\`. User-supplied session IDs bounded. POSTs require \`rags:agent\` and \`rags:generate\`, plus configured features; scopes are checked before model/DB calls.
- CLI: \`rags agent doctor\` for model configuration readiness; \`rags agent run --goal-file PATH --model ...\` with explicit local bearer/URL and safe defaults or document API as primary interface.
- Dashboard: add "Agents" operator section with run status and approval form only if scoped admin session; no exposed provider keys or agent execution code.
- Tests for 2-tenant isolation, invalid model alias, missing tools/rights, malformed tool arguments, provider transient errors, tool budget, run stop conditions, concurrent CAS, restart/resume, deny/approve/replay, branch/parallel order, checkpoint hash, memory TTL and dedupe, transcript injection isolation, feature flag off, offline no-network default, actual PostgreSQL+pgvector schema, and installed wheel.
- Use existing 80% line / 72% branch coverage floors, benchmark/quality gates, real sidecar suites, and final Codex Review. GitHub comments must contain exactly \`@codex review\`.
- Do not claim production operations, 40 individual provider live integration checks, arbitrary action execution, end-user token management, distributed workers, cost accounting, OTel SDK automatic instrumentation, or RLS in this release. List remaining follow-up work explicitly.
