# RAGs v0.9 — model contracts and execution safety

Date: 2026-11-23. Implement from `2010ea5`; preserve the Python service and the
existing knowledge/execution boundaries. This specification is written before
implementation. Model benchmark evidence is reported separately.

## Problems found in the v0.8 code

1. Both the direct OpenAI-compatible adapter and the router send `max_tokens`.
   Tinker's documented request limit is `max_completion_tokens`; its compatibility
   API ignores unlisted request fields. An adapter can therefore appear to work
   while failing to enforce the requested generation limit.
2. A new model name defaults to a tiktoken lookup even when no matching encoding
   is available. Unicode code-point counting is neither that encoding nor a
   conservative count for multilingual inputs.
3. An agent handles tool calls before it checks `finish_reason=length`, which can
   execute a truncated tool call. A run's fingerprint omits tool argument schemas,
   idempotency flags, model mappings and execution limits, so an old approval can
   survive changes to its execution contract.
4. Router JSON decoding accepts non-standard nonfinite numbers and sends tools
   outside its message-byte budget. Embedding and completion envelopes are not
   consistently checked for operation, termination, exact result count or shape.
5. Direct generation, embedding and image-caption adapters buffer upstream bodies
   through `httpx.post` before validation. Per-read timeouts do not bound a slowly
   delivered response, and compressed content can inflate before size checks.

## Changes and compatibility

- Add a small, pure model capability layer shared by the direct provider and the
  model router. Existing OpenAI-compatible endpoints retain their established
  token-limit field unless an explicit profile is selected. The exact published
  `thinkingmachines/Inkling-2-Mini` identifier selects the Tinker chat profile.
- Operator configuration owns protocol and reasoning effort. Callers cannot
  supply endpoints, credentials, profiles or arbitrary upstream request fields.
  Tinker accepts text/tool generation; an Inkling alias cannot be used as an
  embedding model. Embeddings remain independently configured.
- Tinker generation uses an explicit `RAGS_GENERATION_API_KEY` or its file form;
  it never inherits ambient OpenAI or Tinker credentials. Without an explicit
  generation endpoint, the Tinker profile resolves to the canonical Tinker base,
  preserving the independently selected legacy embedding endpoint/credential.
- Tinker uses `max_completion_tokens`; reasoning effort can be explicitly fixed.
  No JSON/schema guarantees are inferred from `response_format` because Tinker
  documents that field as ignored. Validate all JSON and tool arguments locally.
- Add `utf8_bytes` budgeting for unsupported model tokenizers, preserving complete
  Unicode boundaries. Report `utf8_bytes` / `conservative_bytes`, never exact model
  tokens. It is a conservative text budget, not provider billing or full chat
  framing measurement. Source chunking remains independently configurable.
- Preserve bounded upstream `reasoning_content` only in the internal agent run
  checkpoint and resend it on tool continuation, as Tinker recommends. It is
  excluded from public reply serialization, checkpoint exports, API summaries,
  conversation history and observations. Existing database access/retention rules
  apply to these internal continuation records.
- Reject truncated tool calls before executing or requesting approval. Bind
  checkpoints to the registered tool schemas, permissions, idempotency policy,
  approved model mapping and bounded runtime configuration. A changed contract
  rejects resume; it never silently applies an old approval to a new contract.
- Bound direct model requests before network I/O (1 MiB for text/embeddings,
  3,000,000 bytes for image payloads, including JSON framing) and responses at
  512 KiB. Reject redirects and compressed responses without reading their bodies.
  One wall deadline covers connection, headers and every body read. A shared
  transport adapter preserves the synchronous provider interface and explicit
  test/client injection; no alternate unbounded transport is retained.
- The v0.9 agent fingerprint intentionally differs from v0.8. Existing pending
  agent approvals and other resumable agent checkpoints require a new run and
  review after upgrading, even when operator configuration is unchanged.
  Read-only access to historical runs remains available under the original
  tenant/owner permissions; only continued execution requires a matching contract.
- Keep the router, agent and memory enabled by default. Configuration and scopes
  still gate model calls. Add an installed `inkling` profile with explicit model,
  endpoint, tokenizer and independent offline embedding selection; no secret is
  written by the package.

## Required verification

Unit tests cover profile bounds, request construction, strict JSON, usage
normalization, unsupported embeddings, Unicode byte boundaries, and truncated
responses. Integration tests exercise RAG generation, tool calling, approval,
checkpoint changes, tenant isolation and configuration-driven startup. Transport
tests cover whole-request budgets, oversized/chunked/compressed responses, slow
drip/stalled I/O, resource closure and calls from existing asynchronous contexts.
Core pytest cases guard the real synchronous/asynchronous HTTP transports before
network I/O: only literal loopback destinations and injected transports are
allowed. Nonlocal HTTP requires an explicitly selected external-resource test.
Run the
complete core suite without skips, real PostgreSQL contracts where available,
static analysis and installed CLI checks. Record optional resource exclusions.
No mocked or fixture measurement may be labelled as live Inkling performance.

## Sources

- https://thinkingmachines.ai/model-card/inkling-2-mini/
- https://tinker-docs.thinkingmachines.ai/inference/openai/overview/
- https://tinker-docs.thinkingmachines.ai/inference/openai/chat-completions/
