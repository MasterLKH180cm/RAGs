# Self-hosted Inkling 2 Mini implementation contract

## Goal and evidence boundary

Inkling 2 Mini has downloadable Apache-2.0 weights. Tinker is an optional hosted
deployment, not a prerequisite for using the model. Preserve the existing hosted
profile and add an explicit, reproducible local path through the production
generation provider, agent router, and benchmark. A missing Tinker credential
describes only the hosted run; local readiness must report hardware/runtime/model
availability separately.

Primary references, checked 2026-11-23:

- https://thinkingmachines.ai/model-card/inkling-2-mini/
- https://huggingface.co/thinkingmachines/Inkling-2-Mini
- https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF
- https://unsloth.ai/docs/models/inkling-2-mini.md

## Authentication and wire contract

1. Authentication is an explicit `required` (default) or `none` choice.
   Unauthenticated mode is valid only for an operator-selected literal loopback
   IP endpoint with a clean HTTP(S) URL, explicit local HTTP opt-in where needed,
   and an explicitly supported OpenAI-compatible protocol. DNS names, private LAN
   addresses, credentials/query/fragment in URLs and public endpoints are rejected.
2. Unauthenticated mode sends no Authorization header, reads no ambient provider
   secret, and rejects contradictory explicitly configured credentials. It does
   not invent dummy API keys. Loopback traffic bypasses ambient HTTP proxies and
   follows no redirects. Existing hosted credential defaults remain unchanged.
3. Local generation and benchmark use explicit `openai` protocol / `max_tokens`
   and unset reasoning effort. The canonical model name alone must not silently
   select Tinker wire semantics in the local preset. Hosted Tinker continues to
   use its own fields and dedicated credentials.
4. The local server must use the model's supported template and parsers so that
   tool calls and private reasoning remain distinct from final text. Application
   validation, truncation rejection, response/request limits, deadlines, tool
   authorization and approval fencing remain in effect. Bind changed execution
   configuration into the approval fingerprint.
5. Embedding configuration remains independent. The Inkling generation profile
   never creates an embedding capability. Container-to-host/service inference
   uses required authentication and an explicitly configured endpoint; a container
   loopback address never denotes the host or another container.

## Local benchmark and provenance

- Add an explicit local preset / auth choice to the existing bounded benchmark;
  keep its 42-call upper bound, six synthetic cases, production router, privacy
  rules, whole-run deadline and strict comparator.
- Local no-key runs must reach the configured local endpoint rather than fail
  with `missing_model_credential`. Connection/model failures are not passes.
- Record a strict, bounded operator-supplied deployment manifest for local live
  comparisons: runtime/build, model repository/revision/artifact hash, weight
  precision or quantization, context and server concurrency, and hardware class.
  Mark it as operator supplied; an HTTP model label does not independently prove
  the weights. Do not log URLs, secret paths, prompts, outputs or reasoning.
- Comparison discloses deployment differences. Quantized local performance is
  not claimed as native-checkpoint, hosted-service, full retrieval or full agent
  workflow performance. Fixture reports remain ineligible for live comparisons.

## Reproducible deployment

Provide a pinned mainline-compatible llama.cpp GGUF example and preparation /
readiness tooling. Verify model metadata and runtime support before suggesting
commands. Download only public model artifacts, pin revision and expected digest,
and check space/memory before a large download. Separate runtime/weight readiness
from inference evidence. A bounded hardware preflight cannot prove model quality.

The current execution host has about 9.73 GiB total RAM, about 6.35 GiB available
at inspection, no swap and no GPU. The original checkpoint is about 20 GB, and
the selected mainline IQ4_NL artifact is 17,627,940,768 bytes. This host cannot
hold the selected weights plus runtime/KV memory. Do not force swapping, use an
unverified incompatible low-bit format, or relabel fixture timings as model data.

## Acceptance

- Real loopback HTTP tests prove no-key generation and router chat/tool contract
  behavior, no Authorization/proxy use, explicit OpenAI fields, local validation,
  truncation rejection, and no ambient hosted-secret access.
- Negative tests cover unsafe endpoints, conflicting credentials, unsupported
  auth values, hosted/local profile confusion and approval-contract changes.
- Benchmark tests cover local preset, manifest validation and comparison
  disclosure while retaining hosted and fixture boundaries.
- Readiness checks produce factual machine-readable local evidence and a clear
  hardware/runtime/weights blocker without any Tinker-key requirement.
- Run complete core suites, existing gates and real cross-service acceptance;
  publish tested changes and preserve the existing hosted-attempt reports.
