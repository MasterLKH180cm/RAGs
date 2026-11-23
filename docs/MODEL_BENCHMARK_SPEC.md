# Live model contract and performance benchmark

## Purpose and scope

Measure the production `rags.agents.router.ModelRouter` against an explicitly
selected model endpoint. The default target is the verified
`thinkingmachines/Inkling-2-Mini` identifier on Tinker's OpenAI-compatible
sampling endpoint. The explicit `inkling-local` preset uses downloadable weights
through an operator-run OpenAI-compatible server on literal loopback. Embedding
configuration and indexes are independent and are
not changed. This is a small synthetic contract/behavior benchmark, not a RAG
retrieval evaluation, full agent workflow evaluation, or general intelligence
score. Database and two-service benchmarks remain separate.

## Invariants

1. Authentication defaults to `required`; credentials come from an environment
   variable or its `_FILE` companion. Missing required credentials produce a
   `blocked` report before any request. Explicit `none` authentication is permitted
   only for a clean literal-loopback endpoint, explicit `openai` protocol and unset
   effort. It reads no ambient credential or credential file and sends no
   Authorization header. There is no mock or extractive fallback.
2. Every request goes through the production router. The harness must not
   implement another provider HTTP adapter. Offline tests explicitly use an HTTP
   fixture; those results establish contracts, never model speed or quality.
3. Work is bounded: at most 12 synthetic cases, 10 repetitions, concurrency drawn
   from 1/2/4, at most 3 warmup rounds, output limit 1..4096, request timeout
   0.05..90 seconds, and a finite overall run deadline. No hidden retries.
4. Warmup samples, measured samples, errors, and deadline-exhausted samples are
   separate. Successful throughput is validated successes divided by the entire
   measured phase wall time. Latency percentiles use successful measured samples
   only and disclose the sample count. Output-token throughput is reported only
   when every successful measured sample has valid provider usage.
5. A `length` finish reason is a failure even when the text parses. JSON uses
   strict Pydantic schemas (extra fields forbidden); booleans are not numbers.
   Grounded cases require exact synthetic answers and citation IDs; tool cases
   require one approved read-only function with exact validated arguments. No
   tool is executed. Provider HTTP success alone never means a case passed.
6. Reports contain no prompts, completions, tool argument values, API keys,
   endpoint URLs, exception messages, or environment dumps. Store source/version
   hashes, dataset/config identities, selected public model ID, endpoint
   category/fingerprint, aggregate metrics, finite sanitized usage counters,
   stable failure codes, and case IDs only. Errors never print provider bodies.
7. Only operator-supplied per-million token prices produce an estimated cost.
   Cached-input pricing is used only with valid cached counters. Unknown usage
   or pricing stays null with an explicit reason. Estimates are not invoices.
8. Comparisons require matching dataset, harness/adapter source identities,
   repetitions, warmups, concurrency, output caps, effort, timeout, and validation
   contract. Different endpoint fingerprints are disclosed. A failed/blocked run
   cannot produce a speedup claim. Small samples are always disclosed.
9. Live loopback runs require a strict operator-supplied deployment manifest.
   Missing provenance produces `blocked / missing_deployment_manifest` before
   any request. HTTP contract fixtures are explicitly exempt and remain
   ineligible for live comparisons. Manifest data and its digest are reported;
   the harness does not claim to verify that the server loaded those weights.
10. Report schema `model-contract-benchmark-v2` uses `request_upper_bound`, since
    local inference requests need not be paid provider calls. Comparisons reject
    older schemas and disclose deployment fields, authentication and protocol
    differences. Quantization, hardware, context and server-slot differences are
    experimental conditions, not evidence of native-checkpoint model equivalence.

## Interface

`python -m rags.model_benchmark run --output report.json` runs the built-in
synthetic cases with explicit CLI controls and an optional validated dataset.
The hosted default key environment name is `TINKER_API_KEY`; literal secrets are
never CLI arguments. `--model` and `--base-url` allow a baseline through the same
adapter. `--reasoning-effort` is explicit and `--protocol` selects an approved
request profile (`auto`, `openai`, or `tinker`). Plain HTTP requires an explicit
loopback development flag and cannot target arbitrary remote plaintext hosts.

`--preset inkling-local` selects `http://127.0.0.1:8080/v1`, the canonical model
identifier, `--auth-mode none`, explicit `--protocol openai`, unset reasoning
effort and loopback HTTP permission. Explicit CLI values can override a preset,
but all endpoint/authentication invariants still apply. If required authentication
is selected for this preset, the default key name is `INKLING_LOCAL_API_KEY`
(or its `_FILE` companion), without fallback to hosted-provider credentials.
`--key-env` overrides a preset's credential name only when explicitly supplied;
no-auth mode never reads it. `--deployment-manifest`
accepts at most 16 KiB of strict JSON, rejects extra fields and duplicate keys,
and snapshots validated values before measurements.

The manifest schema is `model-deployment-v1` with `source: operator_supplied`.
Required fields are `runtime_name`, `runtime_version`, `runtime_revision`,
`model_repository`, `model_revision`, `model_artifact`, `model_sha256`,
`model_bytes`, `quantization`, `context_tokens`, `server_parallel_slots`, `server_threads`,
`hardware_class`, `hardware_label`, `host_memory_bytes`, `accelerator_count`,
and `accelerator_memory_bytes`. Revision IDs are full lowercase 40-digit Git
hashes; artifact digests are lowercase SHA-256. Labels are bounded public
identifiers, artifact names are basenames, and numeric values are bounded strict
integers. Hardware classes are `cpu`, `cuda`, `rocm`, `metal`, `tpu`, or `other`.
CPU manifests must declare zero accelerators and zero accelerator memory.
No arbitrary notes, hostnames, local paths, endpoint URLs or secret fields are
accepted. Operators must use public non-secret labels. Optional manifests can
also describe authenticated remote self-hosted deployments.

`python -m rags.model_benchmark compare baseline.json target.json --output ...`
validates report structure, checks measurement parity, and records comparable
phase metrics or an explicit rejection. It never calls a model.

## Initial dataset

The immutable default dataset contains short synthetic JSON extraction,
grounded answer, insufficient-evidence/refusal, untrusted-document instruction,
and read-only tool-selection cases. A JSON snapshot in `examples/benchmark*`
allows inspection and reproducible custom runs. Dataset hashes cover prompts,
expectations, and validation kind. Reported case IDs are bounded identifiers.

## Acceptance

- Unit tests cover bounds, booleans/nonfinite inputs, schema/semantic checks,
  percentiles/throughput, incomplete usage/pricing, parity rejection, and redaction.
- HTTP fixture tests pass through the actual router, including malformed output,
  provider failures, truncation, tool arguments, and cached usage.
- A credential-free invocation writes an honest blocked artifact without any
  network request or traceback when authentication is required. Explicit local
  no-auth fixtures reach the actual production router without reading secrets;
  local live invocations require a valid manifest before connecting.
- Tests reject malformed, oversized, contradictory or tampered manifests and
  disclose valid deployment differences in comparisons. With one server slot,
  client concurrency 2/4 measures queueing rather than simultaneous decoding.
- Real model performance is claimed only from an independently identified live
  report whose requests actually completed and passed the declared validators.
