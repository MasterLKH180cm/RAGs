# Live model contract and performance benchmark

## Purpose and scope

Measure the production `rags.agents.router.ModelRouter` against an explicitly
selected model endpoint. The default target is the verified
`thinkingmachines/Inkling-2-Mini` identifier on Tinker's OpenAI-compatible
sampling endpoint. Embedding configuration and indexes are independent and are
not changed. This is a small synthetic contract/behavior benchmark, not a RAG
retrieval evaluation, full agent workflow evaluation, or general intelligence
score. Database and two-service benchmarks remain separate.

## Invariants

1. A live run requires a credential supplied through an environment variable or
   its `_FILE` companion. Missing credentials produce a `blocked` report and a
   nonzero exit status before any request. There is no mock or extractive fallback.
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

## Interface

`python -m rags.model_benchmark run --output report.json` runs the built-in
synthetic cases with explicit CLI controls and an optional validated dataset.
The default key environment name is `TINKER_API_KEY`; literal secrets are never
CLI arguments. `--model` and `--base-url` allow a baseline through the same
adapter. `--reasoning-effort` is explicit and `--protocol` selects an approved
request profile (`auto`, `openai`, or `tinker`). Plain HTTP requires an explicit
loopback development flag and cannot target arbitrary remote plaintext hosts.

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
  network request or traceback.
- Real model performance is claimed only from an independently identified live
  report whose requests actually completed and passed the declared validators.
