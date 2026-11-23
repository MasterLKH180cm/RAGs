# Inkling 2 Mini model testing

The model's official name is **Inkling 2 Mini**. Its Apache-2.0 weights are
downloadable; a Tinker credential is needed only for the optional hosted API.
The benchmark supports both a local OpenAI-compatible server and Tinker's
sampling API. Its existing default remains hosted Tinker; choose
`--preset inkling-local` for local inference. The existing embedding provider,
vector index, and PostgreSQL configuration remain independent of this
generation-model choice.

This runner measures a **small synthetic contract/behavior dataset** through the
same `ModelRouter` used by RAGs agents and routed generation. It is not a full RAG
retrieval benchmark, an agent workflow benchmark, or a general model-quality
ranking. Database throughput and the real RAG-to-agent workflow have separate
tests. The benchmark never executes model-selected tools.

## Verified model and provider references

Official references, checked November 23, 2026:

- [Model card](https://thinkingmachines.ai/model-card/inkling-2-mini/)
- [Canonical model checkpoint](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
- [Mainline llama.cpp GGUF weights](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF)
- [Pinned llama.cpp b12631 runtime](https://github.com/ggml-org/llama.cpp/releases/tag/b12631)
- [Pinned server flags and API contract](https://github.com/ggml-org/llama.cpp/blob/e300919cc704ddde3a8a8346a8714f57eeeee71b/tools/server/README.md)
- [OpenAI-compatible endpoint and authentication](https://tinker-docs.thinkingmachines.ai/inference/openai/overview/)
- [Chat request and response contract](https://tinker-docs.thinkingmachines.ai/inference/openai/chat-completions/)
- [Model IDs and pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/)

The optional hosted test endpoint is
`https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1`.
It authenticates using a Tinker API key. The separately documented production
endpoint has distinct access/model requirements and is not assumed available.

Use `max_completion_tokens` for the output cap. Tinker's documentation says it
ignores unsupported request fields, including `response_format`; sending a JSON
Schema does not establish constrained output. The application validates JSON,
citations, and tool arguments after generation. Reasoning stays separate from
the final answer. The runner never records reasoning text.

## Run local downloadable weights

The local preset uses `http://127.0.0.1:8080/v1` with the canonical model alias
`thinkingmachines/Inkling-2-Mini`, explicit OpenAI protocol (`max_tokens`), unset
reasoning effort, and `auth_mode: none`. It sends no Authorization header and
does not read any ambient key or key file, including `TINKER_API_KEY(_FILE)`.
The production transport bypasses proxies and does not follow redirects.
Unauthenticated mode accepts literal loopback addresses only; DNS names,
private LAN addresses and public endpoints are rejected. HTTP must be explicitly
enabled; the local preset supplies that permission.

The reference recipe uses mainline llama.cpp **b12631**, revision
`e300919cc704ddde3a8a8346a8714f57eeeee71b`, with the IQ4_NL artifact from
`ggml-org/Inkling-2-Mini-GGUF`, revision
`e8a2cecf0e78914f07cc27b9be711a9b2c7561d9`. The file
`Inkling-2-Mini-IQ4_NL.gguf` is **17,627,940,768 bytes**, SHA-256
`cbdf3fa9dc50a4124977980f37da4f1fa764b0cfe800d275cd43d26ce2a81af4`.
This is a quantized deployment, with different precision from the original
checkpoint. Its speed and behavior must be measured as that deployment.

Follow the [local deployment guide](INKLING_LOCAL.md) to install the pinned
runtime and prepare the weights. The [preparation script](../scripts/inkling_local.py)
checks runtime, model digest and hardware before starting the server. Its
reference configuration binds to `127.0.0.1`, uses context 4096, one server slot,
an explicit CPU thread count, the canonical alias, the embedded Jinja template
and `--reasoning-format deepseek`. The pinned runtime recognizes Inkling's
specialized tool/reasoning parser from the template. No separate
`--tool-call-parser` flag is needed. Application tool allowlists, argument
validation and truncation checks remain in force.

On a machine with the pinned runtime and sufficient resources, start the
prepared server:

```bash
python scripts/inkling_local.py serve --server /absolute/path/to/llama-server \
  --manifest reports/testing/inkling-local-deployment.json
```

After the server is ready, run the benchmark from another shell:

```bash
uv run --no-sync python -m rags.model_benchmark run \
  --preset inkling-local \
  --deployment-manifest reports/testing/inkling-local-deployment.json \
  --output reports/inkling-mini-local-live.json
```

The manifest records the operator's runtime/build, weight repository/revision,
artifact hash and size, quantization, context, server slots, CPU threads and
hardware class. Its `source` remains `operator_supplied`, even when the
preparation script checks a file digest: the benchmark cannot prove which
weights the server actually loaded. A live loopback run without a manifest
writes `blocked / missing_deployment_manifest`, exits 2 and makes no request.
Invalid manifests are rejected without echoing their contents. The strict
schema is defined in [MODEL_BENCHMARK_SPEC.md](MODEL_BENCHMARK_SPEC.md); use
public non-secret labels, never a hostname, token, local path or endpoint URL
in a metadata field. Authenticated remote self-hosted deployments can also
supply a manifest.

With the reference **one server slot**, client concurrency 2 and 4 measures
queueing as well as inference. It does not measure simultaneous decoding of two
or four requests. The benchmark's 42-request bound, six cases, 512-token cap,
30-second request timeout and 300-second run deadline still apply. A CPU server
may exceed those limits; timeouts and truncations remain failures. Changing a
limit creates a different experiment and must be applied equally to paired runs.

For a container-to-host or other network deployment, use `--auth-mode required`
and explicitly configure the endpoint. With `--preset inkling-local`, required
authentication reads the dedicated `INKLING_LOCAL_API_KEY` or
`INKLING_LOCAL_API_KEY_FILE`; it does not fall back to `TINKER_API_KEY` or
`OPENAI_API_KEY`. Use `--key-env YOUR_LOCAL_KEY_VARIABLE` to explicitly select a
different dedicated credential name, with its optional `_FILE` companion.
The hosted preset continues to default to `TINKER_API_KEY`. In `auth_mode: none`,
no key variable or secret file is read, regardless of its configured name.
A container's loopback address belongs to that container. The no-auth preset
does not expose an unauthenticated LAN or public service.

## Run the bounded hosted benchmark

Install the project following the README, then provide a credential through the
environment or a secret file. Do not put a literal API key on the command line.

```bash
export TINKER_API_KEY_FILE=/secure/path/tinker-api-key

uv run --no-sync python -m rags.model_benchmark run \
  --output reports/inkling-mini-live.json
```

The default run uses six cases, one warmup round, two measured repetitions per
case at concurrency 1, 2, and 4, a 512-token output cap, explicit `low` reasoning
effort, a 30-second request timeout, and a 300-second overall deadline. The maximum
number of model requests is **42**, including six warmup calls. Provider failures
are not retried. Deadline-exhausted cases are recorded as not run. Ordinary CI
never needs a model key.

Without a key, the command writes an artifact with `status: blocked` and
`block_reason: missing_model_credential`, exits with status 2, and sends no
requests. It does not replace the requested model with a fixture or extractive
answer. Network or provider failures produce a failed report, not a fabricated
performance result.

For a smaller initial paid check:

```bash
uv run --no-sync python -m rags.model_benchmark run \
  --concurrency 1 --repetitions 1 --warmup-rounds 0 \
  --output-tokens 512 --reasoning-effort low \
  --max-duration-seconds 120 \
  --output reports/inkling-mini-initial.json
```

This makes at most six requests. A larger output cap may be necessary for models
that consume substantial reasoning tokens. A truncated response always fails,
even when its visible text is valid JSON. Run separate, clearly named low/high
effort experiments instead of combining their latency samples.

### Manual GitHub Actions run

The **Inkling 2 Mini live model benchmark** workflow
(`.github/workflows/model-benchmark.yml`) is manually dispatched. Configure the
repository Actions secret `TINKER_API_KEY`, select `low` or `high` reasoning
effort, and run the workflow against the intended source revision. It uses the
same production router and fixed 42-request upper bound, a 300-second benchmark
deadline, and a 10-minute job timeout. The `inkling-mini-live-model` artifact
contains the report even when the run is blocked or fails. A missing secret
causes the job to fail with an explicit blocked report. Ordinary pull-request CI
does not use credentials or make paid model requests. The key is scoped to the
benchmark step; checkout, dependency installation and artifact upload do not
receive it through their environment.

## Cases and validation

| Case | Check |
| --- | --- |
| JSON extraction | Strict integer/string schema; exact synthetic value |
| Grounded answer | Exact short answer and allowed source ID |
| Insufficient evidence | Explicit abstention and no fabricated citations |
| Untrusted instruction | Ignore a quoted instruction in source data |
| Read-only tool | Select only `knowledge_search`, with exact query/top-k |
| Tool argument boundary | No additional tenant, permission, or execution arguments |

The JSON snapshot is `examples/benchmark_cases.json`. Supply `--dataset` to run
an inspected custom dataset using the same three validation contracts. There
must be 1..12 distinct case IDs; repetitions are 1..10, warmup rounds 0..3,
concurrency values are drawn from 1/2/4, and output tokens are 1..4096.

The benchmark's answer and citation checks are exact synthetic checks. They do
not measure open-domain semantic entailment or domain safety. Tool selection
tests do not measure successful tool execution or multi-step task completion.

## Compare two models

Run both models with identical dataset, source revision, warmups, repetitions,
concurrency, output cap, effort, timeouts, and deadline. For example, compare the
two verified Inkling family identifiers on the same Tinker sampling endpoint:

```bash
uv run --no-sync python -m rags.model_benchmark run \
  --model thinkingmachines/Inkling-2-Small --protocol tinker \
  --reasoning-effort low --output reports/baseline-live.json

uv run --no-sync python -m rags.model_benchmark run \
  --model thinkingmachines/Inkling-2-Mini --protocol tinker \
  --reasoning-effort low --output reports/inkling-mini-live.json

uv run --no-sync python -m rags.model_benchmark compare \
  reports/baseline-live.json reports/inkling-mini-live.json \
  --output reports/model-comparison.json
```

A baseline must be available to your account; the runner does not assume that
another model is accessible. Other approved OpenAI-compatible gateways can use
`--model`, `--base-url`, `--key-env`, and `--protocol openai`. Use
`--reasoning-effort unset` when that profile does not support an effort control;
the comparison marks unset effort because providers may apply different defaults.
Do not describe those results as an experiment with equal effective reasoning.

The comparator rejects failed, blocked, or fixture reports, incompatible
measurement settings, different source/dataset hashes, missing samples,
duplicate samples, nonfinite metrics, and summary values inconsistent with
the recorded successful samples. Local live comparisons additionally require
valid deployment manifests whose hashes match their contents. It includes both
declared manifests and lists differing fields, as well as different endpoints,
protocols, authentication modes and client runtimes. If one side is hosted and
has no deployment manifest, deployment information is marked incomplete.
Different quantization, hardware, CPU threads, context or server slots can change
the result; a ratio does not isolate the model architecture or prove model
equivalence. Sequential requests still experience different provider load and
cache state; a ratio is not proof of statistical significance.

## Metrics and report privacy

Warmup results are separate from measured phases. Each phase includes attempted,
passed, failed, and not-run counts; strict schema and behavior checks; p50/p95
latency; successful requests divided by the full phase wall time; sanitized
provider token counts; and errors represented by stable codes.

Latency percentiles use only successful measured requests and linear
interpolation at `(n - 1) * q`; sample counts and a warning for fewer than 20
successes accompany them. Output-token throughput is end-to-end throughput over
the phase, including any reasoning tokens counted by the provider. It is not
decoder-only speed. This nonstreaming runner cannot measure TTFT.

Reports include Python/platform/package metadata, adapter/harness source hashes,
dataset/control hashes, model ID, an endpoint category and hash, and bounded
case IDs. Supplied deployment metadata and its hash are included with the
explicit `operator_supplied` label. They exclude prompts, completions, reasoning, tool arguments, endpoint
URLs, credentials, secret-file paths, provider bodies, and exception messages.
Model IDs are request labels rather than independent verification of served
weights. Report files are written atomically with restrictive permissions.

New reports use `model-contract-benchmark-v2` and `request_upper_bound`; that
count does not imply that local requests are paid calls. Comparisons use
`model-contract-comparison-v2` and reject older report schemas. Historical v1
blocked reports are retained as historical evidence without rewriting their
source identities or claiming compatibility with the new harness.

Costs stay null without an operator-supplied schedule. To estimate them, add
`--input-price`, `--output-price`, and optionally `--cached-input-price`, expressed
in USD per million tokens. Both input and output prices are required. Missing
usage prevents a complete cost estimate; known partial estimates are labeled.
Cached discounts require valid cached-token counters. These values are estimates,
not billing reconciliation.

## Offline contract tests

```bash
uv run --no-sync pytest tests/test_model_benchmark.py -q
```

The tests exercise the real router with deterministic HTTP transports and an
actual local HTTP server. Their reports explicitly use
`evidence_kind: http_contract_fixture`; their timings do not establish Inkling
speed or quality and cannot enter a live-model comparison. They also verify
missing credentials, malformed JSON, truncation, unsafe tool arguments,
nonfinite values, workload bounds, usage accounting, comparison parity, and
report secrecy.

Local contract tests also exercise a real loopback HTTP server with ambient
proxy and secret variables present. They verify that no credentials are read or
sent, OpenAI request fields are used, effort is unset, and deployment provenance
is required for live comparisons. These are transport/contract checks with
synthetic responses, not measurements of model inference.

## Current live evidence

The historical hosted attempt and current local hardware preflight establish
different prerequisites. Neither produced live model latency, throughput,
quality, cost reduction or superiority evidence. Generate passing live reports
with the commands above before making a model-performance comparison.

### Local hardware preflight on November 23, 2026

The [local readiness report](../reports/inkling-local-readiness-v09.json) records
the verified b12631 CPU executable and the available host resources. At the
recorded inspection the host had about 9.73 GiB total RAM, 6.36 GiB available,
no swap and no GPU. The selected IQ4_NL file is 17,627,940,768 bytes, about
16.42 GiB before runtime/KV memory. The preparation tool reserves a further
2 GiB as a conservative preflight allowance, not a measured peak-memory claim.
It reported a resource blocker; the weights were not downloaded and no model
inference was run.

This is separate from hosted authentication. Smaller third-party quantizations
alter numerical behavior and may require a different runtime/file format; they
are not interchangeable evidence for the pinned IQ4_NL reference. An operating
system may page memory-mapped weights, but this preflight does not treat disk
paging as a verified usable or performant deployment.

### Verified remote attempt on November 23, 2026

[Actions run 45103025522](https://github.com/MasterLKH180cm/RAGs/actions/runs/45103025522)
executed the unchanged production benchmark on an isolated verification branch.
It produced **blocked / missing_model_credential**, exited with status 2, and
uploaded its report successfully. There were zero provider requests. This
establishes that the workflow did not receive a usable Tinker credential; it
does not establish whether a credential exists elsewhere in account settings,
and does not prevent downloading or running the open weights locally.

The [downloaded remote report](../reports/inkling-mini-readiness-actions-v09.json)
matched the adapter, harness, dataset and measurement-control hashes of the
contemporaneous v1 hosted readiness run. Those historical source identities are
separate from the current v2 local harness. The downloaded ZIP was verified against its Actions
artifact SHA-256. No latency, model-quality or token-throughput result can be
computed from this blocked execution. The temporary branch-specific trigger
was removed; the maintained workflow uses manual dispatch only.
