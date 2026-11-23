# Inkling 2 Mini model testing

The model's official name is **Inkling 2 Mini**. The default benchmark target is
`thinkingmachines/Inkling-2-Mini`, served through Tinker's OpenAI-compatible
sampling API. The existing embedding provider, vector index, and PostgreSQL
configuration remain independent of this generation-model choice.

This runner measures a **small synthetic contract/behavior dataset** through the
same `ModelRouter` used by RAGs agents and routed generation. It is not a full RAG
retrieval benchmark, an agent workflow benchmark, or a general model-quality
ranking. Database throughput and the real RAG-to-agent workflow have separate
tests. The benchmark never executes model-selected tools.

## Verified provider requirements

Official references, checked November 23, 2026:

- [Model card](https://thinkingmachines.ai/model-card/inkling-2-mini/)
- [Canonical model checkpoint](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
- [OpenAI-compatible endpoint and authentication](https://tinker-docs.thinkingmachines.ai/inference/openai/overview/)
- [Chat request and response contract](https://tinker-docs.thinkingmachines.ai/inference/openai/chat-completions/)
- [Model IDs and pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/)

The test endpoint is
`https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1`.
It authenticates using a Tinker API key. The separately documented production
endpoint has distinct access/model requirements and is not assumed available.

Use `max_completion_tokens` for the output cap. Tinker's documentation says it
ignores unsupported request fields, including `response_format`; sending a JSON
Schema does not establish constrained output. The application validates JSON,
citations, and tool arguments after generation. Reasoning stays separate from
the final answer. The runner never records reasoning text.

## Run the bounded live benchmark

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
number of provider calls is **42**, including six warmup calls. Provider failures
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
does not use credentials or make paid model requests.

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
the recorded successful samples. It discloses different endpoints/client
runtimes. Sequential requests still experience different provider load and
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
case IDs. They exclude prompts, completions, reasoning, tool arguments, endpoint
URLs, credentials, secret-file paths, provider bodies, and exception messages.
Model IDs are request labels rather than independent verification of served
weights. Report files are written atomically with restrictive permissions.

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

## Current live evidence

If this workspace has no model credential or reachable deployment, only the
blocked artifact and offline contract tests are available. No throughput,
latency, quality, cost reduction, or superiority claim can be drawn from a
blocked artifact. Generate passing live reports with the commands above before
making a model-performance comparison.
