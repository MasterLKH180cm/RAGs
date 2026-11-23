# Local Qwen generation and measurement contract

## Scope

Add an optional CPU deployment of the official Qwen3-1.7B Q8_0 GGUF and explicit
direct/LiteLLM benchmark presets. Preserve the existing Inkling recipes, default
extractive generation, local embedding identity, authorization, and frozen Qwen
answerability profile. This generation deployment is distinct from the original
Transformers bfloat16 YES-minus-NO scorer; success here cannot promote that profile.

Official artifact metadata was checked on 2026-11-23 at
https://huggingface.co/Qwen/Qwen3-1.7B-GGUF and its Hugging Face model API.

| Field | Fixed value |
| --- | --- |
| Repository | `Qwen/Qwen3-1.7B-GGUF` |
| Revision | `90862c4b9d2787eaed51d12237eafdfe7c5f6077` |
| Artifact | `Qwen3-1.7B-Q8_0.gguf` |
| Bytes | `1834426016` |
| SHA-256 | `061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a` |
| Runtime | llama.cpp `b12631`, revision `e300919cc704ddde3a8a8346a8714f57eeeee71b` |
| Served model | `qwen/Qwen3-1.7B` |
| Gateway model | `gateway/qwen3-1.7b-q8_0-v1` |

## Preparation and execution

1. Reuse explicit, immutable artifact descriptors for resource checks, hashing,
   downloads, and deployment manifests. Never mutate one model's globals to
   prepare another model. Existing Inkling CLI behavior remains compatible.
2. Before downloading, require Linux x86_64, enough available CPU memory for
   weights plus a 2 GiB reserve, and enough disk space plus a 256 MiB reserve.
   These are recipe limits, not measured universal hardware requirements.
3. Download only the revision-pinned artifact through validated HTTPS Hugging
   Face redirects, within a whole-operation deadline. Enforce the exact byte
   count and SHA-256 before atomic publication; reject existing mismatches and
   symlinks and remove partial files on failure.
4. Verify the pinned runtime version/revision and executable hash. Bind to
   literal loopback, use CPU only, 4096 context tokens, one parallel slot,
   1–8 threads, Jinja, reasoning off, temperature zero, seed 42, no prompt cache,
   and zero cache RAM. Remove ambient `LLAMA_ARG_*` overrides and
   `LLAMA_API_KEY` before launch so the default local authentication policy
   cannot change implicitly.
5. Readiness is hardware/artifact evidence and records zero model requests.
   Write the existing strict deployment manifest plus a separate bounded public
   execution contract for sampling, template, and cache settings before serving.
   A manifest is an operator claim, not independent HTTP weight attestation.

## Benchmark presets and interpretation

`qwen-local` uses `http://127.0.0.1:8080/v1`, explicit OpenAI-compatible protocol,
`max_tokens`, unset reasoning effort, and explicit no-auth loopback operation.
An authenticated local override reads only `QWEN_LOCAL_API_KEY` or its `_FILE`
equivalent. `qwen-litellm` requires verified HTTPS, a dedicated gateway credential,
and a complete deployment manifest before credential access or network traffic.
Neither preset reads hosted Inkling credentials or changes model-name validation.

Use the same running upstream, immutable manifests, benchmark client/runtime and
workload for direct-versus-gateway comparison. Record failures and quality gates;
do not discard failing samples to obtain a passing performance comparison.
Five repetitions of six cases give 30 measured samples per concurrency. With one
server slot, concurrency 2/4 measures queueing, not parallel decoding. Nonstreaming
measurements do not establish time to first token or decoder-only token speed.

## Acceptance

- Existing Inkling preparation regressions continue to pass.
- Qwen tests cover pinned identity, bounded resources and downloads, corruption
  and symlink rejection, fixed execution flags and ambient override removal.
- Benchmark regressions cover both new presets, dedicated credentials and
  manifest-before-secret behavior without model downloads or inference.
- Root acceptance separately runs actual Qwen generation, tool behavior and
  complete RAG/Agentic/LiteLLM workflows; preparation tests are never labeled as
  model-performance evidence.

## Commands

From the repository, with the core environment installed:

```bash
uv run --locked python scripts/qwen_local.py download \
  --server /absolute/path/to/llama-server
uv run --locked python scripts/qwen_local.py serve \
  --server /absolute/path/to/llama-server --threads 8
```

The setup tool accepts `--cache-dir`, `--output`, `--manifest`, and
`--execution-contract` for explicit locations. It rejects a symlink as the final
model file; an existing Hugging Face cache snapshot can be copied from its
resolved artifact into the selected cache directory and will be rehashed.

In another terminal, while that server is running:

```bash
uv run --locked python -m rags.model_benchmark run --preset qwen-local \
  --deployment-manifest reports/testing/qwen-local-deployment.json \
  --repetitions 5 --warmup-rounds 1 --concurrency 1 2 4 \
  --timeout-seconds 90 --max-duration-seconds 3600 \
  --output reports/testing/qwen-direct.json
```

For an authenticated local server, set the dedicated `QWEN_LOCAL_API_KEY_FILE`
and pass `--auth-mode required`. For the independently provisioned TLS gateway,
use `--preset qwen-litellm`, its approved `--base-url`, its public `--ca-file`, and
`LITELLM_BENCHMARK_API_KEY_FILE`; retain the same upstream deployment manifest
and measurement parameters. Credentials and credential paths do not belong in
published benchmark or execution reports.
