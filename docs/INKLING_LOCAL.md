# Run downloadable Inkling 2 Mini locally

Inkling 2 Mini is released with **Apache-2.0 open weights**. You can download and
run it on your own hardware without a Tinker API key. The hosted `inkling`
profile remains available; `inkling-local` explicitly selects a self-hosted
OpenAI-compatible endpoint and independent local authentication settings.

This guide pins a **Linux/WSL x86_64 CPU reference deployment** for text, native
tool selection and reasoning separation. It uses a quantized GGUF model. It does
not establish native-checkpoint performance, GPU performance, or vision support.

## Choose hardware and precision

The publisher distributes a roughly 20 GB mixed NVFP4/MXFP8 checkpoint and
documents Apple silicon with at least 32 GB memory, DGX Spark, and other GPUs
with 32 GB or more VRAM recommended. Those are hardware options, not a
requirement to use the hosted API.

The pinned CPU recipe uses **IQ4_NL, 17,627,940,768 bytes** of model weights.
Provision sufficient RAM for weights, runtime and KV cache as well as the OS;
a 32 GB system is a practical starting point. The preparation script requires
available CPU RAM at least the file size plus a 2 GiB reserve and free storage
at least the file size plus 256 MiB. This is a conservative preparation policy,
not an empirically validated universal memory minimum. It does not count GPU
VRAM as CPU RAM or force swapping when the check fails.

Other quantizations are separate experiments. Some Unsloth files use a different
looped-layer layout and require the corresponding Unsloth runtime build; their
`Mainline_Support` files target upstream llama.cpp. Do not mix runtime and GGUF
layouts. A smaller 1/2-bit download does not establish identical model quality.

Primary references checked November 23, 2026:

- [Publisher model card and license](https://thinkingmachines.ai/model-card/inkling-2-mini/)
- [Original checkpoint](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
- [llama.cpp-maintained GGUF model](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF)
- [Unsloth precision/runtime compatibility guide](https://unsloth.ai/docs/models/inkling-2-mini.md)

## Pinned artifacts

| Item | Pin |
| --- | --- |
| Runtime | llama.cpp `b12631` |
| Runtime source | `e300919cc704ddde3a8a8346a8714f57eeeee71b` |
| Linux CPU archive | `llama-b12631-bin-ubuntu-x64.tar.gz`, 23,225,891 bytes |
| Archive SHA-256 | `26fbc713d49bea58c826160f3ce0e75bff1d8df8df544823fd40ff329db6155e` |
| Model repository | `ggml-org/Inkling-2-Mini-GGUF` |
| Model revision | `e8a2cecf0e78914f07cc27b9be711a9b2c7561d9` |
| Model file | `Inkling-2-Mini-IQ4_NL.gguf` |
| Model SHA-256 | `cbdf3fa9dc50a4124977980f37da4f1fa764b0cfe800d275cd43d26ce2a81af4` |

The runtime archive digest comes from its
[official release](https://github.com/ggml-org/llama.cpp/releases/tag/b12631).
The model digest and size come from the pinned repository's
[LFS pointer](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF/raw/e8a2cecf0e78914f07cc27b9be711a9b2c7561d9/Inkling-2-Mini-IQ4_NL.gguf).
Do not replace these pins with a generic `latest` release: Inkling support is
new, and an older stable runtime may not understand its looped architecture.

## 1. Install the verified CPU runtime

Run in Bash on Linux or WSL. These commands download the approximately 23 MB
runtime archive, verify it, then extract it into a versioned cache directory.
The command stops on a failed checksum.

```bash
set -euo pipefail
INKLING_RUNTIME_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/rags/llama-b12631-runtime"
mkdir -p "$INKLING_RUNTIME_DIR"
curl --fail --location --proto '=https' --proto-redir '=https' --max-time 120 \
  'https://github.com/ggml-org/llama.cpp/releases/download/b12631/llama-b12631-bin-ubuntu-x64.tar.gz' \
  --output "$INKLING_RUNTIME_DIR/runtime.tar.gz"
printf '%s  %s\n' \
  '26fbc713d49bea58c826160f3ce0e75bff1d8df8df544823fd40ff329db6155e' \
  "$INKLING_RUNTIME_DIR/runtime.tar.gz" | sha256sum --check -
tar -xzf "$INKLING_RUNTIME_DIR/runtime.tar.gz" -C "$INKLING_RUNTIME_DIR"
export INKLING_SERVER="$INKLING_RUNTIME_DIR/llama-b12631/llama-server"
"$INKLING_SERVER" --version
```

Expected version: build `12631`, commit `e300919cc`. The executable was actually
downloaded, digest-checked and launched with `--version` and `--help` in the
validation environment. Loading the model is a separate step and was not
completed on that resource-constrained host.

## 2. Check resources and download the weights

From the RAGs checkout, after installing the project dependencies:

```bash
uv run --no-sync python scripts/inkling_local.py check \
  --server "$INKLING_SERVER" || test "$?" -eq 2

uv run --no-sync python scripts/inkling_local.py download \
  --server "$INKLING_SERVER"
```

`check` performs no network request or model inference. Exit status 2 means a
documented blocker, including missing weights, insufficient available RAM,
insufficient disk space, or a missing/mismatched runtime. Before the first
download, `model_weights_missing` is expected; address resource/runtime blockers
before running `download`.

`download` checks resources before making a large download, uses the exact model
revision, verifies size and SHA-256, and publishes the file only after validation.
It preserves an existing mismatched file and removes its own incomplete download.
No Hugging Face or Tinker credential is needed for these public artifacts.

Weights default to `~/.cache/rags/inkling-iq4`. To choose another filesystem,
pass the same `--cache-dir /path/to/model-cache` to `check`, `download` and
`serve`. The resource check measures free space on the selected cache filesystem.
Keep model weights and runtime archives outside Git.

## 3. Start the local model server

```bash
uv run --no-sync python scripts/inkling_local.py serve \
  --server "$INKLING_SERVER" --threads 8 \
  --manifest reports/testing/inkling-local-deployment.json
```

The helper validates the pinned weights/runtime, writes a deployment manifest,
then starts `llama-server` with:

- Literal loopback binding `127.0.0.1:8080` and canonical served model alias
  `thinkingmachines/Inkling-2-Mini`.
- CPU execution, 4,096 total context tokens, one server slot, and an explicit
  thread count. Automatic memory-fitting changes and device offload are disabled.
- Embedded Jinja chat template and `--reasoning-format deepseek`, which returns
  private reasoning in `reasoning_content`. The pinned mainline runtime includes
  the Inkling tool parser; do not add a vLLM/SGLang-specific tool-parser argument.
- Web UI disabled, and ambient `LLAMA_ARG_*` overrides removed so they cannot
  silently alter the reference launch settings.

The manifest describes the requested deployment. Its source is deliberately
`operator_supplied`: file verification and a declared configuration do not prove
which weights a separately running HTTP service actually loaded.

In another terminal, wait for the server to finish loading:

```bash
curl --fail --max-time 5 http://127.0.0.1:8080/health
```

## 4. Connect RAGs and Agentic

Create a fresh RAGs configuration using `rags init --profile inkling-local`.
The packaged profile sets the direct generation endpoint and agent router to
`http://127.0.0.1:8080/v1`, explicitly selects OpenAI wire semantics, leaves
reasoning effort unset and uses `auth_mode=none`. It does not download a model
or claim that a server is running. Existing embedding configuration is separate.

Unauthenticated mode is restricted to explicitly configured literal loopback
endpoints, emits no Authorization header, and reads no ambient hosted-model key.
The default authentication mode for other profiles remains `required`. RAGs
tenant API credentials, tool authorization and human approval remain separate.

Agentic provides an `inkling-local` JSON route preset; use its
`examples/inkling-local.routes.json` and local deployment README section. It sends
validated JSON planning/synthesis requests, while the RAGs native agent has a
separate tool-calling contract.

If clients run inside Docker containers, their `127.0.0.1` is the container
itself. For a host/service endpoint, use required authentication with a dedicated
local-server credential and an explicit reachable URL. The strict loopback
unauthenticated mode does not treat private LAN addresses or service names as
equivalent to loopback.

## 5. Measure through the production router

With the server healthy and the generated manifest available:

```bash
uv run --no-sync python -m rags.model_benchmark run \
  --preset inkling-local \
  --deployment-manifest reports/testing/inkling-local-deployment.json \
  --output reports/testing/inkling-local-live.json
```

The preset selects the local endpoint, explicit OpenAI protocol, `max_tokens`,
unset reasoning effort and no authentication. It does not require
`TINKER_API_KEY`. Missing deployment metadata, an unavailable server, a truncated
answer or invalid tool arguments cannot produce a passing live report.

The bounded benchmark retains six synthetic cases and its warmup/repetition
controls. With this one-slot server, client concurrency 2/4 measures queueing as
well as inference; it does not imply concurrent decoding. CPU runs may need the
documented request/deadline options or a smaller initial workload. Keep timeout,
output and workload settings equal when comparing runs and retain failures.
See [INKLING_BENCHMARK.md](INKLING_BENCHMARK.md) for limits and metrics.

Reports bind the manifest's runtime/model revisions, artifact hash, quantization,
context, server slots/threads and hardware. They explicitly disclose deployment
differences. Quantized CPU measurements must retain that label; they cannot be
presented as hosted Tinker or original-checkpoint performance.

## Current execution evidence

The [local readiness report](../reports/inkling-local-readiness-v09.json) records
about 9.73 GiB total RAM, about 6.36 GiB available at inspection, the successfully
launched pinned CPU runtime, and missing model weights. Preparation is blocked
by available memory for this reference recipe. The 17.63 GB model was not
downloaded into an environment that cannot hold it with runtime overhead.

This is hardware preflight evidence with zero model requests, not model latency
or quality evidence. The earlier `missing_model_credential` reports describe
only the hosted Tinker attempts. Neither those reports nor this local hardware
limit makes a Tinker key mandatory for self-hosted use.

For GPU deployment, follow the model-specific
[vLLM recipe](https://recipes.vllm.ai/thinkingmachines/Inkling-2-Mini) or
[SGLang recipe](https://docs.sglang.io/cookbook/autoregressive/ThinkingMachines/Inkling-2-Mini),
including their supported runtime build, device and parser settings. Those GPU
paths are documented upstream and were not executed in this CPU-only environment.
