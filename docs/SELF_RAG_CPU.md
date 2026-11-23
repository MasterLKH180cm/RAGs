# Trained Self-RAG CPU execution

The `self_rag` research adapter executes the actual trained
`selfrag/selfrag_llama2_7b` checkpoint on a CPU, selects an authorized passage
using its reflection-token probabilities, and retains that checkpoint's answer.
It is a bounded, quantized port of the official short-form inference algorithm.
It does not use the ordinary chat provider to imitate Self-RAG reflection.

The implementation contract is [RRC-03](RESEARCH_COMPLETION_SPEC.md). Its code is
in [the adapter](../rags/research/selfrag_adapter.py),
[the native runtime](../rags/research/selfrag_runtime.py), and
[the artifact provisioner](../scripts/selfrag_local.py).

## Pinned identities

| Component | Exact identity |
| --- | --- |
| Official algorithm | [AkariAsai/self-rag](https://github.com/AkariAsai/self-rag/blob/1fcdc420e48f50a7d7ab1ece5494221b93252e99/retrieval_lm/run_short_form.py), commit `1fcdc420e48f50a7d7ab1ece5494221b93252e99` |
| Official trained weights and tokenizer | [selfrag/selfrag_llama2_7b](https://huggingface.co/selfrag/selfrag_llama2_7b/tree/190261383b0779ff66d2f95a73c7ad267d94b820), revision `190261383b0779ff66d2f95a73c7ad267d94b820` |
| Native runtime and converter | [llama.cpp b12631](https://github.com/ggml-org/llama.cpp/tree/e300919cc704ddde3a8a8346a8714f57eeeee71b), commit `e300919cc704ddde3a8a8346a8714f57eeeee71b` |
| Supported native platform | Linux x86_64; CPU only |
| Served model | `selfrag_llama2_7b.Q4_K_M.gguf`, 4,081,095,936 bytes |
| Served model SHA-256 | `e6bcabadccb962ddda7f1899f1a684cb3f84dd98c07778f034d6b43d0f245e85` |
| Adapter metadata | `self-rag`, `git-1fcdc42`, adapter `v10` |
| Request/result protocol | `authorized-evidence-v2` |

Both original weight shards, every tokenizer/config file, the converter archive,
the corrected F16 intermediate, the quantizer, and all application-loaded native
libraries have fixed digests in the provisioner/runtime. The manifest records
these identities. A successful preparation is artifact evidence; its
`model_inference_executed` field deliberately remains `false`.

### Reflection-token metadata correction

The pinned llama.cpp converter treats a vocabulary of size 32,016 as CodeLlama
and writes `tokenizer.ggml.eot_token_id=32010`. In this checkpoint, ID 32010 is
`[Utility:4]`. Stopping on that ID would truncate a valid Self-RAG reflection
sequence. The provisioner changes that one GGUF metadata value to the official
EOS ID `2` before quantization. It preserves every tensor and token ID, then
verifies the corrected F16 and final Q4 artifacts against fixed hashes.

At model initialization, the runtime checks the spelling and tokenization of
all 16 added tokens (IDs 32000–32015), verifies that none is an end-of-generation
token, and verifies EOS ID 2. The recorded negative-source diagnostic ends with
`[Utility:4]</s>`, exercising the previously incorrect stop-token case.

## Inference and source contract

The mode is always-retrieve, open-answer, short-form inference. Every authorized
input passage is evaluated independently with the reference instruction prompt
and `[Retrieval]<paragraph>…</paragraph>` framing. Its exact text is retained;
the optional reference title is empty because this boundary supplies passages.
Greedy decoding uses the complete 32,016-token logit vector, and each output step
retains the raw logits for all reflection controls.

The reference reward is retained:

```text
reward = relevance + support + 0.5 * utility
relevance = P([Relevant] | relevance controls)
support = P([Fully supported] | support controls)
          + 0.5 * P([Partially supported] | support controls)
utility = sum(weight[i] * P([Utility:i] | utility controls))
weight = [-1, -0.5, 0, 0.5, 1]
```

Probabilities are a stable softmax within each token group, computed from raw
logits at the corresponding output position. This is mathematically the same
conditional normalization as exponentiating full-vocabulary log probabilities
and dividing within the group. The sequence-likelihood term is disabled. An
absent optional support or utility decision contributes zero, as in the
reference short-form scoring function. The result records how many decisions
were present.

The following are explicit, stricter admission rules for this port, beyond the
reference's weighted ranking:

- An `[Irrelevant]` first decision, a `[No Retrieval]` continuation, or an explicit
  `[No support / Contradictory]` decision makes the **whole answer ineligible**.
  No answer fragment is repaired into an apparently grounded response.
- Partial support keeps the reference weight; it does not establish semantic
  truth. The public response does not infer `supported=true` from a reflection
  score or a source citation.
- The output must finish with EOS. Invalid ordering, repeated critique decisions,
  non-finite logits, unsupported multi-hop controls, invalid UTF-8, and output
  exhaustion produce a structured error. No incomplete generation is published.
- If every candidate abstains, the result contains no hits or trained answer.
  The main API returns its standard insufficient-evidence response and never
  invokes an ordinary generator to replace the refusal.

The highest-scoring eligible answer wins, with its original source ID as a stable
tie-break. This short-form mode returns one source even when `top_k` is larger.
`grounded_answer = {text, cited_ids}` preserves the trained plain text and exact
authorized ID. The outer RAG pipeline validates both against the original
snapshot, rechecks admission and complete source packing, and appends original
citations only after those checks. Source text, source ID, document revision and
tenant association do not come from model-generated labels.

### Execution bounds

| Bound | Behavior |
| --- | --- |
| Authorized passages | At most 8; a larger supplied snapshot is rejected, never silently sliced |
| Complete prompt per passage | At most 1,536 checkpoint tokens, including question and framing |
| Generated tokens per passage | At most 128, including reflection tokens and EOS |
| Native context | Full longest prompt + 128 output tokens, rounded up to 256; maximum 1,792, one sequence |
| Native threads | Operator setting 1–8, default 4 |
| Weight loading | `LLAMA_LOAD_MODE_NONE=0`, `LLAMA_LAZY_MODE_OFF=0`; full tensor bytes in owned CPU buffers |
| Model ownership | One active model per worker process; released after each request |
| Request deadline | Fixed research worker process is terminated and its directory removed on timeout |

The entire snapshot is tokenized and checked before allocating the KV context or
running the first inference. The allocation includes the checkpoint's BOS token
and the entire 128-token output allowance; a 1,536-token prompt still receives
1,792 context positions. A 37-token prompt needs only 256 positions. RoPE
parameters and native default KV precision remain unchanged. Source
text is never truncated. Literal reserved checkpoint control tokens in questions
or passages are rejected so source text cannot inject structural markers. This
check does not claim to solve arbitrary natural-language prompt injection.

The native loader verifies artifacts before loading them, loads the verified
`ggml` dependencies by absolute path, and selects a compatible backend from the
verified CPU libraries. It does not enumerate arbitrary native plugin paths.
The operating system and its standard C/C++/OpenMP libraries remain part of the
operator's trusted environment. Requests cannot select a model, library path,
conversion command or alternate provider.

Weight loading is explicit because the native default `AUTO` selected a
file-backed mapping. A controlled real-API diagnostic on the test filesystem
observed millions of major page faults during a short answer and exceeded the
120-second service deadline. The corrected runtime reads the same verified tensor
bytes into owned CPU buffers and disables lazy reads. This changes loading and
memory ownership, not tensor values, tokenization, decoding or reflection scoring.
It is not `mlock`: normal operating-system swapping can still occur under memory
pressure. The worker's single-model capacity and process cleanup remain in force.

## Reproduce the model

Prepare the ordinary application environment, an isolated conversion environment,
and at least 40 GB of available workspace. Cold preparation temporarily retains
the approximately 13.5 GB official weight shards, 13.5 GB F16 intermediate, and
4.1 GB quantized artifact. The converter memory-maps the official shards; it does
not require all original weights to fit in RAM at once.

Run from the repository root. Set `SELFRAG_WORK_ROOT` to a suitable absolute path:

```bash
export SELFRAG_WORK_ROOT="$PWD/../selfrag-local"
mkdir -p "$SELFRAG_WORK_ROOT/runtime" "$SELFRAG_WORK_ROOT/model"
uv sync --locked --extra dev
uv venv --python python3.12 "$SELFRAG_WORK_ROOT/conversion"
uv pip install --python "$SELFRAG_WORK_ROOT/conversion/bin/python" \
  --torch-backend cpu -r requirements/research-selfrag-convert.txt

curl --fail --location --proto '=https' --proto-redir '=https' --max-time 120 \
  https://github.com/ggml-org/llama.cpp/releases/download/b12631/llama-b12631-bin-ubuntu-x64.tar.gz \
  --output "$SELFRAG_WORK_ROOT/runtime/runtime.tar.gz"
printf '%s  %s\n' \
  26fbc713d49bea58c826160f3ce0e75bff1d8df8df544823fd40ff329db6155e \
  "$SELFRAG_WORK_ROOT/runtime/runtime.tar.gz" | sha256sum --check -
tar -xzf "$SELFRAG_WORK_ROOT/runtime/runtime.tar.gz" \
  -C "$SELFRAG_WORK_ROOT/runtime"

export RAGS_SELFRAG_LLAMA_DIR="$SELFRAG_WORK_ROOT/runtime/llama-b12631"
export RAGS_SELFRAG_MODEL_PATH="$SELFRAG_WORK_ROOT/model/selfrag_llama2_7b.Q4_K_M.gguf"
export RAGS_SELFRAG_THREADS=4

uv run --no-sync python scripts/selfrag_local.py prepare \
  --cache-dir "$SELFRAG_WORK_ROOT/model" \
  --llama-dir "$RAGS_SELFRAG_LLAMA_DIR" \
  --python "$SELFRAG_WORK_ROOT/conversion/bin/python"
```

Preparation checks the recorded identities, downloads only the pinned official
files, converts with the pinned source, fixes the EOT metadata, and quantizes with
the pinned `Q4_K_M` quantizer. Intermediate files are atomically published only
after verification. Existing verified files are reusable. An incompatible source
or intermediate fails rather than being silently accepted.

The conversion dependencies are isolated in
[research-selfrag-convert.txt](../requirements/research-selfrag-convert.txt).
The deployed adapter requires the ordinary application and native bundle; it
does not import PyTorch or Transformers. Queries never download or convert models.

```bash
uv run --no-sync python scripts/selfrag_local.py inspect \
  --cache-dir "$SELFRAG_WORK_ROOT/model"
```

`inspect` verifies the final model identity and reports its manifest. Use the
actual gates below to verify execution.

## Service configuration

Set the three `RAGS_SELFRAG_*` variables above in the research sidecar's operator
environment. Configure `RAGS_RESEARCH_ENGINE=self_rag`, a tenant-key mapping,
and a private research work directory according to the
[research-sidecar deployment instructions](../README.md).
The main API's `RAGS_RESEARCH_BACKENDS_JSON` entry must use the `self_rag` engine
and its matching per-tenant sidecar credentials. The implementation uses protocol
v2 because the result carries the trained answer as well as selected IDs.

Allow enough deadline for full model verification, fresh model loading and all
supplied passages. The HTTP gate uses a 120-second sidecar deadline for its
single-passage synthetic requests. It also tests a deliberately short 0.1-second
deadline to prove process cleanup. Those settings are workload-specific, not a
throughput or production concurrency claim.

## Verification and interpretation

The resource-heavy tests are marked `integration`, `external` and
`research_selfrag`. Opt in explicitly; a core offline suite cannot silently count
them as trained-checkpoint evidence.

```bash
uv run --no-sync python -m pytest -q \
  tests/test_selfrag.py tests/test_selfrag_local.py --require-no-skips

RAGS_TEST_SELFRAG=1 RAGS_SELFRAG_REPORT_PATH=reports/actual-selfrag.json \
  uv run --no-sync python -m pytest -q \
  tests/test_selfrag_runtime_integration.py --require-no-skips \
  --junitxml reports/actual-selfrag.junit.xml

uv run --no-sync python scripts/e2e_selfrag_v10.py \
  --report reports/actual-selfrag-http.json
```

The direct gate uses one exact refund passage and one unrelated passage. It checks
the actual trained answer, exact source ID, completed reflection decisions,
negative-source abstention, actual token counts, and artifact identities. The HTTP
gate executes the real API, sidecar and subprocess model, then checks source
updates, historical reads, tenant isolation, deletion, refusal, authentication,
protocol validation and timeout cleanup. Its ordinary generator is extractive;
it records zero ordinary-provider calls on these queries, so it cannot replace
the trained answer with a chat-model substitute.

The checked-in reports are:

- [Actual direct execution](../reports/research_selfrag_v10.json) and
  [JUnit result](../reports/research_selfrag_v10.junit.xml).
- [Actual HTTP lifecycle](../reports/research_selfrag_http_v10.json).
- [Preparation and diagnostic history](../reports/research_selfrag_provenance_v10.json),
  including the initial strict-parser failure, raw synthetic positive and negative
  model outputs, converter metadata correction, and measured preparation costs.
- [CPU loading comparison](../reports/research_selfrag_cpu_comparison_v10.json),
  with raw before/after profiles and the retained timeout and initialization
  failures that led to the loading and KV-allocation corrections.

The final local run passed **60 unit cases, one actual checkpoint gate using two
passages, and all 11 HTTP lifecycle checks**, with no skipped cases in the selected
gates. Actual direct execution took 21.86 seconds, used four native threads and a
native-observed 256-token context, and reached approximately 4.28 GB peak process
RSS. The complete HTTP lifecycle took 77.23 seconds; its four positive queries
took 13.12–16.11 seconds each, and its actual negative-source refusal took 16.52
seconds at the sidecar. The ordinary generation provider was not called.

In the single controlled positive API diagnostic, end-to-end time changed from
296.04 to 14.16 seconds. The complete generation interval's major page faults
changed from 5,005,150 to 5; autoregressive decoding alone changed from 4,889,185
to 0. The plain answer and 37-input/16-output token counts matched. The post-fix
token IDs also matched the separately retained earlier raw positive diagnostic;
the slow before profile did not record its full token stream. File-cache state
was not forcibly equalized, so this is a scoped diagnostic comparison.

An intermediate attempt using owned buffers with the full 1,792-token KV cache
received SIGKILL during allocation. The per-process kernel OOM cause was not
available. The final runtime sizes KV from the complete actual snapshot while
retaining the full accepted input and output bounds. The ordinary final HTTP gate
passed with the original 120-second service deadline; the 300-second diagnostic
deadline was used only to measure the pre-fix slow case and its paired follow-up.

These tests establish actual implementation execution and bounded source
preservation. They do not establish paper benchmark parity, quantized/full-precision
equivalence, production load capacity, or domain/language quality. The port does
not implement adaptive retrieval decisions, long-form beam search, closed-option
answer aggregation, or training. Latencies include artifact hashing and model
loading and should not be reported as steady-state token generation throughput.
