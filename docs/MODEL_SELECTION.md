# Choose embedding models and tokenizers

Use `rags models list`, `rags models show <id>`, and `rags tokenizers list` to
explore without credentials, provider calls or weight downloads. Run
`rags configure` in a terminal for numbered choices, or use explicit options
for repeatable setup. The admin Settings page offers the same choices and a
configuration command; it does not mutate a running deployment.

## Quick choices

| Goal | Preset IDs | Optional runtime |
| --- | --- | --- |
| Learn the API entirely offline | `offline` | None; hashing demonstration |
| Compact English retrieval | `bge-small-en`, `bge-base-en` | `encoders` / FastEmbed |
| Chinese retrieval | `bge-small-zh` | `encoders` |
| Multilingual sentences or retrieval | `multilingual-minilm`, `multilingual-e5-large` | `encoders` |
| Code retrieval | `jina-code` | `encoders` |
| Nomic or Qwen embeddings | `nomic-v1.5`, `qwen3-0.6b` | `encoders` |
| Sentence Transformers | `st-minilm`, `st-mpnet`, `st-multilingual-e5-base`, `st-multilingual-e5-small`, `st-bge-m3` | `sentence-transformers` |
| OpenAI embeddings | `openai-small`, `openai-large`, `openai-ada` | Provider credential |
| Native local Ollama | `ollama-nomic`, `ollama-mxbai`, `ollama-bge-m3`, `ollama-embeddinggemma` | Running Ollama and pulled model |
| An existing approved gateway alias | `gateway` | Existing gateway authorization and model contract |

These are starting configurations, not a ranking or a claim that all models
were benchmarked. Test retrieval with your own language, document lengths and
questions. BGE-M3 here produces one dense vector, not sparse or multi-vector
output. Model licenses and deployment requirements remain model-specific; each
catalog entry links its upstream source. Other model IDs can be used through a
compatible configured provider without editing the catalog.

## Install and configure

If you followed the README's Git installation, activate the same Python
environment. You can add FastEmbed without a local source checkout or `uv`:

```bash
python -m pip install "rags-platform[encoders,tokenizers] @ git+https://github.com/MasterLKH180cm/RAGs.git"
```

For a source checkout, install [uv](https://docs.astral.sh/uv/getting-started/installation/)
and run the following in the repository root for FastEmbed:

```bash
uv sync --locked --extra encoders --extra tokenizers
```

Alternatively, install CPU Sentence Transformers without CUDA packages from
that source checkout:

```bash
uv sync --locked --extra tokenizers
uv pip install --torch-backend cpu -e '.[sentence-transformers,tokenizers]'
```

For the README installation without a checkout, install `uv`, activate the
existing Python environment, and use this CPU command instead:

```bash
uv pip install --torch-backend cpu "rags-platform[sentence-transformers,tokenizers] @ git+https://github.com/MasterLKH180cm/RAGs.git"
```

A later `uv sync` can remove extras not selected by that command; repeat
the CPU installation afterward when using this recipe. Standard installs
need no optional embedding runtime.
Installing a runtime is separate from downloading a model. First use of a
trained encoder can fetch its weights; populate `RAGS_MODEL_CACHE` beforehand
for disconnected deployments. Sentence Transformers disables remote Python
code and requires safetensors weights, using a bounded two-entry model cache.

```bash
rags configure --embedding st-multilingual-e5-base --tokenizer unicode \
  --output rags.models.env
rags doctor --env-file .env --model-config rags.models.env --no-db
rags serve --env-file .env --model-config rags.models.env
```

Create the base `.env` once with `rags init` if needed. Model overlays never
copy credentials or feature policies. **Precedence is process environment,
then model overlay, then base file.** Use a fresh terminal when replacing
previously exported selections. Optional embedding fields are explicitly reset
by each overlay so old prefixes, dimensions or token-window settings do not
silently carry over from the base file.

An offline deployment profile may disable `embedding.fastembed` or
`embedding.openai`. Explicitly enable the desired entry in the operator's
feature file. `doctor` reports disabled selected providers and missing optional
runtime/artifact files. It does not run inference, fetch tokenizer assets,
validate a provider credential online or prove model availability.

`configure` never overwrites its output. Choose a new output filename for another
selection. Files use mode `0600`; existing paths and leaf symlinks are rejected.

## Tokenizers are three independent decisions

| Setting | Purpose | CLI option |
| --- | --- | --- |
| `RAGS_TOKENIZER_JSON` | Document splitting and source offsets | `--tokenizer` |
| `RAGS_GENERATION_TOKENIZER_JSON` | Generated-answer context packing | `--generation-tokenizer` |
| `RAGS_EMBEDDING_TOKENIZER_JSON` + `RAGS_EMBEDDING_MAX_TOKENS` | Explicit custom embedding input budget | `--embedding-tokenizer` + `--embedding-max-tokens` |

The embedding tokenizer is normally model-native for FastEmbed and Sentence
Transformers. Selecting a document tokenizer does not replace the encoder's
native tokenizer. Native Ollama receives `truncate:false` and rejects an input
that exceeds its actual model window.

| Tokenizer preset | What it counts |
| --- | --- |
| `unicode` | Unicode code points; offline fallback units, not model tokens |
| `utf8-bytes` | UTF-8 bytes; explicitly conservative operational budget |
| `cl100k-base`, `o200k-base`, `p50k-base`, `r50k-base` | Exact tiktoken text encoding selected by the operator |
| `huggingface` | Exact local `tokenizer.json` text encoding |
| `sentencepiece` | Exact local SentencePiece `.model` text encoding |

The installed tiktoken library may need its encoding assets cached on first use.
The two local artifact backends never download assets or execute remote code.
Use a tokenizer artifact from the actual model deployment:

```bash
rags configure --embedding st-bge-m3 \
  --tokenizer huggingface --tokenizer-path /absolute/models/bge-m3/tokenizer.json \
  --generation-tokenizer cl100k-base --output bge.models.env
```

The generation choice above is suitable only if your generation model uses
that encoding. A matching tokenizer name alone does not account for chat
envelopes or automatic BOS/EOS tokens: keep generation output/envelope reserves.
For explicit remote embedding limits, supply the **usable text budget after
model-specific overhead**. Byte budgets are labeled as bytes, not promoted to
exact provider token counts.

Local artifacts are bounded regular files, identified by SHA-256 of the exact
bytes loaded. Saved native padding/truncation is disabled. Source windows slice
the original Unicode text, preserving citations even if normalization prevents
a lossless token decode. The runtime guarantees a fitting source window, not
the largest possible window for every tokenizer. Paths are omitted from public
tokenizer profiles. HTTP ingestion may select a local artifact only when its
configuration matches the operator's configured document tokenizer.

## Native Ollama

```bash
ollama pull bge-m3
rags configure --embedding ollama-bge-m3 --tokenizer unicode --output ollama.models.env
rags serve --env-file .env --model-config ollama.models.env
```

This embedding adapter uses `http://127.0.0.1:11434/api/embed`, with
`truncate:false`. It sends no ambient hosted key in explicit no-auth mode.
The native API is distinct from the existing OpenAI-compatible `/v1` profile.
Remove a previous dedicated embedding key/key-file selection before using
no-auth mode; conflicting credentials are rejected rather than silently ignored.
Use an explicit authenticated deployment when the endpoint is outside literal
loopback. A container's `127.0.0.1` refers to that container, not its host.

## Custom models, task roles and dimensions

```bash
rags configure --provider sentence_transformers \
  --model intfloat/multilingual-e5-base \
  --query-prefix 'query: ' --document-prefix 'passage: ' \
  --output custom.models.env
```

Pin a verified model revision with `--revision <commit>` for reproducible
Sentence Transformers deployments. For remote deployments or mutable aliases,
set `--space-id <deployment-version>`. A model name cannot establish that an
operator has not changed its weights behind the same endpoint or alias.

Presets supply E5, Nomic and Qwen task prefixes explicitly. Sentence Transformers
uses `encode_query`/`encode_document` and the appropriate model prompt when no
explicit prefix overrides it. Multiline Qwen query instructions round-trip
through quoted overlays without executing shell content.

For supported models, `--dimensions` requests an output dimension. Returned
dimensions must match the request and remain stable. Dimension reduction does
not imply that every model has been trained for reduced-dimensional retrieval.
`text-embedding-ada-002` has fixed dimensions; choose a compatible model for
dimension selection.

If an existing document chunk exceeds a newly selected model's input window,
reduce `max_tokens` and `overlap_tokens` in `RAGS_CHUNKING_JSON` and reindex.
For a compact Chinese encoder with Unicode chunking, a starting configuration
is `{"strategy":"structure","max_tokens":256,"overlap_tokens":32}`.
Actual native-tokenizer checks still decide whether every complete chunk fits;
the service reports an error instead of silently dropping source text.

Approved LiteLLM aliases can target Cohere, Voyage, Gemini or other supported
upstreams. Provider availability depends on the configured gateway. For example,
after an operator configures the `embeddings` alias to a Cohere embedding model:

```bash
rags configure --provider router --model embeddings \
  --query-input-type search_query --document-input-type search_document \
  --embedding-tokenizer huggingface \
  --embedding-tokenizer-path /absolute/models/cohere/tokenizer.json \
  --embedding-max-tokens 500 --space-id cohere-deployment-v1 \
  --output gateway.models.env
```

The `500` value is an example conservative usable budget, not a declaration of
the selected provider's maximum. Supply the matching tokenizer and verify your
deployment's full request contract. Voyage roles are typically `query` and
`document`; configure the roles supported by your selected gateway model.
The CLI does not create gateway aliases, install providers or acquire keys.
OpenAI-compatible custom services require the same explicit tokenizer/window
contract when the model is not one of the known OpenAI embedding names.

## Safely switch an existing index

1. Create a new model overlay and check required runtime, authorization and
   feature permissions. Stop the current server before using the same database
   or port with a replacement process.
2. Restart with the new selection. Model, endpoint, explicit deployment ID,
   prefixes, input types, dimensions and revision participate in vector-space
   identity where applicable. A changed identity rejects incompatible dense
   queries instead of mixing spaces.
3. Explicitly reindex each source document using its current revision. A stored
   local tokenizer artifact must still match its recorded hash, or the request
   must explicitly select the new operator-approved tokenizer configuration.
4. Create a new collection and re-add documents to generate the new vectors.
   A collection's embedding binding is immutable; there is no implicit in-place
   conversion. Keep the old collection until the new one passes your retrieval
   checks, then change callers to the new collection.

A tokenizer artifact replacement takes effect in a new provider instance after
restart; identity and counting use the same loaded artifact snapshot. This
prevents a new fingerprint from being combined with an old cached counter.

## Containers and operational limits

The generated overlay uses shell quoting understood by `rags --model-config`.
Do not pass it directly as Docker Compose's `--env-file`: the two parsers differ
for values containing apostrophes. Copy the selected values into your existing
Compose configuration using its dotenv syntax, or export them explicitly.

Standalone `compose.yaml` and the companion Agentic stack forward all embedding
selection fields. Add `encoders`, `sentence-transformers` or `tokenizers` to
`RAGS_EXTRAS` and rebuild when selecting an optional runtime. Mount tokenizer
artifacts read-only and configure paths as seen **inside the container**.
Provider key files and model artifacts are not copied into a generated overlay.

The embedding cache bounds the number of models, not resident memory. Choose
models that fit your host. Native encoding is synchronous CPU work; a configured
HTTP deadline cannot forcibly stop an arbitrary native inference kernel.
Repository validation distinguishes real local encoding from synthetic remote
transport tests and does not claim new model-quality or throughput results.
The [runtime review](EMBEDDING_RUNTIME_NOTES.md) explains the repaired defects;
the [combined local verification report](testing-results-model-selection.json)
records test counts, complete-package coverage and execution boundaries.

### Upstream contracts

- [FastEmbed supported models](https://qdrant.github.io/fastembed/examples/Supported_Models/)
- [Sentence Transformers retrieval methods](https://www.sbert.net/docs/sentence_transformer/usage/usage.html)
- [E5 model card](https://huggingface.co/intfloat/multilingual-e5-base)
- [Nomic model card](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- [Qwen3 embedding model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [Ollama native embedding contract](https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings)
- [OpenAI embedding API](https://developers.openai.com/api/reference/python/resources/embeddings/methods/create)
