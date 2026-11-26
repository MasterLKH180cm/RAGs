# Choose embeddings and tokenizers

RAGs lets you choose the embedding model, the counter used to split source text,
and the counter used to budget generation prompts separately. The default remains
offline hashing with extractive answers. Nothing is downloaded by discovery,
configuration planning, `init`, or `doctor`.

Run installation examples from a matching RAGs source checkout with its Python
environment activated, as in the [quick start](../README.md#quick-start). The
package is not yet published on PyPI. Wheel deployments should install the same
release's wheel with the required extras.

## Find a suitable starting point

```bash
rags models list
rags models list --language zh
rags models show st-e5-small
rags tokenizers list
rags tokenizers show huggingface
```

Add `--json` for structured output. `rags models list --runtime fastembed` filters
by execution backend. These commands work without credentials, a database or
optional model packages. The catalog is a small set of supported starting points,
not a quality ranking or a restriction on operator-configured models.

| Preset | Intended use | Runtime and installation | Vector dimensions | Complete input window |
| --- | --- | --- | --- | --- |
| `offline-hash` | Setup, API and integration checks | Base package; no download | 384 | No trained model token window |
| `bge-small-en` | English retrieval | FastEmbed; `rags-platform[encoders]` | 384 | 512 model tokens |
| `bge-small-zh` | Chinese retrieval | FastEmbed; `rags-platform[encoders]` | 512 | 512 model tokens |
| `fastembed-minilm` | Short English passages | FastEmbed; `rags-platform[encoders]` | 384 | 256 model tokens |
| `st-e5-small` | Multilingual retrieval, including Chinese and English | Sentence Transformers; `rags-platform[sentence-transformers]` | 384 | 512 model tokens |
| `st-minilm` | Short English passages with the pinned PyTorch runtime | Sentence Transformers; `rags-platform[sentence-transformers]` | 384 | 256 model tokens |
| `ollama-nomic` | An existing local Ollama installation | Separate Ollama server and `ollama pull nomic-embed-text` | 768 by model default | Checked by the server; no local exact counter claimed |
| `openai-small` | Hosted multilingual embeddings | Base package; operator supplies credentials | 1536 by model default | 8191 model tokens |

The language and model-window descriptions come from the official model cards
linked in [Sources](#sources). Evaluate your own corpus, especially Traditional
Chinese, mixed languages, code and specialist vocabulary. Local models avoid a
per-request hosted inference bill but still use your CPU/GPU, memory, disk and
download bandwidth. Hosted providers charge according to their own current
pricing. No latency or accuracy advantage is implied by this table.

FastEmbed loads ONNX artifacts on first use unless already cached. Sentence
Transformers uses a larger optional PyTorch environment and immutable Hub
revisions. Its adapter supports standard Transformer → Pooling with an optional
Normalize module, safetensors and no remote Python code. A model requiring custom remote modules
is not automatically compatible. A model with the same name in two runtimes is
not assumed to produce interchangeable vectors.

## Start a new deployment

For Chinese retrieval with the existing offline answer mode:

```bash
python -m pip install -e '.[encoders]'
mkdir chinese-rag
rags init --profile offline --embedding bge-small-zh --directory chinese-rag
rags doctor --env-file chinese-rag/.env --no-db
rags serve --env-file chinese-rag/.env
```

For multilingual retrieval, replace the extra with `sentence-transformers` and
the preset with `st-e5-small`. The E5 preset sets `query: ` and `passage: `
prefixes, including for non-English text, and pins its model revision. These
prefixes count toward the embedding input budget.

The deployment `--profile` controls generation, storage and other operational
defaults. `--embedding` overrides the embedding choice only. For example,
`--profile openai --embedding bge-small-zh` uses local Chinese embeddings and
the profile's hosted generation settings. Supply generation credentials through
the environment or configured secret files; no secret is a selection argument.

New selected model profiles generate consistent embedding feature permissions.
They choose an automatic source counter and conservative initial chunk budgets:
at most 384 units, or a smaller known encoder window with a 32-token reserve,
and overlap at most 64 units. Existing calls without new selection options keep
their previous defaults. Complete input, including instructions and generated
context, is still validated before encoding; a reserve does not guarantee that
arbitrary added context fits.

For an explicit terminal wizard, use:

```bash
rags init --interactive --directory chinese-rag
```

Use a fresh empty configuration directory. Interactive mode requires both input
and output to be terminals, asks you to select and confirm a model, and creates
no files on cancellation. Scripts should use explicit flags.

## Understand the three counting roles

| Role | Setting or CLI flag | What it controls |
| --- | --- | --- |
| Source chunking | `RAGS_TOKENIZER_JSON`, `--source-tokenizer` | How original text is split and overlapped, while retaining original character offsets. |
| Embedding input validation | The embedding provider and its input contract | Whether the complete query/document, including prefixes and special tokens, fits the actual encoder. It is checked even when source chunking uses a different counter. |
| Generation prompt budget | `RAGS_GENERATION_TOKENIZER_JSON`, `--generation-tokenizer` | How much raw prompt text can be assembled for answer generation. Chat/tool framing needs its own allowance. |

Available source selections are `auto`, `embedding`, `unicode`, `utf8_bytes`,
`cl100k_base`, `o200k_base`, `huggingface:ALIAS` and `sentencepiece:ALIAS`.
Generation accepts the same explicit text counters except `auto` and `embedding`.
Omit a flag to retain its current setting; selecting an embedding preset
implicitly chooses `--source-tokenizer auto` unless you specify another source
counter.

`auto` resolves to the configured embedding counter when an exact native or
verified hosted counter is available. Otherwise it selects conservative UTF-8
bytes. Unicode counts characters; byte counts are conservative units; neither
claims to be exact model tokens. Native Ollama sends `truncate:false`, so the
server rejects overflow instead of silently shortening input. That is a different
guarantee from exact local counting.

### Use your own tokenizer artifact

Install the extra with `python -m pip install -e '.[tokenizers]'`. Prepare an operator-owned JSON file such as
`tokenizers.json`, replacing the example path and digest with a real local file
and its SHA-256:

```json
{
  "my-counter": {
    "backend": "huggingface",
    "path": "/absolute/operator/path/tokenizer.json",
    "sha256": "REPLACE_WITH_64_LOWERCASE_HEX_CHARACTERS"
  }
}
```

Use `backend:"sentencepiece"` for a serialized SentencePiece `.model`. The
configuration command reads the registry without loading the artifact:

```bash
rags config plan --env-file .env \
  --tokenizer-registry tokenizers.json \
  --source-tokenizer huggingface:my-counter
rags config write --env-file .env \
  --tokenizer-registry tokenizers.json \
  --source-tokenizer huggingface:my-counter --output .env.next
```

The same flags work with `init`. Runtime loading reads one bounded regular file,
checks its digest, and constructs the counter from those same bytes. It does
not follow artifact symlinks or download a substitute. Per-document requests
select a registered alias; they cannot name a host file. Raw-text counting
disables truncation, padding, sampling/dropout and automatic special-token
insertion. Source slices stay tied to the original text even if the tokenizer
normalizes it.

Preserve an alias's artifact and digest while historical processing profiles
refer to it. Register a new alias and make an explicit new processing selection
when changing an artifact. Rebinding a name does not make old indexes compatible.

## Plan a change without overwriting configuration

In a fresh terminal, activate the intended Python environment. Exported process
variables take precedence over `--env-file`; a plan lists conflicting variable
names without printing their values.

```bash
rags config plan --env-file .env --embedding bge-small-zh --json
rags config write --env-file .env --embedding bge-small-zh --output .env.next
rags doctor --env-file .env.next --no-db
```

`plan` is read-only. `write` creates a new file with mode `0600`, refuses an
existing output or symlink, and preserves the original configuration, unrelated
assignments and credentials. It never edits feature files. Existing explicit
feature restrictions remain effective: a blocked plan lists the required
permissions and refuses to write. Review the operator's feature policy before
making an explicit permission change, or initialize an independently configured
deployment with the selected model.

Explicit existing chunk settings are preserved. If a new exact encoder has a
smaller window than that budget, supply a deliberate adjustment, for example:

```bash
rags config plan --env-file .env --embedding st-minilm \
  --chunk-tokens 224 --chunk-overlap 28
```

Planning validates selection structure and permissions; it does not prove
endpoint reachability, artifact integrity, model compatibility or quality.
`doctor` distinguishes installed dependencies, registered/present artifacts and
configured policies, and explicitly says runtime validation has not run. Restart
with the reviewed candidate, then perform a real ingest/query check before
switching application traffic. Do not treat a successful plan or doctor check as
a performance benchmark.

The admin **Settings** page displays the current model and both text counters,
offers the shared catalog and registered aliases, and generates equivalent plan,
write and doctor commands. Preview changes do not alter the running service.

## Configure another model or endpoint

Use `--embedding-config model.json` instead of a preset. The file contains one
embedding contract with fields accepted by the operator profile registry. For a
Sentence Transformers model, provide a supported Hub repository and its exact
40-character revision. For example, the pinned MiniLM contract is:

```json
{
  "provider": "sentence_transformers",
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
  "normalize": true,
  "input_policy": "exact"
}
```

```bash
rags init --embedding-config model.json --chunk-tokens 224 --chunk-overlap 28
```

For a custom compatible hosted endpoint, specify `provider:"openai"`, `model`,
`base_url`, and a verified counter contract when the model has no known built-in
window. The counter requires `tokenizer`, `token_limit`, and `special_tokens`:

```json
{
  "provider": "openai",
  "model": "your-verified-encoder",
  "base_url": "https://models.example/v1",
  "input_policy": "exact",
  "tokenizer": {"backend": "huggingface", "model": "my-counter"},
  "token_limit": 512,
  "special_tokens": 2
}
```

The example's token limit and special-token count must be replaced with the
actual encoder's contract. Explicitly declare `special_tokens:0` only if verified
for that endpoint. A local counter artifact alone does not establish how a remote
encoder inserts BOS/EOS or preprocesses instructions. Supply direct credentials
through `RAGS_EMBEDDING_API_KEY` or its supported secret-file setting; do not put
credentials in `model.json`. `api_key_env` is reserved for approved profile
registries, where the server resolves that reference. Direct no-auth is an
explicit policy permitted only for validated loopback endpoints.

Other supported contract fields include `dimensions` where supported,
`query_prefix`, `document_prefix`, `normalize` and `space_id`. The native Ollama
adapter supports `input_policy:"server_reject"`; arbitrary compatible endpoints
cannot claim that policy. Unsupported combinations fail explicitly. An Ollama
tag or self-hosted alias that can change its weights needs an operator-managed
`space_id` change and re-embedding when its underlying model changes.

## Let collection users choose approved models

The deployment operator registers bounded aliases in
`RAGS_EMBEDDING_PROFILES_JSON`. For example:

```json
{
  "chinese-small": {"provider":"fastembed","model":"BAAI/bge-small-zh-v1.5"},
  "english-small": {"provider":"fastembed","model":"BAAI/bge-small-en-v1.5"}
}
```

The runtime, model artifacts and corresponding feature permissions must be
available. Authenticated discovery exposes safe public metadata:

```python
import os
from rags import HttpClient

with HttpClient(os.environ["RAGS_URL"], api_key=os.environ["RAGS_API_KEY"]) as client:
    print(client.list_models())
    print(client.list_tokenizers())
    collection = client.create_collection("chinese-notes", embedding_profile="chinese-small")
    collection.add(ids=["backup"], documents=["備份每天執行，並且加密保存。"])
    print(collection.query(query_texts=["備份如何保存？"], n_results=1))
```

The HTTP routes are `GET /v1/models` and `GET /v1/tokenizers`. The CLI equivalent
for creation is `rags collection create chinese-notes --embedding-profile chinese-small`.
Collection callers choose an approved alias, never an endpoint, credential or
host path. An alias is resolved after authorization, and its effective vector
identity is bound to the collection. Rebinding an alias cannot silently change
the meaning of stored vectors.

### Move existing records into another vector space

Changing embedding models, revisions, dimensions, normalization or input
instructions requires newly generated vectors. A collection fork preserves its
old vectors and binding. To copy stored text and re-embed it into a fresh target:

```python
import os
from rags import HttpClient
from rags.client import ReembeddingError

with HttpClient(os.environ["RAGS_URL"], api_key=os.environ["RAGS_API_KEY"]) as client:
    source = client.get_collection("english-notes")
    try:
        progress = source.reembed("multilingual-notes", embedding_profile="multilingual-approved",
                                  batch_size=100, expected_generation=source.to_dict()["generation"])
        print(progress)
    except ReembeddingError as exc:
        print(exc.progress)  # Retained target ID/generation and copied record count.
        raise
```

The corresponding command is:

```bash
rags collection reembed english-notes --name multilingual-notes \
  --embedding-profile multilingual-approved --batch-size 100 --expected-generation 7
```

Use the source's actual current generation instead of the example `7`, and an
alias registered by your operator. This is a bounded batch workflow, not one
transaction spanning an entire collection. It checks source and target
generations, preserves document text, metadata and configured sparse vectors,
and never overwrites the source or an existing target name. Records must have
stored text to generate new embeddings.

On failure, a partially populated target remains available for inspection.
The progress report records the last acknowledged generation and copied count.
After a transport failure, a server commit may have completed without a response;
refresh the target to inspect its actual generation, count and documents.
Do not route searches to that target until the workflow completed and
you validated its results. If the source changed, start a fresh target to obtain
a consistent copy. Delete an abandoned target only after review, using its
current expected generation; no cleanup automatically deletes concurrent work.

Document indexes use explicit reindexing with current revision checks; see
[ingestion and reindexing](README_REFERENCE.md#ingest-and-query). Changing only
the generation tokenizer adjusts prompt budgets without re-embedding vectors.

## Validate behavior and performance

After the runtime is installed and the service restarted, use a new small
collection with representative text, queries and expected matches. Verify input
overflow errors, language handling, matching dimensions and result quality.
Record first-use download/load time separately from warmed requests. Report the
hardware, model revision, provider, chunk sizes, corpus, concurrency, failures
and measured latency percentiles alongside retrieval metrics. Hosted fixture
tests or a dependency check are not live model performance measurements.

### Recorded local-model validation

The [selection validation report](testing-results-embedding-selection.json)
records the actual five pretrained local presets, complete-input rejection,
source-preserving ingest/reindex, artifact revisions and file hashes, and the
full application test results. The latency smoke used a shared x86-64 CPU host,
Python 3.12, two configured OpenMP/MKL threads, and ten warm samples per role:

| Preset | Document median | Query median | Query p95 |
| --- | ---: | ---: | ---: |
| `bge-small-en` | 37.714 ms | 32.896 ms | 47.450 ms |
| `bge-small-zh` | 27.946 ms | 19.817 ms | 70.212 ms |
| `fastembed-minilm` | 4.522 ms | 3.841 ms | 5.552 ms |
| `st-e5-small` | 25.109 ms | 21.272 ms | 26.796 ms |
| `st-minilm` | 11.626 ms | 9.554 ms | 13.226 ms |

These figures include adapter input validation. They do not measure retrieval
quality, memory use, concurrent capacity or a production latency guarantee.
With ten samples, this estimator's p95 is the maximum. Ollama transport tests
use a real HTTP fixture process; they do not measure live Ollama model inference.

The `model-selection-runtime` CI job provisions weights with
[`prepare_embedding_models.py`](../scripts/prepare_embedding_models.py), then
runs the actual model tests and
[`benchmark_embedding_selection.py`](../scripts/benchmark_embedding_selection.py)
inside an outbound-network-free Linux namespace. This separates downloads from
inference and blocks runtime telemetry at the OS boundary. Sentence Transformers
uses pinned revisions; FastEmbed's resolved revisions and file hashes are
recorded in the artifact manifest. Running the benchmark script by itself does
not create that network boundary.

The [benchmark contract](MODEL_BENCHMARK_SPEC.md) describes reproducible model
measurements. The frozen [Qwen admission results](README_REFERENCE.md#evidence-admission-and-refusal)
remain historical and unchanged; adding selectable models does not promote a
failed quality profile.

## Sources

- [FastEmbed supported models](https://qdrant.github.io/fastembed/examples/Supported_Models/)
- [BGE small English model card](https://huggingface.co/BAAI/bge-small-en-v1.5)
- [BGE small Chinese model card](https://huggingface.co/BAAI/bge-small-zh-v1.5)
- [Multilingual E5 small model card](https://huggingface.co/intfloat/multilingual-e5-small)
  and [pinned revision](https://huggingface.co/intfloat/multilingual-e5-small/tree/614241f622f53c4eeff9890bdc4f31cfecc418b3)
- [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
  and [pinned revision](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41)
- [Ollama embedding API and overflow behavior](https://docs.ollama.com/api/embed)
  and [Nomic Embed Text](https://ollama.com/library/nomic-embed-text)
- [Nomic model instructions](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- [OpenAI embedding documentation](https://platform.openai.com/docs/guides/embeddings)
