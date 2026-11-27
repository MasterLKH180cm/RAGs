# Embedding runtime changes and verification

The implementation follows [the committed specification](EMBEDDING_TOKENIZER_SPEC.md).
The existing eighteen findings in [the prior review](CODE_SMELL_REVIEW.md) were
already repaired on the starting branch. This change preserves those regression
contracts and repairs the additional boundaries below.

## Behavior

- Providers now support the existing offline hash, FastEmbed, OpenAI-compatible
  and approved gateway paths, plus optional CPU Sentence Transformers and the
  native Ollama `/api/embed` API.
- Every production embedding path validates finite, nonempty, bounded vectors
  and stable dimensions. OpenAI-compatible single-input responses must contain
  one result; a present result index must be exactly integer zero.
- Query/document prefixes, provider input types, requested dimensions, explicit
  model revision and embedding tokenizer settings are bound into a hashed model
  identity. Existing document/collection checks reject incompatible profiles
  before making model calls. Default existing model identities are retained.
  Jina v3's corrected passage task deliberately changes its identity.
- Document ingestion and collection batches validate all planned complete
  embedding inputs before the first embedding request. Research and collections
  use the same production provider counter and its accurately labeled policy.
- FastEmbed uses the actual untruncated tokenizer, including special tokens,
  for reviewed ordinary ONNX text architectures. Document calls use
  `passage_embed`; query calls use `query_embed`. Unsupported custom/static or
  multimodal architectures require a separate reviewed counting contract.
- Sentence Transformers 5.7 uses the actual text preprocessing path for prompts,
  whitespace, normalization and special tokens. Shorter declared preprocessing
  limits are honored. Counting and inference disable truncation; shared tokenizer
  use is serialized and counting restores native truncation/padding state.
  Text-only standard Transformer models are supported. Other architectures fail
  with an explicit capability error. Loading requires safetensors and disables
  remote Python code execution.
- Native Ollama always receives `truncate: false`; its server enforces the model
  window. Its default local input ceiling is a transport byte bound, not an exact
  model-token count. A sanitized HTTP 400 becomes an input/settings error.
  Provider-reported `prompt_eval_count` is retained as usage.
- OpenAI's three recognized embedding model families retain their explicit
  `cl100k_base`/8,191-token contract. Other compatible/gateway models require an
  operator-declared tokenizer and usable text budget. Local tokenizer artifacts
  count text without automatic model envelope/BOS/EOS overhead: reserve that
  overhead when choosing the usable budget. UTF-8 budgets are labeled as byte
  budgets and do not claim universal tokenizer equivalence.
- No-authentication embedding requests require an explicit literal-loopback
  endpoint. They send no ambient hosted credential and bypass environment
  proxies. Authenticated Ollama requires its dedicated embedding credential.
- A provider instance shares one locked explicit-tokenizer snapshot between its
  fingerprint and counter. Replace an artifact, then restart to use a new
  snapshot. Restart and explicit reindex/new-collection rules remain in force.

Mutable remote model names, gateway aliases and local checkpoint directories
still need operator deployment versioning. Pin a Sentence Transformers revision
and update `RAGS_EMBEDDING_SPACE_ID` when changing an unpinned deployment behind
an unchanged name. Catalog selection does not download model weights.
Native CPU inference is cooperative; a completed late computation is rejected,
but arbitrary native computation is not forcibly interrupted by Python.

## Reproduced defects repaired

| Defect | Protection and regression |
| --- | --- |
| FastEmbed documents used a generic task, including the wrong passage task for Jina v3. | Separate passage/query APIs and identity invalidation. Existing provider role test plus new runtime tests. |
| Document embedding could silently inherit native tokenizer truncation. | Complete-input checks, batch preflight and actual trained BGE/MiniLM gates. |
| Modern ST preprocessing preserves spaces while a stripped-text counter underestimated them. | Actual native preprocessing and no-truncation inference; tiny real ByteLevel/GPT2 fixture proves nine source tokens are rejected at an eight-token limit before inference. |
| Single-input compatible endpoints could return multiple vectors or an incorrect index. | Strict cardinality/index validation with typed provider failures. |
| Gateway extreme integer vectors could raise uncaught `OverflowError`. | Sanitized invalid-vector handling and a regression using `10**400`. |
| Native Ollama input rejection was hidden by generic HTTP sanitization. | The existing bounded transport is retained with provider-specific status mapping; actual loopback HTTP verifies no source is committed. |
| Configured tokenizer file replacement could separate a new fingerprint from an old cached counter. | One loaded snapshot serves both identity and counting; regression verifies fresh instances alone adopt the new snapshot. |
| Optional encoder construction could enable unrelated diagnostic traffic. | ONNX and Hugging Face telemetry are disabled before construction for all encoder paths. |

## Executed verification

These are local observations from this change; the final combined repository
gate is reported separately by the integration owner.

- 54 new ordinary runtime contract cases and four real loopback HTTP/persistence
  cases passed. Existing targeted suites passed 189 cases. These selections
  overlap with subsequent combined verification and must not be summed.
- A combined regression selection passed **211 cases**, with no selected skips;
  two actual PostgreSQL parametrizations were explicitly deselected for that
  command. This includes the four new actual local Sentence Transformers cases.
- Actual trained FastEmbed BGE semantic retrieval and collection lifecycle:
  **2/2 passed**. An earlier test attempt was blocked when ONNX tried to contact
  a diagnostic endpoint. That attempt is not counted as a pass. The corrected
  lifecycle explicitly opts out of telemetry before loading the runtime.
- Actual locally constructed tiny GPT2/ByteLevel/Sentence Transformers runtime:
  **4/4 passed**, covering whitespace, prefixes, special tokens, shorter
  preprocessing windows, and preservation of runtime settings. Its weights are
  synthetic; this verifies behavior, not retrieval quality.
- Actual trained `sentence-transformers/all-MiniLM-L6-v2`, pinned to publisher
  revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`: **1/1 passed**. The gate
  checks normalized 384-dimensional vectors, document and collection retrieval,
  oversize rejection and changed-profile rejection.
- Both repository Ruff selections and `git diff --check` passed for owned files.

The hosted provider role/dimension tests and native Ollama HTTP checks use
explicit synthetic transport fixtures. No paid/live hosted inference or actual
Ollama checkpoint inference was executed. No throughput or quality improvement
is inferred from these smoke checks. Frozen Qwen evidence is unchanged.

Official API references used to confirm implementation contracts:
[Sentence Transformers model API](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html),
[FastEmbed retrieval API](https://qdrant.github.io/fastembed/qdrant/Retrieval_with_FastEmbed/),
[Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md), and
[MiniLM publisher metadata](https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2).
