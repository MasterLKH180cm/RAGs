# Embedding and tokenizer choice with completed maintainability repairs

Status: specification committed before implementation.
Baseline: `50436763b63f895054b315f9732eb1979901f80b`.
Requested on 2026-11-26: finish identified repairs, add varied embedding models
and tokenizers, and make selection friendly and explicit for users.

## Outcomes

1. Users can discover, compare, select and validate embedding/tokenizer choices
   without hand-editing several coupled settings or silently downloading models.
2. Local English, Chinese and multilingual embeddings, a broader optional
   Sentence Transformers runtime, native Ollama embeddings, compatible hosted
   endpoints and approved gateway aliases have explicit execution contracts.
3. Source chunk counting, embedding input validation and generation prompt
   budgeting remain separate roles. No encoder may silently truncate a complete
   source/query because the selected chunk counter was a different tokenizer.
4. Operator-registered embedding profiles can be selected when creating a
   collection. Tenant callers cannot supply endpoints, credentials or host paths.
   Existing collections retain their immutable vector-space binding.
5. The actionable SQLite, connection-lifecycle, cache and dispatcher observations
   from CODE_SMELL_REVIEW.md receive measured, bounded improvements. Existing
   confirmed defects remain covered by their regressions.

## Scope and compatibility

Keep the default offline hashing/extractive behavior, existing CLI invocations,
public response contracts, tenant/owner checks, exact source citations and
router/agent/memory default-on policy. Preserve existing tokenizer profile
dictionaries and legacy vector identities when no new option is selected.

Keep frozen admission profiles, datasets, results, quality thresholds and
promotion state unchanged. New optional dependency/runtime support must not
upgrade or rewrite the frozen research/admission recipes. No coverage exclusions,
lowered thresholds, weakened resource checks or synthetic substitutions for
existing actual-model jobs are permitted.

The research subprocess remains a lifecycle boundary and trusted Python callbacks
remain responsible for their own external effects. This work does not promise
arbitrary thread cancellation, a distributed control plane or OS sandboxing.

## Embedding contract

### Runtime choices

Preserve `local`, `fastembed`, `openai` and `router`; add optional
`sentence_transformers` and native `ollama` adapters. FastEmbed catalog entries
must use a verified supported execution/preprocessing family. Unsupported model
families fail explicitly instead of receiving a guessed tokenizer contract.

Sentence Transformers uses a tested optional version, pinned Hub revisions,
standard supported transformer/pooling modules, safetensors and
`trust_remote_code=False`. Loading is lazy and bounded by a small synchronized
cache. Exact input validation must include the selected query/document prompt,
normalization and special-token behavior; reject overflow before encoding.

Native Ollama uses `/api/embed`, requests `truncate:false`, validates the complete
response and distinguishes server-enforced overflow rejection from locally exact
token counting. Its catalog must not describe server validation as an exact
local counter. Explicit no-auth is limited to validated loopback endpoints;
remote endpoints require configured authentication.

Compatible hosted endpoints and gateway aliases remain configurable. For custom
models without a known local input window, require an explicit verified counter
contract or a supported server-rejection contract; do not silently assume the
OpenAI text-embedding window or accept unknown truncation behavior.

### Shared validation and identity

Move reusable FastEmbed input-window logic out of the research-specific layer
into a provider-level module. Keep compatibility exports for research users.
Preserve role-specific document/query preprocessing and validate complete input
for documents as well as collections. Validate every returned vector for numeric
type, finiteness, dimension and result-count consistency.

Model revision, requested dimensions, normalization, input instructions and
tokenizer/preprocessing changes must affect a new explicit embedding identity.
Legacy identity stays byte-for-byte stable when the new configuration is absent.
Generation-tokenizer changes alone do not require re-embedding stored text.

Source chunking may select another counter; embedding preflight is still bound
to the actual encoder. Friendly automatic source selection uses the configured
embedding counter where supported, with truthful count labels and exact source
character slices. It must not make arbitrary model loading available through
per-document request parameters.

### Approved collection profiles

Use a bounded, structurally validated operator profile registry. Each alias maps
to supported embedding settings; credentials remain server-side. Collection
creation accepts a profile alias in addition to existing configured/external
vector modes. SDK and CLI expose a convenient `embedding_profile` selection.

Resolve an approved profile after authorization. Bind its effective immutable
identity to the collection. Select a request-local provider without mutating
shared service settings. Rebinding an alias must not cause old collections to
silently query a new vector space. Unknown/disabled profiles fail before model
loading or network work. Forks preserve the original binding and never imply
re-embedding. Support explicit copy/re-embedding into a new collection with
documented generation/revision checks rather than changing old vectors in place.

## Tokenizers

Preserve Unicode character counting, UTF-8 byte counting and Tiktoken. Add generic
optional `huggingface` and `sentencepiece` backends using local registered
`tokenizer.json` and serialized `.model` bytes respectively.

An operator registry maps a bounded alias to backend, absolute path and SHA-256.
Requests select aliases, never arbitrary paths. Read a regular file once within
a size bound, verify the bytes and construct the tokenizer from those same
bytes. No remote code, automatic download or fallback to another tokenizer.

Disable truncation, padding, sampling/dropout and automatic special-token
insertion for raw-text counting. Label exact encoding counts separately from
chat framing and provider envelope costs. Persist artifact digest, runtime
version and wrapper semantics without exposing host paths.

Source spans, truncation and suffixes must be original Python-character slices.
Native decode is not assumed to round-trip normalized text. Bound candidate
windows, recount final slices, preserve normalized-away content, revalidate
overlap and require forward progress. Existing materialization limits remain.

Extend reindex profile reconstruction to preserve the artifact digest. Alias
rebinding requires an explicit new processing selection rather than silently
reinterpreting historical settings. Pass operator registry settings through all
supported generation/source consumers and the explicit research-worker whitelist.

## Discovery and user experience

A shared, side-effect-free catalog drives CLI, authenticated API/SDK discovery
and the admin settings UI. Each entry identifies language coverage, dimensions,
input policy/window, execution location, optional dependency, cost category,
source documentation and limitations. Configuration, installed dependency,
artifact presence and successful validation are distinct states.

Add `rags models list/show`, `rags tokenizers list/show`, orthogonal `rags init`
selection flags, and an explicit TTY-only interactive setup. A base installation
can list choices without credentials, databases, optional native imports or
network calls. Cancellation leaves no partial configuration; secrets are never
prompt arguments or public output.

Add read-only configuration planning and creation of a new private candidate
configuration. Preserve the old files and unrelated settings. Existing explicit
feature restrictions cannot be silently widened. Show required extras, setup
commands, restart instructions and index consequences. New init selections must
keep generated provider permissions consistent with the chosen model.

Fix doctor reporting for router generation aliases and distinguish configured
from usable runtime/tokenizer policy. Doctor/listing must not run inference or
download weights. Explicit validation commands may perform requested live work
and report actual measurements as such.

The admin model/tokenizer selector previews validated configuration choices and
generates commands/drafts. It does not silently mutate a shared running process.
Client-visible discovery omits secrets, registry paths and secret-file names.

## Remaining maintainability repairs

1. Push SQLite revision/authorized-ID predicates into parameterized SQL before
   materializing candidates; use one explicit read snapshot for multi-read scope.
   Preserve BM25/cosine formulas, result ordering and temporal source identities.
2. Share SQLite/PostgreSQL connection lifecycle between document and asset
   stores. Preserve each store's schema ownership and explicit foreign-key policy.
   Rollback/close on failures; no import-time connections or new ORM.
3. Replace the late-interaction cache's entry-only clear-all policy with a
   synchronized LRU bounded by retained payload bytes and entry count. Oversized
   entries remain usable without caching. This is not an RSS or model-weight cap.
4. Extract cohesive retrieval handlers while retaining common preflight,
   authorized scope and result assembly. Do not build a generic plugin framework.
   Verify the advertised strategy inventory against implemented paths.

## Validation and delivery

Use focused behavior regressions, real local tokenizer libraries and tiny
deterministic artifacts for Unicode/source/normalization boundaries. Add actual
representative trained encoder checks in explicitly provisioned CI environments.
Provider contract fixtures remain clearly distinguished from live hosted model
performance. Native Ollama overflow and response handling require real HTTP
process-boundary tests even when the upstream is a deterministic fixture.

Test discovery without optional dependencies, generated feature permissions,
interactive cancellation/non-TTY behavior, private no-clobber candidate files,
custom counter/identity mismatches, approved-profile authorization, concurrent
profile selection, tenant isolation, original citations, PostgreSQL vector
binding, reindex round-trips and model-switch guidance.

Measure SQLite candidate materialization and cache byte/eviction behavior rather
than claiming generic throughput improvements. Run complete-package unit,
integration, PostgreSQL, HTTP/wheel/README and coverage gates; preserve all existing
actual-resource jobs. Verify the PR head, merge, then verify master CI.

Deliver the implementation, concise selection guide, updated review disposition,
reproducible validation/performance evidence and PR/CI links. Any GitHub comment,
if needed, must contain exactly `@codex review`.
