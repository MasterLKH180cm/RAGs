# RAGs

**Source-preserving retrieval with configurable chunking, versioned indexes, bounded model workflows, and explicit capability reporting.**

[![CI](https://github.com/MasterLKH180cm/RAGs/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/MasterLKH180cm/RAGs/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

RAGs provides an authenticated FastAPI knowledge service and a read-only MCP bridge. Use it directly or connect [agentic-system](https://github.com/MasterLKH180cm/agentic-system) for durable workflows. The default runs without paid model credentials, using clearly identified hashing embeddings and extractive evidence output.

Version 0.3 adds typed evidence decisions, separately calibrated relevance and answerability profiles, and isolated services executing pinned RAPTOR and LightRAG code. It builds on v0.2's six chunking strategies, token budgets, parent/summary indexes, graph traversal, structured queries, temporal revisions, federation, and pretrained encoders. The [twenty-family coverage table](#twenty-family-coverage) distinguishes native mechanisms, tested upstream execution, and remaining external requirements. An unavailable capability returns an explicit error instead of silently running another strategy.

## Contents

- [Quick start](#quick-start)
- [Ingest and query](#ingest-and-query)
- [Architecture and flows](#architecture-and-flows)
- [Chunking and token budgets](#chunking-and-token-budgets)
- [Twenty-family coverage](#twenty-family-coverage)
- [Evidence admission and refusal](#evidence-admission-and-refusal)
- [RAPTOR and LightRAG services](#raptor-and-lightrag-services)
- [Structured and temporal queries](#structured-and-temporal-queries)
- [Pretrained encoders and image assets](#pretrained-encoders-and-image-assets)
- [Model providers and federation](#model-providers-and-federation)
- [MCP and API reference](#mcp-and-api-reference)
- [Validation and measured limits](#validation-and-measured-limits)
- [Deployment and project map](#deployment-and-project-map)

## Quick start

### Docker

Requires Git and Docker Compose with `--wait` support.

```bash
git clone https://github.com/MasterLKH180cm/RAGs.git
cd RAGs
cp .env.example .env
docker compose up -d --build --wait
```

Open [API documentation](http://127.0.0.1:8000/docs). The supplied Compose file binds to localhost, stores data in a named volume, and uses development token `local-development-key`.

```bash
curl --fail-with-body http://127.0.0.1:8000/health \
  -H 'Authorization: Bearer local-development-key'
```

`docker compose down` stops services while retaining the volume. To run both APIs and the workflow worker, follow the [companion full-stack setup](https://github.com/MasterLKH180cm/agentic-system#quick-start).

### Python

Requires Python 3.11+ and SQLite FTS5. From the cloned repository in a Bash-compatible shell:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,pdf,mcp,vision]'
export RAGS_ENV=development
export RAGS_API_KEY=local-development-key
uvicorn rags.api:app --host 127.0.0.1 --port 8000 --reload
```

Python reads exported process variables. Copying `.env` alone does not configure this command; standalone Compose loads it through `env_file`. Run API examples in another terminal.

## Ingest and query

### Add an identified source

```bash
curl --fail-with-body http://127.0.0.1:8000/documents \
  -H 'Authorization: Bearer local-development-key' \
  -H 'Content-Type: application/json' \
  -d '{"document_id":"session-v1","title":"Session policy","source":"manual","text":"# Sessions\nSessions expire after 30 minutes of inactivity. Refresh tokens rotate on login."}'
```

The response includes the document ID, source revision, index profile, and chunk count. Repeating unchanged source content and processing settings returns `unchanged: true`. Reusing an ID with changed content creates a new revision and closes the previous revision's validity interval.

### Retrieve evidence

```bash
curl --fail-with-body http://127.0.0.1:8000/query \
  -H 'Authorization: Bearer local-development-key' \
  -H 'Content-Type: application/json' \
  -d '{"question":"When do sessions expire?","mode":"hybrid","top_k":3}'
```

Responses expose requested/executed mode, engine, ranking score kind, execution trace, citations, generation-budget information, and a typed `outcome`/`admission` decision. Citations identify the source document, revision, index profile, exact source text, and character offsets. `"generate":false` still assesses evidence; it returns `answer:null`. An evidence refusal has `outcome:"insufficient_evidence"`, `supported:false`, a fixed `refusal_reason`, and no usable citations.

Inspect runtime availability with authenticated `GET /strategies`. `requires_provider`, `requires_index`, and `requires_backend` explain why a capability cannot currently run. Unknown strategy names are rejected.

The complete HTTP request body is capped at 2,100,000 bytes before JSON parsing or multipart spooling, with a 30-second total receive deadline. Individual document and image limits still apply.

### Reindex without overwriting a concurrent change

```bash
curl --fail-with-body http://127.0.0.1:8000/documents/session-v1/reindex \
  -H 'Authorization: Bearer local-development-key' \
  -H 'Content-Type: application/json' \
  -d '{"expected_revision":"replace-with-current-revision","chunking":{"strategy":"recursive","max_tokens":256,"overlap_tokens":32},"tokenizer":{"backend":"tiktoken","encoding":"cl100k_base"}}'
```

Use the revision returned by ingestion or `GET /documents/session-v1`. A stale revision returns `409`. Indexing work finishes before the transactional swap; provider failure preserves the previous index. Deletion purges the document's revisions and derived indexes.

## Architecture and flows

### Service boundaries

```mermaid
flowchart TD
    App["Application or agentic-system"] -->|Bearer HTTP| API["FastAPI and tenant authentication"]
    Client["MCP client"] --> MCP["stdio bridge"]
    MCP --> API
    API --> Docs["Document ingestion and retrieval"]
    API --> Assets["Original image asset retrieval"]
    Docs <--> DB[("SQLite sources, revisions, and indexes")]
    Assets <--> DB
    Docs --> Local["Hashing or optional trained text encoders"]
    Assets --> CLIP["Optional paired CLIP encoders"]
    Docs -.-> Models["Configured generation or research backend"]
```

The workflow runtime remains a separate service. Named research HTTP adapters receive only authorized source passages and must return IDs from that supplied set. Image assets use separate endpoints and original-image citations.

### Ingestion and reindexing

```mermaid
flowchart TD
    Request["Source and processing configuration"] --> Scope["Authenticate and read current revision"]
    Scope --> Same{"Source and index profile unchanged?"}
    Same -->|Yes| Unchanged["Return unchanged"]
    Same -->|No| Chunk["Tokenize and preserve source spans"]
    Chunk --> Index["Embed and build requested derived indexes"]
    Index --> Check{"Source revision still current?"}
    Check -->|No| Conflict["Return conflict; retain existing index"]
    Check -->|Yes| Commit["Atomically commit revision, indexes, and audit"]
```

`index_features:["hierarchy"]` builds recursive vector clusters and source-linked summaries; at least two leaf chunks are needed. Summaries are extractive by default and model-generated when a generator is configured. Explicit `Entity A -> Entity B` lines are indexed as relations. `index_features:["graph"]` additionally enables bounded model extraction when a generator is configured.

### Retrieval and answer production

```mermaid
flowchart TD
    Query["Question, mode, filters, and time"] --> Scope["Resolve authorized source revisions"]
    Scope --> Ready{"Required capability available?"}
    Ready -->|No| Error["Return an explicit requirement error"]
    Ready -->|Yes| Retrieve["Execute the declared retrieval mechanism"]
    Retrieve --> Review{"Model-driven refinement requested?"}
    Review -->|Within bounded loop| Retrieve
    Review -->|Ready or limit reached| Evidence["Resolve original source evidence"]
    Evidence --> Admission{"Evidence admitted?"}
    Admission -->|No| Refusal["Typed refusal; no citations or generation"]
    Admission -->|Yes or explicitly unassessed| Generate{"Generate an answer?"}
    Generate -->|No| Result["Return evidence and execution trace"]
    Generate -->|Yes| Pack["Pack full prompt within input budget"]
    Pack --> Answer["Extract evidence or call configured generator"]
    Answer --> Validate["Check cited source IDs or report insufficient evidence"]
    Validate --> Result
    Refusal --> Result
```

Source-record validity and correct citation IDs do not establish semantic entailment. Compatibility mode without a scoring profile returns explicitly unassessed ranked candidates. A configured profile adds a query-level gate over original sources; it preserves their ranking on acceptance and removes all usable citations on refusal. Prompt packing can independently refuse when admitted evidence does not fit.

## Chunking and token budgets

Pass `chunking` and `tokenizer` on ingestion/reindexing, or set `RAGS_CHUNKING_JSON` and `RAGS_TOKENIZER_JSON` defaults.

| Strategy | Actual behavior | Requirement |
| --- | --- | --- |
| `fixed` | Tokenizer-bounded windows and overlap with exact source offsets | Configured tokenizer |
| `recursive` | Paragraph/newline/sentence/word splitting, then bounded fallback | Configured tokenizer |
| `structure` | Markdown heading paths and code/list/table block boundaries | Text structure; no PDF layout claim |
| `parent_child` | Separate parent and child source records; deduplicated parent expansion | Parent budget greater than child budget |
| `semantic` | Sentence-vector discontinuities plus hard chunk limits | Trained OpenAI or FastEmbed embedding adapter |
| `contextual` | Generated document context indexed with source text in BM25 and vectors | Configured generator; over-budget context fails explicitly |

Default source processing uses `structure`, 850 Unicode code points per chunk, and 120 units of overlap. Unicode units are **not model tokens**. For exact encoding counts, select `{"backend":"tiktoken","model":"known-model"}` or an explicit, correct encoding. Unknown model mappings fail. Processing metadata records tokenizer identity, counts, source spans, block types, and generated context separately.

Generation has its own `RAGS_GENERATION_TOKENIZER_JSON`. OpenAI generation defaults to the configured chat model's tiktoken mapping; other defaults explicitly count Unicode units. Select the correct generation encoding when using compatible servers with custom model names. `RAGS_CONTEXT_BUDGET`, `RAGS_OUTPUT_BUDGET`, and `RAGS_ENVELOPE_BUDGET` control full-prompt packing, output reservation, and provider framing allowance. Headers, instructions, question, and delimiters are counted; an oversized question fails instead of being silently dropped.

The index fingerprint includes chunking/tokenizer settings, model identities, embedding endpoint/dimensions, and prompt revision. Legacy rows remain marked `legacy` until explicitly reindexed; incompatible vector profiles return a reindex requirement.

## Twenty-family coverage

**Native** means the stated mechanism is implemented. **Provider-dependent** paths have control-flow tests; live paid-model behavior remains unverified. **Pretrained encoder** paths have actual local checkpoint tests. **Pinned upstream** means the upstream code executes through an isolated service, with real embeddings and explicitly deterministic generation fixtures in verification. **External contract** means the named upstream engine still needs implementation and execution evidence.

| Original family | Entry point | Implementation and boundary |
| --- | --- | --- |
| Naive RAG | `naive` | Native single dense retrieval followed by evidence assembly |
| Sparse RAG | `sparse` | Native FTS5 BM25 ranking |
| Dense RAG | `dense` | Native cosine retrieval; hashing, OpenAI, or trained FastEmbed vectors are identified explicitly |
| Hybrid RAG | `hybrid` | Native sparse/dense RRF; no hidden lexical reranker |
| RAG-Fusion | `fusion` | Provider-generated distinct queries, independent retrieval, and RRF |
| HyDE | `hyde` | Provider-generated hypothetical documents and embedding centroid; hypotheses never become citations |
| Multi-vector / ColBERT | `multivector`, `colbert` | Actual FastEmbed ColBERT token matrices and normalized sum-MaxSim; checkpoint required |
| Contextual RAG | `contextual` | Searches generated context in both indexes; contextual ingestion required |
| Hierarchical / RAPTOR | `parent_child`, `hierarchical`; `raptor` | Native parent expansion and recursive clustered summaries; separate pinned RAPTOR service runs actual UMAP/GMM tree building and upstream retrieval, resolving summaries to original leaves |
| GraphRAG | `graph`; `full_graphrag` | Native two-hop graph traversal, label-propagation communities, and source lineage; Microsoft GraphRAG is external |
| LightRAG | `lightrag` | Pinned 1.5.7 service runs actual entity/relation extraction, graph merging and hybrid retrieval; internal chunks resolve to authorized original sources |
| HippoRAG | `hipporag` | External contract; actual upstream graph/PPR engine is not bundled or verified |
| Self-RAG | `self_reflective`; `self_rag` | Native bounded draft/critique/refine loop; trained reflection-token Self-RAG remains external |
| CRAG | `corrective`; `crag` | Native bounded evidence evaluation and corrective retrieval; the paper-specific engine remains external |
| Adaptive RAG | `adaptive` | Provider router chooses a validated, available mechanism and records its reason |
| Agentic RAG | `agentic` | Provider search planning, evidence review, and bounded additional searches |
| Multimodal RAG | `/assets/*`, `caption`; `multimodal_native` | Actual paired CLIP asset retrieval; separate text-caption path; optional external contract; no audio/video or image-answer generation |
| Structured RAG | `structured` | Typed row/metadata predicates and safe numeric aggregations; no arbitrary SQL |
| Federated RAG | `federated` | Operator-allowlisted sources, distinct tenant credentials, bounded responses, and namespaced RRF |
| Temporal / real-time RAG | `temporal` | Versioned validity/as-of reads, explicit revision updates, and recency ranking; no automatic source-refresh/streaming connector |

`fusion`, `hyde`, `corrective`, `adaptive`, `agentic`, and `self_reflective` require a generator. Corrective, agentic, and self-reflective loops allow at most three iterations; fusion allows at most five generated queries and HyDE at most three hypothetical documents. Model responses are validated; unknown sources and citation IDs are not accepted. These mechanisms are not claims of research-paper reproduction or benchmark parity.

## Evidence admission and refusal

Every text query returns the same evidence contract, including retrieval-only queries and empty authorized scopes:

| Field | Meaning |
| --- | --- |
| `outcome` | `success` or `insufficient_evidence`; transport/model/configuration errors remain HTTP errors |
| `supported` | `null` for unassessed or relevance-only evidence; `true` for an explicit supporting model assessment or exact structured result; `false` for refusal |
| `admission.decision` | `accept`, `abstain`, or `unassessed` |
| `admission.basis` | Ranked, calibrated relevance, calibrated answerability, explicit model review, exact structure, empty scope, or context budget |
| `admission.profile_id` | Identity of the frozen scorer, artifacts, runtime, prompt/window, threshold and calibration evidence |
| `refusal_reason` | `no_candidates`, `below_admission_threshold`, `model_rejected`, `context_budget_exhausted`, or `null` |

`RAGS_ADMISSION_POLICY=compatibility` is the default. Without a profile it makes no calibrated relevance claim. `RAGS_ADMISSION_POLICY=calibrated_required` requires an applicable operator-selected profile for ordinary source evidence. `RAGS_ADMISSION_PROFILE` names that local JSON file. Missing dependencies or changed artifacts fail explicitly; request parameters cannot select a profile, threshold or model.

The relevance scorer uses actual multilingual BGE ONNX inference. The separate answerability scorer uses pinned Qwen3-1.7B CPU inference and a fixed source-only YES/NO instruction, scored by its next-token logit difference. Profiles identify the exact token window and any source-prefix truncation. Scores are not probabilities. Acceptance preserves the candidate ranking/set; there is no hidden lexical veto or common cutoff for cosine, RRF and learned logits.

Calibration and validation are separate: [48 frozen calibration cases](examples/admission_calibration_v03.json) and [32 topic-disjoint held-out cases](examples/admission_holdout_v03.json), balanced between English and Traditional Chinese. The first BGE profile retained 23/24 calibration positives but refused only 1/24 negatives; [its report](reports/admission-calibration-v03.json) remains a visible failed answerability candidate. A profile's presence is not evidence that it passed the quality gates. The [v0.3 specification](docs/SPEC_V03.md) fixes the objective, holdout targets and amendment history before additional scoring implementation.

**The frozen Qwen profile failed the quality gates and is not enabled by default.** The [final verification report](reports/admission-verification-qwen-v03.json) passed all 21 correctness checks but only 7 of 11 quality checks. Its threshold and prompt were not changed after held-out evaluation.

| Frozen verification | Measured result | Specified target |
| --- | --- | --- |
| Held-out answerable cases retained | 15/16 | At least 15/16; passed |
| Held-out unanswerable cases refused | 9/16 | At least 13/16; failed |
| Held-out semantic-only cases retained | 7/8 | 8/8; failed |
| Original hybrid raw / admitted Recall@3 | 0.85 / 0.70 in both generation modes | Preserve 0.85; failed in both modes |
| Original no-answer cases refused | 4/4 in both generation modes | At least 3/4; passed |
| Exact structured / temporal cases | 6/6 and 4/4 in both generation modes | Preserve all; passed |

Five of 20 answerable hybrid questions were refused. Two already lacked relevant raw candidates; three lost useful evidence at the admission gate. The regression uses 34 applicable cases from the original 36-case dataset; the two graph-only cases retain their original mode labels. These are synthetic measurements, not a production-quality claim.

Exact structured aggregates are separate from learned relevance. A valid zero carries bounded `structured_result.provenance.datasets`; no authorized dataset produces a refusal. Federation quarantines refused sources before merging other admitted evidence. The companion Agent preserves these decisions through synthesis, durable results, SSE and MCP.

The [actual Qwen HTTP-to-Agent report](reports/agent_admission_http_v03.json) verifies one accepted and one refused frozen calibration fixture through separate APIs, a worker, REST/SSE and both MCP bridges. The refused case makes no synthesis request or usage reservation. This checks the integration contract; it is not an additional quality benchmark.

## RAPTOR and LightRAG services

Each service hosts one operator-selected engine in its own dependency environment. RAPTOR uses official source commit `7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767`; LightRAG uses release 1.5.7 at commit `28ff1b05f2ac3f3e6fa14dd2cd33656579bd0c9c`. The installer pins dependencies and verifies the official source/wheel; it does not install the unrelated PyPI package named `raptor`.

```bash
# Python 3.12, separate from the main service environment:
bash scripts/install_research.sh lightrag .venv-lightrag
# Or: bash scripts/install_research.sh raptor .venv-raptor ./upstream-raptor
export RAGS_ENV=development
export RAGS_RESEARCH_ENGINE=lightrag
export RAGS_RESEARCH_KEYS_JSON='{"local-research-key":"local"}'
export RAGS_RESEARCH_TIMEOUT_SECONDS=120
export RAGS_RESEARCH_CONCURRENCY=1
export RAGS_EMBEDDING_PROVIDER=fastembed
export RAGS_FASTEMBED_MODEL=BAAI/bge-small-en-v1.5
export RAGS_MODEL_CACHE="$PWD/data/models"
export RAGS_MODEL_PROVIDER=openai
# Export your OPENAI_API_KEY and compatible OPENAI_CHAT_MODEL separately.
.venv-lightrag/bin/uvicorn rags.research_service:create_app --factory --host 127.0.0.1 --port 8010
```

Configure the main RAG API process separately and restart it:

```bash
export RAGS_RESEARCH_BACKENDS_JSON='{"lightrag":{"url":"http://127.0.0.1:8010/retrieve","tenant_tokens":{"local":"local-research-key"},"timeout_seconds":150,"require_pinned_upstream":true}}'
```

The localhost example requires development/test mode; production backend URLs require HTTPS and strong sidecar credentials. Keep the client timeout longer than the sidecar's execution deadline. `timeout_seconds` is bounded to 1–360 seconds; the sidecar wall deadline is 0.1–300 seconds. The sidecar concurrency limit is per service process, so start one Uvicorn worker for the stated capacity.

Requests carry at most 64 original passages and 200,000 source characters. Each request gets a fresh process and temporary index built only from its authorized source revision snapshot. The service validates tenant identity, finite scores, original IDs and pinned implementation metadata; it bounds response bytes and terminates process groups on timeout, cancellation or completion. Source-derived indexes are removed afterward. Public model weights can use a shared cache. Embedding inputs exceeding the actual encoder window fail explicitly rather than silently truncating.

RAPTOR projects selected summary nodes through upstream child lineage to original leaves and labels its projected reciprocal-rank scores. LightRAG executes graph extraction/merging and hybrid graph retrieval, then verifies internal chunk content against original passages. The runtime catalog labels returned version metadata `pinned_upstream_reported`; the separate actual execution reports supply verification evidence. A configured URL or health probe alone does not verify an algorithm.

## Structured and temporal queries

Structured ingestion accepts `rows` instead of `text`:

```json
{
  "document_id": "sales",
  "title": "Regional sales",
  "rows": [{"region":"north","amount":10},{"region":"south","amount":20}]
}
```

Send this query to authenticated `POST /query`:

```json
{
  "question": "Total sales in the north",
  "mode": "structured",
  "generate": false,
  "row_filters": [{"field":"region","op":"eq","value":"north"}],
  "aggregate": {"operation":"sum","field":"amount"}
}
```

The computed value is in `structured_result.aggregate`. Supported operations are `count`, `sum`, `mean`, `min`, and `max`; predicates are `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `in`, and string `contains`. JSON/CSV uploaded as ordinary text does not automatically become typed rows.

Use timezone-qualified `valid_from` / `valid_to` during ingestion and `as_of` during queries. Intervals include `valid_from` and exclude `valid_to`. Historical selection happens before metadata filtering, so filters cannot resurrect an older version. `GET /documents/{id}/revisions` lists authorized revisions. `filters` supports source equality, scalar metadata equality, and `valid_after` / `valid_before` bounds on revision start time.

## Pretrained encoders and image assets

These optional features load real local ONNX checkpoints:

```bash
python -m pip install -e '.[encoders,vision]'
export RAGS_MODEL_CACHE=./data/models
# Semantic single-vector embeddings:
export RAGS_EMBEDDING_PROVIDER=fastembed
export RAGS_FASTEMBED_MODEL=BAAI/bge-small-en-v1.5
# Token-matrix late interaction:
export RAGS_LATE_INTERACTION_MODEL=answerdotai/answerai-colbert-small-v1
# Paired image/text retrieval:
export RAGS_MULTIMODAL_MODEL=clip-vit-b-32
```

Restart the service after changing configuration. Reindex text documents when changing their embedding profile. Initial downloads require network access and disk space; paired CLIP files total roughly 0.6 GB. The `encoders` dependency is constrained to the tested FastEmbed minor version because CLIP artifact inspection uses version-specific implementation details.

The default Docker image includes PDF, MCP, and vision support. To include local encoders, build with `docker compose build --build-arg RAGS_EXTRAS=pdf,mcp,vision,encoders`, then run `docker compose up -d --wait` with the corresponding variables in `.env`.

Image assets retain the original bytes and vector, unlike the `/images` caption endpoint:

```bash
curl --fail-with-body http://127.0.0.1:8000/assets/images \
  -H 'Authorization: Bearer local-development-key' \
  -F 'file=@diagram.png' -F 'title=Architecture diagram'

curl --fail-with-body http://127.0.0.1:8000/assets/search \
  -H 'Authorization: Bearer local-development-key' \
  -H 'Content-Type: application/json' \
  -d '{"query":"an architecture diagram","top_k":5}'
```

`POST /assets/search/image` searches with an uploaded image. Authenticated get/delete/reindex routes live under `/assets/images/{id}`. Asset citations include an authenticated original-image URI and cosine score. Inputs are single-frame JPEG/PNG/WebP, at most 2,000,000 bytes and 25 million pixels. The CLIP text path enforces its actual 77-token limit including special tokens and is primarily intended for English descriptions.

## Model providers and federation

| Setting | Values / purpose |
| --- | --- |
| `RAGS_EMBEDDING_PROVIDER` | `local` (default), `openai`, `fastembed` |
| `RAGS_MODEL_PROVIDER` | `extractive` (default), `openai`, `anthropic` |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL` | Credential and compatible API endpoint |
| `OPENAI_EMBEDDING_MODEL`, `OPENAI_CHAT_MODEL` | Configured OpenAI model identities |
| `ANTHROPIC_API_KEY`, `ANTHROPIC_CHAT_MODEL` | Anthropic generation/caption configuration |
| `RAGS_MAX_ITERATIONS` | Bounded model refinement, 1–3 |

Set credentials locally, outside Git. OpenAI embeddings and generation are independent of Anthropic generation and MCP client choice. After editing standalone Compose `.env`, rerun `docker compose up -d --build --wait` to recreate the service with the new environment.

`RAGS_SOURCES_JSON` maps source IDs to fixed RAG API URLs and `tenant_keys` mappings. A `federated` request selects those IDs through `sources`; it cannot supply a URL or tenant. Credentials are distinct per tenant. Remote responses are bounded to 2 MB, redirects/compressed responses are rejected, and a failed source does not produce a misleading partial success.

`RAGS_RESEARCH_BACKENDS_JSON` separately configures research adapters using `url`, optional `model` label, `tenant_tokens`, and bounded `timeout_seconds`. Set `require_pinned_upstream:true` for the supplied RAPTOR/LightRAG services. The `authorized-evidence-v1` protocol sends only scoped original passages and accepts IDs from that set. HippoRAG, Microsoft GraphRAG, trained Self-RAG and paper-specific CRAG still require separate upstream integrations; their generic adapter is not an implementation of those engines.

## MCP and API reference

The stdio bridge exposes `search_knowledge` and `list_knowledge_documents`:

```bash
export RAGS_URL=http://127.0.0.1:8000
export RAGS_API_KEY=local-development-key
python -m rags.mcp_server
```

Start the HTTP service first and use a Python environment with `.[mcp]` installed. The process waits for an MCP client over stdio. See [Codex configuration](docs/codex-mcp.toml) and [Claude Code configuration](docs/claude-mcp.json); use an absolute virtual-environment Python path when necessary.

| Routes | Purpose |
| --- | --- |
| `/live`; `/health`; `/strategies` | Public liveness; authenticated configuration and capabilities |
| `/documents`, `/documents/{id}` | Ingest, list, read, delete |
| `/documents/{id}/revisions`, `/documents/{id}/reindex` | Historical metadata and guarded reindexing |
| `/documents/upload` | UTF-8 text/Markdown/JSON/CSV and PDF text extraction |
| `/query` | Native or explicitly configured backend retrieval |
| `/images` | Optional model-generated caption ingestion; text index only |
| `/assets/*` | Separate original-image lifecycle and aligned retrieval |

Text uploads are capped at 2,000,000 bytes; extracted document text at 200,000 characters; PDF extraction at 150 pages. Scanned PDFs need an external OCR step.

## Validation and measured limits

The current local core suite passes **335 unit tests and 152 integration tests**
(**487 unique cases, zero selected skips**). Five actual encoder/research tests
are explicitly deselected here and have separately provisioned CI jobs. Every
pytest case has one primary layer, and a machine-readable inventory identifies
each node ID and its resource requirements. See the committed
[test specification](docs/TEST_SPEC.md) and [test/code review](docs/TEST_REVIEW.md).

| Coverage of the complete production Python package | Before this iteration | Current local result | Required floor |
| --- | ---: | ---: | ---: |
| Statements | 75.88% | **80.66%** | 80% |
| Branches | 68.51% | **74.83%** | 72% |

Both measurements use ordinary parent-process tests, including every production
module in the denominator. Optional models and independent MCP/worker processes
have separate execution evidence; their coverage is not folded into this figure.
No new module omissions or coverage exclusions were added. CI publishes JUnit,
collection inventories and per-file JSON/XML/HTML coverage. A selected
skip/xfail/xpass, an unclassified test or a missing production coverage file
fails the required gate.

Ordinary CI runs actual Qwen acceptance/refusal on two preselected calibration cases and audits the immutable complete quality report. It checks all file/dataset/profile identities and recomputes the recorded gates, while clearly labelling those metrics as historical. The full 472-forward benchmark is a separate manual workflow option, `full_admission_benchmark`, using the unchanged `--enforce-quality` command and a bounded CPU budget. Any scoring/retrieval change or proposed profile promotion requires a new complete benchmark; a successful contract smoke or recorded-report audit cannot substitute for it. See [the CI resource specification](docs/SPEC_V03.md#9-ci-resource-and-evidence-amendment).

```bash
python -m pip install -e ".[dev,pdf,mcp,vision]"
RAGS_ENV=test python -m pytest -q -m unit --require-no-skips \
  --test-inventory reports/testing/inventory-unit.json \
  --junitxml reports/testing/junit-unit.xml \
  --cov=rags --cov-branch --cov-config=pyproject.toml --cov-report=json:reports/testing/coverage-unit.json
RAGS_ENV=test python -m pytest -q -m "integration and not external" --require-no-skips \
  --test-inventory reports/testing/inventory-integration.json \
  --junitxml reports/testing/junit-integration.xml \
  --cov=rags --cov-branch --cov-config=pyproject.toml --cov-append \
  --cov-report=json:reports/testing/coverage-core.json \
  --cov-report=xml:reports/testing/coverage-core.xml \
  --cov-report=html:reports/testing/coverage-html
python scripts/check_coverage.py
ruff check rags tests scripts conftest.py --select E4,E7,E9,F
ruff check rags --select B,RUF021
RAGS_ENV=test python -m rags.evaluate --dataset examples/eval_v02.json --k 3 --generate
```

Model downloads are opt-in for the integration tests:

```bash
RAGS_TEST_COLBERT=1 RAGS_MODEL_CACHE=./data/models python -m pytest tests/test_backends.py -q -m encoders --require-no-skips
RAGS_TEST_MULTIMODAL=1 RAGS_MODEL_CACHE=./data/models python -m pytest tests/test_multimodal.py -q -m encoders --require-no-skips
RAGS_TEST_ENCODERS=1 RAGS_MODEL_CACHE=./data/models python -m pytest tests/test_semantic_encoder.py -q -m encoders --require-no-skips
```

The [36-question synthetic dataset](examples/eval_v02.json) covers retrieval, structured rows, temporal versions, and no-answer cases. The [recorded local report](docs/evaluation-v02-summary.json) includes applicable recall/MRR/nDCG, exact structured/time checks, answer metrics, refusal behavior, citation-record validity, and latency. Unmeasured provider usage, cost, and semantic citation entailment remain `null` with reasons.

| Recorded observation | Result and implication |
| --- | --- |
| Offline hybrid Recall@3 | `0.85` on applicable retrieval cases; a synthetic regression baseline |
| Offline hybrid no-answer refusal recall | `0.0`; unrelated highest-ranked evidence can still be returned |
| Structured exact results | `6/6` dedicated cases |
| Temporal source selection | `4/4` dedicated cases |
| Source citation-record validity | `1.0` for measured available modes; this checks records/spans, not semantic support |
| Extractive answer exact match | `0.0`; evidence excerpts do not match short reference answers verbatim |

The [recorded v0.2 CI run](https://github.com/MasterLKH180cm/RAGs/actions/runs/44798397794) passed both jobs: the ordinary suite finished with **141 passed, 3 skipped**, and the real-encoder job finished with **39 passed**, including all three opt-in model checks. The encoder job also repeats relevant adapter tests, so these counts must not be added as unique tests. BGE smoke verifies actual 384-dimensional semantic chunking and dense retrieval. ColBERT smoke uses a 32 × 96 query matrix and checks a three-document retrieval fixture. Paired CLIP smoke checks simple image/text retrieval and image self-matching. These establish that checkpoints and adapters execute; they do not establish domain quality.

The historical v0.2 companion HTTP/worker/MCP flow also passed with Agent SQLite and PostgreSQL; RAG storage remained SQLite. Version 0.3 additionally executes actual upstream engines through the full HTTP and Agent boundary: [RAPTOR report](reports/research_raptor_http_v03.json) and [LightRAG report](reports/research_lightrag_http_v03.json). Both include tenant isolation, updates/as-of/deletion, process cleanup, independent worker and terminal REST/SSE/MCP checks. These use real BGE and explicitly deterministic generation fixtures; live paid-generation quality remains unverified.

See the [v0.3 implementation review](docs/IMPLEMENTATION_REVIEW_V03.md), [v0.3 specification](docs/SPEC_V03.md), [upstream ledger](docs/RESEARCH_ENGINES.md) and [architecture](docs/ARCHITECTURE.md) for current evidence. The [v0.2 review](docs/IMPLEMENTATION_REVIEW.md) remains a historical record.

## Deployment and project map

Production requires `RAGS_KEYS_JSON` mapping strong API keys to tenant IDs; production keys must be at least 24 characters. Authentication precedes source selection, hierarchy expansion, graph traversal, historical reads, and asset access. The local development keys are examples for localhost.

SQLite WAL, full vector scans, bounded local worker threads for federation, and on-demand research encodings target small deployments. Multi-node operation requires different storage/indexing infrastructure. A public deployment also needs identity integration, document-level policy where required, quotas, TLS, backups, and monitoring. No universal answer-grounding guarantee or production quality claim is made by the smoke tests.

| Module | Responsibility |
| --- | --- |
| [`api.py`](rags/api.py), [`registry.py`](rags/registry.py) | Authenticated HTTP contracts and explicit capabilities |
| [`tokenization.py`](rags/tokenization.py), [`chunking.py`](rags/chunking.py) | Counting, prompt packing, source-preserving chunk plans |
| [`storage.py`](rags/storage.py) | Source revisions, transactional derived indexes, tenant filtering |
| [`retrieval.py`](rags/retrieval.py), [`native.py`](rags/native.py) | Distinct retrieval paths, bounded loops, graph/structured algorithms |
| [`providers.py`](rags/providers.py), [`backends.py`](rags/backends.py) | Model adapters, actual late interaction, upstream protocol |
| [`admission.py`](rags/admission.py), [`answerability.py`](rags/answerability.py), [`admission_eval.py`](rags/admission_eval.py) | Typed evidence decisions, immutable scorer profiles, calibration and held-out evaluation |
| [`research_service.py`](rags/research_service.py), [`research_worker.py`](rags/research_worker.py), [`research/`](rags/research/) | Bounded process isolation and pinned RAPTOR/LightRAG adapters |
| [`vision.py`](rags/vision.py), [`multimodal.py`](rags/multimodal.py) | Caption ingestion and separate original-image retrieval |
| [`evaluate.py`](rags/evaluate.py), [`tests/`](tests/) | Measured evaluation and behavioral integration tests |

Licensed under the [MIT License](LICENSE).
