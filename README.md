# RAGs

**Source-preserving retrieval with configurable chunking, versioned indexes, bounded model workflows, and explicit capability reporting.**

[![CI](https://github.com/MasterLKH180cm/RAGs/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/MasterLKH180cm/RAGs/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

RAGs provides an authenticated FastAPI knowledge service and a read-only MCP bridge. Use it directly or connect [agentic-system](https://github.com/MasterLKH180cm/agentic-system) for durable workflows. The default runs without paid model credentials, using clearly identified hashing embeddings and extractive evidence output.

Version 0.2 adds six chunking strategies, token-budget packing, real parent/summary indexes, graph traversal, structured queries, temporal revisions, federation, and optional pretrained encoders. The [twenty-family coverage table](#twenty-family-coverage) distinguishes native mechanisms from model-dependent features and upstream research integrations. An unavailable capability returns an explicit error instead of silently running another strategy.

## Contents

- [Quick start](#quick-start)
- [Ingest and query](#ingest-and-query)
- [Architecture and flows](#architecture-and-flows)
- [Chunking and token budgets](#chunking-and-token-budgets)
- [Twenty-family coverage](#twenty-family-coverage)
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

Responses expose requested/executed mode, engine, ranking score kind, execution trace, citations, and generation-budget information. Citations identify the source document, revision, index profile, exact source text, and character offsets. `"generate":false` returns retrieval evidence without an answer.

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
    Evidence --> Generate{"Generate an answer?"}
    Generate -->|No| Result["Return evidence and execution trace"]
    Generate -->|Yes| Pack["Pack full prompt within input budget"]
    Pack --> Answer["Extract evidence or call configured generator"]
    Answer --> Validate["Check cited source IDs or report insufficient evidence"]
    Validate --> Result
```

Source-record validity and correct citation IDs do not establish semantic entailment. The default dense/hybrid paths return the highest-ranked candidates without a learned relevance threshold; the evaluation below records their weak no-answer behavior.

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

**Native** means the stated mechanism is implemented. **Provider-dependent** paths have control-flow tests; live paid-model behavior remains unverified. **Pretrained encoder** paths have actual local checkpoint smoke tests. **External contract** means the adapter is implemented, while the named upstream engine remains a separately deployed and unverified requirement.

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
| Hierarchical / RAPTOR | `parent_child`, `hierarchical`; `raptor` | Native parent expansion and recursive clustered summaries; full RAPTOR is an external contract |
| GraphRAG | `graph`; `full_graphrag` | Native two-hop graph traversal, label-propagation communities, and source lineage; Microsoft GraphRAG is external |
| LightRAG | `lightrag` | External contract; actual upstream indexing/retrieval is not bundled or verified |
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

`RAGS_RESEARCH_BACKENDS_JSON` separately configures named upstream adapters using `url`, `model`, and `tenant_tokens`. The `authorized-evidence-v1` protocol sends only scoped source passages and accepts returned source IDs from that set. The adapter is a retrieval integration boundary: installing it does not install or verify RAPTOR, LightRAG, HippoRAG, Microsoft GraphRAG, trained Self-RAG, or CRAG.

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

```bash
RAGS_ENV=test pytest -q
ruff check rags tests --select E4,E7,E9,F
RAGS_ENV=test python -m rags.evaluate --dataset examples/eval_v02.json --k 3 --generate
```

Model downloads are opt-in for the integration tests:

```bash
RAGS_TEST_COLBERT=1 RAGS_MODEL_CACHE=./data/models pytest tests/test_backends.py -q
RAGS_TEST_MULTIMODAL=1 RAGS_MODEL_CACHE=./data/models pytest tests/test_multimodal.py -q
RAGS_TEST_ENCODERS=1 RAGS_MODEL_CACHE=./data/models pytest tests/test_semantic_encoder.py -q
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

The ordinary suite completed locally with **126 passed, 3 skipped**; the three opt-in actual-model checks also passed separately. BGE smoke verifies actual 384-dimensional semantic chunking and dense retrieval. ColBERT smoke uses a 32 × 96 query matrix and checks a three-document retrieval fixture. Paired CLIP smoke checks simple image/text retrieval and image self-matching. These establish that checkpoints and adapters execute; they do not establish domain quality.

The companion real HTTP/worker/MCP acceptance flow also passed with both Agent SQLite and PostgreSQL 16.2 storage; RAG storage remained SQLite in both runs. Paid generation providers and named upstream research services have not been verified through live integration runs.

See the [implementation review](docs/IMPLEMENTATION_REVIEW.md), [v0.2 specification](docs/SPEC_V02.md), and [architecture](docs/ARCHITECTURE.md) for requirement-level evidence and remaining work.

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
| [`vision.py`](rags/vision.py), [`multimodal.py`](rags/multimodal.py) | Caption ingestion and separate original-image retrieval |
| [`evaluate.py`](rags/evaluate.py), [`tests/`](tests/) | Measured evaluation and behavioral integration tests |

Licensed under the [MIT License](LICENSE).
