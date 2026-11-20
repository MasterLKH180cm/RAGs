# RAGs

**A retrieval API that turns documents into searchable evidence—with ten retrieval modes, tenant-scoped access, and an offline default.**

[![CI](https://github.com/MasterLKH180cm/RAGs/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/MasterLKH180cm/RAGs/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

RAGs handles document ingestion, retrieval, and source-attributed responses. Use it directly through FastAPI, connect an MCP client, or pair it with [agentic-system](https://github.com/MasterLKH180cm/agentic-system) for durable agent workflows. The default setup uses local hashing vectors and extractive answers, so you can run it without model credentials.

## Contents

- [Capabilities](#capabilities)
- [Architecture](#architecture)
- [Quick start](#quick-start)
- [Ingest and query](#ingest-and-query)
- [Data flows](#data-flows)
- [Retrieval strategies](#retrieval-strategies)
- [Chunking and tokenization](#chunking-and-tokenization)
- [Model configuration](#model-configuration)
- [MCP integration](#mcp-integration)
- [Testing and evaluation](#testing-and-evaluation)
- [Deployment boundaries](#deployment-boundaries)
- [Project map](#project-map)

## Capabilities

| Area | Available today |
| --- | --- |
| Document lifecycle | Ingest, list, retrieve, replace, and delete documents; maintain chunks, indexes, and audit records transactionally |
| Retrieval | SQLite FTS5 BM25, cosine similarity, reciprocal rank fusion (RRF), lexical reranking, and strategy variants |
| Evidence | Return chunk and document IDs, source, title, section, relevance score, and supporting text |
| Providers | Local hashing or OpenAI embeddings; extractive, OpenAI-compatible, or Anthropic text generation |
| Inputs | Text, Markdown, JSON, CSV, PDF text extraction, and optional model-generated image captions |
| Access | Bearer tokens resolve to tenants before document access and retrieval |
| Integration | Authenticated HTTP API and read-only MCP search/list tools |

Version 0.1 covers a subset of the requested RAG approaches. Ten selectable modes share a retrieval pipeline; several are lightweight variants. The [coverage matrix](#retrieval-strategies) records implemented, partial, and missing approaches, and [chunking and tokenization](#chunking-and-tokenization) documents the actual text processing.

## Architecture

```mermaid
flowchart TD
    App[Application or agentic-system] -->|Bearer token over HTTP| API[FastAPI service]
    Client[Codex or Claude Code] -->|stdio| MCP[MCP bridge]
    MCP -->|Bearer token over HTTP| API
    API --> Auth[Resolve tenant and validate input]
    Auth --> Service[Ingestion and retrieval service]
    Service <--> DB[("SQLite WAL: documents, FTS5, vectors, audit")]
    Service --> Local[Local hashing and extractive answers]
    Service -.-> Remote[Optional OpenAI or Anthropic APIs]
```

The HTTP service owns the knowledge store. The MCP bridge forwards requests to that service; the separate agentic-system runtime owns workflow execution.

## Quick start

### Docker

Requires Git, Docker, and Docker Compose with `--wait` support.

```bash
git clone https://github.com/MasterLKH180cm/RAGs.git
cd RAGs
cp .env.example .env
docker compose up -d --build --wait
```

Open [API documentation](http://127.0.0.1:8000/docs). The supplied Compose configuration binds to `127.0.0.1:8000` and keeps data in a named volume. Its development token is `local-development-key`.

```bash
curl --fail-with-body http://127.0.0.1:8000/health \
  -H 'Authorization: Bearer local-development-key'
```

Stop the service with `docker compose down`; the named volume is retained. For the API plus workflow worker, follow the [agentic-system full-stack setup](https://github.com/MasterLKH180cm/agentic-system#quick-start).

### Python

Requires Python 3.11+ with SQLite FTS5. From the cloned repository, in a Bash-compatible shell:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,pdf,mcp,vision]'
export RAGS_ENV=development
export RAGS_API_KEY=local-development-key
uvicorn rags.api:app --host 127.0.0.1 --port 8000 --reload
```

Python reads process environment variables; copying `.env` alone does not configure this command. Compose reads `.env` through its `env_file` setting. Run the following API examples in another terminal.

## Ingest and query

### 1. Add a document

```bash
curl --fail-with-body http://127.0.0.1:8000/documents \
  -H 'Authorization: Bearer local-development-key' \
  -H 'Content-Type: application/json' \
  -d '{"document_id":"session-v1","title":"Session policy","source":"manual","text":"# Sessions\nRefresh tokens rotate on login. Sessions expire after 30 minutes."}'
```

Reuse `document_id` to replace a document. An identical body, title, source, metadata, and existing index for the configured embedding model produce `unchanged: true`.

### 2. Retrieve evidence

```bash
curl --fail-with-body http://127.0.0.1:8000/query \
  -H 'Authorization: Bearer local-development-key' \
  -H 'Content-Type: application/json' \
  -d '{"question":"When does the session expire?","mode":"hybrid","top_k":3}'
```

The JSON response includes `mode_requested`, `mode_used`, `answer`, and `citations`. In the default extractive mode, the answer contains excerpts from the session policy, with chunk IDs. Each citation includes `document_id`, `chunk_id`, `title`, `source`, `section`, `score`, and `text`. Use `"generate": false` to return evidence with `answer: null`.

### Uploads and API reference

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/live` | Public process liveness |
| `GET` | `/health`, `/strategies` | Authenticated configuration and available retrieval modes |
| `POST` | `/documents` | Ingest or replace a document |
| `POST` | `/documents/upload` | Upload a UTF-8 TXT, MD, JSON, CSV, or text-based PDF file |
| `POST` | `/images` | Validate an image and index a model-generated caption; optional dependencies/provider required |
| `GET` | `/documents`, `/documents/{document_id}` | List or read authorized documents |
| `DELETE` | `/documents/{document_id}` | Delete an authorized document and its index entries |
| `POST` | `/query` | Retrieve evidence and optionally generate an answer |

Uploads use multipart field `file` and are limited to 2,000,000 bytes. Document text is limited to 200,000 characters; PDF extraction is limited to 150 pages. PDFs need extractable text; scanned pages require an external OCR step.

Image ingestion accepts JPEG, PNG, and WebP. Install the `vision` extra and select a vision-capable OpenAI or Anthropic model. The Python setup above includes this extra; the current Docker image includes only `pdf` and `mcp`. Captions are indexed as text, and original image pixels are not stored.

## Data flows

### Document ingestion

```mermaid
flowchart TD
    Request[Document or supported upload] --> Auth[Authenticate and resolve tenant]
    Auth --> Text[Validate or extract text]
    Text --> Same{Same document and current index?}
    Same -->|Yes| Unchanged[Return unchanged]
    Same -->|No| Split[Split by headings and overlapping windows]
    Split --> Embed["Embed title, section, and chunk text"]
    Embed --> Write["Transaction: replace document and indexes; append audit"]
    Write --> Result[Return document ID and chunk count]
```

Explicit `Entity A -> Entity B` lines become graph edges during ingestion. Switching embedding providers or models requires re-ingesting the source documents.

### Default hybrid retrieval

```mermaid
flowchart TD
    Query[Question and bearer token] --> Auth[Resolve tenant and validate query]
    Auth --> Sparse[BM25 text search]
    Auth --> Dense[Embed query and compare vectors]
    Sparse --> Fusion[Reciprocal rank fusion]
    Dense --> Fusion
    Fusion --> Rank[Lexical reranking and top-k selection]
    Rank --> Generate{Generate answer?}
    Generate -->|No| Evidence[Evidence with null answer]
    Generate -->|Yes| Answer[Extractive excerpts or configured model]
    Answer --> Response[Answer and source citations]
    Evidence --> Response
```

The two search branches show data dependencies; the current implementation calls sparse and dense retrieval sequentially. With generation enabled, an empty result produces an insufficient-evidence response. Model prompts request answers grounded in retrieved text; citations are evidence references, not an automatic guarantee of factual correctness.

## Retrieval strategies

Set `mode` on `/query`. Available values are `sparse`, `dense`, `hybrid`, `fusion`, `hyde`, `contextual`, `hierarchical`, `graph`, `corrective`, and `adaptive`; all use the common lexical reranking stage. A selectable name does not establish that a research algorithm has been reproduced.

**Status:** Implemented means the described mechanism exists in code; Partial means only the listed subset or approximation exists; Not implemented means there is no corresponding retrieval mechanism. These statuses do not claim production readiness or benchmark parity.

| Requested approach | Status | Current implementation and remaining gap |
| --- | --- | --- |
| Naive RAG | Implemented | Shared retrieve-then-answer path with source citations; use `sparse` or `dense` for a single retrieval source. There is no separate `naive` mode. |
| Sparse RAG | Implemented | `sparse`: SQLite FTS5 BM25 candidates, followed by the shared lexical reranker. |
| Dense RAG | Implemented | `dense`: cosine similarity over one vector per chunk. Local vectors are deterministic hashes; trained semantic embeddings require the OpenAI adapter. |
| Hybrid RAG | Implemented | `hybrid`: BM25 and vector ranks combined with RRF, followed by lexical reranking. |
| RAG-Fusion | Partial | `fusion`: a small fixed synonym map supplies query alternatives for RRF; no model-generated diverse queries. |
| HyDE | Partial | `hyde`: the core hypothetical-answer mechanism exists when a model provider is configured and is combined with sparse retrieval. Offline mode falls back to the original question; live-provider behavior has not been verified by the offline tests. |
| Multi-vector / ColBERT | Not implemented | Each chunk stores one vector; no per-token vectors, late interaction, or ColBERT index. |
| Contextual RAG | Partial | `contextual` uses hybrid retrieval. Title and section are prepended to embeddings for every mode; no generated chunk-specific explanatory context. |
| Hierarchical / RAPTOR | Partial | `hierarchical` adds adjacent chunks to generation context. The RAPTOR pipeline is not implemented: no parent-summary tree or recursive clustering. |
| GraphRAG | Partial | `graph` boosts documents matching explicit `A -> B` edges. The GraphRAG pipeline is not implemented: no graph traversal, community detection, or community summaries. |
| LightRAG | Not implemented | No dedicated LightRAG indexing or retrieval implementation. |
| HippoRAG | Not implemented | No HippoRAG retrieval implementation or graph-based personalized PageRank. |
| Self-RAG | Not implemented | No reflection tokens, retrieval critic, or self-evaluation loop. |
| CRAG | Partial | `corrective` checks question-term overlap and expands vector retrieval; no learned evidence evaluator or external corrective search. |
| Adaptive RAG | Partial | `adaptive` routes relationship-related keywords to `graph` and other questions to `hybrid`; no learned router. |
| Agentic RAG | Partial | Companion [agentic-system](https://github.com/MasterLKH180cm/agentic-system) executes bounded retrieval/synthesis DAGs; no built-in iterative retrieval, reflection, or replanning loop. |
| Multimodal RAG | Partial | Optional `/images` indexes a vision model's text caption; no native image-vector, audio, or video retrieval. |
| Structured RAG | Not implemented | JSON/CSV files are ingested as text; no schema-aware table engine, SQL generation, or structured-data query connector. |
| Federated RAG | Not implemented | Queries use one local tenant-scoped store; no routing or merging across independent knowledge sources. |
| Temporal / Real-time RAG | Not implemented | Manual upserts and timestamps exist, but no time-aware retrieval, historical snapshots, validity intervals, streaming ingestion, or automatic source refresh. |

Implementation: [`rags/retrieval.py`](rags/retrieval.py), [`rags/storage.py`](rags/storage.py), [`rags/providers.py`](rags/providers.py), and [`rags/vision.py`](rags/vision.py). See [architecture and non-goals](docs/ARCHITECTURE.md) for additional boundaries.

## Chunking and tokenization

The current splitter is heading-aware character windowing. In [`split_text`](rags/retrieval.py), each Markdown section is split into windows of up to **850 Python characters** with **120-character overlap**. When possible, the end moves to a space in the latter half of the window. These are function defaults, not API-selectable chunking strategies or model-token counts.

| Requested chunking approach | Status | What the code does |
| --- | --- | --- |
| Fixed-size | Implemented for characters | Bounded overlapping character windows, with a space boundary preference; no token-sized window option. |
| Recursive | Not implemented | No recursive separator hierarchy or recursive splitter. |
| Semantic | Not implemented | No sentence embeddings, similarity breakpoints, or topic-boundary detection. |
| Structure-aware | Partial | Recognizes Markdown `#` through `######` headings; no layout-aware PDF, HTML, table, or code parser. |
| Parent-child | Not implemented | Stores a document ID and ordinal per chunk; adjacent-context expansion does not create separate parent/child chunk levels. |
| Contextual | Partial | Prepends document title and section to each embedding input; no generated context or context-driven chunk boundaries. |

Token processing is separate from chunking:

- **Sparse index:** SQLite FTS5 uses the `unicode61` tokenizer. Query terms come from Python Unicode `\w+`, case folding, and a small English stop-word set; at most 12 distinct terms are joined with `OR`.
- **Local vectors:** regular-expression word features and overlapping character trigrams are hashed into 384 dimensions. This is neither a trained tokenizer nor a semantic embedding model.
- **Model tokens:** no `tiktoken`, SentencePiece, or provider token-counting adapter is used. External providers tokenize their own inputs. There is no model-aware input budget, token-based chunking, multilingual word segmenter, or token-cost accounting.
- **Generation context:** selected text is truncated to 14,000 characters. This character cap does not guarantee that a configured model's token window will be respected.

Sources: [`rags/retrieval.py`](rags/retrieval.py), [`rags/providers.py`](rags/providers.py), and [`rags/storage.py`](rags/storage.py).

## Model configuration

| Setting | Default | Options or purpose |
| --- | --- | --- |
| `RAGS_EMBEDDING_PROVIDER` | `local` | `local` or `openai` |
| `RAGS_MODEL_PROVIDER` | `extractive` | `extractive`, `openai`, or `anthropic` |
| `RAGS_DB_PATH` | `data/rags.db` | SQLite database location |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model when using OpenAI |
| `OPENAI_CHAT_MODEL` | `gpt-4.1-mini` | Generation model when using OpenAI |
| `ANTHROPIC_CHAT_MODEL` | `claude-sonnet-4-5` | Generation model when using Anthropic |

These model names are repository defaults. Select a model available to your provider account. To enable OpenAI embeddings and generation, put the following in your local Compose `.env`, or export the same variables in the Python service shell:

```dotenv
RAGS_EMBEDDING_PROVIDER=openai
RAGS_MODEL_PROVIDER=openai
OPENAI_API_KEY=your-local-secret
```

For Anthropic generation, set `RAGS_MODEL_PROVIDER=anthropic` and `ANTHROPIC_API_KEY`. Embeddings remain independently configurable: the available embedding adapters are local hashing and OpenAI; Anthropic is a generation/caption adapter here.

After editing Compose `.env`, rerun `docker compose up -d --build --wait` to recreate the service with the new environment. For Python, export the new variables and restart Uvicorn. Re-ingest documents after changing embeddings. Keep credentials outside Git.

## MCP integration

The optional stdio bridge exposes `search_knowledge` and `list_knowledge_documents` for Codex, Claude Code, and other MCP clients. Start the HTTP service first, then use a Python environment with the `mcp` extra installed:

```bash
export RAGS_URL=http://127.0.0.1:8000
export RAGS_API_KEY=local-development-key
python -m rags.mcp_server
```

The process waits for an MCP client over stdio. In a client configuration, register command `python` with arguments `["-m", "rags.mcp_server"]`, or use the absolute path to your virtual environment's Python. Supply the two environment variables locally. See [Codex configuration](docs/codex-mcp.toml) and [Claude Code configuration](docs/claude-mcp.json).

Client choice and model provider are independent: using Codex or Claude Code through MCP does not select the server's embedding or generation model.

## Testing and evaluation

With the Python development extras installed:

```bash
RAGS_ENV=test pytest -q
ruff check rags tests --select E4,E7,E9,F
RAGS_ENV=test python -m rags.evaluate --dataset examples/eval_baseline.json --k 3
```

[CI](.github/workflows/ci.yml) runs static checks, integration tests, an offline labelled retrieval evaluation, and a real HTTP health check. The evaluator reports Recall@K, Precision@K, MRR, and mean latency using a temporary database and local providers. The small [sample dataset](examples/eval_baseline.json) demonstrates the evaluation format; use held-out domain questions for quality comparisons. Offline tests do not verify live external model calls.

| Evaluation requirement | Current coverage |
| --- | --- |
| Retrieval relevance | Offline Recall@K, Precision@K, and MRR over labelled document IDs; the sample contains three cases. |
| Answer quality | Not implemented: no automated groundedness, correctness, citation-validity, or refusal-quality evaluator. |
| Latency | Mean retrieval latency only; no answer-generation timing or p50/p95 report. |
| Cost | Not implemented: no provider usage collection, token counts, or monetary cost report. |

## Deployment boundaries

The development configuration is for local use. Production defaults to requiring `RAGS_KEYS_JSON`, a JSON mapping from API keys to tenant IDs; production keys must contain at least 24 characters. Configure it through your runtime or secret manager. Tenants are derived from credentials and cannot be selected in request bodies.

Before exposing a deployment to users, provide identity integration, document-level permissions where required, TLS, traffic limits, backups, and operational monitoring. SQLite and full vector scans suit small deployments; horizontal scaling requires a shared database and suitable indexes. Local hashing provides a deterministic baseline rather than trained semantic embeddings. Image captions need verification against the original; native image-vector, audio, and video retrieval are not implemented.

See [Architecture](docs/ARCHITECTURE.md) for storage, trust boundaries, and index migrations.

## Project map

| Path | Responsibility |
| --- | --- |
| [`rags/api.py`](rags/api.py) | HTTP routes, authentication, and input limits |
| [`rags/retrieval.py`](rags/retrieval.py) | Chunking, ingestion, retrieval strategies, and evidence assembly |
| [`rags/storage.py`](rags/storage.py) | Tenant-scoped SQLite storage and indexes |
| [`rags/providers.py`](rags/providers.py), [`rags/vision.py`](rags/vision.py) | Embedding, generation, and optional caption adapters |
| [`rags/mcp_server.py`](rags/mcp_server.py) | MCP stdio tools |
| [`rags/evaluate.py`](rags/evaluate.py), [`tests/`](tests/) | Offline evaluation and automated tests |

Licensed under the [MIT License](LICENSE).
