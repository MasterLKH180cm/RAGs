# RAGs — runnable retrieval platform

A tenant-scoped FastAPI knowledge retrieval service that runs **offline by default**
and can use OpenAI embeddings and OpenAI-compatible / Anthropic text generation.
Designed to serve the [agentic-system](https://github.com/MasterLKH180cm/agentic-system)
workflow runtime through an authenticated API.

> Version 0.1 is a working foundation, not a claim to fully reproduce every
> published RAG research method. Capability and limitations are explicit below.

## Start locally

Python 3.11+:

    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev,pdf,mcp]"
    cp .env.example .env
    export RAGS_ENV=development
    export RAGS_API_KEY=local-development-key
    uvicorn rags.api:app --reload

Or after copying and configuring .env:

    docker compose up --build

The compose service is bound to localhost, not exposed publicly.

## Index a document

    curl -X POST http://127.0.0.1:8000/documents \
      -H 'Authorization: Bearer local-development-key' \
      -H 'Content-Type: application/json' \
      -d '{"title":"Session policy","document_id":"session-v1","source":"manual", "text":"# Login\nRefresh tokens rotate on login. Sessions expire after 30 minutes."}'

    curl -X POST http://127.0.0.1:8000/query \
      -H 'Authorization: Bearer local-development-key' \
      -H 'Content-Type: application/json' \
      -d '{"question":"When does the session expire?","mode":"hybrid"}'

The response contains evidence chunk IDs, source, title, section and scores.

## Strategies

| Strategy | Actual implementation / boundary |
| --- | --- |
| sparse | SQLite FTS5 BM25 |
| dense | offline hashing similarity; real OpenAI embeddings optional |
| hybrid | BM25 + cosine vector retrieval, RRF, lexical second-stage reranking |
| fusion | deterministic query alternatives + multi-retrieval RRF |
| hyde | hypothetical answer when an LLM is configured; offline fallback otherwise |
| contextual | heading/title context included in indexed embeddings |
| hierarchical | adjacent child chunks returned with selected chunks |
| graph | explicit A -> B edges; lightweight graph-assisted retrieval, NOT full GraphRAG |
| corrective | evidence-term overlap triggers expansion; NOT published CRAG |
| adaptive | simple intent-based retrieval selection, NOT trained Adaptive-RAG |

The API also handles text extraction from TXT, Markdown, JSON, CSV and PDF uploads.
Image/audio understanding and true multimodal retrieval are NOT implemented.

## Choose real providers

Set environment variables in your runtime, not in Git:

    RAGS_EMBEDDING_PROVIDER=openai
    RAGS_MODEL_PROVIDER=openai
    OPENAI_API_KEY=...
    OPENAI_EMBEDDING_MODEL=text-embedding-3-small
    OPENAI_CHAT_MODEL=gpt-4.1-mini

Or set RAGS_MODEL_PROVIDER=anthropic and ANTHROPIC_API_KEY plus
ANTHROPIC_CHAT_MODEL for Claude generation. Codex is a **developer agent**, not
a synonym for an OpenAI text-model API.

**Important:** switching embedding models requires re-ingesting documents.

## Codex and Claude Code

Both can use this RAG service through a standard MCP stdio server:

    pip install -e ".[mcp]"
    RAGS_URL=http://127.0.0.1:8000 \
      RAGS_API_KEY=local-development-key \
      python -m rags.mcp_server

Configure an MCP stdio server in the Codex or Claude Code client with command
"python", args ["-m","rags.mcp_server"], and environment variables RAGS_URL and
RAGS_API_KEY provided **through local configuration or a secret manager**.
Do not commit private keys in .mcp.json or config.toml.
AGENTS.md and CLAUDE.md contain consistent implementation instructions.

## Security / scaling

Production **fails startup** until RAGS_KEYS_JSON maps strong, unique API keys
to tenant IDs. Example: RAGS_KEYS_JSON='{"<strong-secret>":"tenant-acme"}'.
Use a trusted OIDC/API Gateway, document-level permissions, traffic quotas,
TLS, backups, secrets and centralized audit before exposing to real users.
See [Architecture](docs/ARCHITECTURE.md).

## Validation

    RAGS_ENV=test pytest -q
    ruff check rags tests --select E4,E7,E9,F

CI runs real HTTP server smoke tests as well as tenant-isolation and retrieval
integration tests. No external model key is needed for offline test mode.
