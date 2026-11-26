# RAGs

**Build a searchable knowledge base and return answers linked to their original sources.**

[![CI](https://github.com/MasterLKH180cm/RAGs/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/MasterLKH180cm/RAGs/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

RAGs gives you a Python SDK, an authenticated HTTP API and a CLI for storing,
searching and querying knowledge. It supports tenant isolation, versioned sources,
metadata filters and bounded agent workflows. Start locally with SQLite and no
model credentials; use the supplied PostgreSQL/pgvector deployment when you need
an application service.

The default uses deterministic hashing embeddings and extractive answers. It is
useful for learning the data flow without a model account. For semantic retrieval
or generated answers, configure a trained embedding model or generation provider
and evaluate it on your data.

[Quick start](#quick-start) · [Choose an interface](#choose-an-interface) ·
[Models](#use-a-model-provider) · [Production](#deploy-with-postgresql) ·
[Troubleshooting](#troubleshooting) · [Documentation](#documentation)

## Choose an interface

| You want to… | Start here |
| --- | --- |
| Store records with IDs, metadata and one vector per record | **Collections SDK/API** — the quick start below. |
| Ingest long documents, preserve source spans and manage revisions | **Document API** — [ingest and query a document](#ingest-and-query-a-document). |
| Use collections inside a local Python process | **PersistentClient** — [local SDK example](docs/COLLECTIONS.md#start-locally). |
| Call retrieval from an MCP client | **Read-only MCP bridge** — [configuration](docs/README_REFERENCE.md#mcp-and-api-reference). |
| Run approved tools, model routing and resumable workflows | **Agent Platform** — [agent guide](docs/README_REFERENCE.md#agent-platform-model-routing-workflows-and-memory-v08). Use [agentic-system](https://github.com/MasterLKH180cm/agentic-system) for its separate API and worker. |

Collections and documents share authentication and evidence policies. They keep
separate records: collection text is not automatically chunked or inserted into
the document API.

## Quick start

### 1. Install in a Python environment

You need **Python 3.11+ and Git**. These commands use Bash on Linux, macOS or WSL.
They install the SDK and CLI together from this repository into one isolated
Python environment.

```bash
mkdir my-rags
cd my-rags
python3 -m venv .venv
source .venv/bin/activate
python -m pip install "rags-platform @ git+https://github.com/MasterLKH180cm/RAGs.git"
```

The same environment provides `rags` and `rags.client`. Prefer an isolated CLI
installation? See the [uv tool and profile options](docs/README_REFERENCE.md#installable-cli-no-clone-needed).
For a source checkout and locked development dependencies, see [development](#development-and-validation).

### 2. Create configuration and start the service

<!-- rags-smoke: setup -->
```bash
rags init --profile offline
rags doctor --env-file .env --no-db
```

`init` creates a private `.env` with a random API key and a feature configuration
file. It preserves existing files. `doctor --no-db` checks configuration before
the database exists. Start the server in this terminal:

<!-- rags-smoke: serve -->
```bash
rags serve --env-file .env
```

Keep it running. The service listens at <http://127.0.0.1:8000>; interactive API
documentation is at <http://127.0.0.1:8000/docs>. SQLite data is saved under
`data/rags.db` in this working directory and survives restarts.

### 3. Store records and get your first answer

Open a **second terminal in `my-rags`**, activate the same environment, and load
the generated client credential. Only source the `.env` you just generated or a
configuration file you trust; keep it out of version control.

```bash
source .venv/bin/activate
```

<!-- rags-smoke: client_env -->
```bash
set -a
source .env
set +a
export RAGS_URL=http://127.0.0.1:8000
```

Run this SDK example in that second terminal:

<!-- rags-smoke: sdk -->
```bash
python - <<'PYTHON'
import os
from rags.client import HttpClient

with HttpClient(os.environ["RAGS_URL"], api_key=os.environ["RAGS_API_KEY"]) as client:
    notes = client.get_or_create_collection("product-notes")
    notes.upsert(
        ids=["backups", "retention"],
        documents=["Backups are encrypted at rest.", "Logs are retained for 30 days."],
        metadatas=[{"team": "security"}, {"team": "operations"}],
    )
    hits = notes.search(query_texts=["encrypted backups"], mode="lexical")
    print("Search:", hits.rows()[0]["document"])
    answer = notes.ask("How long are logs retained?", mode="lexical")
    print("Answer:", answer["answer"])
    print("Citations:", answer["citations"])
PYTHON
```

You should see the backup sentence, an answer containing **30 days**, and source
citations. `upsert` makes this example safe to repeat with the same record IDs.
`lexical` uses BM25 text ranking; it needs no embedding or generation API call.
Try `mode="hybrid"` after choosing an appropriate embedding model.

You can inspect the same collection from the CLI:

```bash
rags collection list --env-file .env
rags records get product-notes --ids retention --env-file .env
```

Continue with [collection writes, filters, embeddings, search and forks](docs/COLLECTIONS.md).
Raw search results report `ranked_unassessed`; a ranking score alone is not proof
that an answer is supported. [Evidence admission](docs/README_REFERENCE.md#evidence-admission-and-refusal)
explains the distinction and refusal behavior.

## Ingest and query a document

Use this API when a source needs automatic chunking, historical revisions or
source character offsets. With the server and second-terminal environment from
the quick start still active, add a document:

<!-- rags-smoke: document_add -->
```bash
curl --fail-with-body "$RAGS_URL/documents" \
  -H "Authorization: Bearer $RAGS_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"document_id":"session-policy","title":"Session policy","source":"manual","text":"Sessions expire after 30 minutes of inactivity. Refresh tokens rotate on login."}'
```

The response includes a revision and chunk count. Now ask a question:

<!-- rags-smoke: document_query -->
```bash
curl --fail-with-body "$RAGS_URL/query" \
  -H "Authorization: Bearer $RAGS_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"question":"When do sessions expire?","mode":"hybrid","top_k":3}'
```

Expect an answer containing **30 minutes** and citations to `session-policy`.
Responses also identify the requested/executed retrieval mode, evidence decision
and context budget. Set `"generate":false` to assess and retrieve evidence without
producing an answer. An evidence refusal returns `outcome:"insufficient_evidence"`
and no usable citations.

Changed content under the same document ID creates a revision; unchanged content
and processing settings return `unchanged:true`. See [ingestion and safe reindexing](docs/README_REFERENCE.md#ingest-and-query)
and [chunking options](docs/README_REFERENCE.md#chunking-and-token-budgets) before
changing models or index settings.

## Use a model provider

Embeddings and answer generation are configured separately. Changing only the
chat model does not improve the embedding index. Discover local English, Chinese,
multilingual and hosted choices without downloading anything:

```bash
rags models list --language zh
rags models show st-e5-small
rags tokenizers list
```

For a new deployment with local Chinese embeddings:

```bash
python -m pip install -e '.[encoders]'
mkdir chinese-rag
rags init --embedding bge-small-zh --directory chinese-rag
rags doctor --env-file chinese-rag/.env --no-db
```

Use `--interactive` for a terminal wizard, or select embedding and source/generation
tokenizers with independent flags. The [model selection guide](docs/MODEL_SELECTION.md)
compares runtimes and costs, explains custom models and registered tokenizer
artifacts, and shows how to preview an existing configuration change with
`rags config plan`. Model changes require deliberate re-embedding; collection
forks retain their original vector space.

Open a **fresh terminal** and
activate your Python environment before switching profiles: exported variables
take precedence over `--env-file`. Select a profile in a new, empty configuration
directory, then supply its required models or credentials:

```bash
mkdir model-config
rags init --profile ollama --directory model-config
rags doctor --env-file model-config/.env --no-db
```

`init` prepares configuration; it does not install a model server or download
weights. Stop the first server before using the same port.

| Goal | Profile or guide | What you supply |
| --- | --- | --- |
| Offline demonstration | `offline` | Nothing; hashing embeddings and extractive answers. |
| Trained local semantic retrieval | [Model and tokenizer selection](docs/MODEL_SELECTION.md) | `encoders` or `sentence-transformers` extra, selected model weights and index configuration. |
| Local Ollama models | `ollama` | Running Ollama plus the configured chat and embedding models. |
| OpenAI-compatible hosted models | `openai` | Separate embedding/generation credentials and endpoint/model settings. |
| Anthropic generation | `anthropic` | Generation credential; embeddings remain local unless changed separately. |
| Self-hosted Qwen | [Qwen setup and benchmark](docs/SELF_HOSTED_QWEN_SPEC.md) | Pinned runtime and Qwen3-1.7B Q8_0 weights; hardware preflight. |
| Inkling 2 Mini | `inkling` or `inkling-local`; [setup](docs/README_REFERENCE.md#inkling-2-mini) | Tinker credential, or a compatible self-hosted runtime and weights. |
| Shared model routing | `litellm`; [gateway guide](docs/LITELLM.md) | Running gateway, verified HTTPS, credential and approved immutable aliases. |

Router, agent and memory switches default to enabled. External execution still
requires a configured provider/gateway, approved models and the caller's scopes.
An unavailable model or backend produces an explicit requirement error.

**Model compatibility and answer quality are separate checks.** The frozen Qwen
answerability profile failed its quality gates (7 of 11 passed) and is not enabled by default;
its [results and limitations](docs/README_REFERENCE.md#evidence-admission-and-refusal)
remain visible. The local Qwen generation recipe is a separate model use case.
Use the [live benchmark contract](docs/MODEL_BENCHMARK_SPEC.md) for measured latency,
throughput and quality; a missing-credential or hardware preflight is not a
performance result.

## Deploy with PostgreSQL

The supplied Docker Compose stack runs PostgreSQL/pgvector and the RAG API with
separate database/application credentials. It exposes only loopback HTTP and
keeps the database private. You need Git, [uv](https://docs.astral.sh/uv/getting-started/installation/)
and Docker Compose with `--wait` support. From a fresh checkout:

```bash
git clone https://github.com/MasterLKH180cm/RAGs.git
cd RAGs
uv run --no-project --python 3.12 python scripts/bootstrap_compose.py
docker compose up -d --build --wait
set -a
source data/client.env
set +a
curl --fail-with-body "$RAGS_URL/health" \
  -H "Authorization: Bearer $RAGS_API_KEY"
```

Health should report `storage.backend=postgresql` and
`storage.vector_search=pgvector_exact`. This deployment also starts with offline
models. Native `rags serve` uses SQLite unless a PostgreSQL DSN is configured;
it never silently falls back from a selected PostgreSQL backend.

Before exposing a service, configure tenant roles, TLS/reverse proxy, quotas,
backups and monitoring for your application. Legacy keys without an explicit
role map have admin-equivalent access. Use [authentication and OAuth2](docs/README_REFERENCE.md#tenant-authorization-and-oauth2),
[full deployment instructions](docs/README_REFERENCE.md#postgresql-and-pgvector-with-docker),
and [migration/backups](docs/README_REFERENCE.md#database-migration-and-backups).
Changing a database URL does not migrate existing data.

Collection forks share immutable payloads and copy record mappings in O(record
count). Vector ranking is exact, with bounded scans; no ANN, high-availability or
Chroma Cloud throughput claim is made. See [collection limits and measurements](docs/COLLECTIONS_VALIDATION.md).
The collection API uses RAGs contracts; it is not Chroma wire compatibility.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `rags` is not found, or Python cannot import `rags` | Activate the environment where you installed the package. `python -m rags --help` checks that interpreter. A `uv tool` install does not install the SDK into another Python environment. |
| `init` says existing configuration was preserved | Reuse the existing files, or initialize a new empty directory. It intentionally does not overwrite credentials. |
| `doctor` warns that the schema is uninitialized | Use `--no-db` before first startup. Serving creates the schema; doctor is read-only. |
| HTTP 401 | Load the generated client credential in the requesting terminal and send `Authorization: Bearer …`. The server and client must use the matching configuration. |
| HTTP 403 | Check the token's role/scopes and effective feature permissions. A feature flag does not grant identity permissions. |
| A new profile seems ignored | Exported environment values override `--env-file`. Start a fresh terminal and use the intended profile file. |
| `requires_provider`, `requires_index` or `requires_backend` | Inspect authenticated `/strategies` and `/capabilities`; configure the dependency that the selected strategy requires. |
| Retrieval is weak or refuses an answer | Confirm the text exists in the correct collection/document store and tenant. Check filters, embedding identity and evidence decisions; evaluate a trained encoder for semantic matching. |
| `requires_reindex` or an embedding-space mismatch | Keep the stored embedding model/version and query encoder aligned. Follow the documented reindex procedure rather than mixing vectors from different spaces. |
| HTTP 409 during an update | Read the current revision/generation, reconcile the change and retry with that value. Blind retries can overwrite a concurrent writer's intent. |
| Port 8000 is occupied | Stop the previous server, or use `rags serve --env-file .env --port 8001` and update `RAGS_URL` in the client terminal. |

## Development and validation

From a checkout, install the locked development dependencies. Repeat all desired
extras on each `uv sync`; optional research/model runtimes have separate recipes.

```bash
uv sync --locked --extra dev --extra pdf --extra mcp --extra vision --extra postgres
uv run --no-sync python -m pytest -q -m unit --require-no-skips
uv run --no-sync python -m pytest -q -m "integration and not external" --require-no-skips
uv run --no-sync python scripts/check_readme.py --live --report reports/testing/readme-onboarding.json
```

The README verifier checks local documentation links and runs the marked
onboarding examples against a temporary loopback service, including authentication
and restart persistence. It uses the installed package and does not download
models. Actual PostgreSQL, encoders, upstream research engines, browser and
live-model checks are separate gates; the commands above do not run all of them.
See [test policy and evidence](docs/TEST_REVIEW.md), [collection validation](docs/COLLECTIONS_VALIDATION.md)
and the [uv workflow](docs/README_REFERENCE.md#uv-environment-workflow).

## Documentation

| Topic | Guide |
| --- | --- |
| SDK, CLI, filters, embedding spaces and dataset forks | [Collections](docs/COLLECTIONS.md) |
| Model choices, tokenizer artifacts, safe switching and approved collection profiles | [Model selection](docs/MODEL_SELECTION.md) |
| Architecture, retrieval modes and capability boundaries | [Architecture](docs/ARCHITECTURE.md) · [Twenty-family coverage](docs/README_REFERENCE.md#twenty-family-coverage) |
| Source chunking, citations, admission and historical queries | [Detailed reference](docs/README_REFERENCE.md#chunking-and-token-budgets) |
| Agent routing, approvals, workflows and memory | [Agent Platform](docs/README_REFERENCE.md#agent-platform-model-routing-workflows-and-memory-v08) |
| Administration, metrics and telemetry | [Dashboard and OpenTelemetry](docs/README_REFERENCE.md#admin-dashboard-and-opentelemetry-v07) |
| Settings, feature permissions and provider endpoints | [Configuration reference](docs/README_REFERENCE.md#feature-and-operational-configuration) |
| MCP and HTTP endpoint map | [API reference](docs/README_REFERENCE.md#mcp-and-api-reference) · [local interactive docs](http://127.0.0.1:8000/docs) |
| RAPTOR, LightRAG, GraphRAG, HippoRAG, Self-RAG and CRAG | [Research engines and requirements](docs/RESEARCH_ENGINES.md) |
| Deployment, migration, backup and recorded validation | [Operations and validation reference](docs/README_REFERENCE.md) |
| Code review findings, fixes and validation evidence | [Whole-repository review](docs/CODE_SMELL_REVIEW.md) |

Licensed under the [MIT License](LICENSE).
