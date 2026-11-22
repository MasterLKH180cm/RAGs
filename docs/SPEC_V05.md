# SPEC V05 — Installable CLI, read-only diagnostics, configuration profiles, and PostgreSQL retrieval profiling

Status: accepted for implementation on a feature branch. This document is written before executable changes.

## Scope and compatibility

- Ship an installable `rags` console script via the existing `rags-platform` Hatch wheel, usable with `pip`, `uv tool install`, and `uvx --from git+https://github.com/MasterLKH180cm/RAGs`. Do not claim publication to PyPI.
- Preserve `uv.lock`, Python >=3.11 compatibility, API contracts, feature gates, default hashing/extractive behavior, research-sidecar isolation, and the distinction between native SQLite and PostgreSQL+pgvector.
- No new mandatory runtime dependencies, destructive database operations, live provider calls, or implicit fallbacks. Tooling does not bypass any existing API authentication, feature, tenant, revision, or model checks.
- A separate Agent service is not bundled into the RAG wheel.

## Commands and exit codes

1. `rags init --profile {offline,postgres,openai,ollama,anthropic} --directory DIR`
   - Copy a versioned operator profile into a new `.env` file and a nonsecret feature manifest, creating a random >=32-byte API key with a token-to-tenant map. Never overwrite files/symlinks or echo secrets.
   - Both files must be created atomically on a best-effort basis (rollback newly created files on error); .env permission 0600. `--directory` must be an existing directory; do not change permissions of existing paths.
   - Provider secrets are never manufactured or embedded. Placeholder PostgreSQL DSNs must be edited before use, and deployment must fail closed.
2. `rags doctor --env-file PATH [--no-db] [--json]`
   - Load shell-quoted KEY=VALUE lines without code execution; explicit process environment overrides file values. Reject malformed lines, duplicate keys, symlinks, and oversized config.
   - Validate `Settings.from_env`, provider endpoint/model compatibility, credentials present (not key contents), enabled/disabled feature counts and PostgreSQL/SQLite selection.
   - By default perform ONLY read-only database introspection. For PostgreSQL, verify reachability, `vector` extension and types of the two application vector columns. For SQLite, open an existing file in `mode=ro`; absence means `not_initialized`, not creation.
   - No model inference, no secret values/passwords/DSN echoes, no migrations, schema initialization, or persistent writes. Bound timeouts; machine JSON plus readable text. Exit 0 on ready, 1 on warning/error, 2 on CLI misuse.
3. `rags serve --env-file PATH [--host 127.0.0.1] [--port 8000]`
   - Load and validate settings before starting the existing FastAPI ASGI app through Uvicorn. Loopback default. `--host 0.0.0.0`/non-loopback requires `--allow-public` to avoid accidental exposure.
   - Native SQLite and PostgreSQL remain explicit; serve is not a database bootstrapper.
4. `rags benchmark --env-file PATH [--documents N] [--chunks-per-document N] [--iterations N] [--json]`
   - Requires explicit PostgreSQL DSN and postgres extra, extension already installed. Use disposable TEMPORARY TABLES in the selected database, with synthetic deterministic data, not real patient/client records. No network model calls or production table writes.
   - Warm up and report old vs scoped SQL path p50/p95 latencies, output row counts and query profile (including the same server-side filtered join used by `Store.scope`). Never present synthetic observations as production throughput.

## PostgreSQL read path optimization

- `Store.scope()` must choose one valid revision per document first, apply the same Python metadata/source/validity predicates, and use a *single database connection and snapshot* for revision and chunk reads.
- SQLite retains compatibility and post-filtering. PostgreSQL pushes down the selected (document_id,revision) keys to a parameterized `jsonb_to_recordset` join instead of loading every tenant chunk and discarding most; empty selection short-circuits.
- `include_embeddings=False` must never fetch vectors. Do not change dense exact cosine, profile compatibility validation, sparse ordering, document admission, or post-read metadata decoration.
- No blanket vector approximate index. Capture repeated measurements before considering pool/HNSW tuning.

## Security, tests, and CI release gates

- Unit tests: env parsing/precedence/duplicate guard; private `init` and rollback; profile completeness; non-leaking `doctor`; read-only SQLite; rejected PostgreSQL/pgvector mismatch; argument validation; public bind safeguard.
- Real PostgreSQL integration: compare scoped rows against SQLite semantics across tenant, metadata, source, historical validity and revision collisions; verify zero rows require no chunk query, no vectors when excluded, retained exact search.
- Package test: build Hatch wheel, install it into an isolated venv, invoke `rags --help`, `rags init` and `rags doctor --no-db` with no checkout import path.
- Reproducible benchmark: CI runs small synthetic sample against actual PostgreSQL+pgvector and uploads JSON as evidence; do not make performance thresholds a flaky functional gate.
- Preserve existing unit/integration/E2E and coverage floors. Document measured vs hypothetical claims clearly.

## Exclusions

PyPI release, production load guarantees, pooled connections, ANN vector search, cloud provider credential probes and an Agent CLI are outside this RAG wheel change unless separately implemented and verified.
