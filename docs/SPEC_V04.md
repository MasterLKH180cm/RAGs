# v0.4: explicit feature configuration and PostgreSQL vector storage

Status: specified before implementation, 2026-11-22.
Baseline: merged v0.3 `8a53129c2cddf813d4956e4f2a94ea46366e78d8`.

## Requested outcome

Make existing optional functionality operator-controlled, document installation
and use, and make PostgreSQL with pgvector the normal deployed data store.
The changes span storage, API/model configuration, feature boundaries and real
service tests. They deliberately exceed a small single-file change; each concern
must remain in a focused module. Existing research/quality limitations remain.

## Configuration contract

- Operational settings are validated at process startup. Existing environment
  names remain supported; the README and environment template must enumerate
  settings, defaults, prerequisites and restart requirements.
- `RAGS_FEATURES_FILE` optionally names a UTF-8 JSON object of boolean feature
  overrides. `RAGS_FEATURES_JSON` overrides its individual entries. Unspecified
  entries retain existing availability. Unknown keys, non-booleans, malformed
  documents and unreadable explicitly selected files fail configuration.
- A single manifest defines the feature names and defaults. Cover ingestion,
  upload, reindex, deletion, query, API documentation, MCP, model generation,
  federation, research engines, captioning, original-image retrieval, each
  retrieval strategy, each chunker, optional index builders and tokenizers.
  Existing provider/backend/config prerequisites remain necessary when a flag
  is true. A flag never implements an unavailable upstream engine.
- Apply gates in service/model methods as well as HTTP/MCP entry points, before
  provider calls or mutations. Disabled operations return typed
  `feature_disabled` failures (HTTP 403); disabled documentation is unregistered.
  Internal retrieval dependencies must also respect their flags. Do not silently
  substitute another provider, strategy or store when a selected feature fails.
- Authentication, tenant authorization, input bounds, citation provenance,
  revision consistency and required evidence policies remain mandatory controls.
  Optional calibrated scoring may be disabled only when policy does not require
  it; an incompatible required policy must fail configuration.
- Authenticated capability/health output reports resolved flags and storage
  implementation without credentials, connection URLs or secret-file contents.
- Separate RAG embedding and generation settings:
  `RAGS_EMBEDDING_BASE_URL`, `RAGS_EMBEDDING_API_KEY`,
  `RAGS_EMBEDDING_MODEL`, `RAGS_GENERATION_BASE_URL`,
  `RAGS_GENERATION_API_KEY`, `RAGS_GENERATION_MODEL`. They override the relevant
  legacy OpenAI/Anthropic settings; provider selectors remain
  `RAGS_EMBEDDING_PROVIDER` and `RAGS_MODEL_PROVIDER`. Secret values may use
  matching `_FILE` settings; simultaneous nonempty value/file is rejected.
  Endpoint identities participate in existing index compatibility checks.

## PostgreSQL and pgvector

- `RAGS_DATABASE_URL` selects PostgreSQL; an explicitly selected PostgreSQL
  backend never falls back to SQLite. Keep `RAGS_DB_PATH` for explicit standalone
  development/legacy compatibility and existing offline test fixtures.
- The supplied deployment uses PostgreSQL 17 with released pgvector 0.8.7.
  Bootstrap installs the vector extension before least-privilege application
  connections initialize the application schema. No database host port is
  exposed in the default Compose stack.
- Persist document/chunk history, graph/source provenance, audit records and
  image assets in PostgreSQL. Text/image vectors use the actual `vector` type.
  Their dimensions remain attached to the model/index profile; do not impose a
  single global dimension that prevents a safe model transition/reindex.
- Dense queries execute pgvector cosine-distance SQL. Tenant and the exact
  authorized revision/filter candidate set constrain the query before ranking
  and limit. Check all authorized candidates for incompatible model, endpoint
  or dimensions and return `requires_reindex`; never hide bad rows by dropping
  them. Preserve deterministic tie ordering and zero-vector cosine semantics.
- Validate vector dimensions and finite float32 representation before writes.
  pgvector's representable dimension limit applies explicitly. Never corrupt an
  existing source when provider output cannot be stored.
- Preserve atomic source/revision/vector/full-text/graph/audit updates, CAS,
  historical/as-of visibility, deletion and concurrent-writer behavior.
  Serialize conflicting document mutations; ordinary reads must not take a
  database-wide writer lock. Use explicit revision ordinals for timestamp ties.
- PostgreSQL sparse search uses its full-text search and `ts_rank_cd` with a GIN
  index; document this backend-specific ranking instead of calling it SQLite
  BM25. Hybrid search still combines ranked lists with RRF.
- This version uses exact vector search, with relational/filter and full-text
  indices. It does not claim an ANN/HNSW performance result. Research sidecars
  retain their explicitly temporary upstream-specific indexes.
- Provide an explicit, transactional SQLite-to-PostgreSQL import command that
  preserves IDs, revisions, provenance and image bytes, refuses a nonempty
  target and never deletes or silently modifies the SQLite source. Selecting a
  new DSN alone is not a migration. Document backup, quiescence and rollback.

## Installation and acceptance

- Provide a complete standalone PostgreSQL Compose path and coordinated Agent
  deployment with separate RAG/Agent databases and application roles. Forward
  documented configuration, including feature files/maps, model routes and
  credentials, consistently. Optional dependencies must be installed when used.
- Preserve existing strict unit/core/external classification, inventories,
  no-selected-skip checks and fixed statement/branch coverage floors. Include
  actual PostgreSQL coverage when measuring the new database implementation;
  never exclude it to meet a floor.
- Add negative controls for flags/invalid configuration and positive controls
  for enabled functionality, provider separation, internal strategy dependencies
  and disabled MCP/HTTP paths. No external model billing is needed for these.
- Actual PostgreSQL tests require `RAGS_TEST_POSTGRES_URL`; missing prerequisites
  fail an explicitly selected gate. Verify vector column types and real SQL
  ranking, tenant/filter decoys, incompatible profiles, zero vectors, updates,
  CAS/rollback, historical visibility, deletion, images and SQLite import.
- Real HTTP E2E uses `RAGS_E2E_POSTGRES_URL`, checks reported backend identity,
  restart persistence and source lifecycle, and flows through an independent
  Agent worker and MCP. Retain existing browser/recovery/research/admission CI.
- Complete independent review, publish exact-revision evidence, and merge only
  after required CI succeeds. No model-quality profile is promoted by this work.
