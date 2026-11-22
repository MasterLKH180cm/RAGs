# uv workflow and deployment clarity

Status: implementation specification, 2026-11-22.

## Goal

Use uv to reduce dependency installation and repeated environment setup work,
while preserving the existing API, optional integrations, recorded model profiles,
test coverage requirements, and PostgreSQL/pgvector deployment.

## Required behavior

- Commit a generated `uv.lock` and a Python 3.12 development pin. Keep the
  supported Python requirement unchanged unless resolution proves an existing
  dependency cannot support it; document any necessary compatibility change.
- Use uv 0.13.9 in CI and Docker. Pin setup-uv to verified commit
  `68559e99cc807c1eb7e9238fd43a15bba460eec2` (v10.3.3), and cache uv downloads
  using the project lock and isolated requirements as cache inputs.
- Install core development/CI environments from the lock with explicit extras.
  Ordinary offline setup must not install all research/model extras. Retain
  existing tests, inventories, resource boundaries and coverage floors.
- Docker must use uv, preserve the unprivileged runtime user and configurable
  extras, reuse the dependency layer when only source changes, and invoke the
  installed interpreter without resolving packages during application startup.
- Use uv for the isolated RAPTOR, LightRAG and answerability installers, retaining
  the existing pins, trusted upstream identities, CPU variants and separate
  environments. Preserve the frozen model/profile compatibility checks.
- README commands must identify the environment they run in, avoid accidentally
  removing selected extras, and explain how to install/update uv and refresh the
  lock. Leave historical evidence intact; update active setup instructions.
- Make the storage distinction explicit: default Compose uses PostgreSQL 17 and
  pgvector 0.8.7; direct local startup without a DSN uses SQLite. Show authenticated
  health verification and the existing full-stack verification command.
- Describe native pgvector exact ranking accurately. Core storage does not use
  FAISS; the isolated RAPTOR dependency set contains faiss-cpu, while this
  integration selects the upstream tree retriever.

## Validation

Generate and check the lock; sync an isolated environment using the documented
extras; run the existing unit/integration tests and lint; preserve real encoder,
research, model-runtime and PostgreSQL/Compose CI jobs. Record warm repeat-sync
time only as an observed installation baseline, not as API acceleration or a
pip-versus-uv speedup claim. Verify the real Compose deployment in CI if local
Docker is unavailable.

## Scope limits

uv changes installation and environment management, not SQL query capacity,
tokenizer algorithms or model inference speed. PostgreSQL indexing, pooling,
transaction locking and production load testing remain separate measured work.
Do not change quality targets or reclassify the failed historical quality profile.
