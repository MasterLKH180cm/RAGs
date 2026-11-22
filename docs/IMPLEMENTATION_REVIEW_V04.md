# v0.4 implementation and independent review

Status: independent review complete; all six automatic CI jobs passed.
The [specification](SPEC_V04.md), commit `a2c8b24`, precedes implementation
against baseline `8a53129`. See [PR #4](https://github.com/MasterLKH180cm/RAGs/pull/4)
and its [revision-specific checks](https://github.com/MasterLKH180cm/RAGs/pull/4/checks).

## Implemented boundaries

A validated boolean manifest controls existing optional functionality. Environment
JSON overrides file entries. HTTP, MCP and service methods enforce dependencies
before provider work or mutation; enabled flags still require actual backends.
Authentication, tenant scope, revision consistency and required evidence policy
remain mandatory. Embedding/generation endpoints, keys and models are independent;
capabilities omit secrets and DSNs.

## PostgreSQL, vectors and migration

Compose uses PostgreSQL 17/pgvector 0.8.7, administrator-installed extension and
separate application role. SQLite is explicit compatibility, never fallback.

Text and image vectors use native `vector` columns. Exact cosine SQL applies
tenant and authorized revision/filter IDs before ranking and limit. Preflight
checks every authorized model, endpoint/profile and dimension; mismatches require
reindexing. Finite float32 vectors support 1–16,000 dimensions; zero similarity is
preserved. Source inspection can omit vectors. PostgreSQL uses GIN full-text search
and `ts_rank_cd`; SQLite retains FTS5/BM25. No ANN acceleration is claimed.

Document mutations serialize conflicting tenant/document writes and atomically
commit source, revision, vector, full-text, graph and audit changes. Reads take no
global writer lock. Revision ordinals preserve timestamp ties; CAS, history and
tenant-scoped deletion remain enforced. Image bytes and encoder identity commit
with their vectors.

`python -m rags.migrate --sqlite SOURCE` opens SQLite read-only, refuses a
nonempty PostgreSQL target and imports application rows atomically without
inference, preserving IDs, history, provenance and image bytes. Failed imports
may retain schema bootstrap, never partial rows. Quiescence, backups and explicit
deployment switching remain operator responsibilities.

## Independent review corrections

| Finding | Correction and regression contract |
| --- | --- |
| Captioning preceded chunker/counter checks | Shared preflight runs before vision; prior sources remain unchanged. |
| Internal generation bypassed counter flags with `generate:false` | Strategy dependencies check generation counters; extractive packing checks before embedding. |
| Standalone image defaults ignored environment flags | Defaults use `FeatureFlags.from_env()`; JSON/file regressions prove direct/HTTP rejection without encoding or source changes. |
| Research counter permissions were incomplete | RAPTOR/LightRAG require tiktoken; generation/embedding counters are checked before corresponding work. |
| Revision history had unstable timestamp ties | Explicit backend tie ordering preserves history through initialization. |
| Restrictive umask prevented container configuration reads | Explicit new-file/directory permissions preserve private credentials; incompatible existing directories fail unchanged. |
| README credentials/ranking contradicted deployment | MCP retains generated tokens; SQLite links and backend-specific sparse/contextual descriptions were corrected. |

## Verification ledger

The reviewed implementation `f3c9506b850c0196517861b11cf51ca56d6ed991` passed
all six automatic jobs in [CI 44996488847](https://github.com/MasterLKH180cm/RAGs/actions/runs/44996488847).
JUnit and collection artifacts record **411 unit, 300 ordinary integration and
21 actual PostgreSQL tests**, with zero failures, errors or selected skips.
Complete-package coverage is **3,837/4,626 statements (82.94%)** and
**1,345/1,732 branches (77.66%)**, including PostgreSQL execution across all
30 production modules. Unchanged floors are 80% and 72%; the existing 22
protocol exclusions remain unchanged.

Separate successful jobs execute actual pretrained encoders, RAPTOR and LightRAG
HTTP lifecycles, admission runtime/report checks, and a real Compose deployment
with restrictive `umask 077`, native vector columns and restart persistence.
The manual full quality benchmark was not selected and is not counted as passed.
[Machine-readable results](testing-results-v04.json) bind counts, artifact digest,
per-file coverage and production SHA-256 hashes to this exact implementation.
Subsequent documentation-only commits retain those source identities; the PR's
current checks certify the complete final revision.

## Unchanged quality limits

Frozen profiles, datasets and quality reports were not promoted or retuned.
The Qwen profile remains disabled by default after failing its recorded quality
gates. Protocol/storage success establishes neither semantic grounding nor paid
provider quality. Research indexes remain temporary; throughput, ANN recall and
high availability are unmeasured.
