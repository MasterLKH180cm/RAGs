# Collection validation and measured performance

Measured on 2026-11-26. The specification was committed before implementation;
this report covers the additive collection interface and the existing package
regression gates. The implementation retains the established 80% line /72% branch
minimums and the complete-package coverage/exclusion checks.

## Verification scope

The final local core run passed 1,632 unit tests, 742 integration tests and 55
actual PostgreSQL/pgvector tests without skipped selected cases. Three additional real HTTP/CLI subprocess cases passed against SQLite.
The PostgreSQL total includes the corresponding actual HTTP/CLI lifecycle.

The complete 63-module Python package measured 86.10% line coverage and 80.07%
branch coverage. The ordinary local core result excludes optional model-resource
and browser tests. CI also executes the actual BGE collection lifecycle and
full-input token-window rejection, existing trained encoder/research/Qwen runtime
gates, actual browser acceptance, and the Compose deployment gate. Selected
external gates require execution rather than silently skipping missing resources.

The standalone wheel was installed into an isolated environment and exercised
through authenticated local create/add/search/fork/update/ask operations. Public
client import never initializes the API application or loads a model. Required
Ruff gates and `git diff --check` passed.

Important regressions include cross-tenant access, explicit scopes and feature
gates before work, duplicate and oversized JSON, atomic provider/SQL failures,
concurrent CAS, consistent snapshots during physical version cleanup, shared
fork references, whole-batch filter validation, safe regex timeouts, bounded
mutation responses before commit, source-version citations, real pgvector column
types and transactional SQLite import. Unicode control-response and accumulated
provider-usage overflow both fail without committing a mutation.

## Reproduce the gates

```bash
uv sync --locked --extra dev --extra postgres --extra pdf --extra mcp --extra vision
python -m pytest -q -m unit --require-no-skips
python -m pytest -q -m "integration and not external" --require-no-skips
# Provision a disposable PostgreSQL database with the vector extension first.
RAGS_TEST_POSTGRES_URL='postgresql://...' \
  python -m pytest -q -m postgres --require-no-skips
python -m pytest -q tests/test_collection_http_e 2 e.py -m "not postgres" --require-no-skips
```

The CI workflow records JUnit, explicit test inventories, complete-package
coverage, actual model evidence and both synthetic collection benchmarks. Model
quality results are separate from these collection correctness/latency tests.
Existing frozen Qwen profiles, datasets, scores and promotion thresholds were
not altered or relabelled by this change.

## Synthetic measurements

Both backends used 200 records,64 float32 dense dimensions,2 sparse entries per
record,50-record mutation batches,10 measured repetitions,2 search warmups and
seed 20261126. Queries request up to 10 results. The environment was Linux x86_64,
Python 3.12.14, with 9 visible CPUs. The benchmark is single-client and synchronous.

Timings include the authenticated local Python SDK, application service and
database. They exclude HTTP transport, model encoding, setup and post-operation
assertions. Vectors were precomputed synthetic inputs; model calls were zero.
All 8 correctness checks passed and cleanup verified zero generated-tenant rows.
These are descriptive samples from a shared environment; ten observations do
not establish a production tail-latency SLA or comparative Chroma performance.

| Operation | SQLite p 50 / p 95 (ms) | PostgreSQL p 50 / p 95 (ms) |
| --- | ---: | ---: |
| Dense cosine search | 44.93 / 53.73 | 30.47 / 37.57 |
| BM25 search | 7.19 / 11.06 | 21.84 / 27.77 |
| Sparse dot-product search | 7.03 / 13.32 | 23.58 / 29.77 |
| BM25 + dense RRF | 46.78 / 56.97 | 29.90 / 33.68 |
| Sparse + dense RRF | 46.74 / 63.69 | 28.31 / 39.01 |
| Five-query batch | 58.24 / 77.80 | 45.61 / 124.29 |
| 50-record lookup | 2.17 / 62.82 | 8.95 / 14.97 |
| 50-record metadata update | 21.35 / 26.03 | 60.50 / 91.10 |
| 200-record fork | 2.46 / 3.01 | 16.61 / 21.53 |

Initial insertion throughput was 1809 records/second for SQLite
and 497 records/second for PostgreSQL. Insertion has four
50-record samples; the listed operations each have ten. Raw timing samples,
measurement boundaries, correctness checks and cleanup results are preserved in
[SQLite JSON](../reports/collections-sqlite.json) and
[PostgreSQL JSON](../reports/collections-postgres.json).

```bash
python scripts/benchmark_collections.py --records 200 --dimensions 64 \
  --batch-size 50 --iterations 10 --warmup 2 --report reports/collections-sqlite.json
RAGS_BENCHMARK_POSTGRES_URL='postgresql://...' \
  python scripts/benchmark_collections.py --postgres --records 200 --dimensions 64 \
  --batch-size 50 --iterations 10 --warmup 2 --report reports/collections-postgres.json
```

See [the usage guide](COLLECTIONS.md) and [implementation contract](COLLECTIONS_SPEC.md)
for record/scan limits, embedding spaces, missing/null filter semantics and the
O(record-count) fork mapping copy. No cloud-scale, ANN, learned sparse encoder or
model-quality capability is inferred from this bounded synthetic test.
