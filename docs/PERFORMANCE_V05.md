# V0.5 Synthetic PostgreSQL Retrieval Performance

Run: [GitHub Actions 45041879522](https://github.com/MasterLKH180cm/RAGs/actions/runs/45041879522), UTC 2026-11-22. The actual PostgreSQL/pgvector integration suite and the full project wheel-install test were also successful in this run.

## Mechanism

Before: `Store.scope` read every chunk for the tenant into Python and discarded chunks whose source revision or document metadata was not selected. After: documents are still selected and metadata-checked using the original authorization and valid-time semantics, then the `(document_id, revision)` keys are passed to a parameterized PostgreSQL `jsonb_to_recordset` join. Both reads share one read-only REPEATABLE READ transaction. The existing `chunks_revision` B-tree index enables efficient scoped reads, and `include_embeddings=False` avoids transferring vector data. Exact pgvector distance, tenant predicates, score ordering, and profile validation remain unchanged.

## Measurements

GitHub-hosted Ubuntu runner, real PostgreSQL 17 with pgvector 0.8.7, synthetic deterministic records in TEMPORARY tables, one connection, 3 warmups, 10% of documents selected. Each sample includes server execution, client fetch, and the old path's Python-side filtering. Values in milliseconds.

| Synthetic dataset | Approach | p50 | p95 | Authorized output rows | Timed samples |
| --- | --- | ---: | ---: | ---: | ---: |
| 40 documents × 4 chunks (160 chunks) | Old tenant-wide scan + Python filtering | 0.375 | 0.678 | 16 | 5 |
| 40 documents × 4 chunks (160 chunks) | New SQL scoped revision join | 0.392 | 0.602 | 16 | 5 |
| 40 documents × 4 chunks (160 chunks) | Exact scoped pgvector cosine top-10 | 0.449 | 0.642 | 10 | 5 |
| 500 documents × 8 chunks (4,000 chunks) | Old tenant-wide scan + Python filtering | 3.548 | 3.918 | 400 | 10 |
| 500 documents × 8 chunks (4,000 chunks) | New SQL scoped revision join | 1.484 | 1.622 | 400 | 10 |
| 500 documents × 8 chunks (4,000 chunks) | Exact scoped pgvector cosine top-10 | 1.324 | 1.516 | 10 | 10 |

On the 4,000-chunk workload, the revision-scoped query was **2.39× faster on p50 latency (58.2% reduction)** and **2.42× faster at p95 (58.6% reduction)**. On the 160-chunk workload there was **no stable improvement** (new p50 was marginally slower). This suggests scope pushdown primarily helps when large numbers of chunks would otherwise be transferred and discarded.

These are **isolated synthetic microbenchmark results**, not clinical production workload claims, model-generated retrieval quality benchmarks, or statistically robust p99/throughput measurements. Exact cosine is a separate workload and is not comparable to old scope fetch as a speedup. The query plan and precise measurements are preserved in CI's `rags-core-test-evidence` artifact as `pgvector-synthetic-benchmark.json` and `pgvector-synthetic-benchmark-large.json`.

## Reproduce

```bash
uv sync --locked --extra postgres --extra dev
# Use a disposable PostgreSQL/pgvector database with CREATE TEMP privilege.
export RAGS_DATABASE_URL='postgresql://...'
uv run --no-sync rags benchmark --documents 40 --chunks-per-document 4 --iterations 5 --json
uv run --no-sync rags benchmark --documents 500 --chunks-per-document 8 --iterations 10 --json
```

This does not write application tables. PostgreSQL must have the `vector` extension installed by the administrator. The timing is influenced by hardware, PostgreSQL planning/statistics, data distribution and client connection costs; performance thresholds are intentionally not CI pass/fail gates. The existing query path still must inspect source revision metadata in the application. Evaluate connection pooling, server-side filtering strategies, ANN recall tradeoffs and longitudinal source cardinality only with actual workloads and separate regression tests.
