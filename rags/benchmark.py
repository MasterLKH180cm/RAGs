"""Isolated, reproducible PostgreSQL/pgvector read-path benchmark.

Only creates TEMP tables: no production table, model call or source data is
changed. Numbers measure a synthetic workload, not production throughput.
"""
import json
import math
from statistics import median
from time import perf_counter

from rags.postgres import cosine_distance, is_postgres, vector_literal


def _summary(samples: list[float], rows: int) -> dict:
    ordered = sorted(samples)
    return {"p50_ms": round(median(ordered), 3),
            "p95_ms": round(ordered[math.ceil(0.95 * (len(ordered) - 1))], 3),
            "rows": rows, "samples": len(samples)}


def benchmark(dsn: str, *, documents: int = 200, chunks_per_document: int = 8,
              iterations: int = 20) -> dict:
    if not is_postgres(dsn):
        raise ValueError("Benchmark requires an explicit PostgreSQL URL (no SQLite fallback)")
    if not 10 <= documents <= 2000 or not 1 <= chunks_per_document <= 16:
        raise ValueError("Benchmark documents=10..2000 and chunks-per-document=1..16")
    if documents * chunks_per_document > 10_000:
        raise ValueError("Benchmark is limited to 10000 synthetic chunks")
    if not 3 <= iterations <= 100:
        raise ValueError("Benchmark iterations must be between 3 and 100")
    try:
        import psycopg
    except ImportError as exc:
        raise ValueError("Install rags-platform[postgres] to run the pgvector benchmark") from exc
    selected = {f"document-{index:05d}" for index in range(max(1, documents // 10))}
    pairs = json.dumps([{"document_id": name, "revision": "r1"} for name in sorted(selected)],
                       separators=(",", ":"))
    identity = "benchmark-tenant"
    # Exactly the predicates used by Store.scope; only the unoptimized path
    # transfers extra rows for Python-side authorization filtering.
    old_sql = (
        "SELECT c.id,c.document_id,c.revision,c.content FROM rags_bench_chunks c "
        "WHERE c.tenant=%s"
    )
    new_sql = (
        "SELECT c.id,c.document_id,c.revision,c.content FROM rags_bench_chunks c "
        "JOIN jsonb_to_recordset(%s::jsonb) AS scope(document_id text, revision text) "
        "ON scope.document_id=c.document_id AND scope.revision=c.revision "
        "WHERE c.tenant=%s ORDER BY c.id"
    )
    vector = vector_literal([1.0] + [0.0] * 31)
    dense_sql = (
        "WITH q AS (SELECT %s::vector AS embedding) "
        "SELECT c.id," + cosine_distance("c.embedding") + " AS distance "
        "FROM rags_bench_chunks c "
        "JOIN jsonb_to_recordset(%s::jsonb) AS scope(document_id text, revision text) "
        "ON scope.document_id=c.document_id AND scope.revision=c.revision "
        "CROSS JOIN q WHERE c.tenant=%s ORDER BY distance,c.id LIMIT 10"
    )
    queries = {
        "unfiltered_tenant_scan": (old_sql, (identity,)),
        "scoped_revision_join": (new_sql, (pairs, identity)),
        "exact_scoped_cosine": (dense_sql, (vector, pairs, identity)),
    }
    samples = {key: [] for key in queries}
    row_counts = {key: None for key in queries}
    with psycopg.connect(dsn, connect_timeout=5) as connection:
        connection.execute("SET LOCAL statement_timeout='20s'")
        extension = connection.execute(
            "SELECT extversion FROM pg_extension WHERE extname='vector'"
        ).fetchone()
        if extension is None:
            raise ValueError("The database must have the pgvector extension installed")
        version = connection.execute("SHOW server_version").fetchone()[0]
        connection.execute(
            "CREATE TEMP TABLE rags_bench_chunks "
            "(id text PRIMARY KEY,tenant text NOT NULL,document_id text NOT NULL,"
            "revision text NOT NULL,content text NOT NULL,embedding vector(32) NOT NULL) "
            "ON COMMIT DROP"
        )
        rows = []
        for index in range(documents):
            doc = f"document-{index:05d}"
            values = vector_literal([float(((index + dimension) % 7) + 1) / 10 for dimension in range(32)])
            for ordinal in range(chunks_per_document):
                rows.append((doc + ":" + str(ordinal), identity, doc, "r1",
                             f"synthetic source {index} passage {ordinal} " + "plain text " * 8, values))
        with connection.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO rags_bench_chunks "
                "(id,tenant,document_id,revision,content,embedding) "
                "VALUES (%s,%s,%s,%s,%s,%s::vector)",
                rows,
            )
        connection.execute(
            "CREATE INDEX ON rags_bench_chunks(tenant,document_id,revision)"
        )
        connection.execute("ANALYZE rags_bench_chunks")
        expected = len(selected) * chunks_per_document
        for iteration in range(iterations + 3):
            # Alternate the order to limit cache-warming bias.
            cases = list(queries.items())
            if iteration % 2:
                cases.reverse()
            for name, (sql, params) in cases:
                begin = perf_counter()
                fetched = connection.execute(sql, params).fetchall()
                if name == "unfiltered_tenant_scan":
                    fetched = [row for row in fetched if row[1] in selected and row[2] == "r1"]
                elapsed = (perf_counter() - begin) * 1000
                count = len(fetched)
                if count != (min(10, expected) if name == "exact_scoped_cosine" else expected):
                    raise ValueError("Synthetic benchmark result parity failed")
                if iteration >= 3:
                    samples[name].append(elapsed)
                    row_counts[name] = count
        plans = {}
        for name, (sql, params) in queries.items():
            result = connection.execute(
                "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql, params
            ).fetchone()[0]
            data = result[0]
            plans[name] = {
                "execution_ms": round(data["Execution Time"], 3),
                "shared_hit_blocks": data["Plan"].get("Shared Hit Blocks", 0),
                "temp_read_blocks": data["Plan"].get("Temp Read Blocks", 0),
            }
    return {
        "workload": "synthetic_temp_tables",
        "database": "postgresql+pgvector",
        "postgres_version": version,
        "pgvector_version": extension[0],
        "documents": documents,
        "chunks_per_document": chunks_per_document,
        "candidate_fraction": round(len(selected) / documents, 4),
        "vector_dimensions": 32,
        "warmups": 3,
        "iterations": iterations,
        "measurements": {name: _summary(samples[name], row_counts[name]) for name in queries},
        "explain": plans,
        "disclaimer": "Synthetic single-connection read-path comparison; not production capacity or p99.",
    }
