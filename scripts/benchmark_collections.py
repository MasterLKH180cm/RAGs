"""Measure bounded synthetic collection operations; this is not model-quality evidence."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import platform
import random
import secrets
import statistics
import sys
import tempfile
from time import perf_counter
import uuid

# Support direct source-checkout execution as well as an installed package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rags.client import ClientError, PersistentClient  # noqa: E402
from rags.config import Settings  # noqa: E402
from rags.storage import Store  # noqa: E402


def require(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def bounded_integer(minimum: int, maximum: int):
    def parse(value: str) -> int:
        try:
            result = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError("Expected an integer") from None
        if not minimum <= result <= maximum:
            raise argparse.ArgumentTypeError(f"Expected {minimum}..{maximum}")
        return result
    return parse


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(values: list[float]) -> dict:
    require(bool(values), "A reported operation must have actual timing samples")
    return {"samples": len(values), "p50_ms": percentile(values, 0.5),
            "p95_ms": percentile(values, 0.95), "min_ms": min(values), "max_ms": max(values),
            "mean_ms": statistics.fmean(values), "raw_ms": values}


def measure(samples: dict[str, list[float]], name: str, action):
    started = perf_counter()
    result = action()
    samples.setdefault(name, []).append((perf_counter() - started) * 1000)
    return result


def cleanup(store: Store, tenants: tuple[str, str]) -> dict:
    """Only this invocation's unguessable synthetic namespaces are removed."""
    remaining = {}
    with store.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        for table in ("rag_collection_records", "rag_collection_versions", "rag_collections", "audit"):
            # Names are fixed constants; values always use database parameters.
            if not list(connection.execute(f"PRAGMA table_info({table})")):
                continue
            connection.execute(f"DELETE FROM {table} WHERE tenant IN (?,?)", tenants)
            row = connection.execute(f"SELECT COUNT(*) AS count FROM {table} WHERE tenant IN (?,?)", tenants).fetchone()
            remaining[table] = row["count"]
    require(all(count == 0 for count in remaining.values()), "Synthetic tenant cleanup was incomplete")
    return {"scope": "only_this_invocation_generated_tenants", "remaining_rows": remaining, "passed": True}


def configuration(database: str, keys: dict[str, str]) -> Settings:
    # Model credentials and operator authentication environment are intentionally
    # not loaded. All timed dense/sparse operations use caller-supplied vectors.
    return Settings(db_path=database, keys=keys, key_roles={key: ("admin",) for key in keys},
                    embedding_provider="local", model_provider="extractive", openai_key="",
                    openai_base_url="https://api.openai.com/v1", embedding_model="offline-hash-v1",
                    chat_model="extractive", anthropic_key="", anthropic_model="")


def synthetic_records(count: int, dimensions: int, seed: int) -> dict:
    generator = random.Random(seed)
    vectors = [[1.0] + [0.0] * (dimensions - 1)]
    vectors.extend([[generator.uniform(-0.75, 0.75) for _ in range(dimensions)] for _ in range(count - 1)])
    documents = [f"Synthetic record {index:05d} discusses retention policy and daily backup." for index in range(count)]
    documents[0] += " benchmarkanchor"
    return {
        "ids": [f"record-{index:05d}" for index in range(count)], "documents": documents,
        "metadatas": [{"group": index % 10, "iteration": -1} for index in range(count)],
        "embeddings": vectors,
        "sparse_embeddings": [{"indices": [0, index + 1], "values": [10.0 if index == 0 else 1.0, 0.25],
                               "space_id": "benchmark-sparse-v1"} for index in range(count)],
    }


def validate_workload(args) -> None:
    # Conservative UTF-8 JSON sizing before allocating a corpus. This workload
    # deliberately uses float-valued vectors, not artificially short JSON ints.
    estimated_record_bytes = args.dimensions * 24 + 1024
    if args.records * estimated_record_bytes > 16 * 1024 * 1024:
        raise ValueError("Reduce records or dimensions to fit the bounded 16 MiB scan workload")
    if min(args.records, args.batch_size) * estimated_record_bytes > 2 * 1024 * 1024:
        raise ValueError("Reduce batch size or dimensions to fit the bounded 2 MiB request workload")


def run_workload(database: str, args) -> dict:
    validate_workload(args)
    unique = uuid.uuid4().hex
    tenants = ("collection-bench-a-" + unique, "collection-bench-b-" + unique)
    keys = (secrets.token_urlsafe(32), secrets.token_urlsafe(32))
    config = configuration(database, dict(zip(keys, tenants, strict=True)))
    store = Store(database)
    data = synthetic_records(args.records, args.dimensions, args.seed)
    samples: dict[str, list[float]] = {}
    checks = {}
    started = perf_counter()
    report = {
        "schema_version": 1, "kind": "synthetic_collection_performance", "passed": False,
        "started_at": datetime.now(UTC).isoformat(),
        "backend": "postgresql_pgvector" if store.is_postgres else "sqlite",
        "scope": "synchronous_authenticated_persistent_sdk_service_database",
        "evidence": "synthetic correctness and latency; no model quality or Chroma performance comparison",
        "hardware": {"platform": platform.system(), "machine": platform.machine(),
                     "cpu_count": os.cpu_count(), "cpu_model": platform.processor() or None,
                     "affinity_cpu_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                     "python": platform.python_version()},
        "workload": {"records": args.records, "dimensions": args.dimensions, "sparse_entries_per_record": 2,
                     "batch_size": args.batch_size, "iterations": args.iterations, "seed": args.seed,
                     "search_warmup_iterations": args.warmup, "query_limit": min(args.records, 10),
                     "embedding_source": "synthetic_precomputed", "external_model_calls": 0,
                     "concurrency": 1, "percentile_method": "linear_interpolation"},
        "checks": checks,
    }
    binding = {"mode": "external", "model": "benchmark-encoder-v1",
               "space_id": "benchmark-dense-v1", "dimensions": args.dimensions}
    try:
        with PersistentClient(database, keys[0], settings=config) as client, PersistentClient(
            database, keys[1], settings=config,
        ) as other:
            collection = client.create_collection("benchmark", embedding=binding, sparse_space_id="benchmark-sparse-v1")
            isolated = other.create_collection("benchmark", embedding=binding)
            isolated.add(ids=[data["ids"][0]], documents=["Private unrelated tenant evidence."],
                         embeddings=[data["embeddings"][0]], embedding_space_id="benchmark-dense-v1")
            for start in range(0, args.records, args.batch_size):
                batch = {name: values[start:start + args.batch_size] for name, values in data.items()}
                outcome = measure(samples, "add_batch", lambda batch=batch: collection.add(
                    **batch, embedding_space_id="benchmark-dense-v1"))
                require(outcome["added"] == len(batch["ids"]) and outcome["usage"]["calls"] == 0,
                        "Precomputed ingestion count or provider usage was incorrect")
            require(collection.count() == args.records, "Committed record count does not match workload")
            status = collection.indexing_status()
            require(status["record_count"] == args.records and status["pending_operations"] == 0,
                    "Indexing status does not match committed records")
            checks["committed_counts_and_transactional_index"] = True
            report["committed_record_count"] = collection.count()
            report["index"] = status["index"]
            total_add_seconds = sum(samples["add_batch"]) / 1000
            report["initial_write_records_per_second"] = args.records / total_add_seconds

            sparse = {"indices": [0], "values": [1.0], "space_id": "benchmark-sparse-v1"}
            common = {"limit": min(args.records, 10), "select": ["score", "distance"]}
            dense = {**common, "mode": "dense", "query_embedding": data["embeddings"][0],
                     "embedding_space_id": "benchmark-dense-v1"}
            lexical = {**common, "mode": "lexical", "query_text": "benchmarkanchor"}
            sparse_query = {**common, "mode": "sparse", "query_sparse": sparse}
            hybrid = {**dense, "mode": "hybrid", "query_text": "benchmarkanchor"}
            hybrid_sparse = {**dense, "mode": "hybrid", "query_sparse": sparse}
            queries = {"dense_search": dense, "lexical_bm25_search": lexical, "sparse_search": sparse_query,
                       "hybrid_bm25_rrf_search": hybrid, "hybrid_sparse_rrf_search": hybrid_sparse}
            search_generation = collection.refresh().to_dict()["generation"]

            def verify(result, *, expected_generation=search_generation):
                require(result["usage"]["calls"] == 0, "A precomputed query unexpectedly invoked a provider")
                require(result["generation"] == expected_generation, "Search reported an unexpected generation")
                for index in range(len(result["results"])):
                    rows = result.rows(index)
                    require(bool(rows) and rows[0]["id"] == data["ids"][0], "Deterministic top result is incorrect")
                    require(all(rows[position]["score"] >= rows[position + 1]["score"]
                                for position in range(len(rows) - 1)), "Search scores are not descending")

            for _ in range(args.warmup):
                for query in queries.values():
                    verify(collection.search(**query))
            for _ in range(args.iterations):
                for name, query in queries.items():
                    verify(measure(samples, name, lambda query=query: collection.search(**query)))
                verify(measure(samples, "search_batch", lambda: collection.search(searches=list(queries.values()))))
            checks["dense_bm25_sparse_and_hybrid_top_result"] = True
            checks["single_generation_batched_search"] = True
            checks["no_model_calls"] = True

            ids = data["ids"][:args.batch_size]
            for iteration in range(args.iterations):
                rows = measure(samples, "get_batch", lambda: collection.get(ids=ids, include=["document", "metadata"]))
                require([row["id"] for row in rows["items"]] == ids, "Batched lookup lost or reordered records")
                updated = measure(samples, "metadata_update_batch", lambda iteration=iteration: collection.update(
                    ids=ids, metadatas=[{"group": index % 10, "iteration": iteration} for index in range(len(ids))]))
                require(updated["updated"] == len(ids) and updated["usage"]["calls"] == 0,
                        "Metadata-only update must retain vectors without provider calls")
            checks["batched_lookup_and_update"] = True

            require(other.list_collections_page()["total"] == 1 and isolated.count() == 1,
                    "Same-name collections leaked across tenants")
            require(not isolated.search(**lexical).rows(), "Lexical search leaked another tenant's text")
            try:
                other.get_collection(collection_id=collection.id)
            except ClientError as error:
                require(error.code == "collection_not_found" and error.status_code == 404,
                        "Cross-tenant access must fail as not found")
            else:
                raise AssertionError("Cross-tenant collection became visible")
            checks["tenant_isolation"] = True

            original = collection.get(ids=[ids[0]])["items"][0]
            source_generation = collection.refresh().to_dict()["generation"]
            survivor = None
            for iteration in range(args.iterations):
                branch = measure(samples, "fork_mapping_snapshot", lambda iteration=iteration: collection.fork(
                    f"branch-{iteration}", expected_generation=source_generation))
                require(branch.count() == args.records and branch.to_dict()["generation"] == 0,
                        "Fork count or starting generation is incorrect")
                require(branch.get(ids=[ids[0]])["items"][0]["version"] == original["version"],
                        "Fork did not share the immutable record version")
                branch.update(ids=[ids[0]], metadatas=[{"branch_iteration": iteration}])
                require(branch.get(ids=[ids[0]])["items"][0]["version"] != original["version"]
                        and collection.get(ids=[ids[0]])["items"][0] == original,
                        "A fork mutation changed its source")
                if iteration + 1 == args.iterations:
                    survivor = branch
                else:
                    client.delete_collection(collection_id=branch.id)
            checks["shared_immutable_versions_and_fork_mutation_isolation"] = True
            client.delete_collection(collection_id=collection.id, expected_generation=source_generation)
            require(survivor is not None and survivor.count() == args.records,
                    "Source deletion removed fork data")
            verify(survivor.search(**hybrid), expected_generation=1)
            checks["fork_survives_source_deletion"] = True
            client.delete_collection(collection_id=survivor.id)
            require(isolated.count() == 1, "Deleting one tenant's datasets changed another tenant")
            report["latency"] = {name: summarize(values) for name, values in samples.items()}
            report["measured_operations"] = sum(len(values) for values in samples.values())
            report["passed"] = all(checks.values())
    finally:
        report["cleanup"] = cleanup(store, tenants)
    report["elapsed_seconds"] = perf_counter() - started
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--sqlite", type=Path, help="Optional database file; default is a temporary SQLite database")
    backend.add_argument("--postgres", action="store_true", help="Read RAGS_BENCHMARK_POSTGRES_URL or RAGS_TEST_POSTGRES_URL")
    parser.add_argument("--records", type=bounded_integer(2, 10000), default=250)
    parser.add_argument("--dimensions", type=bounded_integer(2, 4096), default=32)
    parser.add_argument("--batch-size", type=bounded_integer(1, 100), default=50)
    parser.add_argument("--iterations", type=bounded_integer(1, 1000), default=10)
    parser.add_argument("--warmup", type=bounded_integer(0, 20), default=2)
    parser.add_argument("--seed", type=bounded_integer(0, 2 ** 32 - 1), default=20261126)
    parser.add_argument("--report", type=Path, help="Write the complete JSON report, including timing samples")
    args = parser.parse_args(argv)
    try:
        with ExitStack() as stack:
            if args.postgres:
                database = os.getenv("RAGS_BENCHMARK_POSTGRES_URL") or os.getenv("RAGS_TEST_POSTGRES_URL")
                if not database or not database.startswith(("postgresql://", "postgres://")):
                    parser.error("--postgres requires RAGS_BENCHMARK_POSTGRES_URL or RAGS_TEST_POSTGRES_URL")
            elif args.sqlite is not None:
                database = str(args.sqlite)
            else:
                directory = stack.enter_context(tempfile.TemporaryDirectory(prefix="rags-collection-benchmark-"))
                database = str(Path(directory) / "records.db")
            report = run_workload(database, args)
    except Exception as error:
        # Connection strings may contain credentials; never print database or
        # driver exception text. A failed run must not claim benchmark success.
        report = {"schema_version": 1, "kind": "synthetic_collection_performance", "passed": False,
                  "error": error.code if isinstance(error, ClientError) else type(error).__name__}
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
