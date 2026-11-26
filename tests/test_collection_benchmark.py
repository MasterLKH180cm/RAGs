"""Benchmark correctness and generated-tenant cleanup are part of acceptance."""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from rags.client import PersistentClient
from scripts import benchmark_collections as benchmark


pytestmark = pytest.mark.integration
ROOT = Path(__file__).parents[1]


def existing_dataset(path):
    config = benchmark.configuration(str(path), {"existing-owner-key": "existing-tenant"})
    with PersistentClient(path, "existing-owner-key", settings=config) as client:
        collection = client.create_collection("must-survive")
        collection.add(ids=["protected"], documents=["Existing unrelated tenant evidence."])
        return collection.id, collection.get()["items"]


def verify_only_existing_dataset(path, identifier, records):
    config = benchmark.configuration(str(path), {"existing-owner-key": "existing-tenant"})
    with PersistentClient(path, "existing-owner-key", settings=config) as client:
        assert client.get_collection(collection_id=identifier).get()["items"] == records
    with sqlite3.connect(path) as connection:
        for table in ("rag_collections", "rag_collection_versions", "rag_collection_records", "audit"):
            tenants = connection.execute(f"SELECT DISTINCT tenant FROM {table}").fetchall()
            assert tenants == [("existing-tenant",)]


def test_benchmark_subprocess_measures_real_work_and_preserves_existing_tenant(tmp_path):
    path = tmp_path / "existing.db"
    identifier, records = existing_dataset(path)
    report_path = tmp_path / "report.json"
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("RAGS_", "OPENAI_", "ANTHROPIC_"))}
    env.update(RAGS_EMBEDDING_PROVIDER="openai", OPENAI_API_KEY="ambient-test-key-must-not-be-used")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/benchmark_collections.py"), "--sqlite", str(path),
         "--records", "25", "--dimensions", "8", "--batch-size", "10", "--iterations", "3",
         "--warmup", "1", "--report", str(report_path)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "ambient-test-key" not in result.stdout and "ambient-test-key" not in result.stderr
    report = json.loads(report_path.read_text())
    assert json.loads(result.stdout) == report
    assert report["passed"] and report["backend"] == "sqlite"
    assert report["committed_record_count"] == 25 and report["initial_write_records_per_second"] > 0
    assert len(report["checks"]) == 8 and all(report["checks"].values())
    assert report["cleanup"]["passed"] and set(report["cleanup"]["remaining_rows"].values()) == {0}
    assert report["workload"]["external_model_calls"] == 0
    assert len(report["latency"]) == 10 and report["measured_operations"] == 30
    for metric in report["latency"].values():
        assert metric["samples"] == 3 and len(metric["raw_ms"]) == 3
        assert 0 < metric["min_ms"] <= metric["p50_ms"] <= metric["p95_ms"] <= metric["max_ms"]
    verify_only_existing_dataset(path, identifier, records)


def test_benchmark_failure_still_cleans_only_generated_tenants(tmp_path, monkeypatch):
    path = tmp_path / "failed.db"
    identifier, records = existing_dataset(path)
    original = benchmark.measure

    def fail_during_fork(samples, name, action):
        result = original(samples, name, action)
        if name == "fork_mapping_snapshot":
            raise RuntimeError("Injected failure after committed fork")
        return result

    monkeypatch.setattr(benchmark, "measure", fail_during_fork)
    args = argparse.Namespace(records=4, dimensions=4, seed=123, batch_size=2, iterations=1, warmup=0)
    with pytest.raises(RuntimeError, match="after committed fork"):
        benchmark.run_workload(str(path), args)
    verify_only_existing_dataset(path, identifier, records)


def test_benchmark_rejects_oversized_workload_before_creating_database(tmp_path):
    path = tmp_path / "unopened.db"
    args = argparse.Namespace(records=10000, dimensions=4096, seed=1, batch_size=100, iterations=1, warmup=0)
    with pytest.raises(ValueError, match="16 MiB"):
        benchmark.run_workload(str(path), args)
    assert not path.exists()

