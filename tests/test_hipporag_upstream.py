"""Explicit pinned HippoRAG/BGE CPU gate; generation remains a labelled fixture."""
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rags.registry import CapabilityError
from rags.research.hipporag_adapter import SOURCE_COMMIT, search

pytestmark = [
    pytest.mark.integration, pytest.mark.external, pytest.mark.encoders, pytest.mark.research_hipporag,
    pytest.mark.skipif(os.getenv("RAGS_TEST_RESEARCH_UPSTREAMS") not in {"hipporag", "all"},
                       reason="Explicit real HippoRAG/BGE CPU gate"),
]


@pytest.fixture(scope="module")
def provider():
    from hipporag_fixture_provider import HippoRAGFixtureProvider
    return HippoRAGFixtureProvider()


def test_actual_openie_fact_selection_and_ppr_preserve_source_snapshots(tmp_path, provider):
    from research_fixture_provider import PASSAGES, QUESTION
    started = time.perf_counter()
    result = search(QUESTION, PASSAGES, 20, tmp_path, provider)
    elapsed = time.perf_counter() - started
    observed = result["diagnostics"]
    assert result["implementation"]["source_commit"] == SOURCE_COMMIT
    assert observed["ppr_calls"] == 1 and observed["retrieval_path"] == "personalized_pagerank"
    assert observed["graph_nodes"] > len(PASSAGES) and observed["graph_edges"] > 0
    assert observed["extracted_triples"] == 16 and 1 <= observed["linked_facts"] <= 5
    assert observed["generation_calls"] == 2 * len(PASSAGES) + 1
    assert observed["query_embedding_calls"] == 1 and observed["embedding_dimensions"] == 384
    assert observed["embedding_token_limit"] == 512
    assert observed["generation_provider"].startswith("fixture:")
    assert {hit["chunk_id"] for hit in result["hits"]} == {item["id"] for item in PASSAGES}
    assert all(math.isfinite(hit["score"]) and 0 <= hit["score"] <= 1 for hit in result["hits"])

    duplicate_dir = tmp_path / "duplicates"
    duplicate_dir.mkdir()
    duplicated = [PASSAGES[0], {"id": "duplicate-original", "text": PASSAGES[0]["text"]}, PASSAGES[1]]
    duplicate = search(QUESTION, duplicated, 3, duplicate_dir, provider)
    assert {hit["chunk_id"] for hit in duplicate["hits"]} == {item["id"] for item in duplicated}
    assert duplicate["diagnostics"]["unique_passages"] == 2
    assert duplicate["diagnostics"]["source_count"] == 3

    replacement_dir = tmp_path / "replacement"
    replacement_dir.mkdir()
    replacement = [{"id": "different-tenant-same-id", "text": PASSAGES[4]["text"]}]
    replaced = search("What does LunarTeam operate?", replacement, 1, replacement_dir, provider)
    assert replaced["hits"][0]["chunk_id"] == "different-tenant-same-id"
    assert replaced["diagnostics"]["source_count"] == 1
    assert replaced["diagnostics"]["extracted_triples"] == 1

    destination = os.getenv("RAGS_RESEARCH_REPORT_PATH")
    if destination:
        root = Path(provider.encoder.model._model_dir)
        artifacts = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                     for name in [provider.encoder.model.model_description.model_file, "tokenizer.json", "tokenizer_config.json"]}
        versions = ["fastembed", "onnxruntime", "tokenizers", "numpy", "tiktoken", "igraph", "torch", "pandas", "pyarrow"]
        report = {
            "schema": "actual-upstream-verification-v1", "engine": "hipporag",
            "verified_at": datetime.now(UTC).isoformat(), "python": platform.python_version(),
            "verification": "actual_upstream_algorithm_real_bge_embeddings_deterministic_generation_fixture",
            "not_verified": ["trained_generation_quality", "paper_benchmark_reproduction", "paid_provider_quality"],
            "runtime_versions": {name: importlib.metadata.version(name) for name in versions},
            "embedding_artifacts_sha256": artifacts, "algorithm_elapsed_seconds": elapsed,
            "result": result, "duplicate_snapshot": duplicate, "replacement_snapshot": replaced,
        }
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


@pytest.mark.parametrize("operation", ["ner", "triples", "filter"])
def test_actual_upstream_swallowed_provider_failure_is_rethrown(tmp_path, provider, monkeypatch, operation):
    from hipporag_fixture_provider import hipporag_completion
    from research_fixture_provider import PASSAGES, QUESTION

    def fail_selected(system, user):
        selected = (system.startswith("Your task is to extract named entities") if operation == "ner"
                    else system.startswith("Your task is to construct an RDF") if operation == "triples"
                    else system.startswith("Your input fields are:"))
        if selected:
            raise RuntimeError("Synthetic provider fault, no private data")
        return hipporag_completion(system, user)

    monkeypatch.setattr(provider, "complete", fail_selected)
    with pytest.raises(CapabilityError) as error:
        search(QUESTION, PASSAGES[:2], 2, tmp_path, provider)
    assert error.value.code == "research_provider_failed"


def test_actual_upstream_dense_fallback_is_an_explicit_error(tmp_path, provider, monkeypatch):
    from hipporag_fixture_provider import hipporag_completion
    from research_fixture_provider import PASSAGES, QUESTION

    def no_facts(system, user):
        if system.startswith("Your input fields are:"):
            return '[[ ## fact_after_filter ## ]]\n{"fact":[]}\n[[ ## completed ## ]]'
        return hipporag_completion(system, user)

    monkeypatch.setattr(provider, "complete", no_facts)
    with pytest.raises(CapabilityError) as error:
        search(QUESTION, PASSAGES[:2], 2, tmp_path, provider)
    assert error.value.code == "insufficient_graph_evidence"


def test_actual_modified_upstream_source_fails_before_inference(tmp_path, provider, monkeypatch):
    import rags.research.hipporag_adapter as adapter
    monkeypatch.setattr(adapter, "SOURCE_FILES", {"HippoRAG.py": "0" * 64})
    calls = (provider.documents, provider.queries, provider.generations)
    with pytest.raises(CapabilityError) as error:
        search("question", [{"id": "a", "text": "OrionLab develops HarborSensor."}], 1, tmp_path, provider)
    assert error.value.code == "requires_backend"
    assert (provider.documents, provider.queries, provider.generations) == calls
    assert list(tmp_path.iterdir()) == []
