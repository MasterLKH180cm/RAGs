"""Explicit real Microsoft GraphRAG/BGE gate; synthetic generation is labelled."""
import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.external, pytest.mark.research_graphrag]


@pytest.mark.skipif(os.getenv("RAGS_TEST_RESEARCH_UPSTREAMS") not in {"full_graphrag", "all"},
                    reason="Explicit real Microsoft GraphRAG/BGE CPU gate")
def test_actual_standard_index_communities_local_graph_retrieval_and_duplicate_lineage(tmp_path, monkeypatch):
    from graphrag.index.operations.extract_graph.graph_extractor import GraphExtractor
    from graphrag_fixture_provider import GraphRAGFixtureProvider
    from research_fixture_provider import PASSAGES, QUESTION

    from rags.research.graphrag_adapter import PIPELINE_STAGES, SOURCE_COMMIT, search

    parser = GraphExtractor._process_result
    parsed_weights = []

    def observe_parser(self, *args, **kwargs):
        entities, relationships = parser(self, *args, **kwargs)
        assert len(entities) >= 2 and len(relationships) >= 1
        weights = relationships["weight"].tolist()
        assert weights == [8.0] * len(weights)
        parsed_weights.extend(weights)
        return entities, relationships

    monkeypatch.setattr(GraphExtractor, "_process_result", observe_parser)
    start = time.monotonic()
    provider = GraphRAGFixtureProvider()
    passages = [*PASSAGES[:8], {"id": "water-duplicate", "text": PASSAGES[0]["text"]}]
    result = asyncio.run(search(QUESTION, passages, 20, tmp_path, provider))
    evidence = result["diagnostics"]
    assert result["implementation"]["source_commit"] == SOURCE_COMMIT
    assert evidence["pipeline_stages"] == len(PIPELINE_STAGES)
    assert evidence["query_mode"] == "local_graph_context"
    assert evidence["graph_nodes"] >= 8 and evidence["graph_edges"] >= 8
    assert evidence["communities"] > 0 and evidence["community_reports"] == evidence["communities"]
    assert evidence["source_count"] == 9 and evidence["text_units"] == 8
    assert evidence["generation_calls"] > 8 and evidence["query_embedding_calls"] > 0
    assert evidence["embedding_dimensions"] == 384 and evidence["embedding_token_limit"] == 512
    identifiers = {hit["chunk_id"] for hit in result["hits"]}
    assert {"water-1", "water-duplicate"} <= identifiers <= {item["id"] for item in passages}
    assert evidence["retrieved_entities"] > 0 and evidence["retrieved_relationships"] > 0
    assert list(tmp_path.iterdir()) == []

    # A second isolated snapshot cannot inherit the first request's graph or IDs.
    second = tmp_path / "second"
    second.mkdir()
    replacement = [{"id": "replacement", "text": "AmberLab develops RiverSensor for river water monitoring."}]
    other = asyncio.run(search("Which organization develops RiverSensor?", replacement, 5, second, provider))
    assert [hit["chunk_id"] for hit in other["hits"]] == ["replacement"]
    assert list(second.iterdir()) == []
    assert len(parsed_weights) == 9
    elapsed = time.monotonic() - start
    path = os.getenv("RAGS_RESEARCH_REPORT_PATH")
    if path:
        root = Path(provider.encoder.model._model_dir)
        files = [provider.encoder.model.model_description.model_file, "tokenizer.json", "tokenizer_config.json"]
        report = {
            "schema": "actual-upstream-verification-v1", "engine": "full_graphrag",
            "verification": "actual_standard_index_local_graph_context_real_bge_deterministic_generation",
            "not_verified": ["trained_generation_quality", "paper_benchmark_reproduction", "global_search", "drift_search"],
            "python": platform.python_version(), "elapsed_seconds": elapsed,
            "runtime_versions": {name: importlib.metadata.version(name) for name in (
                "graphrag", "graphrag-llm", "graspologic-native", "lancedb", "pandas", "fastembed", "onnxruntime", "tiktoken")},
            "embedding_artifacts_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files},
            "checks": {"full_standard_pipeline": True, "original_source_projection": True,
                       "duplicate_source_ids_preserved": True, "isolated_replacement_snapshot": True,
                       "request_indexes_removed": True, "extraction_records_and_weights_preserved": True},
            "extraction_fidelity": {"parsed_relationships": len(parsed_weights),
                                    "expected_weight": 8.0, "observed_weights": sorted(set(parsed_weights))},
            "result": result, "replacement_result": other,
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


@pytest.mark.skipif(os.getenv("RAGS_TEST_RESEARCH_UPSTREAMS") not in {"full_graphrag", "all"},
                    reason="Explicit real Microsoft GraphRAG/BGE CPU gate")
@pytest.mark.parametrize("stage", ["extraction", "community_report", "entity_embedding"])
def test_actual_upstream_cannot_swallow_provider_failure_and_leave_a_private_index(tmp_path, stage):
    from graphrag_fixture_provider import GraphRAGFixtureProvider
    from research_fixture_provider import PASSAGES, QUESTION

    from rags.registry import CapabilityError
    from rags.research.graphrag_adapter import search

    provider = GraphRAGFixtureProvider()
    complete, embed = provider.complete, provider.embedding
    def failing_completion(system, user):
        if ((stage == "extraction" and "-Real Data-" in user)
                or (stage == "community_report" and "# Real Data" in user)):
            return "malformed-output-must-not-be-hidden"
        return complete(system, user)
    def failing_embedding(text):
        return [0.0] * 384 if stage == "entity_embedding" and text != PASSAGES[0]["text"] else embed(text)
    provider.complete, provider.embedding = failing_completion, failing_embedding
    with pytest.raises(CapabilityError) as error:
        asyncio.run(search(QUESTION, PASSAGES[:2], 5, tmp_path, provider))
    assert error.value.code == "invalid_upstream_output"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(os.getenv("RAGS_TEST_RESEARCH_UPSTREAMS") not in {"full_graphrag", "all"},
                    reason="Explicit real Microsoft GraphRAG/BGE budget gate")
def test_actual_upstream_keeps_token_budget_refusal_instead_of_hiding_it_as_model_failure(tmp_path):
    from graphrag_fixture_provider import GraphRAGFixtureProvider
    from research_fixture_provider import PASSAGES, QUESTION

    from rags.registry import CapabilityError
    from rags.research.graphrag_adapter import search

    provider = GraphRAGFixtureProvider()
    provider.settings.context_budget = 1500
    with pytest.raises(CapabilityError) as error:
        asyncio.run(search(QUESTION, PASSAGES[:2], 5, tmp_path, provider))
    assert error.value.code == "context_budget_exceeded" and error.value.status_code == 422
    assert provider.generations == 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(os.getenv("RAGS_TEST_RESEARCH_UPSTREAMS") not in {"full_graphrag", "all"},
                    reason="Explicit real Microsoft GraphRAG parser gate")
@pytest.mark.parametrize("variant", ["plain", "field_spacing", "quoted_names", "html_identity", "control_identity"])
def test_actual_upstream_parser_preserves_validated_records_names_and_final_weight(variant):
    from graphrag.index.operations.extract_graph.graph_extractor import GraphExtractor

    from rags.research.common import verify_source
    from rags.research.graphrag_adapter import SOURCE_FILES, validate_extraction

    verify_source("graphrag", SOURCE_FILES)
    value = ('("entity"<|>A<|>ORGANIZATION<|>Entity A)##'
             '("entity"<|>B<|>ORGANIZATION<|>Entity B)##'
             '("relationship"<|>A<|>B<|>A uses B<|>8)\n<|COMPLETE|>')
    names, descriptions = ["A", "B"], ["Entity A", "Entity B"]
    if variant == "field_spacing":
        value = value.replace('("entity"', '( "entity" ').replace('("relationship"', '( "relationship" ')
        value = value.replace("<|>", " <|> ")
    elif variant == "quoted_names":
        value = value.replace("<|>A<|>", '<|>"A"<|>').replace("<|>B<|>", '<|>"B"<|>')
        names = ['"A"', '"B"']
    elif variant == "html_identity":
        value = value.replace("<|>A<|>", "<|>A&amp;B<|>", 1).replace("<|>A<|>B<|>", "<|>a&b<|>B<|>")
        value = value.replace("Entity A", "Entity &amp; A\x01")
        names, descriptions = ["A&B", "B"], ["Entity & A", "Entity B"]
    elif variant == "control_identity":
        value = value.replace("<|>A<|>", "<|>A\x01<|>", 1)
    canonical = validate_extraction(value)
    entities, relationships = GraphExtractor(None, "", 0)._process_result(canonical, "original", "<|>", "##")
    assert entities.to_dict("records") == [
        {"title": name, "type": "ORGANIZATION", "description": description, "source_id": "original"}
        for name, description in zip(names, descriptions, strict=True)
    ]
    assert relationships.to_dict("records") == [
        {"source": names[0], "target": names[1], "description": "A uses B", "source_id": "original", "weight": 8.0},
    ]
