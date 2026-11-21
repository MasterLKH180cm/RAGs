"""Pinned upstream execution is opt-in; ordinary tests need no model download."""
import asyncio
import math
import os
from types import SimpleNamespace

import pytest

from rags.registry import CapabilityError
from rags.research.common import ProviderBridge, private_directory, validate_request, verify_source
from rags.research.lightrag_adapter import project_chunks, validate_extraction, validate_keywords
from rags.research.raptor_adapter import project_leaves


def stub_provider(**kwargs):
    """Protocol stub used only for boundary tests, never an actual-model claim."""
    values = {"settings": SimpleNamespace(embedding_provider="fastembed"), "has_generation": True,
              "embedding_model": "test:trained", "model_identity": "fixture:generation",
              "embedding": lambda text: [3.0, 4.0], "query_embedding": lambda text: [4.0, 3.0],
              "embedding_token_count": lambda text, query=False: len(text), "embedding_token_limit": 512,
              "complete": lambda system, user: "bounded result"}
    values.update(kwargs)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("question,passages,top_k", [
    ("", [{"id": "a", "text": "text"}], 1),
    ("question", [], 1),
    ("question", [{"id": "../bad", "text": "text"}], 1),
    ("question", [{"id": "a", "text": "one"}, {"id": "a", "text": "two"}], 1),
    ("question", [{"id": "a", "text": "x" * 200001}], 1),
    ("question", [{"id": "a", "text": "\ud800"}], 1),
    ("question", [{"id": "a", "text": "text"}], True),
])
def test_rejects_invalid_authorized_snapshot(question, passages, top_k):
    with pytest.raises(CapabilityError):
        validate_request(question, passages, top_k)


def test_request_preserves_unicode_and_requires_fresh_index(tmp_path):
    assert validate_request("哪一項？", [{"id": "a", "text": "繁體中文👩🏽‍💻"}], 1) == {"a": "繁體中文👩🏽‍💻"}
    private_directory(tmp_path, "index")
    with pytest.raises(CapabilityError, match="fresh"):
        private_directory(tmp_path, "index")


def test_missing_or_changed_upstream_source_is_explicit():
    with pytest.raises(CapabilityError, match="not installed"):
        verify_source("this_pinned_research_module_does_not_exist", {})
    with pytest.raises(CapabilityError, match="differs"):
        verify_source("rags", {"__init__.py": "0" * 64})


@pytest.mark.parametrize("provider", [
    stub_provider(settings=SimpleNamespace(embedding_provider="local")),
    stub_provider(embedding_model="offline-hash-v1"),
    stub_provider(has_generation=False),
    stub_provider(model_identity="extractive-v1"),
])
def test_rejects_demo_providers_before_import_or_index(provider, tmp_path):
    from rags.research.raptor_adapter import search
    with pytest.raises(CapabilityError) as error:
        search("question", [{"id": "a", "text": "content"}], 1, tmp_path, provider)
    assert error.value.code == "requires_provider"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("vector", [[0, 0], [float("nan"), 1], [float("inf"), 1], [True, 1], ["1", 2], []])
def test_bridge_rejects_invalid_model_vectors(vector):
    bridge = ProviderBridge(stub_provider(embedding=lambda text: vector))
    with pytest.raises(CapabilityError, match="vector"):
        bridge.embed("source")


def test_bridge_uses_separate_query_encoding_and_full_prompt_budget():
    bridge = ProviderBridge(stub_provider())
    assert bridge.embed("same") == pytest.approx([0.6, 0.8])
    assert bridge.embed("same", query=True) == pytest.approx([0.8, 0.6])
    assert bridge.embed("same") == pytest.approx([0.6, 0.8])
    assert bridge.embedding_calls == 2 and bridge.query_embedding_calls == 1
    with pytest.raises(CapabilityError, match="model token window"):
        bridge.embed("x" * 513)
    assert bridge.embedding_calls == 2
    with pytest.raises(CapabilityError, match="budget"):
        bridge.complete("x" * 15000, system="s" * 1500)


def test_raptor_projection_uses_all_descendant_original_leaves():
    nodes = {0: SimpleNamespace(text="one", children=set()),
             1: SimpleNamespace(text="two", children=set()),
             2: SimpleNamespace(text="Generated summary is not original evidence.", children={0, 1})}
    result = project_leaves([{"node_index": 2}], nodes, {0: "a", 1: "b"}, {"a": "one", "b": "two"}, 3)
    assert result == [{"chunk_id": "a", "score": 1}, {"chunk_id": "b", "score": 1}]


@pytest.mark.parametrize("mutation", ["cycle", "unknown", "changed", "empty"])
def test_raptor_projection_rejects_broken_lineage(mutation):
    nodes = {0: SimpleNamespace(text="one", children=set()), 1: SimpleNamespace(text="summary", children={0})}
    if mutation == "cycle":
        nodes[1].children = {1}
    elif mutation == "unknown":
        nodes[1].children = {999}
    elif mutation == "changed":
        nodes[0].text = "replacement"
    else:
        nodes[1].children = set()
    with pytest.raises(CapabilityError):
        project_leaves([{"node_index": 1}], nodes, {0: "a"}, {"a": "one"}, 3)


@pytest.mark.parametrize("record,content", [
    ({"full_doc_id": "foreign", "content": "one"}, "one"),
    ({"full_doc_id": "a", "content": "changed"}, "one"),
    ({"full_doc_id": "a", "content": "one"}, "changed"),
    (None, "one"),
])
def test_lightrag_projection_requires_exact_authorized_source(record, content):
    async def get_by_ids(ids):
        return [record]
    with pytest.raises(CapabilityError):
        asyncio.run(project_chunks([{"chunk_id": "internal", "content": content}],
                                   SimpleNamespace(get_by_ids=get_by_ids), {"a": "one"}, 1))


@pytest.mark.parametrize("value", ['{}', 'not json', '{"high_level_keywords": [1], "low_level_keywords": []}'])
def test_invalid_keyword_generation_is_not_a_domain_refusal(value):
    with pytest.raises(CapabilityError) as error:
        validate_keywords(value)
    assert error.value.code == "invalid_model_output"


def test_extraction_format_bounds_and_unknown_entity_references():
    validate_keywords('{"high_level_keywords": [], "low_level_keywords": []}')
    validate_extraction("<|COMPLETE|>")
    with pytest.raises(CapabilityError):
        validate_extraction("Unstructured answer pretending extraction succeeded.")
    with pytest.raises(CapabilityError):
        validate_extraction("relation<|#|>A<|#|>B<|#|>uses<|#|>A uses B\n<|COMPLETE|>")


def real_engine_enabled(name):
    return os.getenv("RAGS_TEST_RESEARCH_UPSTREAMS") in {name, "all"}


@pytest.mark.skipif(not real_engine_enabled("raptor"), reason="Explicit real RAPTOR/BGE CPU gate")
def test_actual_raptor_umap_gmm_summary_retrieval_and_source_lineage(tmp_path):
    from research_fixture_provider import PASSAGES, QUESTION, RealBGEFixtureProvider, record_actual_report

    from rags.research.raptor_adapter import SOURCE_COMMIT, search
    provider = RealBGEFixtureProvider()
    result = search(QUESTION, PASSAGES, 20, tmp_path, provider)
    evidence = result["diagnostics"]
    assert result["implementation"]["source_commit"] == SOURCE_COMMIT
    assert evidence["summary_layers"] >= 1
    assert evidence["summary_nodes"] >= 1 and evidence["selected_summary_nodes"] >= 1
    assert evidence["tree_nodes"] > len(PASSAGES)
    assert evidence["generation_calls"] > 0 and evidence["query_embedding_calls"] > 0
    assert evidence["embedding_dimensions"] == 384
    assert evidence["embedding_token_limit"] == 512
    assert evidence["generation_provider"].startswith("fixture:")
    assert result["hits"] and all(math.isfinite(hit["score"]) for hit in result["hits"])
    assert {hit["chunk_id"] for hit in result["hits"]} <= {item["id"] for item in PASSAGES}
    bridge = ProviderBridge(provider)
    with pytest.raises(CapabilityError, match="model token window"):
        bridge.embed("river " * 600)
    assert bridge.embedding_calls == 0
    record_actual_report("raptor", result, provider)
    print({"engine": "raptor", "diagnostics": evidence})
    small_dir = tmp_path / "small-snapshot"
    small_dir.mkdir()
    small = search(QUESTION, PASSAGES[:4], 4, small_dir, provider)
    assert small["diagnostics"]["summary_layers"] == 0
    assert {hit["chunk_id"] for hit in small["hits"]} == {item["id"] for item in PASSAGES[:4]}


@pytest.mark.skipif(not real_engine_enabled("lightrag"), reason="Explicit real LightRAG/BGE CPU gate")
def test_actual_lightrag_extracts_and_retrieves_hybrid_graph_with_source_mapping(tmp_path):
    from research_fixture_provider import PASSAGES, QUESTION, RealBGEFixtureProvider, record_actual_report

    from rags.research.lightrag_adapter import SOURCE_COMMIT, search
    provider = RealBGEFixtureProvider()
    sources = PASSAGES[:4]
    result = asyncio.run(search(QUESTION, sources, 4, tmp_path, provider))
    evidence = result["diagnostics"]
    assert result["implementation"]["source_commit"] == SOURCE_COMMIT
    assert evidence["graph_nodes"] >= 4 and evidence["graph_edges"] >= 4
    assert evidence["retrieved_entities"] > 0 and evidence["retrieved_relationships"] > 0
    assert evidence["generation_calls"] >= 5 and evidence["query_embedding_calls"] > 0
    assert evidence["embedding_dimensions"] == 384
    assert evidence["embedding_token_limit"] == 512
    assert evidence["generation_provider"].startswith("fixture:")
    assert result["hits"] and {hit["chunk_id"] for hit in result["hits"]} <= {item["id"] for item in sources}
    assert all(math.isfinite(hit["score"]) for hit in result["hits"])
    bridge = ProviderBridge(provider)
    with pytest.raises(CapabilityError, match="model token window"):
        bridge.embed("river " * 600)
    assert bridge.embedding_calls == 0
    record_actual_report("lightrag", result, provider)
    print({"engine": "lightrag", "diagnostics": evidence})
