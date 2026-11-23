"""Offline validation and original-source projection for the HippoRAG boundary."""
import json
from types import SimpleNamespace

import pytest
from test_research_upstreams import stub_provider

from rags.registry import CapabilityError
from rags.research.common import ProviderBridge
from rags.research.hipporag_adapter import (
    FACT_END,
    FACT_START,
    BoundedLLM,
    group_sources,
    project_passages,
    search,
    validate_extraction,
    validate_fact_filter,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("kind,value", [
    ("named_entities", '{"named_entities":["OrionLab","HarborSensor"]}'),
    ("named_entities", '["OrionLab","HarborSensor"]'),
    ("named_entities", '{"named_entities":[]}'),
    ("triples", '{"triples":[["OrionLab","develops","HarborSensor"]]}'),
    ("triples", '{"triples":[]}'),
])
def test_extraction_accepts_bounded_upstream_formats(kind, value):
    validate_extraction(value, kind)


@pytest.mark.parametrize("kind,value", [
    ("named_entities", 'prefix {"named_entities":[]}'),
    ("named_entities", '{"named_entities":[],"named_entities":["forged"]}'),
    ("named_entities", '{"named_entities":[false]}'),
    ("named_entities", '{"named_entities":[" "]}'),
    ("named_entities", json.dumps({"named_entities": ["entity"] * 65})),
    ("triples", '{"triples":[["a","b"]]}'),
    ("triples", '{"triples":[["a","b",true]]}'),
    ("triples", '{"triples":[],"extra":[]}'),
    ("triples", '{"triples":[["a","b",NaN]]}'),
])
def test_extraction_rejects_malformed_outputs_before_upstream_swallow(kind, value):
    with pytest.raises(CapabilityError) as error:
        validate_extraction(value, kind)
    assert error.value.code == "invalid_model_output"


def fact_prompt(facts):
    return '[[ ## question ## ]]\nquestion\n\n[[ ## fact_before_filter ## ]]\n' + json.dumps({"fact": facts}) + '\n\nRespond with the corresponding output fields.'


def fact_response(facts):
    return FACT_START + "\n" + json.dumps({"fact": facts}) + "\n" + FACT_END


def test_fact_filter_accepts_exact_subset_and_empty_decision():
    candidates = [["OrionLab", "develops", "HarborSensor"], ["HarborSensor", "monitors", "RiverDelta"]]
    validate_fact_filter(fact_response(candidates[:1]), fact_prompt(candidates))
    validate_fact_filter(fact_response([]), fact_prompt(candidates))


@pytest.mark.parametrize("response", [
    fact_response([["OrionLab", "develops", "ForeignDevice"]]),
    fact_response([["OrionLab", "develops", "HarborSensor"]] * 2),
    FACT_START + '\n{"fact":[["OrionLab","develops"]]}\n' + FACT_END,
    FACT_START + '\n{"fact":[],"fact":[]}\n' + FACT_END,
    '{"fact":[]}',
])
def test_fact_filter_rejects_unknown_facts_instead_of_fuzzy_repair(response):
    with pytest.raises(CapabilityError) as error:
        validate_fact_filter(response, fact_prompt([["OrionLab", "develops", "HarborSensor"]]))
    assert error.value.code == "invalid_model_output"


class PassageStore:
    def __init__(self, content="original source"):
        self.content = content

    def get_hash_id(self, text):
        assert text == "original source"
        return "upstream-key"

    def get_row(self, key):
        assert key == "upstream-key"
        return {"content": self.content}


def passage_result(**changes):
    data = {"docs": ["original source"], "doc_scores": [0.25],
            "doc_metadata": [{"source_id": "a", "rags_source_ids": ["a", "b"]}]}
    data.update(changes)
    return SimpleNamespace(**data)


def test_identical_passages_expand_to_all_original_identifiers_with_bound():
    groups = group_sources({"b": "original source", "a": "original source", "c": "other"})
    assert groups == {"original source": ["a", "b"], "other": ["c"]}
    hits = project_passages(passage_result(), PassageStore(), groups, 2)
    assert hits == [{"chunk_id": "a", "score": .25}, {"chunk_id": "b", "score": .25}]
    assert project_passages(passage_result(), PassageStore(), groups, 1) == hits[:1]


@pytest.mark.parametrize("mutation", ["foreign_id", "foreign_text", "changed_store", "nan", "boolean", "negative", "string", "missing_scores", "missing_metadata", "duplicate"])
def test_projection_fails_closed_on_changed_or_foreign_evidence(mutation):
    result, store = passage_result(), PassageStore()
    if mutation == "foreign_id":
        result.doc_metadata[0]["rags_source_ids"] = ["foreign"]
    elif mutation == "foreign_text":
        result.docs[0] = "generated summary"
    elif mutation == "changed_store":
        store.content = "replacement"
    elif mutation == "missing_metadata":
        result.doc_metadata = []
    elif mutation == "missing_scores":
        result.doc_scores = None
    elif mutation == "duplicate":
        result.docs *= 2
        result.doc_scores *= 2
        result.doc_metadata *= 2
    else:
        result.doc_scores = [{"nan": float("nan"), "boolean": True, "negative": -0.1, "string": "0.25"}[mutation]]
    with pytest.raises(CapabilityError):
        project_passages(result, store, {"original source": ["a", "b"]}, 2)


def test_provider_exception_is_recorded_for_upstream_swallow_detection():
    def fail(system, user):
        raise RuntimeError("private credential and source detail")
    llm = BoundedLLM(ProviderBridge(stub_provider(complete=fail)))
    messages = [{"role": "system", "content": "Your task is to extract named entities"},
                {"role": "user", "content": "original source"}]
    with pytest.raises(CapabilityError) as error:
        llm.infer(messages, max_new_tokens=512)
    assert error.value.code == "research_provider_failed"
    assert "private" not in str(error.value)
    with pytest.raises(CapabilityError) as observed:
        llm.check()
    assert observed.value is error.value


def test_inference_checks_prompt_and_output_bounds_before_acceptance():
    llm = BoundedLLM(ProviderBridge(stub_provider(complete=lambda system, user: '{"named_entities":[]}')))
    with pytest.raises(CapabilityError):
        llm.infer([], max_new_tokens=512)
    llm = BoundedLLM(ProviderBridge(stub_provider(complete=lambda system, user: '{"named_entities":[]}')))
    messages = [{"role": "system", "content": "Your task is to extract named entities"},
                {"role": "user", "content": "source"}]
    with pytest.raises(CapabilityError, match="token bound"):
        llm.infer(messages, max_new_tokens=2)


def test_demo_provider_fails_before_upstream_import_or_index(tmp_path):
    provider = stub_provider(settings=SimpleNamespace(embedding_provider="local"))
    with pytest.raises(CapabilityError) as error:
        search("question", [{"id": "a", "text": "source"}], 1, tmp_path, provider)
    assert error.value.code == "requires_provider"
    assert list(tmp_path.iterdir()) == []
