"""CRAG branch/provenance contracts; fixture logits are not model evidence."""
import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rags.registry import CapabilityError
from rags.research import crag_adapter as crag
from rags.research_protocol import ResearchRequest, external_source_id, validate_result

pytestmark = pytest.mark.unit


def provider(response=None):
    def complete(system, user):
        if response is not None:
            return response
        evidence = json.loads(user)["evidence"]
        return json.dumps({"answer": "The office is in Taipei.",
                           "cited_ids": list(dict.fromkeys(item["id"] for item in evidence)), "abstain": False})
    return SimpleNamespace(
        settings=SimpleNamespace(generation_tokenizer_config={"backend": "unicode"}, context_budget=16000,
                                 output_budget=1024, envelope_budget=256),
        model_identity="fixture:crag-generation", require_generation=lambda: None, complete=complete,
    )


class Evaluator:
    def __init__(self, value):
        self.value, self.calls = value, 0

    def score(self, question, passage):
        self.calls += 1
        return self.value


def external(namespace="tenant-a"):
    text, url = "The office is in Taipei.", "https://example.org/office"
    digest = hashlib.sha256(text.encode()).hexdigest()
    return {"id": external_source_id(namespace, url, digest), "url": url, "title": "Office",
            "text": text, "content_sha256": digest, "fetched_at": "2026-11-23T00:00:00+00:00"}


@pytest.mark.parametrize("scores,expected", [
    ([], "incorrect"), ([1.0, -2.0], "correct"), ([0.592], "correct"),
    ([0.59199], "ambiguous"), ([-0.995], "ambiguous"), ([-0.99501], "incorrect"),
])
def test_reference_threshold_boundaries(scores, expected):
    assert crag.correction_branch(scores) == expected


@pytest.mark.parametrize("score", [True, "0.6", float("nan"), float("inf")])
def test_invalid_evaluator_scores_fail(score):
    with pytest.raises(CapabilityError):
        crag.correction_branch([score])


@pytest.mark.parametrize("count,expected", [(1, [1]), (9, [9]), (50, [50]), (51, [51]), (60, [50, 10]), (101, [50, 51])])
def test_decomposition_retains_exact_source_spans_and_short_tail(count, expected):
    text = "\n ".join("word" + str(i) for i in range(count))
    strips = crag.source_strips(text)
    assert [len(strip.split()) for strip in strips] == expected
    assert all(strip in text for strip in strips)
    assert " ".join(" ".join(strips).split()) == " ".join(text.split())


def test_decomposition_resource_cap():
    with pytest.raises(CapabilityError, match="sixteen"):
        crag.source_strips("word " * 850)


@pytest.mark.parametrize("score,expected,external_calls", [(1.0, "correct", 0), (0.0, "ambiguous", 1), (-1.1, "incorrect", 1)])
def test_all_correction_branches_preserve_authorized_source_and_cleanup(tmp_path, score, expected, external_calls):
    calls = []
    async def acquire(question, namespace, hosts, bridge):
        calls.append((question, namespace, hosts))
        return [external(namespace)]
    passages = [{"id": "original", "text": "The office is in Taipei."}]
    request = ResearchRequest(protocol="authorized-evidence-v2", engine="crag", namespace="tenant-a",
                              question="Where is the office?", top_k=3, passages=passages,
                              external_hosts=["example.org"])
    result = asyncio.run(crag.search(request.question, passages, 3, tmp_path, provider(),
                                    namespace=request.namespace, external_hosts=request.external_hosts,
                                    evaluator_factory=lambda: Evaluator(score), acquire_sources=acquire))
    result = validate_result(result, request)
    assert result["diagnostics"]["correction_branch"] == expected
    assert len(calls) == external_calls
    ids = {hit["chunk_id"] for hit in result["hits"]}
    assert ("original" in ids) is (expected != "incorrect")
    assert (external()["id"] in ids) is bool(external_calls)
    assert set(result["grounded_answer"]["cited_ids"]) <= ids
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("score", [0.0, -1.1])
def test_missing_external_authorization_never_substitutes_internal_answer(tmp_path, score):
    with pytest.raises(CapabilityError) as error:
        asyncio.run(crag.search("Where is the office?", [{"id": "a", "text": "Taipei office"}], 2,
                               tmp_path, provider(), namespace="tenant-a", external_hosts=[],
                               evaluator_factory=lambda: Evaluator(score)))
    assert error.value.code == "requires_external_source"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("response", [
    "not json", '{"answer":"x","cited_ids":["foreign"],"abstain":false}',
    '{"answer":"x","cited_ids":[],"abstain":false}',
    '{"answer":"x","cited_ids":["a"],"abstain":true}',
    '{"answer":"x","cited_ids":["a"],"abstain":false,"extra":1}',
])
def test_generation_never_invents_evidence_or_repairs_invalid_output(tmp_path, response):
    with pytest.raises(CapabilityError, match="invalid grounded answer"):
        asyncio.run(crag.search("Where is the office?", [{"id": "a", "text": "Taipei office"}], 1,
                               tmp_path, provider(response), namespace="tenant-a", external_hosts=[],
                               evaluator_factory=lambda: Evaluator(1.0)))
    assert list(tmp_path.iterdir()) == []


def test_explicit_generator_refusal_removes_all_usable_evidence(tmp_path):
    response = '{"answer":"","cited_ids":[],"abstain":true}'
    result = asyncio.run(crag.search("Where is the office?", [{"id": "a", "text": "Taipei office"}], 1,
                                    tmp_path, provider(response), namespace="tenant-a", external_hosts=[],
                                    evaluator_factory=lambda: Evaluator(1.0)))
    assert result["hits"] == [] and result["grounded_answer"] is None


def test_external_tenant_mismatch_rejected_before_evaluation(tmp_path):
    evaluated = Evaluator(-1.1)
    async def acquire(*args):
        return [external("different-tenant")]
    with pytest.raises(ValueError, match="tenant"):
        asyncio.run(crag.search("Where is the office?", [{"id": "a", "text": "Taipei office"}], 1,
                               tmp_path, provider(), namespace="tenant-a", external_hosts=["example.org"],
                               evaluator_factory=lambda: evaluated, acquire_sources=acquire))
    assert evaluated.calls == 1
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("mutation", ["missing", "size", "hash", "symlink", "directory"])
def test_artifact_identity_is_checked_before_optional_model_import(tmp_path, monkeypatch, mutation):
    content = b"pinned test artifact"
    monkeypatch.setattr(crag, "ARTIFACTS", {"config.json": (len(content), hashlib.sha256(content).hexdigest())})
    target = tmp_path / "config.json"
    if mutation == "size":
        target.write_bytes(content + b"extra")
    elif mutation == "hash":
        target.write_bytes(b"X" * len(content))
    elif mutation == "symlink":
        real = tmp_path / "real"
        real.write_bytes(content)
        target.symlink_to(real)
    elif mutation == "directory":
        target.mkdir()
    with pytest.raises(CapabilityError, match="artifacts"):
        crag.verify_artifacts(tmp_path)


def test_missing_absolute_runtime_path_fails_before_inference(monkeypatch):
    monkeypatch.delenv("RAGS_CRAG_MODEL_PATH", raising=False)
    with pytest.raises(CapabilityError, match="absolute"):
        crag.Evaluator()
    with pytest.raises(CapabilityError):
        crag.Evaluator(str(Path("relative")))
