"""CRAG ambiguous decisions preserve both evidence groups after top-k packing."""
import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest

from rags.registry import CapabilityError
from rags.research.crag_adapter import search
from rags.research_protocol import ResearchRequest, external_source_id, validate_result


pytestmark = pytest.mark.unit


class Scores:
    def __init__(self, internal=0.0, external=-1.1):
        self.internal, self.external, self.calls = internal, external, 0

    def score(self, question, text):
        self.calls += 1
        return self.external if text.startswith("External") else self.internal


def provider(prompts):
    def complete(system, user):
        evidence = json.loads(user)["evidence"]
        prompts.append(evidence)
        return json.dumps({"answer": "The office location has changed.", "abstain": False,
                           "cited_ids": list(dict.fromkeys(item["id"] for item in evidence))})

    return SimpleNamespace(
        settings=SimpleNamespace(generation_tokenizer_config={"backend": "unicode"}, context_budget=16000,
                                 output_budget=1024, envelope_budget=256),
        model_identity="fixture:crag-generation", require_generation=lambda: None, complete=complete,
    )


def snapshot():
    text, url = "External reference says the office has moved to Kaohsiung.", "https://example.org/office"
    digest = hashlib.sha256(text.encode()).hexdigest()
    return {"id": external_source_id("tenant-a", url, digest), "url": url, "title": "Office",
            "text": text, "content_sha256": digest, "fetched_at": "2026-11-23T00:00:00+00:00"}


def request(passages, top_k):
    return ResearchRequest(protocol="authorized-evidence-v2", engine="crag", namespace="tenant-a",
                           question="Where is the office?", top_k=top_k, passages=passages,
                           external_hosts=["example.org"])


@pytest.mark.parametrize("external_score,top_k", [(-1.1, 2), (-1.1, 5), (1.1, 2), (1.1, 5)])
def test_ambiguous_branch_keeps_best_internal_and_external_within_hit_budget(tmp_path, external_score, top_k):
    passages = [{"id": f"internal-{index}", "text": f"Internal source {index} records the office in Taipei."}
                for index in range(6)]
    req = request(passages, top_k)
    prompts, acquisitions = [], []
    external = snapshot()

    async def acquire(*args):
        acquisitions.append(True)
        return [external]

    result = asyncio.run(search(req.question, passages, top_k, tmp_path, provider(prompts),
                                namespace=req.namespace, external_hosts=req.external_hosts,
                                evaluator_factory=lambda: Scores(external=external_score), acquire_sources=acquire))
    validate_result(result, req)
    assert acquisitions == [True]
    assert result["diagnostics"]["correction_branch"] == "ambiguous"
    assert len(prompts) == 1
    generated_ids = {item["id"] for item in prompts[0]}
    hit_ids = {item["chunk_id"] for item in result["hits"]}
    assert "internal-0" in generated_ids and external["id"] in generated_ids
    assert generated_ids == hit_ids
    assert len(hit_ids) == top_k
    assert result["external_sources"] == [external]
    assert list(tmp_path.iterdir()) == []


def test_ambiguous_single_hit_budget_fails_before_external_acquisition(tmp_path):
    prompts, acquisitions = [], []

    async def acquire(*args):
        acquisitions.append(True)
        return [snapshot()]

    with pytest.raises(CapabilityError) as error:
        asyncio.run(search("Where is the office?", [{"id": "internal", "text": "Internal office in Taipei."}],
                           1, tmp_path, provider(prompts), namespace="tenant-a", external_hosts=["example.org"],
                           evaluator_factory=Scores, acquire_sources=acquire))
    assert error.value.code == "research_limit_exceeded"
    assert error.value.status_code == 422
    assert acquisitions == prompts == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("score,branch", [(0.0, "ambiguous"), (-1.1, "incorrect")])
def test_empty_acquisition_abstains_without_falling_back_to_internal_generation(tmp_path, score, branch):
    passages = [{"id": "internal", "text": "Internal source records the office in Taipei."}]
    req = request(passages, 3)
    prompts, acquisitions = [], []

    async def acquire(*args):
        acquisitions.append(True)
        return []

    result = asyncio.run(search(req.question, passages, req.top_k, tmp_path, provider(prompts),
                                namespace=req.namespace, external_hosts=req.external_hosts,
                                evaluator_factory=lambda: Scores(internal=score), acquire_sources=acquire))
    validate_result(result, req)
    assert acquisitions == [True]
    assert prompts == []
    assert result["diagnostics"]["correction_branch"] == branch
    assert result["hits"] == result["external_sources"] == []
    assert result["grounded_answer"] is None
    assert list(tmp_path.iterdir()) == []
