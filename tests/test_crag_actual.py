"""Actual official CRAG evaluator and all correction branches on CPU."""
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rags.registry import CapabilityError
from rags.research import crag_adapter as crag
from rags.research_protocol import ResearchRequest, validate_result

pytestmark = [
    pytest.mark.integration, pytest.mark.external, pytest.mark.research_crag,
    pytest.mark.skipif(os.getenv("RAGS_TEST_RESEARCH_UPSTREAMS") not in {"crag", "all"},
                       reason="Explicit real official CRAG evaluator CPU gate"),
]


@pytest.fixture(scope="module")
def evaluator():
    return crag.Evaluator()


def test_actual_evaluator_runs_all_three_correction_branches_and_reference_padding(tmp_path, evaluator):
    from crag_fixture_provider import (
        FRANCE_ANSWER,
        FRANCE_QUESTION,
        FRANCE_TEXT,
        HAMLET_ANSWER,
        HAMLET_QUESTION,
        HAMLET_TEXT,
        IRRELEVANT_TEXT,
        CRAGFixtureProvider,
        external_france,
    )
    started = time.perf_counter()
    checks = []
    namespace = "actual-crag-tenant"

    class ObservedEvaluator:
        def __init__(self):
            self.calls = 0
            self.observations = []

        def score(self, question, passage):
            score = evaluator.score(question, passage)
            self.calls += 1
            self.observations.append({"question": question, "source_sha256": hashlib.sha256(passage.encode()).hexdigest(),
                                      "logit": score, "input_tokens": len(evaluator.tokenizer(question + " [SEP] " + passage)["input_ids"])})
            return score

    for index, (branch, question, text, expected_answer) in enumerate([
        ("correct", HAMLET_QUESTION, HAMLET_TEXT, HAMLET_ANSWER),
        ("ambiguous", FRANCE_QUESTION, FRANCE_TEXT, FRANCE_ANSWER),
        ("incorrect", FRANCE_QUESTION, IRRELEVANT_TEXT, FRANCE_ANSWER),
    ]):
        work = tmp_path / str(index)
        work.mkdir()
        provider, observed, acquired = CRAGFixtureProvider(), ObservedEvaluator(), []

        async def acquire_fixture(question, requested_namespace, hosts, bridge, acquired=acquired):
            assert requested_namespace == namespace and hosts == ["reference.example.org"]
            rewritten = json.loads(bridge.complete(json.dumps({"question": question}),
                                                   system='Rewrite this question as strict JSON {"query":"..."}.'))
            acquired.append(rewritten["query"])
            return [external_france(requested_namespace)]

        passages = [{"id": "authorized-original", "text": text}]
        call_started = time.perf_counter()
        result = asyncio.run(crag.search(question, passages, 5, work, provider, namespace=namespace,
                                         external_hosts=["reference.example.org"], evaluator_factory=lambda observed=observed: observed,
                                         acquire_sources=acquire_fixture))
        elapsed = time.perf_counter() - call_started
        request = ResearchRequest(protocol="authorized-evidence-v2", engine="crag", namespace=namespace,
                                  question=question, top_k=5, passages=passages, external_hosts=["reference.example.org"])
        validate_result(result, request)
        assert result["diagnostics"]["correction_branch"] == branch
        assert result["diagnostics"]["evaluator_calls"] == observed.calls
        assert crag.correction_branch([observed.observations[0]["logit"]]) == branch
        assert all(math.isfinite(item["logit"]) and item["input_tokens"] <= 512 for item in observed.observations)
        assert result["grounded_answer"]["text"] == expected_answer
        assert bool(acquired) is (branch != "correct")
        assert result["diagnostics"]["generation_calls"] == provider.calls == (1 if branch == "correct" else 2)
        if branch == "correct":
            assert result["grounded_answer"]["cited_ids"] == ["authorized-original"]
            assert not result["external_sources"]
        elif branch == "incorrect":
            assert "authorized-original" not in result["grounded_answer"]["cited_ids"]
            assert {item["id"] for item in result["external_sources"]} == set(result["grounded_answer"]["cited_ids"])
        else:
            assert "authorized-original" in result["grounded_answer"]["cited_ids"]
            assert result["external_sources"]
        assert list(work.iterdir()) == []
        checks.append({"branch": branch, "result": result, "score_observations": observed.observations,
                       "elapsed_seconds": elapsed, "source_snapshot_removed": True,
                       "external_acquisition": "explicit_synthetic_snapshot_fixture" if acquired else "not_required",
                       "live_search_executed": False})

    dynamic_score = evaluator.score(HAMLET_QUESTION, HAMLET_TEXT)
    encoded = evaluator.tokenizer(HAMLET_QUESTION + " [SEP] " + HAMLET_TEXT, return_tensors="pt",
                                  padding="max_length", max_length=512, truncation=False)
    assert encoded["input_ids"].shape[1] == 512
    with evaluator.torch.inference_mode():
        reference_score = float(evaluator.model(**encoded).logits.item())
    difference = abs(dynamic_score - reference_score)
    assert difference <= 1e-5
    report = {
        "schema": "actual-crag-evaluator-verification-v1", "engine": "crag",
        "verified_at": datetime.now(UTC).isoformat(), "python": platform.python_version(),
        "verification": "official_trained_evaluator_cpu_all_three_branches_synthetic_generation_and_external_sources",
        "runtime_versions": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "sentencepiece", "safetensors"]},
        "artifact_manifest": {name: {"bytes": size, "sha256": digest} for name, (size, digest) in crag.ARTIFACTS.items()},
        "thresholds": {"upper": crag.UPPER_THRESHOLD, "lower": crag.LOWER_THRESHOLD, "profile": "popqa_reference"},
        "cases": checks,
        "reference_padding_comparison": {"reference_tokens": 512, "reference_fp32_logit": reference_score,
                                          "dynamic_fp32_logit": dynamic_score, "absolute_difference": difference,
                                          "absolute_tolerance": 1e-5},
        "elapsed_seconds_excluding_model_load": time.perf_counter() - started,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "not_verified": ["live_serper_acquisition", "trained_generation_quality", "paper_benchmark_reproduction"],
    }
    destination = os.getenv("RAGS_RESEARCH_REPORT_PATH")
    if destination:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def test_actual_evaluator_rejects_native_token_overflow_before_forward(evaluator):
    calls = evaluator.calls
    with pytest.raises(CapabilityError) as error:
        evaluator.score("What is the capital of France?", "irrelevant passage " * 600)
    assert error.value.code == "context_budget_exceeded"
    assert evaluator.calls == calls
