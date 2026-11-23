"""Protocol fixtures test decisions and isolation, never stand in for model quality."""
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from threading import Lock
from threading import Barrier
import time
from types import SimpleNamespace

import httpx
import pytest

from rags.admission import (CrossEncoderScorer, EvidenceAdmission, LIMITS, MODEL, PROFILE_SCHEMA,
                            SCORE_CONVENTION, PassageScores, canonical_digest, load_profile,
                            metadata, validate_remote_decision)
from rags.admission_eval import build_profile, choose_threshold, outcome_metrics, write_new
from rags.config import Settings
from rags.evaluate import run
from rags.providers import Providers
from rags.registry import CapabilityError
from rags.retrieval import RAGService, SYSTEM
from rags.storage import Store


class ProtocolScorer:
    """Fixed scores prove control flow only; no encoder or quality claim."""
    def __init__(self, scores):
        self.scores, self.calls = scores, []
        self.identity = {"model": MODEL, "artifacts": {name: "a" * 64 for name in
                         ("onnx/model_int8.onnx", "tokenizer.json", "tokenizer_config.json")},
                         "runtime": {"fixture": "not-real-inference"}, "score_convention": SCORE_CONVENTION,
                         "limits": dict(LIMITS), "aggregation": "max_original_passage"}

    def verify(self, profile):
        if profile["scorer"] != self.identity:
            raise CapabilityError("admission_profile_mismatch", "Fixture identity differs")

    def score(self, question, passages):
        self.calls.append((question, list(passages)))
        scores = self.scores(question, passages) if callable(self.scores) else self.scores
        return PassageScores(tuple(scores[:len(passages)]), (), tuple(10 for _ in passages))


def configured_admission(tmp_path, scorer, *, threshold=0.5):
    value = {"schema_version": PROFILE_SCHEMA, "scorer": scorer.identity, "threshold": threshold,
             "languages": ["en", "zh"], "calibration": {"cases_digest": "b" * 64,
             "objective": "max_specificity_then_retention_then_smallest_threshold_v1", "measurements": {"fixture": True}}}
    value["profile_id"] = canonical_digest(value)
    path = tmp_path / "protocol-profile.json"
    path.write_text(json.dumps(value))
    return EvidenceAdmission("calibrated_required", str(path), scorer=scorer), path


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGS_ENV", "test")
    settings = replace(Settings.from_env(), db_path=str(tmp_path / "admission.db"),
                       embedding_provider="local", model_provider="extractive",
                       admission_policy="compatibility", admission_profile=None)
    return RAGService(Store(settings.db_path), Providers(settings))


def source(service, doc="one", text="A lighthouse flashes every eight seconds.", **kwargs):
    return service.ingest("a", "Coast", text, "test", document_id=doc, **kwargs)


def test_compatibility_exposes_unassessed_and_empty_scope_contract(service):
    source(service)
    result = service.query("a", "lighthouse flashes", generate=False)
    assert result["outcome"] == "success" and result["supported"] is None
    assert result["answer"] is None and result["refusal_reason"] is None
    assert result["admission"] == metadata("unassessed", "ranked_unassessed", 1, 1)
    assert result["retrieval_diagnostics"]["candidate_ids"] == [result["citations"][0]["chunk_id"]]
    for generate in (False, True):
        empty = service.query("foreign", "lighthouse", generate=generate)
        assert empty["outcome"] == "insufficient_evidence" and empty["supported"] is False
        assert empty["refusal_reason"] == "no_candidates" and empty["citations"] == []
        assert (empty["answer"] is None) is (not generate)


def test_max_admission_preserves_order_all_candidates_and_semantic_only(service, tmp_path):
    source(service, text="Young canines rest in individual indoor kennels.")
    source(service, "other", "Visitors can borrow a towel at the entrance.")
    question = "Where do puppies sleep?"
    before = service.query("a", question, top_k=2, generate=False)["citations"]
    scorer = ProtocolScorer([0.9, -2.0])
    service.admission, _ = configured_admission(tmp_path, scorer)
    result = service.query("a", question, top_k=2, generate=False)
    assert result["citations"] == before
    assert result["supported"] is None and result["outcome"] == "success"
    assert result["admission"]["decision"] == "accept"
    assert result["admission"]["accepted_count"] == 2 and result["admission"]["score"] == .9
    assert scorer.calls == [(question, [c["text"] for c in before])]


def test_refusal_removes_citations_and_never_generates(service, tmp_path, monkeypatch):
    source(service)
    scorer = ProtocolScorer([-10.0])
    service.admission, _ = configured_admission(tmp_path, scorer)
    monkeypatch.setattr(service.provider, "complete", lambda *args: pytest.fail("Generation after refusal"))
    for generate in (False, True):
        result = service.query("a", "Missing answer?", generate=generate)
        assert result["outcome"] == "insufficient_evidence" and result["supported"] is False
        assert result["refusal_reason"] == "below_admission_threshold" and result["citations"] == []
        assert result["admission"]["accepted_count"] == 0
        assert result["retrieval_diagnostics"]["candidate_count"] == 1
        assert result["retrieval_diagnostics"]["usable_as_evidence"] is False
        assert "text" not in json.dumps(result["retrieval_diagnostics"])


def test_required_profile_and_explicit_model_rejection(service, tmp_path, monkeypatch):
    source(service)
    service.admission = EvidenceAdmission("calibrated_required")
    with pytest.raises(CapabilityError) as error:
        service.query("a", "lighthouse", generate=False)
    assert error.value.code == "requires_admission_profile"
    scorer = ProtocolScorer([100.0])
    service.admission, _ = configured_admission(tmp_path, scorer)
    raw = service._retrieve("a", "lighthouse", "hybrid", 1, {})
    monkeypatch.setattr(service, "_retrieve", lambda *args: {**raw, "supported": False, "draft_answer": "Never expose this"})
    result = service.query("a", "lighthouse", generate=False)
    assert result["refusal_reason"] == "model_rejected" and not result["citations"]
    assert not scorer.calls and "draft_answer" not in result


def test_positive_model_review_does_not_bypass_required_calibration(service, tmp_path, monkeypatch):
    source(service)
    raw = service._retrieve("a", "lighthouse", "hybrid", 1, {})
    monkeypatch.setattr(service, "_retrieve", lambda *args: {**raw, "supported": True})
    service.admission = EvidenceAdmission("calibrated_required")
    with pytest.raises(CapabilityError) as error:
        service.query("a", "lighthouse", generate=False)
    assert error.value.code == "requires_admission_profile"
    scorer = ProtocolScorer([-1.0])
    service.admission, _ = configured_admission(tmp_path, scorer)
    result = service.query("a", "lighthouse", generate=False)
    assert result["refusal_reason"] == "below_admission_threshold" and result["supported"] is False


def test_model_rejected_reason_survives_an_empty_kept_source_set(service, monkeypatch):
    monkeypatch.setattr(service, "_retrieve", lambda *args: {"chunks": [], "mode_used": "corrective",
                        "supported": False, "trace": [{"action": "evidence_review", "reviewed_candidate_count": 2, "kept_ids": []}]})
    result = service.query("a", "Unanswered question", generate=False)
    assert result["refusal_reason"] == "model_rejected" and result["admission"]["basis"] == "model_review"
    assert result["citations"] == [] and result["outcome"] == "insufficient_evidence"


def test_scoring_sees_only_authorized_selected_revision(service, tmp_path):
    old = source(service, text="The old beacon is amber.", valid_from="2024-01-01T00:00:00Z")
    source(service, text="The new beacon is blue.", valid_from="2025-01-01T00:00:00Z", expected_revision=old["revision"])
    service.ingest("foreign", "Private", "Secret beacon spectrum.", "test", document_id="secret")
    scorer = ProtocolScorer([1.0])
    service.admission, _ = configured_admission(tmp_path, scorer)
    result = service.query("a", "beacon color", "temporal", 1, False, as_of="2024-06-01T00:00:00Z")
    assert result["citations"][0]["revision"] == old["revision"]
    assert scorer.calls[0][1] == ["The old beacon is amber."]
    service.query("unknown", "beacon", generate=False)
    assert len(scorer.calls) == 1


def test_exact_zero_has_provenance_and_no_dataset_refuses(service, monkeypatch):
    ingested = source(service, text="", rows=[{"item": "lamp", "count": 2}])
    service.admission = EvidenceAdmission("calibrated_required")
    monkeypatch.setattr(service.provider, "complete", lambda *args: pytest.fail("Exact result needs no generation"))
    options = {"row_filters": [{"field": "item", "op": "eq", "value": "absent"}], "aggregate": {"operation": "count"}}
    for generate in (False, True):
        zero = service.query("a", "Count absent items", "structured", generate=generate, **options)
        assert zero["supported"] is True and zero["outcome"] == "success" and zero["citations"] == []
        assert zero["admission"]["basis"] == "structured_exact"
        assert zero["structured_result"]["aggregate"]["value"] == 0
        assert zero["structured_result"]["provenance"]["datasets"] == [{
            "document_id": "one", "revision": ingested["revision"], "profile_id": ingested["profile_id"]}]
    absent = service.query("other", "Count absent items", "structured", generate=False, **options)
    assert absent["outcome"] == "insufficient_evidence" and absent["refusal_reason"] == "no_candidates"
    assert absent["structured_result"]["provenance"]["datasets"] == []


def test_structured_dataset_limit_is_not_silent_truncation(service):
    for index in range(21):
        source(service, str(index), text="", rows=[{"n": index}])
    with pytest.raises(CapabilityError) as error:
        service.query("a", "Count rows", "structured", generate=False, aggregate={"operation": "count"})
    assert error.value.code == "structured_dataset_limit"


def test_structured_numeric_overflow_is_not_valid_exact_evidence(service):
    source(service, text="", rows=[{"n": 1e308}, {"n": 1e308}])
    with pytest.raises(CapabilityError) as error:
        service.query("a", "Sum values", "structured", generate=False, aggregate={"operation": "sum", "field": "n"})
    assert error.value.code == "invalid_aggregate"


def test_context_exhaustion_is_typed_refusal(service, monkeypatch):
    source(service)
    question = "beacon"
    fixed = len(f"{SYSTEM}\nQUESTION:\n{question}\nEVIDENCE:\n")
    service.settings = replace(service.settings, context_budget=512, envelope_budget=64, output_budget=512 - 64 - fixed - 1)
    monkeypatch.setattr(service.provider, "complete", lambda *args: pytest.fail("No evidence fits"))
    result = service.query("a", question)
    assert result["refusal_reason"] == "context_budget_exhausted"
    assert result["supported"] is False and result["admission"]["basis"] == "context_budget"
    assert result["citations"] == [] and result["budget"]["included_ids"] == []


def test_profiles_are_content_bound_and_never_overwritten(tmp_path):
    scorer = ProtocolScorer([1])
    _, path = configured_admission(tmp_path, scorer)
    profile = load_profile(str(path))
    profile["threshold"] = 100
    path.write_text(json.dumps(profile))
    with pytest.raises(CapabilityError) as error:
        EvidenceAdmission("compatibility", str(path))
    assert error.value.code == "invalid_admission_profile"
    with pytest.raises(FileExistsError):
        write_new(str(path), {})
    real = CrossEncoderScorer.__new__(CrossEncoderScorer)
    real.identity = {"actual_artifacts": "different"}
    with pytest.raises(CapabilityError) as error:
        real.verify(profile)
    assert error.value.code == "admission_profile_mismatch"


def test_huge_json_integer_profile_threshold_is_a_typed_configuration_error(tmp_path):
    _, path = configured_admission(tmp_path, ProtocolScorer([1]))
    profile = load_profile(str(path))
    profile["threshold"] = 10 ** 1000
    profile["profile_id"] = canonical_digest({key: value for key, value in profile.items() if key != "profile_id"})
    path.write_text(json.dumps(profile))
    with pytest.raises(CapabilityError) as error:
        load_profile(str(path))
    assert error.value.code == "invalid_admission_profile"


@pytest.mark.parametrize("score", [float("nan"), float("inf"), True, "0.9"])
def test_encoder_numeric_output_is_validated_without_real_inference(score):
    real = CrossEncoderScorer.__new__(CrossEncoderScorer)
    real.counter = SimpleNamespace(encode=lambda *args: SimpleNamespace(ids=[1, 2]),
                                   encode_batch=lambda pairs: [SimpleNamespace(ids=[1, 2, 3]) for _ in pairs])
    real.encoder = SimpleNamespace(model=SimpleNamespace(model=object()), rerank=lambda *args, **kwargs: [score])
    real._lock = Lock()
    with pytest.raises(CapabilityError) as error:
        real.score("Where?", ["Source text"])
    assert error.value.code == "invalid_admission_score"


def test_encoder_bounds_apply_before_inference():
    real = CrossEncoderScorer.__new__(CrossEncoderScorer)
    for question, passages in (("q" * 2001, ["source"]), ("query", ["source"] * 21),
                               ("query", ["x" * 200_001]), ("query", [" "])):
        with pytest.raises(CapabilityError) as error:
            real.score(question, passages)
        assert error.value.code == "admission_input_limit"
    real.counter = SimpleNamespace(encode=lambda *args: SimpleNamespace(ids=list(range(LIMITS["pair_tokens"]))))
    with pytest.raises(CapabilityError) as error:
        real.score("query", ["source"])
    assert error.value.code == "admission_question_limit"


def test_concurrent_cold_start_constructs_only_one_large_model(tmp_path, monkeypatch):
    import rags.admission as admission
    import rags.answerability as answerability
    barrier, counter_lock = Barrier(8), Lock()
    calls = []
    def cheap_constructor(cache_dir):
        with counter_lock:
            calls.append(cache_dir)
        # Permit competing callers to enter the factory during a cold miss.
        time.sleep(.025)
        return object()
    monkeypatch.setattr(answerability, "AnswerabilityScorer", cheap_constructor)
    admission._cached_scorer.cache_clear()
    def invoke(index):
        barrier.wait(timeout=5)
        return admission.scorer_for(answerability.QWEN_MODEL, answerability.QWEN_REVISION,
                                    str(tmp_path / "models" / ".") if index % 2 else str(tmp_path / "models"))
    with ThreadPoolExecutor(max_workers=8) as workers:
        results = list(workers.map(invoke, range(8)))
    assert len(calls) == 1 and all(result is results[0] for result in results)
    admission._cached_scorer.cache_clear()


def test_threshold_objective_and_no_holdout_calibration():
    rows = [{"answerable": True, "score": float(index), "category": "literal"} for index in range(24)]
    rows += [{"answerable": False, "score": score, "category": "missing_field"} for score in (-1.0, 0.5)]
    threshold, metrics = choose_threshold(rows)
    assert threshold == 1.0 and metrics["positive_retained"] == 23 and metrics["negative_refused"] == 2
    assert outcome_metrics(rows, threshold)["false_refusals"] == 1
    with pytest.raises(ValueError, match="Only the calibration"):
        build_profile({"split": "holdout"}, ProtocolScorer([]), rows)
    for bad in (float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            choose_threshold([{**rows[0], "score": bad}, rows[-1]])


def _remote(citations, *, refuse=False):
    count = len(citations)
    return {"citations": citations, "outcome": "insufficient_evidence" if refuse else "success",
            "supported": False if refuse else None, "refusal_reason": "no_candidates" if refuse else None,
            "admission": metadata("abstain" if refuse else "unassessed", "no_candidates" if refuse else "ranked_unassessed", count, 0 if refuse else count)}


@pytest.mark.parametrize("change", [
    {"supported": False}, {"outcome": "insufficient_evidence"}, {"refusal_reason": "no_candidates"},
    {"supported": "true"}, {"admission": {"freeform": "not permitted"}},
])
def test_federated_typed_contradictions_fail(change):
    body = _remote([{}])
    body.update(change)
    with pytest.raises(ValueError):
        validate_remote_decision(body, 1)


@pytest.mark.parametrize("value", ["false", 0, 1, {}, [], 0.0])
def test_legacy_federated_support_is_boolean_or_null(value):
    with pytest.raises(ValueError, match="support assessment"):
        validate_remote_decision({"citations": [{}], "supported": value}, 1)


@pytest.mark.parametrize("invalid", [
    {"revision": {}}, {"profile_id": []}, {"title": []}, {"source": {}}, {"section": True},
    {"revision": "r" * 101}, {"profile_id": "p" * 101}, {"title": "x" * 201},
    {"start_char": "0"}, {"end_char": -1}, {"start_char": True}, {"end_char": 10 ** 1000},
    {"start_char": 10, "end_char": 5},
])
def test_federated_invalid_citation_provenance_is_rejected(service, monkeypatch, invalid):
    service.settings = replace(service.settings, federated_sources={"remote": {
        "url": "https://remote.example", "tenant_keys": {"a": "fixed-token"}}})
    original = httpx.AsyncClient
    citation = {"chunk_id": "one", "document_id": "source", "text": "The beacon is amber.", **invalid}
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=_remote([citation]))), **kwargs))
    with pytest.raises(CapabilityError) as error:
        service.query("a", "beacon", "federated", generate=False)
    assert error.value.code == "federated_source_failed"


def test_federated_valid_citation_provenance_is_preserved(service, monkeypatch):
    service.settings = replace(service.settings, federated_sources={"remote": {
        "url": "https://remote.example", "tenant_keys": {"a": "fixed-token"}}})
    original = httpx.AsyncClient
    provenance = {"revision": "rev-1", "profile_id": "p" * 64, "start_char": 12, "end_char": 32,
                  "title": "Coast", "source": "manual", "section": "Navigation"}
    citation = {"chunk_id": "one", "document_id": "source", "text": "The beacon is amber.", **provenance}
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=_remote([citation]))), **kwargs))
    result = service.query("a", "beacon", "federated", generate=False)
    assert result["outcome"] == "success"
    assert all(result["citations"][0][key] == value for key, value in provenance.items())


def test_federation_drops_refused_branch_and_preserves_independent_evidence(service, monkeypatch):
    service.settings = replace(service.settings, federated_sources={name: {
        "url": f"https://{name}.example", "tenant_keys": {"a": "fixed-token"}} for name in ("empty", "useful")})
    original = httpx.AsyncClient
    citation = {"chunk_id": "one", "document_id": "source", "text": "A buoy rings during rough seas."}
    def respond(request):
        body = _remote([], refuse=True) if request.url.host == "empty.example" else {"citations": [citation]}
        return httpx.Response(200, json=body)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(respond), **kwargs))
    result = service.query("a", "buoy", "federated", generate=False)
    assert [c["chunk_id"] for c in result["citations"]] == ["useful:one"]
    assert result["source_decisions"][0]["outcome"] == "insufficient_evidence"
    assert result["source_decisions"][1]["admission"]["decision"] == "unassessed"
    bad = _remote([citation], refuse=True)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=bad)), **kwargs))
    with pytest.raises(CapabilityError) as error:
        service.query("a", "buoy", "federated", generate=False)
    assert error.value.code == "federated_source_failed"


def test_federated_calibration_scores_and_counts_are_not_trusted_blindly():
    base = _remote([{}])
    base["admission"] = metadata("accept", "calibrated_reranker", 1, 1, profile_id="a" * 64, score=1.0, threshold=.5)
    assert validate_remote_decision(base, 1)["supported"] is None
    for field, value in (("score", float("nan")), ("threshold", True), ("accepted_count", 2), ("score", 0.1),
                         ("profile_id", "unsafe"), ("score", 10 ** 1000), ("threshold", 10 ** 1000)):
        altered = copy.deepcopy(base)
        altered["admission"][field] = value
        with pytest.raises(ValueError):
            validate_remote_decision(altered, 1)


def test_answerability_basis_requires_explicit_support_in_federation():
    base = _remote([{}])
    base["admission"] = metadata("accept", "calibrated_answerability", 1, 1, profile_id="a" * 64, score=1.0, threshold=.5)
    with pytest.raises(ValueError):
        validate_remote_decision(base, 1)
    base["supported"] = True
    assert validate_remote_decision(base, 1)["admission"]["basis"] == "calibrated_answerability"


def test_federation_rejects_incomplete_and_wrong_basis_decisions():
    success = _remote([{}])
    del success["supported"]
    with pytest.raises(ValueError):
        validate_remote_decision(success, 1)
    refusal = _remote([], refuse=True)
    refusal["admission"]["basis"] = "model_review"
    with pytest.raises(ValueError):
        validate_remote_decision(refusal, 0)
    refusal["admission"] = metadata("abstain", "calibrated_reranker", 1, 0, profile_id="a" * 64, score=1, threshold=.5)
    refusal["refusal_reason"] = "below_admission_threshold"
    with pytest.raises(ValueError):
        validate_remote_decision(refusal, 0)


def test_api_does_not_accept_caller_admission_configuration(service):
    from fastapi.testclient import TestClient
    from rags.api import create_app
    source(service)
    settings = replace(service.settings, keys={"test-key": "a"}, admission_policy="calibrated_required")
    with TestClient(create_app(settings)) as client:
        headers = {"Authorization": "Bearer test-key"}
        ordinary = client.post("/query", headers=headers, json={"question": "lighthouse", "generate": False})
        assert ordinary.status_code == 503
        for field, value in (("admission_policy", "compatibility"), ("admission_profile", "/arbitrary"), ("threshold", -100)):
            response = client.post("/query", headers=headers, json={"question": "lighthouse", field: value})
            assert response.status_code == 422


def test_evaluation_uses_outcomes_without_generation_and_ignores_env_profile(monkeypatch):
    monkeypatch.setenv("RAGS_ADMISSION_POLICY", "calibrated_required")
    monkeypatch.setenv("RAGS_ADMISSION_PROFILE", "/not-a-real-production-profile")
    data = {"documents": [{"id": "dock", "title": "Dock", "text": "Blue lantern marks the dock."}],
            "cases": [{"question": "Blue lantern", "relevant_document_ids": ["dock"]},
                      {"question": "Invisible marmot?", "relevant_document_ids": [], "unanswerable": True}]}
    for generate in (False, True):
        report = run(data, ["sparse"], k=1, generate=generate)["sparse"]
        assert report["refusal_recall"] == 1.0 and report["false_refusals"] == 0
        assert report["raw_candidates"] == report["admitted_evidence"]
        assert report["admission_profile"] is None
