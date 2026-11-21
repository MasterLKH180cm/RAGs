"""Gate accounting fixtures verify failure visibility, never model quality."""
import copy

from rags.verify_admission import correctness_gates, exit_status, quality_gates, summarize_gates


def regression_fixture():
    common = {"status": "available", "errors": [], "citation_records_checked": 0,
              "valid_citation_records": 0, "measurements": []}
    modes = {"hybrid": {**common, "admitted_evidence": {"recall_at_k": .85}, "true_refusals": 3, "unanswerable_cases": 4},
             "structured": {**common, "structured_exact_accuracy": 1, "structured_cases": 6},
             "temporal": {**common, "temporal_source_accuracy": 1, "temporal_cases": 4}}
    return {name: copy.deepcopy(modes) for name in ("generate_false", "generate_true")}


def holdout_fixture():
    return {"metrics": {"positive_retained": 15, "negative_refused": 13,
                        "semantic_only_retained": 8, "semantic_only_positive": 8}}


def test_quality_failure_has_nonzero_exit_and_explicit_correctness_only_override():
    held, regression = holdout_fixture(), regression_fixture()
    correctness = summarize_gates(correctness_gates(regression))
    assert correctness["passed"]
    assert exit_status(correctness, summarize_gates(quality_gates(held, regression))) == 0
    held["metrics"]["negative_refused"] = 12
    quality = summarize_gates(quality_gates(held, regression))
    assert quality["failed_count"] == 1 and quality["failures"] == ["holdout_negative_refusal"]
    assert exit_status(correctness, quality) == 2
    assert exit_status(correctness, quality, enforce_quality=False) == 0
    assert quality["passed"] is False  # Override affects exit status, never the measurement.
    correctness["passed"] = False
    assert exit_status(correctness, quality, enforce_quality=False) == 1


def test_exact_mechanism_accuracy_cannot_pass_on_only_a_subset():
    regression = regression_fixture()
    regression["generate_false"]["structured"]["structured_cases"] = 5
    regression["generate_true"]["temporal"]["temporal_cases"] = 3
    result = summarize_gates(quality_gates(holdout_fixture(), regression))
    assert result["failures"] == ["generate_false_structured_exact", "generate_true_temporal_exact"]


def test_contract_gate_rejects_refused_citations_and_generation_mode_disagreement():
    regression = regression_fixture()
    invalid = {"id": "case", "outcome": "insufficient_evidence", "refusal_reason": "no_candidates",
               "supported": False, "admission": {"accepted_count": 1, "decision": "abstain"},
               "returned_citation_count": 1, "answer_is_null": True}
    regression["generate_false"]["hybrid"]["measurements"] = [invalid]
    result = summarize_gates(correctness_gates(regression))
    assert result["failures"] == ["generate_false_hybrid_typed_outcomes", "hybrid_generation_mode_consistency"]
