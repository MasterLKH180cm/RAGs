import json
from pathlib import Path

import pytest

from rags.evaluate import reference_diagnostic_summary, reference_diagnostics, run


def test_labelled_retrieval_evaluation_is_repeatable(monkeypatch):
    monkeypatch.setenv("RAGS_ENV", "test")
    path = Path(__file__).resolve().parents[1] / "examples" / "eval_baseline.json"
    data = json.loads(path.read_text())
    result = run(data, ["sparse", "hybrid", "adaptive"], k=3)
    for name in ["sparse", "hybrid"]:
        report = result[name]
        assert report["cases"] == 3
        assert 0 <= report["recall_at_k"] <= 1
        assert 0 <= report["mrr"] <= 1
    assert result["sparse"]["recall_at_k"] >= 0.66
    assert result["adaptive"]["status"] == "unavailable"
    assert result["adaptive"]["recall_at_k"] is None
    assert result["adaptive"]["errors"]


def test_answer_metrics_and_unanswerable_cases(monkeypatch):
    monkeypatch.setenv("RAGS_ENV", "test")
    from rags.evaluate import answer_scores
    assert answer_scores("30 minutes", "30 minutes") == (1.0, 1.0)
    assert answer_scores("30 minutes plus unrelated words", "30 minutes")[1] < 1
    report = run({"documents": [{"id": "alpha", "title": "Alpha", "text": "Alpha facts"}],
                  "cases": [{"id": "no-answer", "question": "Missing zeppelin wings?",
                             "relevant_document_ids": [], "unanswerable": True}]}, ["sparse"], generate=True)
    assert report["sparse"]["evaluated_cases"] == 1
    assert report["sparse"]["refusal_recall"] == 1
    assert report["sparse"]["citation_entailment"] is None


def test_v02_labelled_categories_and_exact_mechanisms(monkeypatch):
    monkeypatch.setenv("RAGS_ENV", "test")
    path = Path(__file__).resolve().parents[1] / "examples" / "eval_v02.json"
    data = json.loads(path.read_text())
    assert len(data["cases"]) >= 30
    assert {c["category"] for c in data["cases"]} == {
        "lexical", "semantic", "multi_hop", "structured", "temporal", "no_answer"}
    result = run(data, ["structured", "temporal", "sparse"], k=3, generate=True)
    assert result["structured"]["structured_exact_accuracy"] == 1
    assert result["temporal"]["temporal_source_accuracy"] == 1
    assert all(r["citation_record_validity"] == 1 for r in result.values())
    assert result["sparse"]["recall_at_k"] >= .65
    assert result["sparse"]["citation_entailment"] is None


def test_offline_evaluation_never_uses_configured_external_engines(monkeypatch):
    monkeypatch.setenv("RAGS_ENV", "test")
    monkeypatch.setenv("RAGS_SOURCES_JSON", json.dumps({"remote": {
        "url": "http://forbidden.invalid", "tenant_keys": {"evaluation": "fixture"}}}))
    monkeypatch.setenv("RAGS_RESEARCH_BACKENDS_JSON", json.dumps({"lightrag": {
        "url": "http://forbidden.invalid", "tenant_tokens": {"evaluation": "fixture"}}}))
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline evaluation must not open an HTTP client")
    monkeypatch.setattr("httpx.Client", forbidden)
    data = {"documents": [{"id": "alpha", "title": "Alpha", "text": "Alpha facts"}],
            "cases": [{"question": "Alpha facts?", "relevant_document_ids": ["alpha"]}]}
    result = run(data, ["hybrid", "federated", "lightrag", "colbert"])
    assert result["hybrid"]["status"] == "available"
    for mode in ("federated", "lightrag", "colbert"):
        assert result[mode]["status"] == "unavailable"
        assert result[mode]["errors"][0]["code"] == "offline_evaluation"
        assert result[mode]["reference_diagnostics"]["cases"] == 0


@pytest.mark.unit
@pytest.mark.parametrize(("relevant", "raw", "found", "coverage", "missing", "removed"), [
    (set(), ["other"], [], "not_applicable", [], []),
    ({"a"}, [], [], "none", ["a"], []),
    ({"a"}, ["other"], [], "none", ["a"], []),
    ({"a", "b"}, ["a", "other", "a"], ["a", "a"], "partial", ["b"], []),
    ({"a", "b"}, ["b", "other", "a", "b"], ["a"], "complete", [], ["b"]),
])
def test_reference_coverage_counts_unique_document_ids(relevant, raw, found, coverage, missing, removed):
    diagnostic = reference_diagnostics(relevant, raw, found, outcome="success", refusal_reason=None)
    assert diagnostic["reference_document_ids"] == sorted(relevant)
    assert diagnostic["raw_reference_document_ids"] == list(dict.fromkeys(key for key in raw if key in relevant))
    assert diagnostic["admitted_reference_document_ids"] == list(dict.fromkeys(key for key in found if key in relevant))
    assert diagnostic["raw_reference_coverage"] == coverage
    assert diagnostic["missing_raw_reference_document_ids"] == missing
    assert diagnostic["removed_reference_document_ids"] == removed
    assert diagnostic["reference_removal_cause"] == ("unattributed" if removed else None)


@pytest.mark.unit
@pytest.mark.parametrize(("outcome", "reason", "cause"), [
    ("insufficient_evidence", "below_admission_threshold", "admission_refusal"),
    ("insufficient_evidence", "model_rejected", "admission_refusal"),
    ("insufficient_evidence", "context_budget_exhausted", "context_budget"),
    ("insufficient_evidence", "no_candidates", "unattributed"),
    ("success", None, "unattributed"),
    ("success", "below_admission_threshold", "unattributed"),
])
def test_reference_removal_requires_an_observed_refusal_cause(outcome, reason, cause):
    diagnostic = reference_diagnostics({"a"}, ["a"], [], outcome=outcome, refusal_reason=reason)
    summary = reference_diagnostic_summary([
        {"outcome": outcome, "unanswerable": False, "reference_diagnostics": diagnostic},
    ])
    assert diagnostic["reference_removal_cause"] == cause
    assert summary["reference_removals"]["cases"] == 1
    assert summary["reference_removals"]["document_occurrences"] == 1
    assert summary["reference_removals"]["by_cause"][cause] == {"cases": 1, "document_occurrences": 1}
    assert sum(summary["answerable_refusals_by_raw_reference_coverage"].values()) == int(outcome == "insufficient_evidence")


@pytest.mark.integration
@pytest.mark.parametrize("generate", [False, True])
def test_evaluation_separates_retrieval_coverage_from_admission_refusals(monkeypatch, generate):
    """Actual retrieval/admission control flow; fixed scores are not model-quality evidence."""
    from rags.admission import PassageScores

    monkeypatch.setenv("RAGS_ENV", "test")
    calls = []

    class RefusalScorer:
        def verify(self, profile):
            assert profile["scorer"]["model"] == "Qwen/Qwen3-1.7B"

        def score(self, question, passages):
            calls.append((question, passages))
            return PassageScores(tuple(-100.0 for _ in passages), (), tuple(10 for _ in passages))

    monkeypatch.setattr("rags.admission.scorer_for", lambda *args: RefusalScorer())
    path = Path(__file__).resolve().parents[1] / "profiles" / "admission-qwen-v03.json"
    data = {"documents": [
        {"id": "copper", "title": "Copper", "text": "Copper is the first fixture material."},
        {"id": "zinc", "title": "Zinc", "text": "Zinc is the second fixture material."},
    ], "cases": [
        {"id": "available", "question": "Copper?", "relevant_document_ids": ["copper"]},
        {"id": "miss", "question": "Copper?", "relevant_document_ids": ["zinc"]},
        {"id": "partial", "question": "Copper?", "relevant_document_ids": ["copper", "zinc"]},
        {"id": "no-answer", "question": "Copper?", "relevant_document_ids": [], "unanswerable": True},
    ]}
    report = run(data, ["sparse"], k=1, generate=generate, admission_profile=str(path))["sparse"]
    assert len(calls) == 4
    assert report["false_refusals"] == 3 and report["true_refusals"] == 1
    diagnostic = report["reference_diagnostics"]
    assert diagnostic["scope"] == "document_id_coverage_only"
    assert diagnostic["cases"] == report["evaluated_cases"] == 4
    assert diagnostic["raw_reference_coverage"] == {"not_applicable": 1, "none": 1, "partial": 1, "complete": 1}
    assert diagnostic["answerable_refusals_by_raw_reference_coverage"] == {
        "not_applicable": 0, "none": 1, "partial": 1, "complete": 1,
    }
    assert diagnostic["reference_removals"] == {
        "cases": 2, "document_occurrences": 2,
        "by_cause": {"admission_refusal": {"cases": 2, "document_occurrences": 2},
                     "context_budget": {"cases": 0, "document_occurrences": 0},
                     "unattributed": {"cases": 0, "document_occurrences": 0}},
    }
    rows = {row["id"]: row for row in report["measurements"]}
    assert all(row["raw_candidate_document_ids"] == ["copper"] for row in rows.values())
    assert all(row["retrieved_document_ids"] == [] for row in rows.values())
    assert rows["miss"]["reference_diagnostics"]["removed_reference_document_ids"] == []
    assert rows["partial"]["reference_diagnostics"]["missing_raw_reference_document_ids"] == ["zinc"]


@pytest.mark.unit
def test_frozen_qwen_report_has_reproducible_reference_attribution():
    """Read historical records only; no inference, retuning, or artifact rewrite."""
    root = Path(__file__).resolve().parents[1]
    recorded = json.loads((root / "reports" / "admission-verification-qwen-v03.json").read_text())
    dataset = json.loads((root / "examples" / "eval_v02.json").read_text())
    cases = {case["id"]: case for case in dataset["cases"]}
    for generation in ("generate_false", "generate_true"):
        report = recorded["regression"][generation]["hybrid"]
        rows = []
        for measurement in report["measurements"]:
            case = cases[measurement["id"]]
            diagnostic = reference_diagnostics(
                set(case["relevant_document_ids"]), measurement["raw_candidate_document_ids"],
                measurement["retrieved_document_ids"], outcome=measurement["outcome"],
                refusal_reason=measurement["refusal_reason"],
            )
            rows.append({"id": measurement["id"], "outcome": measurement["outcome"],
                         "unanswerable": bool(case.get("unanswerable")), "reference_diagnostics": diagnostic})
        summary = reference_diagnostic_summary(rows)
        assert summary["raw_reference_coverage"] == {"not_applicable": 4, "none": 3, "partial": 0, "complete": 17}
        assert summary["answerable_refusals_by_raw_reference_coverage"] == {
            "not_applicable": 0, "none": 2, "partial": 0, "complete": 3,
        }
        assert sum(summary["answerable_refusals_by_raw_reference_coverage"].values()) == report["false_refusals"]
        assert {row["id"] for row in rows if row["reference_diagnostics"]["raw_reference_coverage"] == "none"} == {
            "semantic-01", "semantic-03", "semantic-06",
        }
        assert {row["id"] for row in rows if row["reference_diagnostics"]["reference_removal_cause"] == "admission_refusal"} == {
            "semantic-05", "multi_hop-01", "multi_hop-02",
        }
        assert summary["reference_removals"]["cases"] == 3
        assert summary["reference_removals"]["document_occurrences"] == 5
        assert summary["reference_removals"]["by_cause"]["admission_refusal"] == {"cases": 3, "document_occurrences": 5}
