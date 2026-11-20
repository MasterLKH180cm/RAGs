import json
from pathlib import Path

from rags.evaluate import run


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
