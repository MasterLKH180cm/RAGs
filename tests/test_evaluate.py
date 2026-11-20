import json
from pathlib import Path

from rags.evaluate import run


def test_labelled_retrieval_evaluation_is_repeatable(monkeypatch):
    monkeypatch.setenv("RAGS_ENV", "test")
    path = Path(__file__).resolve().parents[1] / "examples" / "eval_baseline.json"
    data = json.loads(path.read_text())
    result = run(data, ["sparse", "hybrid", "adaptive"], k=3)
    for report in result.values():
        assert report["cases"] == 3
        assert 0 <= report["recall_at_k"] <= 1
        assert 0 <= report["mrr"] <= 1
    assert result["sparse"]["recall_at_k"] >= 0.66
