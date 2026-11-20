"""Repeatable, offline retrieval evaluation with labelled relevance judgments.

Usage: RAGS_ENV=test python -m rags.evaluate --dataset examples/eval_baseline.json
No external LLM or embedding charges: always evaluates deterministic offline mode.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import tempfile
import time

from rags.config import Settings
from rags.providers import Providers
from rags.retrieval import MODES, RAGService
from rags.storage import Store


def run(dataset: dict, modes: list[str], k: int = 5) -> dict:
    documents = dataset["documents"]
    cases = dataset["cases"]
    if not cases or not documents:
        raise ValueError("Evaluation requires documents and labelled cases")
    if any(not case.get("relevant_document_ids") for case in cases):
        raise ValueError("Each case needs non-empty relevant_document_ids")
    with tempfile.TemporaryDirectory(prefix="rags-eval-") as directory:
        settings = replace(
            Settings.from_env(),
            db_path=str(Path(directory) / "eval.db"),
            embedding_provider="local",
            model_provider="extractive",
        )
        service = RAGService(Store(settings.db_path), Providers(settings))
        for doc in documents:
            service.ingest(
                "evaluation", doc["title"], doc["text"],
                doc.get("source", "evaluation"),
                document_id=doc["id"],
            )
        reports = {}
        for mode in modes:
            hits, precision, reciprocal, latencies = [], [], [], []
            for case in cases:
                started = time.perf_counter()
                chunks, _ = service.retrieve("evaluation", case["question"], mode, k)
                latencies.append((time.perf_counter() - started) * 1000)
                relevant = set(case["relevant_document_ids"])
                found = [c["document_id"] for c in chunks]
                found_unique = list(dict.fromkeys(found))
                good = sum(item in relevant for item in found_unique[:k])
                hits.append(good / len(relevant))
                precision.append(good / k)
                first = next(
                    (index + 1 for index, item in enumerate(found) if item in relevant),
                    None,
                )
                reciprocal.append(1 / first if first else 0.0)
            reports[mode] = {
                "cases": len(cases),
                "recall_at_k": round(statistics.mean(hits), 4),
                "precision_at_k": round(statistics.mean(precision), 4),
                "mrr": round(statistics.mean(reciprocal), 4),
                "mean_latency_ms": round(statistics.mean(latencies), 2),
                "k": k,
            }
        return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mode", choices=[*MODES, "all"], default="all")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.k <= 20:
        parser.error("--k must be 1..20")
    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    modes = list(MODES) if args.mode == "all" else [args.mode]
    print(json.dumps(run(dataset, modes, args.k), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
