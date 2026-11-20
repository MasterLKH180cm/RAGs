"""Repeatable offline evaluation with explicit unavailable-engine reporting.

The synthetic corpus is a regression set, not a production-quality benchmark.
--generate adds lexical answer/refusal metrics; no paid model calls occur here.
"""
import argparse
from collections import Counter
from dataclasses import replace
import json
import math
from pathlib import Path
import re
import statistics
import tempfile
import time

from rags.config import Settings
from rags.providers import Providers
from rags.registry import CapabilityError, UPSTREAM_IDS
from rags.backends import BackendUnavailable
from rags.retrieval import MODES, RAGService
from rags.storage import Store


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def answer_scores(actual: str, reference: str) -> tuple[float, float]:
    def tokens(text):
        return re.findall(r"[a-z0-9]+|[^\W\x00-\x7f]", text.casefold())
    a, b = tokens(actual), tokens(reference)
    common = sum((Counter(a) & Counter(b)).values())
    f1 = 2 * common / (len(a) + len(b)) if a or b else 1.0
    return float(a == b), f1


def run(dataset: dict, modes: list[str], k: int = 5, *, generate: bool = False) -> dict:
    documents, cases = dataset["documents"], dataset["cases"]
    if not cases or not documents or not 1 <= k <= 20:
        raise ValueError("Evaluation requires documents, labelled cases and k=1..20")
    identifiers = {doc["id"] for doc in documents}
    for case in cases:
        relevant = case.get("relevant_document_ids")
        if not isinstance(relevant, list) or not set(relevant) <= identifiers:
            raise ValueError("Each case needs valid labelled document IDs")
        if not relevant and not case.get("unanswerable", False):
            raise ValueError("Empty relevance labels require unanswerable=true")
    reports = {}
    with tempfile.TemporaryDirectory(prefix="rags-eval-") as directory:
        settings = replace(Settings.from_env(), db_path=str(Path(directory) / "eval.db"),
                           embedding_provider="local", model_provider="extractive")
        service = RAGService(Store(settings.db_path), Providers(settings))
        for doc in documents:
            service.ingest("evaluation", doc["title"], doc["text"], doc.get("source", "evaluation"),
                           metadata=doc.get("metadata"), document_id=doc["id"],
                           **{key: doc[key] for key in ("rows", "valid_from", "valid_to", "chunking", "tokenizer", "index_features") if key in doc})
        for mode in modes:
            selected_cases = [c for c in cases if not c.get("modes") or mode in c["modes"]]
            recall, precision, reciprocal, ndcg, latency, exact, f1s = [], [], [], [], [], [], []
            valid_citations = all_citations = true_refusals = predicted_refusals = required_refusals = 0
            errors, measurements, structured_checks, temporal_checks = [], [], [], []
            for case in selected_cases:
                started = time.perf_counter()
                try:
                    if mode in UPSTREAM_IDS or mode == "federated":
                        raise CapabilityError("offline_evaluation", "Offline evaluation disables external engines and federation")
                    result = service.query("evaluation", case["question"], mode, k, generate,
                                           **case.get("options", {}))
                except (CapabilityError, BackendUnavailable) as exc:
                    errors.append({"case": case.get("id", case["question"]), "reason": str(exc),
                                   "code": getattr(exc, "code", "requires_backend")})
                    continue
                latency.append((time.perf_counter() - started) * 1000)
                citations = result["citations"]
                found = list(dict.fromkeys(c["document_id"] for c in citations))[:k]
                relevant = set(case["relevant_document_ids"])
                good = sum(key in relevant for key in found)
                if relevant:
                    recall.append(good / len(relevant))
                    precision.append(good / k)
                    first = next((index + 1 for index, key in enumerate(found) if key in relevant), None)
                    reciprocal.append(1 / first if first else 0.0)
                    dcg = sum(1 / math.log2(index + 2) for index, key in enumerate(found) if key in relevant)
                    ideal = sum(1 / math.log2(index + 2) for index in range(min(k, len(relevant))))
                    ndcg.append(dcg / ideal)
                authorized = {c["id"]: c for c in service.store.scope(
                    "evaluation", as_of=case.get("options", {}).get("as_of"),
                    filters=case.get("options", {}).get("filters")) if c["role"] in {"leaf", "parent"}}
                for citation in citations:
                    all_citations += 1
                    source = authorized.get(citation["chunk_id"])
                    valid_citations += bool(source and source["content"] == citation.get("text")
                                            and all(source.get(key) == citation.get(key) for key in
                                                    ("document_id", "revision", "profile_id", "start_char", "end_char")))
                if "expected_aggregate" in case:
                    actual = result.get("structured_result", {}).get("aggregate") or {}
                    structured_checks.append(actual.get("value") == case["expected_aggregate"])
                if "expected_text" in case:
                    temporal_checks.append(any(case["expected_text"] in c["text"] for c in citations))
                if generate:
                    answer = result.get("answer") or ""
                    refused = "insufficient evidence" in answer.casefold() or result.get("supported") is False
                    predicted_refusals += refused
                    required_refusals += bool(case.get("unanswerable"))
                    true_refusals += bool(refused and case.get("unanswerable"))
                    if case.get("reference_answer") is not None and not case.get("unanswerable"):
                        em, f1 = answer_scores(answer, str(case["reference_answer"]))
                        exact.append(em)
                        f1s.append(f1)
                measurements.append({"id": case.get("id"), "category": case.get("category", "unspecified"),
                                     "retrieved_document_ids": found, "mode_used": result["mode_used"],
                                     "budget": result.get("budget"), "usage": result.get("usage")})
            def mean(values):
                return round(statistics.mean(values), 4) if values else None
            reports[mode] = {
                "status": ("not_applicable" if not selected_cases else "available" if latency and not errors
                           else "partial" if latency else "unavailable"),
                "cases": len(selected_cases), "evaluated_cases": len(latency), "k": k,
                "recall_at_k": mean(recall), "precision_at_k": mean(precision), "mrr": mean(reciprocal),
                "ndcg_at_k": mean(ndcg), "mean_latency_ms": mean(latency),
                "p50_latency_ms": round(percentile(latency, .5), 3) if latency else None,
                "p95_latency_ms": round(percentile(latency, .95), 3) if latency else None,
                "answer_exact_match": mean(exact), "answer_token_f1": mean(f1s),
                "answer_metric_reason": None if generate else "Generation disabled",
                "citation_record_validity": valid_citations / all_citations if all_citations else None,
                "citation_entailment": None,
                "citation_entailment_reason": "Record existence does not measure semantic support",
                "structured_exact_accuracy": mean(structured_checks),
                "temporal_source_accuracy": mean(temporal_checks),
                "refusal_precision": true_refusals / predicted_refusals if predicted_refusals else None,
                "refusal_recall": true_refusals / required_refusals if required_refusals else None,
                "provider_tokens": None, "estimated_cost_usd": None,
                "usage_reason": "Offline hash/extractive evaluation; no provider usage or pricing measured",
                "latency_scope": "retrieval_and_generation" if generate else "retrieval_only",
                "errors": errors, "measurements": measurements,
            }
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mode", choices=[*MODES, "all"], default="all")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.k <= 20:
        parser.error("--k must be 1..20")
    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    modes = list(MODES) if args.mode == "all" else [args.mode]
    print(json.dumps(run(dataset, modes, args.k, generate=args.generate), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
