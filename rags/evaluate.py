"""Repeatable offline evaluation with explicit unavailable-engine reporting.

The synthetic corpus is a regression set, not a production-quality benchmark.
Domain outcomes drive refusal metrics with or without generation. Optional
--admission-profile explicitly selects real scoring; paid generation stays off.
"""
import argparse
from collections import Counter
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


def retrieval_metrics(found: list[str], relevant: set[str], k: int) -> dict:
    good = sum(key in relevant for key in found)
    first = next((index + 1 for index, key in enumerate(found) if key in relevant), None)
    dcg = sum(1 / math.log2(index + 2) for index, key in enumerate(found) if key in relevant)
    ideal = sum(1 / math.log2(index + 2) for index in range(min(k, len(relevant))))
    return {"recall_at_k": good / len(relevant), "precision_at_k": good / k,
            "mrr": 1 / first if first else 0.0, "ndcg_at_k": dcg / ideal}


def mean_metrics(rows: list[dict]) -> dict:
    return {name: round(statistics.mean(row[name] for row in rows), 4) if rows else None
            for name in ("recall_at_k", "precision_at_k", "mrr", "ndcg_at_k")}


def refusal_metrics(rows: list[dict]) -> dict:
    negatives = sum(row["unanswerable"] for row in rows)
    positives = len(rows) - negatives
    refused = sum(row["outcome"] == "insufficient_evidence" for row in rows)
    true_refused = sum(row["unanswerable"] and row["outcome"] == "insufficient_evidence" for row in rows)
    false_refused = refused - true_refused
    return {"cases": len(rows), "answerable_cases": positives, "unanswerable_cases": negatives,
            "abstentions": refused, "true_refusals": true_refused, "false_refusals": false_refused,
            "abstention_rate": refused / len(rows) if rows else None,
            "refusal_precision": true_refused / refused if refused else None,
            "refusal_recall": true_refused / negatives if negatives else None,
            "false_refusal_rate": false_refused / positives if positives else None}


def run(dataset: dict, modes: list[str], k: int = 5, *, generate: bool = False,
        admission_profile: str | None = None, progress=None) -> dict:
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
        # The regression harness must not inherit provider credentials, source
        # endpoints, tokenizer settings, or admission profiles from production.
        settings = Settings(db_path=str(Path(directory) / "eval.db"), keys={"evaluation-fixture": "evaluation"},
                            embedding_provider="local", model_provider="extractive", openai_key="",
                            openai_base_url="https://api.openai.com/v1", embedding_model="text-embedding-3-small",
                            chat_model="", anthropic_key="", anthropic_model="",
                            admission_policy="calibrated_required" if admission_profile else "compatibility",
                            admission_profile=admission_profile)
        service = RAGService(Store(settings.db_path), Providers(settings))
        for doc in documents:
            service.ingest("evaluation", doc["title"], doc["text"], doc.get("source", "evaluation"),
                           metadata=doc.get("metadata"), document_id=doc["id"],
                           **{key: doc[key] for key in ("rows", "valid_from", "valid_to", "chunking", "tokenizer", "index_features") if key in doc})
        for mode in modes:
            selected_cases = [c for c in cases if not c.get("modes") or mode in c["modes"]]
            admitted_ranking, raw_ranking, decisions, latency, exact, f1s = [], [], [], [], [], []
            valid_citations = all_citations = 0
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
                raw_found = list(dict.fromkeys(result["retrieval_diagnostics"]["candidate_document_ids"]))[:k]
                relevant = set(case["relevant_document_ids"])
                if relevant:
                    admitted_ranking.append(retrieval_metrics(found, relevant, k))
                    raw_ranking.append(retrieval_metrics(raw_found, relevant, k))
                outcome = result.get("outcome")
                if outcome not in {"success", "insufficient_evidence"}:
                    raise ValueError("Query result lacks a valid typed evidence outcome")
                language = case.get("language") or ("zh" if re.search(r"[\u3400-\u9fff]", case["question"]) else "en")
                decisions.append({"outcome": outcome, "unanswerable": bool(case.get("unanswerable")),
                                  "language": language, "category": case.get("category", "unspecified")})
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
                    if case.get("reference_answer") is not None and not case.get("unanswerable"):
                        em, f1 = answer_scores(answer, str(case["reference_answer"]))
                        exact.append(em)
                        f1s.append(f1)
                measurements.append({"id": case.get("id"), "category": case.get("category", "unspecified"),
                                     "language": language, "raw_candidate_document_ids": raw_found,
                                     "retrieved_document_ids": found, "mode_used": result["mode_used"],
                                     "outcome": outcome, "refusal_reason": result["refusal_reason"],
                                     "supported": result["supported"], "admission": result["admission"],
                                     "returned_citation_count": len(citations), "answer_is_null": result["answer"] is None,
                                     "budget": result.get("budget"), "usage": result.get("usage")})
                if progress:
                    progress(mode, len(measurements) + len(errors), len(selected_cases))
            def mean(values):
                return round(statistics.mean(values), 4) if values else None
            reports[mode] = {
                "status": ("not_applicable" if not selected_cases else "available" if latency and not errors
                           else "partial" if latency else "unavailable"),
                "cases": len(selected_cases), "evaluated_cases": len(latency), "k": k,
                **mean_metrics(admitted_ranking), "raw_candidates": mean_metrics(raw_ranking),
                "admitted_evidence": mean_metrics(admitted_ranking), "ranking_cases": len(admitted_ranking),
                "mean_latency_ms": mean(latency),
                "p50_latency_ms": round(percentile(latency, .5), 3) if latency else None,
                "p95_latency_ms": round(percentile(latency, .95), 3) if latency else None,
                "answer_exact_match": mean(exact), "answer_token_f1": mean(f1s),
                "answer_metric_reason": None if generate else "Generation disabled",
                "citation_record_validity": valid_citations / all_citations if all_citations else None,
                "citation_records_checked": all_citations, "valid_citation_records": valid_citations,
                "citation_entailment": None,
                "citation_entailment_reason": "Record existence does not measure semantic support",
                "structured_exact_accuracy": mean(structured_checks),
                "temporal_source_accuracy": mean(temporal_checks),
                "structured_cases": len(structured_checks), "temporal_cases": len(temporal_checks),
                **{key: value for key, value in refusal_metrics(decisions).items() if key != "cases"},
                "refusal_by_language": {lang: refusal_metrics([d for d in decisions if d["language"] == lang])
                                        for lang in sorted({d["language"] for d in decisions})},
                "refusal_by_category": {category: refusal_metrics([d for d in decisions if d["category"] == category])
                                        for category in sorted({d["category"] for d in decisions})},
                "language_assignment": "dataset label, otherwise presence of Han characters",
                "provider_tokens": None, "estimated_cost_usd": None,
                "usage_reason": "Hash/extractive evaluation with optional explicit local admission; no paid provider usage",
                "admission_profile": admission_profile,
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
    parser.add_argument("--admission-profile", help="Explicit frozen local scorer profile; default baseline disables admission models")
    args = parser.parse_args()
    if not 1 <= args.k <= 20:
        parser.error("--k must be 1..20")
    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    modes = list(MODES) if args.mode == "all" else [args.mode]
    print(json.dumps(run(dataset, modes, args.k, generate=args.generate, admission_profile=args.admission_profile), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
