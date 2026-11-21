"""Calibrate on the committed split, freeze a profile, then measure a held-out split."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from rags.admission import CrossEncoderScorer, PROFILE_SCHEMA, canonical_digest, finite, load_profile


OBJECTIVE = "max_specificity_then_retention_then_smallest_threshold_v1"


def read_dataset(path: str) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = value.get("cases")
    if (value.get("schema_version") != "admission-cases-v1" or value.get("split") not in {"calibration", "holdout"}
            or not isinstance(cases, list) or not cases or len(cases) > 1000
            or value.get("cases_digest", {}).get("value") != canonical_digest(cases)):
        raise ValueError("Invalid or modified admission fixture")
    seen = set()
    for case in cases:
        if (not isinstance(case, dict) or not isinstance(case.get("id"), str) or case["id"] in seen
                or type(case.get("answerable")) is not bool or case.get("language") not in {"en", "zh"}
                or not isinstance(case.get("question"), str) or not isinstance(case.get("passages"), list)
                or not case["passages"] or any(not isinstance(p, dict) or not isinstance(p.get("text"), str) for p in case["passages"])):
            raise ValueError("Invalid admission case")
        seen.add(case["id"])
    return value


def score_cases(dataset: dict, scorer, *, progress=None) -> list[dict]:
    rows = []
    for case in dataset["cases"]:
        # Labels, references, IDs, language/category, and rationales never enter inference.
        result = scorer.score(case["question"], [p["text"] for p in case["passages"]])
        if not result.scores or any(not finite(score) for score in result.scores):
            raise ValueError("Invalid calibration score")
        rows.append({"id": case["id"], "answerable": case["answerable"], "language": case["language"],
                     "category": case["category"], "score": max(result.scores),
                     "passage_scores": list(result.scores), "truncated_passages": len(result.truncated_indices),
                     "original_pair_tokens": list(result.original_pair_tokens)})
        if progress:
            progress(len(rows), len(dataset["cases"]))
    return rows


def outcome_metrics(rows: list[dict], threshold: float) -> dict:
    positive = sum(row["answerable"] for row in rows)
    negative = len(rows) - positive
    retained = sum(row["answerable"] and row["score"] >= threshold for row in rows)
    rejected = sum(not row["answerable"] and row["score"] < threshold for row in rows)
    false_refusals = positive - retained
    abstentions = false_refusals + rejected
    semantic = [row for row in rows if row["answerable"] and row["category"] == "semantic_only"]
    return {"cases": len(rows), "positive": positive, "negative": negative,
            "positive_retained": retained, "negative_refused": rejected, "false_refusals": false_refusals,
            "positive_retention": retained / positive if positive else None,
            "refusal_recall": rejected / negative if negative else None,
            "refusal_precision": rejected / abstentions if abstentions else None,
            "false_refusal_rate": false_refusals / positive if positive else None,
            "abstentions": abstentions, "abstention_rate": abstentions / len(rows) if rows else None,
            "semantic_only_positive": len(semantic),
            "semantic_only_retained": sum(row["score"] >= threshold for row in semantic)}


def choose_threshold(rows: list[dict]) -> tuple[float, dict]:
    """Frozen objective: specificity, retention, then smallest tied threshold."""
    if not rows or any(not finite(row.get("score")) or type(row.get("answerable")) is not bool for row in rows):
        raise ValueError("Calibration requires finite scores and boolean labels")
    positive = sum(row["answerable"] for row in rows)
    if not positive or positive == len(rows):
        raise ValueError("Calibration needs positive and negative cases")
    candidates = {float(row["score"]) for row in rows}
    below = math.nextafter(min(candidates), -math.inf)
    if math.isfinite(below):
        candidates.add(below)
    measured = [(threshold, outcome_metrics(rows, threshold)) for threshold in sorted(candidates)]
    feasible = [(threshold, scores) for threshold, scores in measured
                if scores["positive_retained"] >= math.ceil(0.95 * positive)]
    if not feasible:
        raise ValueError("No calibration threshold satisfies positive retention")
    return max(feasible, key=lambda item: (item[1]["negative_refused"], item[1]["positive_retained"], -item[0]))


def build_profile(dataset: dict, scorer, rows: list[dict]) -> dict:
    if dataset["split"] != "calibration":
        raise ValueError("Only the calibration split can choose a threshold")
    threshold, metrics = choose_threshold(rows)
    profile = {"schema_version": PROFILE_SCHEMA, "scorer": scorer.identity, "threshold": threshold,
               "languages": ["en", "zh"], "calibration": {"cases_digest": canonical_digest(dataset["cases"]),
               "objective": OBJECTIVE, "minimum_positive_retention": 0.95, "measurements": metrics}}
    return {**profile, "profile_id": canonical_digest(profile)}


def measurement_report(dataset: dict, profile: dict, rows: list[dict]) -> dict:
    threshold = profile["threshold"]
    metrics = outcome_metrics(rows, threshold)
    result = {"schema_version": "admission-evaluation-v1", "split": dataset["split"],
              "profile_id": profile["profile_id"], "threshold": threshold,
              "cases_digest": canonical_digest(dataset["cases"]), "scorer": profile["scorer"],
              "metrics": metrics, "by_language": {}, "by_category": {}, "measurements": rows,
              "scope": "Synthetic relevance admission; no semantic entailment or production quality claim"}
    for label, field in (("by_language", "language"), ("by_category", "category")):
        result[label] = {value: outcome_metrics([row for row in rows if row[field] == value], threshold)
                         for value in sorted(Counter(row[field] for row in rows))}
    if dataset["split"] == "holdout":
        if metrics["positive"] != 16 or metrics["negative"] != 16:
            raise ValueError("Frozen holdout acceptance requires exactly 16 positive and 16 negative cases")
        result["quality_gate"] = {"passed": metrics["positive_retained"] >= 15 and metrics["negative_refused"] >= 13
                                  and metrics["semantic_only_retained"] == metrics["semantic_only_positive"],
                                  "required_positive_retained": 15, "required_negative_refused": 13,
                                  "required_semantic_only_retained": metrics["semantic_only_positive"],
                                  "no_retuning": True}
    return result


def write_new(path: str, value: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite a frozen profile or measurement with a later fit.
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["calibrate", "evaluate"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--profile", required=True, help="New output profile for calibration, frozen input for evaluation")
    parser.add_argument("--report", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--scorer", choices=["bge", "qwen"], default="bge", help="Calibration scorer; evaluation dispatches from the frozen profile")
    args = parser.parse_args()
    dataset = read_dataset(args.dataset)
    if args.action == "calibrate" and dataset["split"] != "calibration":
        parser.error("Calibration cannot use held-out labels")
    if args.action == "evaluate" and dataset["split"] != "holdout":
        parser.error("Use evaluate only for the frozen holdout; calibration reports already contain calibration metrics")
    if Path(args.report).exists() or (args.action == "calibrate" and Path(args.profile).exists()):
        parser.error("Refusing to overwrite an existing profile or report")
    profile = load_profile(args.profile) if args.action == "evaluate" else None
    use_qwen = args.scorer == "qwen" if profile is None else profile["scorer"]["model"] == "Qwen/Qwen3-1.7B"
    if use_qwen:
        from rags.answerability import AnswerabilityScorer
        scorer = AnswerabilityScorer(args.cache_dir, allow_download=args.action == "calibrate")
    else:
        scorer = CrossEncoderScorer(args.cache_dir, allow_download=args.action == "calibrate")
    if profile:
        scorer.verify(profile)
    rows = score_cases(dataset, scorer)
    if profile is None:
        profile = build_profile(dataset, scorer, rows)
        write_new(args.profile, profile)
    report = measurement_report(dataset, profile, rows)
    write_new(args.report, report)
    print(json.dumps({"profile_id": profile["profile_id"], "threshold": profile["threshold"],
                      "metrics": report["metrics"], "quality_gate": report.get("quality_gate")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
