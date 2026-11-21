"""Replay a frozen profile; expose correctness and quality as separate gates.

Exit 0: enforced gates passed. Exit 1: execution/correctness failure.
Exit 2: quality failure. --no-enforce-quality is explicitly correctness-only;
the resulting report still records failed quality and failed promotion status.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from rags.admission import canonical_digest, load_profile, scorer_for
from rags.admission_eval import measurement_report, read_dataset, score_cases, write_new
from rags.evaluate import run


def gate(name: str, actual, required, passed: bool) -> dict:
    return {"name": name, "actual": actual, "required": required, "passed": bool(passed)}


def quality_gates(held: dict, regression: dict) -> list[dict]:
    metrics = held["metrics"]
    gates = [gate("holdout_positive_retention", metrics["positive_retained"], "at least 15 of 16", metrics["positive_retained"] >= 15),
             gate("holdout_negative_refusal", metrics["negative_refused"], "at least 13 of 16", metrics["negative_refused"] >= 13),
             gate("holdout_semantic_only", metrics["semantic_only_retained"], metrics["semantic_only_positive"],
                  metrics["semantic_only_retained"] == metrics["semantic_only_positive"])]
    for generation, reports in regression.items():
        hybrid, structured, temporal = (reports[name] for name in ("hybrid", "structured", "temporal"))
        recall = hybrid["admitted_evidence"]["recall_at_k"]
        gates.extend([
            gate(f"{generation}_hybrid_admitted_recall_at_3", recall, "at least 0.85", recall is not None and recall >= .85),
            gate(f"{generation}_hybrid_no_answer_refusals", hybrid["true_refusals"], "at least 3 of 4",
                 hybrid["unanswerable_cases"] == 4 and hybrid["true_refusals"] >= 3),
            gate(f"{generation}_structured_exact", {"accuracy": structured["structured_exact_accuracy"], "cases": structured["structured_cases"]},
                 "6/6", structured["structured_cases"] == 6 and structured["structured_exact_accuracy"] == 1),
            gate(f"{generation}_temporal_exact", {"accuracy": temporal["temporal_source_accuracy"], "cases": temporal["temporal_cases"]},
                 "4/4", temporal["temporal_cases"] == 4 and temporal["temporal_source_accuracy"] == 1),
        ])
    return gates


def correctness_gates(regression: dict) -> list[dict]:
    checks = []
    for generation, reports in regression.items():
        for mode, report in reports.items():
            label = f"{generation}_{mode}"
            checks.append(gate(label + "_execution", report["status"], "available without errors",
                               report["status"] == "available" and not report["errors"]))
            checks.append(gate(label + "_citation_records", report["valid_citation_records"], report["citation_records_checked"],
                               report["valid_citation_records"] == report["citation_records_checked"]))
            invalid = []
            for item in report["measurements"]:
                admission = item["admission"]
                refused = item["outcome"] == "insufficient_evidence"
                valid = (admission["accepted_count"] == item["returned_citation_count"]
                         and (generation != "generate_false" or item["answer_is_null"]))
                if refused:
                    valid = valid and item["supported"] is False and admission["decision"] == "abstain" and not item["returned_citation_count"]
                else:
                    valid = valid and item["supported"] is not False and admission["decision"] != "abstain"
                if not valid:
                    invalid.append(item["id"])
            checks.append(gate(label + "_typed_outcomes", invalid, "no invalid decisions", not invalid))
    for mode in ("hybrid", "structured", "temporal"):
        decisions = [{m["id"]: (m["outcome"], m["refusal_reason"]) for m in regression[generation][mode]["measurements"]}
                     for generation in ("generate_false", "generate_true")]
        mismatches = sorted(key for key in decisions[0].keys() | decisions[1].keys()
                            if decisions[0].get(key) != decisions[1].get(key))
        checks.append(gate(mode + "_generation_mode_consistency", mismatches, "identical evidence decisions", not mismatches))
    return checks


def summarize_gates(gates: list[dict]) -> dict:
    failures = [item["name"] for item in gates if not item["passed"]]
    return {"passed": not failures, "passed_count": len(gates) - len(failures),
            "failed_count": len(failures), "failures": failures, "gates": gates}


def exit_status(correctness: dict, quality: dict, enforce_quality: bool = True) -> int:
    if not correctness["passed"]:
        return 1
    return 2 if enforce_quality and not quality["passed"] else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--holdout", default="examples/admission_holdout_v03.json")
    parser.add_argument("--regression", default="examples/eval_v02.json")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--comparison-report", help="Optional already-frozen measurement for a different profile; never used for selection")
    parser.add_argument("--enforce-quality", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if Path(args.report).exists():
        parser.error("Refusing to overwrite a recorded evaluation")
    started = time.perf_counter()
    profile = load_profile(args.profile)
    holdout = read_dataset(args.holdout)
    if holdout["split"] != "holdout":
        parser.error("Final verification requires the frozen holdout split")
    regression_data = json.loads(Path(args.regression).read_text(encoding="utf-8"))
    cache_dir = str(Path(args.cache_dir).resolve())
    os.environ["RAGS_MODEL_CACHE"] = cache_dir
    identity = profile["scorer"]
    revision = identity.get("revision") or canonical_digest(identity["artifacts"])
    scorer = scorer_for(identity["model"], revision, cache_dir)
    scorer.verify(profile)

    def progress(label, completed, total):
        if completed % 4 == 0 or completed == total:
            print(f"{label}: {completed}/{total}", file=sys.stderr, flush=True)

    print("Scoring frozen holdout; no calibration or selection", file=sys.stderr, flush=True)
    rows = score_cases(holdout, scorer, progress=lambda n, total: progress("holdout", n, total))
    held = measurement_report(holdout, profile, rows)
    regression = {}
    for generate in (False, True):
        label = "generate_true" if generate else "generate_false"
        regression[label] = run(regression_data, ["hybrid", "structured", "temporal"], 3, generate=generate,
                                admission_profile=args.profile,
                                progress=lambda mode, n, total, label=label: progress(label + "/" + mode, n, total))
    correctness = summarize_gates(correctness_gates(regression))
    quality = summarize_gates(quality_gates(held, regression))
    report = {"schema_version": "admission-verification-v1", "measured_at": datetime.now(timezone.utc).isoformat(),
              "profile_id": profile["profile_id"], "profile_file_sha256": hashlib.sha256(Path(args.profile).read_bytes()).hexdigest(),
              "frozen_threshold": profile["threshold"], "enforce_quality": args.enforce_quality,
              "correctness": correctness, "quality": quality,
              "promotion_status": "passed" if correctness["passed"] and quality["passed"] else "failed",
              "holdout": held, "regression": regression, "elapsed_seconds": round(time.perf_counter() - started, 3),
              "scope": "Fixed-profile synthetic measurement; no paid generation, no model/prompt/threshold retuning; all inference is local",
              "comparison": None}
    if args.comparison_report:
        comparison = json.loads(Path(args.comparison_report).read_text(encoding="utf-8"))
        report["comparison"] = {"profile_id": comparison["profile_id"], "metrics": comparison["metrics"],
                                "quality_gate": comparison.get("quality_gate"),
                                "disclosure": "BGE holdout command completed before a stop instruction; its report remained unopened until the Qwen profile was frozen. It did not inform Qwen model, prompt, threshold, or acceptance selection."}
    status = exit_status(correctness, quality, args.enforce_quality)
    report["exit_status"] = status
    write_new(args.report, report)
    print(json.dumps({"profile_id": profile["profile_id"], "correctness": {k: v for k, v in correctness.items() if k != "gates"},
                      "quality": {k: v for k, v in quality.items() if k != "gates"},
                      "promotion_status": report["promotion_status"], "enforce_quality": args.enforce_quality,
                      "exit_status": status}, indent=2))
    raise SystemExit(status)


if __name__ == "__main__":
    main()
