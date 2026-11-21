"""Actual two-case inference and an explicitly historical frozen-evidence audit.

This does not repeat held-out quality scoring or approve profile promotion.
Use rags.verify_admission --enforce-quality for the complete benchmark.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from rags.admission import EvidenceAdmission, canonical_digest, load_profile
from rags.admission_eval import measurement_report, read_dataset, write_new
from rags.verify_admission import correctness_gates, quality_gates, summarize_gates


ROOT = Path(__file__).resolve().parents[1]
PROFILE = "profiles/admission-qwen-v03.json"
RECORDED = "reports/admission-verification-qwen-v03.json"
CALIBRATION = "examples/admission_calibration_v03.json"
HOLDOUT = "examples/admission_holdout_v03.json"
REGRESSION = "examples/eval_v02.json"
FROZEN_SHA256 = {
    PROFILE: "b8ed84a9d97f23ab9e49a0b40e8fbbec6b91224e642ef97efb763f71605abf32",
    RECORDED: "a313583f6c6f8cc0ab70c396055be998988552796beb30943fe4d0dbb35a0707",
    CALIBRATION: "1d28f1134a6cd599837e380ba067ff756bb6991fa3cb747d70c1754e7f1830c1",
    HOLDOUT: "c0cb3da867ee7c5cb2cfa79db68e9388ed83a02b1a7f01130953b6e9337167e4",
    REGRESSION: "b088a871deea5ca437a59f0931934cc6398d62cef1636893ae3270d3551a7709",
}
CONTRACT_CASES = (
    ("cal-en-apiary-positive-literal", True),
    ("cal-en-apiary-negative-missing-field", False),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit_recorded(root: Path = ROOT) -> tuple[dict, dict, dict]:
    """Recompute recorded gates without making any new model-quality claim."""
    for name, expected in FROZEN_SHA256.items():
        require(hashlib.sha256((root / name).read_bytes()).hexdigest() == expected,
                "Frozen evidence changed: " + name)
    profile = load_profile(str(root / PROFILE))
    calibration, holdout = (read_dataset(str(root / name)) for name in (CALIBRATION, HOLDOUT))
    require(calibration["split"] == "calibration" and holdout["split"] == "holdout", "Unexpected data split")
    require(profile["calibration"]["cases_digest"] == canonical_digest(calibration["cases"]),
            "Calibration/profile identity mismatch")
    report = json.loads((root / RECORDED).read_text())
    require(report["profile_id"] == profile["profile_id"]
            and report["profile_file_sha256"] == FROZEN_SHA256[PROFILE]
            and report["frozen_threshold"] == profile["threshold"], "Recorded profile identity mismatch")
    rows = report["holdout"]["measurements"]
    require([row["id"] for row in rows] == [case["id"] for case in holdout["cases"]], "Held-out coverage changed")
    for row, case in zip(rows, holdout["cases"], strict=True):
        require(all(row[key] == case[key] for key in ("id", "answerable", "language", "category")),
                "Recorded held-out labels do not match the frozen dataset")
    recomputed_holdout = measurement_report(holdout, profile, rows)
    require(recomputed_holdout == report["holdout"], "Recorded held-out metrics or identity changed")
    cases = json.loads((root / REGRESSION).read_text())["cases"]
    regression = report["regression"]
    require(set(regression) == {"generate_false", "generate_true"}, "Missing generation mode")
    for reports in regression.values():
        require(set(reports) == {"hybrid", "structured", "temporal"}, "Missing regression mode")
        for mode, item in reports.items():
            expected = [case["id"] for case in cases if not case.get("modes") or mode in case["modes"]]
            require([row["id"] for row in item["measurements"]] == expected
                    and item["cases"] == item["evaluated_cases"] == len(expected), "Recorded regression coverage changed")
    correctness = summarize_gates(correctness_gates(regression))
    quality = summarize_gates(quality_gates(recomputed_holdout, regression))
    require(correctness == report["correctness"] and quality == report["quality"], "Recorded gate accounting changed")
    require(correctness["passed"] and not quality["passed"] and report["promotion_status"] == "failed"
            and report["enforce_quality"] is True and report["exit_status"] == 2, "Recorded failure was concealed")
    return profile, calibration, {
        "report_path": RECORDED, "report_sha256": FROZEN_SHA256[RECORDED], "measured_at": report["measured_at"],
        "profile_id": profile["profile_id"], "dataset_file_sha256": {name: FROZEN_SHA256[name]
                                                                       for name in (CALIBRATION, HOLDOUT, REGRESSION)},
        "correctness": correctness, "quality": quality, "promotion_status": report["promotion_status"],
        "scope": "Historical complete local measurement; gates recomputed from recorded values, no fresh quality inference",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir")
    parser.add_argument("--report")
    parser.add_argument("--audit-only", action="store_true", help="Verify historical evidence without loading a model")
    args = parser.parse_args()
    profile, calibration, recorded = audit_recorded()
    if args.audit_only:
        print(json.dumps({"audit_status": "passed", "new_model_inference": False, "recorded_quality": recorded}, indent=2))
        return
    if not args.cache_dir or not args.report:
        parser.error("Actual runtime verification requires --cache-dir and a new --report destination")
    if Path(args.report).exists():
        parser.error("Refusing to overwrite an existing measurement")
    from rags.answerability import AnswerabilityScorer

    started = time.perf_counter()
    scorer = AnswerabilityScorer(str(Path(args.cache_dir).resolve()))
    scorer.verify(profile)
    admission = EvidenceAdmission("calibrated_required", str(ROOT / PROFILE), scorer=scorer)
    by_id = {case["id"]: case for case in calibration["cases"]}
    results = []
    for case_id, expected in CONTRACT_CASES:
        case = by_id[case_id]
        require(case["answerable"] is expected and len(case["passages"]) == 1, "Contract fixture changed")
        # Only the question and original text enter actual inference; no labels, IDs or reference answers.
        decision = admission.assess(case["question"], [{"content": p["text"]} for p in case["passages"]])
        metadata = decision["admission"]
        require(decision["outcome"] == ("success" if expected else "insufficient_evidence")
                and decision["supported"] is expected
                and decision["refusal_reason"] == (None if expected else "below_admission_threshold"),
                "Actual runtime failed its fixed acceptance/refusal contract")
        require(metadata["profile_id"] == profile["profile_id"] and metadata["threshold"] == profile["threshold"]
                and metadata["basis"] == "calibrated_answerability"
                and metadata["decision"] == ("accept" if expected else "abstain")
                and metadata["candidate_count"] == 1 and metadata["accepted_count"] == int(expected),
                "Actual runtime returned inconsistent profile or admission metadata")
        results.append({"case_id": case_id, "passed": True, **decision})
    result = {"schema_version": "admission-runtime-verification-v1", "status": "passed",
              "measured_at": datetime.now(timezone.utc).isoformat(), "profile_id": profile["profile_id"],
              "cases": results, "elapsed_seconds": round(time.perf_counter() - started, 3),
              "scope": "Two preselected calibration contract cases with actual pinned inference; no new held-out quality measurement or promotion",
              "recorded_quality": recorded}
    write_new(args.report, result)
    print(json.dumps({"status": result["status"], "profile_id": result["profile_id"], "cases": results,
                      "scope": result["scope"], "recorded_quality_passed": recorded["quality"]["passed"],
                      "recorded_promotion_status": recorded["promotion_status"]}, indent=2))


if __name__ == "__main__":
    main()
