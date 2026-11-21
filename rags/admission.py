"""Profile-bound relevance admission; ranking and semantic entailment stay separate."""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
import sys
from threading import Lock

from rags.registry import CapabilityError


POLICY_VERSION = "evidence-admission-v1"
PROFILE_SCHEMA = "evidence-admission-profile-v1"
MODEL = "BAAI/bge-reranker-v2-m3-int8"
SCORE_CONVENTION = "onnx_output_0_first_column_logit_no_activation"
LIMITS = {"max_candidates": 20, "max_question_chars": 2000, "max_passage_chars": 200_000,
          "max_total_passage_chars": 200_000, "pair_tokens": 512, "batch_size": 4,
          "cpu_threads": 2, "window": "right_source_prefix_only_second_v1"}
CALIBRATED_BASES = {"calibrated_reranker", "calibrated_answerability"}
BASES = {"no_candidates", "ranked_unassessed", "calibrated_reranker", "calibrated_answerability", "model_review",
         "structured_exact", "context_budget"}
REFUSALS = {"no_candidates", "below_admission_threshold", "model_rejected", "context_budget_exhausted"}
METADATA_KEYS = {"policy_version", "decision", "basis", "profile_id", "score", "threshold",
                 "candidate_count", "accepted_count"}
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_SCORER_BUILD_LOCK = Lock()


def canonical_digest(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def finite(value) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _runtime_identity() -> dict:
    return {name: version(name) for name in ("fastembed", "onnxruntime", "tokenizers")} | {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}", "execution_provider": "CPUExecutionProvider"}


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_scorer_identity(scorer: dict) -> None:
    common = {"model", "artifacts", "runtime", "score_convention", "limits", "aggregation"}
    if not isinstance(scorer, dict) or not common <= scorer.keys() or scorer["aggregation"] != "max_original_passage":
        raise ValueError("Invalid scorer identity")
    if scorer["model"] == MODEL:
        if set(scorer) != common or scorer["score_convention"] != SCORE_CONVENTION or scorer["limits"] != LIMITS:
            raise ValueError("Unsupported reranker settings")
        required = {"onnx/model_int8.onnx", "tokenizer.json", "tokenizer_config.json"}
        permitted = required | {"special_tokens_map.json", "config.json"}
    else:
        from rags.answerability import (ARTIFACT_NAMES, CHAT_SETTINGS, PROMPT_DIGEST, QWEN_LIMITS,
                                       QWEN_MODEL, QWEN_REVISION, QWEN_SCORE, REQUIRED_ARTIFACTS)
        extra = {"revision", "prompt_sha256", "label_ids", "chat_template_sha256", "chat_settings"}
        if (scorer["model"] != QWEN_MODEL or set(scorer) != common | extra or scorer["revision"] != QWEN_REVISION
                or scorer["score_convention"] != QWEN_SCORE or scorer["limits"] != QWEN_LIMITS
                or scorer["prompt_sha256"] != PROMPT_DIGEST or scorer["chat_settings"] != CHAT_SETTINGS
                or not isinstance(scorer["chat_template_sha256"], str) or not _DIGEST.fullmatch(scorer["chat_template_sha256"])):
            raise ValueError("Unsupported answerability settings")
        labels = scorer["label_ids"]
        if (not isinstance(labels, dict) or set(labels) != {"YES", "NO"}
                or any(type(value) is not int or not 0 <= value < 500_000 for value in labels.values())
                or labels["YES"] == labels["NO"]):
            raise ValueError("Invalid assessment label identities")
        required, permitted = REQUIRED_ARTIFACTS, ARTIFACT_NAMES
    artifacts = scorer["artifacts"]
    if (not isinstance(artifacts, dict) or not required <= artifacts.keys() or not artifacts.keys() <= permitted
            or any(not isinstance(value, str) or not _DIGEST.fullmatch(value) for value in artifacts.values())):
        raise ValueError("Invalid artifact identities")


def load_profile(path: str) -> dict:
    """Read an operator-owned immutable profile without loading or fetching a model."""
    try:
        source = Path(path)
        if source.stat().st_size > 100_000:
            raise ValueError("Profile size")
        profile = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(profile, dict) or set(profile) != {
            "schema_version", "profile_id", "scorer", "threshold", "languages", "calibration"
        }:
            raise ValueError("Profile shape")
        if profile["schema_version"] != PROFILE_SCHEMA or not finite(profile["threshold"]):
            raise ValueError("Profile schema or threshold")
        expected = canonical_digest({key: val for key, val in profile.items() if key != "profile_id"})
        if profile["profile_id"] != expected:
            raise ValueError("Profile identity")
        _validate_scorer_identity(profile["scorer"])
        if profile["languages"] != ["en", "zh"]:
            raise ValueError("Unsupported calibration languages")
        calibration = profile["calibration"]
        if (not isinstance(calibration, dict) or not _DIGEST.fullmatch(str(calibration.get("cases_digest", "")))
                or calibration.get("objective") != "max_specificity_then_retention_then_smallest_threshold_v1"
                or not isinstance(calibration.get("measurements"), dict)):
            raise ValueError("Missing calibration evidence")
    except (OSError, UnicodeError, ValueError, TypeError, KeyError) as exc:
        raise CapabilityError("invalid_admission_profile", "Configured admission profile is invalid") from exc
    return profile


@dataclass(frozen=True)
class PassageScores:
    scores: tuple[float, ...]
    truncated_indices: tuple[int, ...]
    original_pair_tokens: tuple[int, ...]


class CrossEncoderScorer:
    """Real ONNX inference; cache model state, never tenant text or scores."""

    def __init__(self, cache_dir: str | None = None, *, allow_download: bool = False):
        try:
            os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            from tokenizers import Tokenizer
            self.encoder = TextCrossEncoder(MODEL, cache_dir=cache_dir, threads=LIMITS["cpu_threads"],
                                            providers=["CPUExecutionProvider"], lazy_load=True,
                                            local_files_only=not allow_download)
            backend = self.encoder.model
            directory = Path(backend._model_dir)
            paths = ["onnx/model_int8.onnx", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json"]
            artifacts = {name: _file_digest(directory / name) for name in paths if (directory / name).is_file()}
            if not {"onnx/model_int8.onnx", "tokenizer.json", "tokenizer_config.json"} <= artifacts.keys():
                raise ValueError("Missing checkpoint artifacts")
            self.identity = {"model": MODEL, "artifacts": artifacts, "runtime": _runtime_identity(),
                             "score_convention": SCORE_CONVENTION, "limits": dict(LIMITS),
                             "aggregation": "max_original_passage"}
            backend._ensure_tokenizer()
            self.counter = Tokenizer.from_str(backend.tokenizer.to_str())
            self.counter.no_truncation()
            self.counter.no_padding()
            backend.tokenizer.enable_truncation(max_length=LIMITS["pair_tokens"], strategy="only_second", direction="right")
            self._lock = Lock()
        except Exception as exc:
            raise CapabilityError("requires_admission_model", "The configured admission checkpoint or encoder runtime is unavailable") from exc

    def verify(self, profile: dict) -> None:
        if profile["scorer"] != self.identity:
            raise CapabilityError("admission_profile_mismatch", "Admission profile does not match actual artifacts, runtime, or scoring settings")

    def score(self, question: str, passages: list[str]) -> PassageScores:
        if (not isinstance(question, str) or not question.strip() or len(question) > LIMITS["max_question_chars"]
                or not isinstance(passages, list) or not 1 <= len(passages) <= LIMITS["max_candidates"]
                or any(not isinstance(text, str) or not text.strip() or len(text) > LIMITS["max_passage_chars"] for text in passages)
                or sum(map(len, passages)) > LIMITS["max_total_passage_chars"]):
            raise CapabilityError("admission_input_limit", "Admission input exceeds its configured bounds", 422)
        # Keep the entire question. Only the passage prefix may be truncated.
        if len(self.counter.encode(question, "").ids) >= LIMITS["pair_tokens"]:
            raise CapabilityError("admission_question_limit", "Question leaves no source tokens inside the admission window", 422)
        with self._lock:
            token_counts = tuple(len(item.ids) for item in self.counter.encode_batch([(question, text) for text in passages]))
            try:
                # FastEmbed lazy load recreates its tokenizer: configure after loading.
                backend = self.encoder.model
                if getattr(backend, "model", None) is None:
                    backend.load_onnx_model()
                    backend.tokenizer.enable_truncation(max_length=LIMITS["pair_tokens"], strategy="only_second", direction="right")
                scores = tuple(self.encoder.rerank(question, passages, batch_size=LIMITS["batch_size"]))
            except Exception as exc:
                raise CapabilityError("admission_inference_failed", "Admission scorer failed; evidence was not accepted", 502) from exc
        if len(scores) != len(passages) or any(not finite(score) for score in scores):
            raise CapabilityError("invalid_admission_score", "Admission scorer returned invalid numeric scores", 502)
        return PassageScores(scores, tuple(i for i, count in enumerate(token_counts) if count > LIMITS["pair_tokens"]), token_counts)


@lru_cache(maxsize=2)
def _cached_scorer(model: str, revision: str, cache_dir: str | None):
    if model == MODEL:
        scorer = CrossEncoderScorer(cache_dir)
        if canonical_digest(scorer.identity["artifacts"]) != revision:
            raise CapabilityError("admission_profile_mismatch", "Reranker artifact revision does not match the profile")
        return scorer
    from rags.answerability import AnswerabilityScorer, QWEN_MODEL, QWEN_REVISION
    if model == QWEN_MODEL and revision == QWEN_REVISION:
        return AnswerabilityScorer(cache_dir)
    raise CapabilityError("requires_admission_model", "Unsupported admission model")


def scorer_for(model: str, revision: str, cache_dir: str | None):
    # functools.lru_cache alone permits duplicate concurrent cold constructions.
    # A shared factory lock prevents several multi-gigabyte model instances;
    # each resulting scorer separately serializes its inference work.
    normalized_cache = str(Path(cache_dir).expanduser().resolve()) if cache_dir else None
    with _SCORER_BUILD_LOCK:
        return _cached_scorer(model, revision, normalized_cache)


def metadata(decision: str, basis: str, count: int, accepted: int, *, profile_id=None, score=None, threshold=None) -> dict:
    return {"policy_version": POLICY_VERSION, "decision": decision, "basis": basis,
            "profile_id": profile_id, "score": score, "threshold": threshold,
            "candidate_count": count, "accepted_count": accepted}


class EvidenceAdmission:
    def __init__(self, policy: str = "compatibility", profile_path: str | None = None, *, scorer=None):
        if policy not in {"compatibility", "calibrated_required"}:
            raise ValueError("Unsupported admission policy")
        self.policy = policy
        self.profile = load_profile(profile_path) if profile_path else None
        self._scorer_override = scorer

    def assess(self, question: str, chunks: list[dict], *, model_supported=None, structured_exact=False) -> dict:
        count = len(chunks)
        if structured_exact:
            return {"admission": metadata("accept", "structured_exact", count, count), "supported": True,
                    "outcome": "success", "refusal_reason": None}
        if not chunks:
            if model_supported is False:
                return self.refuse("model_rejected", "model_review", count)
            return self.refuse("no_candidates", "no_candidates", count)
        if self.policy == "calibrated_required" and not self.profile:
            raise CapabilityError("requires_admission_profile", "Required admission needs a configured calibrated profile")
        scorer = None
        if self.profile:
            identity = self.profile["scorer"]
            revision = identity.get("revision") or canonical_digest(identity["artifacts"])
            scorer = self._scorer_override or scorer_for(identity["model"], revision, os.getenv("RAGS_MODEL_CACHE") or None)
            scorer.verify(self.profile)
        if model_supported is False:
            return self.refuse("model_rejected", "model_review", count)
        if self.profile:
            scored = scorer.score(question, [c["content"] for c in chunks])
            score, threshold = max(scored.scores), self.profile["threshold"]
            admitted = score >= threshold
            basis = "calibrated_reranker" if self.profile["scorer"]["model"] == MODEL else "calibrated_answerability"
            result = {"admission": metadata("accept" if admitted else "abstain", basis, count,
                                             count if admitted else 0, profile_id=self.profile["profile_id"],
                                             score=score, threshold=threshold),
                      "supported": (True if model_supported is True or basis == "calibrated_answerability" else None) if admitted else False,
                      "outcome": "success" if admitted else "insufficient_evidence",
                      "refusal_reason": None if admitted else "below_admission_threshold"}
            limits = self.profile["scorer"]["limits"]
            result["admission_trace"] = {"action": "admission_scoring", "input_token_limit": limits.get("pair_tokens", limits.get("prompt_tokens")),
                                         "truncated_passages": len(scored.truncated_indices),
                                         "truncated_candidate_indices": list(scored.truncated_indices)}
            return result
        supported = model_supported is True
        return {"admission": metadata("accept" if supported else "unassessed", "model_review" if supported else "ranked_unassessed", count, count),
                "supported": True if supported else None, "outcome": "success", "refusal_reason": None}

    @staticmethod
    def refuse(reason: str, basis: str, count: int) -> dict:
        return {"admission": metadata("abstain", basis, count, 0), "supported": False,
                "outcome": "insufficient_evidence", "refusal_reason": reason}


def validate_remote_decision(body: dict, citation_count: int) -> dict:
    """Validate bounded decision metadata before admitting federated evidence."""
    if body.get("supported") is not None and type(body["supported"]) is not bool:
        raise ValueError("Invalid federated support assessment")
    if "outcome" not in body:
        if "admission" in body or "refusal_reason" in body:
            raise ValueError("Incomplete federated decision")
        if body.get("supported") is False:
            return EvidenceAdmission.refuse("model_rejected", "model_review", citation_count)
        return {"outcome": "success", "supported": None, "refusal_reason": None,
                "admission": metadata("unassessed", "ranked_unassessed", citation_count, citation_count)}
    if not {"outcome", "supported", "refusal_reason", "admission"} <= body.keys():
        raise ValueError("Incomplete federated decision")
    outcome, supported, reason, item = (body.get(k) for k in ("outcome", "supported", "refusal_reason", "admission"))
    if (outcome not in {"success", "insufficient_evidence"} or supported is not None and type(supported) is not bool
            or reason is not None and reason not in REFUSALS or not isinstance(item, dict) or set(item) != METADATA_KEYS):
        raise ValueError("Invalid federated decision")
    if (item["policy_version"] != POLICY_VERSION or item["decision"] not in {"accept", "abstain", "unassessed"}
            or item["basis"] not in BASES or any(type(item[k]) is not int or not 0 <= item[k] <= 20 for k in ("candidate_count", "accepted_count"))
            or item["accepted_count"] > item["candidate_count"]
            or any(item[k] is not None and not finite(item[k]) for k in ("score", "threshold"))):
        raise ValueError("Invalid federated admission metadata")
    if item["basis"] in CALIBRATED_BASES:
        if (not isinstance(item["profile_id"], str) or not _DIGEST.fullmatch(item["profile_id"])
                or not finite(item["score"]) or not finite(item["threshold"])):
            raise ValueError("Invalid federated calibrated profile")
    elif any(item[k] is not None for k in ("profile_id", "score", "threshold")):
        raise ValueError("Unexpected federated scoring metadata")
    if outcome == "insufficient_evidence":
        if supported is not False or reason not in REFUSALS or item["decision"] != "abstain" or item["accepted_count"] or citation_count:
            raise ValueError("Refused federated response contains usable evidence")
        expected_bases = {"no_candidates": {"no_candidates"}, "below_admission_threshold": CALIBRATED_BASES,
                          "model_rejected": {"model_review"}, "context_budget_exhausted": {"context_budget"}}[reason]
        if item["basis"] not in expected_bases or reason == "no_candidates" and item["candidate_count"] != 0:
            raise ValueError("Contradictory federated refusal reason")
        if reason == "below_admission_threshold" and item["score"] >= item["threshold"]:
            raise ValueError("Federation refused above its calibrated threshold")
    elif (supported is False or reason is not None or item["decision"] == "abstain"
          or item["accepted_count"] != citation_count or not citation_count):
        raise ValueError("Contradictory successful federated response")
    if item["decision"] == "unassessed" and (item["basis"] != "ranked_unassessed" or supported is not None):
        raise ValueError("Contradictory unassessed federation")
    if item["decision"] == "accept" and item["basis"] not in CALIBRATED_BASES | {"model_review", "structured_exact"}:
        raise ValueError("Unsupported federation acceptance basis")
    if item["decision"] == "accept" and item["basis"] in CALIBRATED_BASES and item["score"] < item["threshold"]:
        raise ValueError("Federation accepted below threshold")
    if item["decision"] == "accept" and item["basis"] in {"model_review", "structured_exact", "calibrated_answerability"} and supported is not True:
        raise ValueError("Federation lacks the declared supporting assessment")
    # This caller sends a hybrid text query without a typed aggregation. A remote
    # must not invent an exact aggregation as justification for its text evidence.
    if item["basis"] == "structured_exact":
        raise ValueError("Unexpected structured result for a federated hybrid query")
    return {"outcome": outcome, "supported": supported, "refusal_reason": reason, "admission": dict(item)}
