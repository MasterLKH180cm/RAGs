"""Trained Self-RAG short-form passage selection and grounded answer retention.

The CPU port follows the reference short-form reflection reward, with stable
conditional softmax over raw logits. It uses always-retrieve, greedy decoding,
weights (relevance=1, support=1, utility=0.5), and no sequence-score term. It does
not reproduce long-form beam search or substitute a generic chat model.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path

from rags.features import FeatureFlags, strategy_features
from rags.registry import CapabilityError
from rags.research.common import private_directory, validate_request
from rags.research.selfrag_runtime import (
    CHECKPOINT_REVISION,
    MAX_INPUT_TOKENS,
    MAX_OUTPUT_TOKENS,
    MODEL_SHA256,
    QUANTIZATION,
    REFLECTION_IDS,
    RUNTIME_REVISION,
    RUNTIME_VERSION,
    TOKEN_IDS,
    Generation,
    SelfRagRuntime,
)

SOURCE_COMMIT = "1fcdc420e48f50a7d7ab1ece5494221b93252e99"
MAX_PASSAGES = 8
IMPLEMENTATION = {"library": "self-rag", "version": "git-1fcdc42",
                  "source_commit": SOURCE_COMMIT, "adapter_version": "v10"}
RELEVANCE = (TOKEN_IDS["[Relevant]"], TOKEN_IDS["[Irrelevant]"])
SUPPORT = (TOKEN_IDS["[Fully supported]"], TOKEN_IDS["[Partially supported]"],
           TOKEN_IDS["[No support / Contradictory]"])
UTILITY = tuple(TOKEN_IDS[f"[Utility:{index}]"] for index in range(1, 6))
NO_RETRIEVAL = TOKEN_IDS["[No Retrieval]"]
ALLOWED_OUTPUT_CONTROLS = frozenset((*RELEVANCE, *SUPPORT, *UTILITY, NO_RETRIEVAL, 2))


@contextmanager
def request_directory(workdir: Path):
    directory = private_directory(workdir, "selfrag")
    try:
        yield
    finally:
        # This adapter never writes source-derived files; it only holds one
        # native context. The fixed worker's outer cleanup remains authoritative.
        directory.rmdir()


def conditional_probabilities(logits: dict[int, float], identifiers: tuple[int, ...]) -> tuple[float, ...]:
    """Normalize a reflection group without exponentiating absolute logit scales."""
    try:
        values = [logits[identifier] for identifier in identifiers]
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
            raise ValueError("invalid logits")
        maximum = max(values)
        weights = [math.exp(value - maximum) for value in values]
        total = sum(weights)
        return tuple(weight / total for weight in weights)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise CapabilityError("invalid_model_output", "Self-RAG reflection logits are incomplete or invalid", 502) from exc


def source_prompt(question: str, text: str) -> str:
    # The trained model's control tokens are trusted framing. Reject literal
    # controls in untrusted input rather than allowing a source to close its own
    # paragraph or forge a reflection decision. Original source text is unmodified.
    for value in (question, text):
        try:
            value.encode("utf-8")
        except UnicodeError as exc:
            raise CapabilityError("invalid_research_input", "Self-RAG requires valid UTF-8 input", 422) from exc
        if any(token in value for token in (*TOKEN_IDS, "<s>", "</s>", "<unk>")):
            raise CapabilityError("invalid_research_input", "Self-RAG source contains reserved model control tokens", 422)
    # The upstream short-form runner renders an optional title and a newline.
    # This API supplies exact passages, so the title is deliberately empty.
    return f"### Instruction:\n{question}\n\n### Response:\n[Retrieval]<paragraph>\n{text}</paragraph>"


def score_generation(generation: Generation) -> dict:
    if (not isinstance(generation, Generation) or generation.ended is not True
            or not 2 <= len(generation.token_ids) <= MAX_OUTPUT_TOKENS
            or generation.token_ids[-1] != 2
            or len(generation.pieces) != len(generation.token_ids)
            or len(generation.reflection_logits) != len(generation.token_ids)
            or generation.token_ids[0] not in RELEVANCE
            or type(generation.input_tokens) is not int or not 1 <= generation.input_tokens <= MAX_INPUT_TOKENS):
        raise CapabilityError("invalid_model_output", "Self-RAG returned an incomplete short-form generation", 502)
    for index, (identifier, piece, logits) in enumerate(zip(
            generation.token_ids, generation.pieces, generation.reflection_logits, strict=True)):
        if (type(identifier) is not int or not 0 <= identifier < 32016 or not isinstance(piece, bytes)
                or not isinstance(logits, dict) or set(logits) != set(REFLECTION_IDS)
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in logits.values())
                or (identifier in RELEVANCE and index != 0)
                or (identifier >= 32000 and identifier not in ALLOWED_OUTPUT_CONTROLS)
                or identifier in {0, 1} or (identifier == 2 and index != len(generation.token_ids) - 1)):
            raise CapabilityError("invalid_model_output", "Self-RAG returned invalid reflection output", 502)
    support_indices = [index for index, token in enumerate(generation.token_ids) if token in SUPPORT]
    utility_indices = [index for index, token in enumerate(generation.token_ids) if token in UTILITY]
    if (len(support_indices) > 1 or len(utility_indices) > 1
            or generation.token_ids.count(NO_RETRIEVAL) > 1
            or (support_indices and utility_indices and support_indices[0] > utility_indices[0])):
        raise CapabilityError("invalid_model_output", "Self-RAG returned repeated or reordered reflection decisions", 502)
    relevance = conditional_probabilities(generation.reflection_logits[0], RELEVANCE)[0]
    support = utility = 0.0
    if support_indices:
        probabilities = conditional_probabilities(generation.reflection_logits[support_indices[0]], SUPPORT)
        support = probabilities[0] + 0.5 * probabilities[1]
    if utility_indices:
        probabilities = conditional_probabilities(generation.reflection_logits[utility_indices[0]], UTILITY)
        utility = sum(weight * probability for weight, probability in zip(
            (-1.0, -0.5, 0.0, 0.5, 1.0), probabilities, strict=True))
    try:
        text = b"".join(piece for token, piece in zip(generation.token_ids, generation.pieces, strict=True)
                        if token not in ALLOWED_OUTPUT_CONTROLS).decode("utf-8").strip()
    except UnicodeError as exc:
        raise CapabilityError("invalid_model_output", "Self-RAG returned incomplete UTF-8 text", 502) from exc
    if any(token in text for token in TOKEN_IDS) or len(text) > 60_000:
        raise CapabilityError("invalid_model_output", "Self-RAG returned invalid answer text", 502)
    return {"score": relevance + support + 0.5 * utility, "text": text,
            # This port deliberately abstains on explicit no-retrieval or
            # contradictory/unsupported decisions even if the weighted reward
            # is high. Preserve the reference reward, but never trim/repair an
            # explicitly ungrounded answer into a grounded-looking fragment.
            "relevant": (generation.token_ids[0] == RELEVANCE[0]
                         and NO_RETRIEVAL not in generation.token_ids
                         and SUPPORT[2] not in generation.token_ids),
            "support_tokens": len(support_indices), "utility_tokens": len(utility_indices)}


def search(question: str, passages: list[dict], top_k: int, workdir: Path, provider=None) -> dict:
    """Return one selected trained answer and its exact original passage ID.

    The ordinary generator is deliberately unused. Actual model artifacts and
    runtime are selected only by operator environment, never request options.
    """
    FeatureFlags.from_env().require(*strategy_features("self_rag"))
    sources = validate_request(question, passages, top_k)
    if len(sources) > MAX_PASSAGES:
        raise CapabilityError("research_limit_exceeded", "Self-RAG short-form execution allows at most eight passages", 422)
    prompts = [(identifier, source_prompt(question, text)) for identifier, text in sources.items()]
    candidates = []
    input_tokens = output_tokens = support_tokens = utility_tokens = 0
    with request_directory(workdir), SelfRagRuntime() as runtime:
        # Validate the complete snapshot before any inference. No oversized later
        # passage may cause a partly executed corpus or an implicit prefix slice.
        lengths = [len(runtime.tokenize(prompt)) for _, prompt in prompts]
        if any(length > MAX_INPUT_TOKENS for length in lengths):
            raise CapabilityError("context_budget_exceeded", "Full Self-RAG source prompt exceeds 1536 model tokens", 422)
        runtime.prepare_context(max(lengths))
        for identifier, prompt in prompts:
            generation = runtime.generate(prompt)
            scored = score_generation(generation)
            input_tokens += generation.input_tokens
            output_tokens += len(generation.token_ids)
            support_tokens += scored["support_tokens"]
            utility_tokens += scored["utility_tokens"]
            # An explicitly irrelevant, unsupported, or empty trained answer
            # supplies no usable source. Never invoke a generator fallback.
            if scored["relevant"] and scored["text"]:
                candidates.append((identifier, scored))
    candidates.sort(key=lambda entry: (-entry[1]["score"], entry[0]))
    selected = candidates[0] if candidates else None
    return {
        "hits": [{"chunk_id": selected[0], "score": selected[1]["score"]}] if selected else [],
        "implementation": dict(IMPLEMENTATION),
        "diagnostics": {
            "source_count": len(sources), "engine_executed": True,
            "reflection_calls": len(prompts), "generated_tokens": output_tokens,
            "reflection_relevance_tokens": len(prompts), "reflection_support_tokens": support_tokens,
            "reflection_utility_tokens": utility_tokens,
            "reflection_abstentions": len(prompts) - len(candidates),
            "checkpoint_sha256": MODEL_SHA256, "runtime_revision": RUNTIME_REVISION,
            "runtime": f"llama.cpp-{RUNTIME_VERSION}-cpu-owned-buffers",
            "checkpoint_revision": CHECKPOINT_REVISION, "checkpoint_quantization": QUANTIZATION,
            "inference_mode": "always_retrieve_short_form",
            "score_kind": "conditional_reflection_reward",
        },
        "grounded_answer": {"text": selected[1]["text"], "cited_ids": [selected[0]]} if selected else None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }
