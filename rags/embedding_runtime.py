"""Lazy optional embedding runtimes and complete-input window contracts.

Counting uses the selected runtime's tokenizer. An operator-supplied UTF-8
budget is explicitly a byte budget, never advertised as an exact token count.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import importlib.util
import os
from threading import Lock, RLock
from typing import Callable

from rags.registry import CapabilityError
from rags.agents.deadline import remaining_timeout


def disable_encoder_telemetry() -> None:
    """Model computation does not authorize optional diagnostics disclosure."""
    os.environ["ORT_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if importlib.util.find_spec("onnxruntime") is not None:
        import onnxruntime
        onnxruntime.disable_telemetry_events()


@dataclass(frozen=True)
class EmbeddingWindow:
    count: Callable
    limit: int
    policy: str

    def validate(self, text: str, *, query: bool = False) -> None:
        units = self.count(text, query=query)
        if type(units) is not int or units < 0:
            raise CapabilityError("invalid_embedding", "Embedding counter returned an invalid value", 502)
        if units > self.limit:
            raise CapabilityError("context_budget_exceeded", "Complete input exceeds the embedding window", 422)


def fastembed_window(encoder) -> EmbeddingWindow:
    """Count the untruncated tokenizer for reviewed FastEmbed text runtimes."""
    from fastembed.text.onnx_embedding import OnnxTextEmbedding
    from fastembed.text.onnx_text_model import OnnxTextModel
    from fastembed.text.pooled_embedding import PooledEmbedding
    from fastembed.text.pooled_normalized_embedding import PooledNormalizedEmbedding
    from fastembed.text.last_token_normalized_embedding import LastTokenNormalizedEmbedding
    from fastembed.text.builtin_sentence_embedding import BuiltinSentenceEmbedding
    from fastembed.text.multitask_embedding import JinaEmbeddingV3
    from tokenizers import Tokenizer

    model = encoder.model
    permitted = {OnnxTextEmbedding, PooledEmbedding, PooledNormalizedEmbedding,
                 LastTokenNormalizedEmbedding, BuiltinSentenceEmbedding, JinaEmbeddingV3}
    if type(model) not in permitted or getattr(type(model), "tokenize", None) is not OnnxTextModel.tokenize:
        raise CapabilityError("requires_provider", "This FastEmbed runtime needs a supported complete-input counter")
    model._ensure_tokenizer()
    tokenizer = model.tokenizer
    limit = (tokenizer.truncation or {}).get("max_length")
    if type(limit) is not int or not 1 <= limit <= 100_000:
        raise CapabilityError("requires_provider", "Embedding model token window is unavailable")
    full = Tokenizer.from_str(tokenizer.to_str())
    full.no_truncation()
    full.no_padding()
    return EmbeddingWindow(lambda text, query=False: len(full.encode(text).ids), limit,
                           "native_tokenizer_reject_oversize")


_SENTENCE_ENCODER_LOCK = Lock()
_SENTENCE_EXECUTION_LOCK = RLock()


@contextmanager
def sentence_execution():
    """Protect mutable tokenizer state shared by cached Sentence Transformers."""
    if not _SENTENCE_EXECUTION_LOCK.acquire(timeout=remaining_timeout(90)):
        raise CapabilityError("model_timeout", "Embedding runtime is busy", 504)
    try:
        remaining_timeout(90)
        yield
        remaining_timeout(90)
    finally:
        _SENTENCE_EXECUTION_LOCK.release()


@lru_cache(maxsize=2)
def _cached_sentence_encoder(model: str, revision: str | None, cache_dir: str | None):
    disable_encoder_telemetry()
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise CapabilityError("requires_provider", "Install the sentence-transformers extra for this embedding provider") from exc
    try:
        return SentenceTransformer(model, revision=revision, cache_folder=cache_dir,
                                   trust_remote_code=False, device="cpu",
                                   model_kwargs={"use_safetensors": True})
    except Exception as exc:
        raise CapabilityError("requires_provider", "Configured Sentence Transformers checkpoint is unavailable") from exc


def sentence_encoder(model: str, revision: str | None = None, cache_dir: str | None = None):
    """Serialize cold cache construction; retain at most two model instances."""
    with _SENTENCE_ENCODER_LOCK:
        return _cached_sentence_encoder(model, revision, cache_dir)


sentence_encoder.cache_clear = _cached_sentence_encoder.cache_clear
sentence_encoder.cache_info = _cached_sentence_encoder.cache_info


def sentence_prompt(encoder, *, query: bool, configured: str) -> str:
    if configured:
        return configured
    prompts = getattr(encoder, "prompts", {})
    names = ("query",) if query else ("document", "passage", "corpus")
    for name in names:
        if name in prompts:
            value = prompts[name]
            if not isinstance(value, str):
                raise CapabilityError("requires_provider", "Embedding prompt is invalid")
            return value
    default = getattr(encoder, "default_prompt_name", None)
    value = prompts.get(default, "")
    if not isinstance(value, str):
        raise CapabilityError("requires_provider", "Embedding prompt is invalid")
    return value


def sentence_window(encoder, query_prefix: str, document_prefix: str) -> EmbeddingWindow:
    """Use the standard Transformer tokenizer with the exact selected prompt."""
    from sentence_transformers.sentence_transformer.modules import Transformer
    first = encoder._first_module()
    if (type(first) is not Transformer or not callable(getattr(first, "preprocess", None))
            or "text" not in first.modality_config or "message" in first.modality_config
            or first.can_flatten_inputs):
        raise CapabilityError("requires_provider", "This Sentence Transformers architecture needs a supported complete-input counter")
    limit = encoder.max_seq_length
    if type(limit) is not int or not 1 <= limit <= 100_000:
        raise CapabilityError("requires_provider", "Embedding model token window is unavailable")
    for key in ("common", "text"):
        declared = first.processing_kwargs.get(key, {}).get("max_length")
        if declared is not None:
            if type(declared) is not int or not 1 <= declared <= 100_000:
                raise CapabilityError("requires_provider", "Embedding preprocessing window is invalid")
            limit = min(limit, declared)

    def count(text: str, *, query: bool = False) -> int:
        prefix = sentence_prompt(encoder, query=query, configured=query_prefix if query else document_prefix)
        # Run the runtime's actual prompt/normalization/special-token pipeline.
        # In particular, do not strip whitespace: byte-level tokenizers can
        # consume one token per space and newer runtimes preserve that input.
        with sentence_execution():
            backend = getattr(encoder.tokenizer, "backend_tokenizer", None)
            truncation = backend.truncation if backend is not None else None
            padding = backend.padding if backend is not None else None
            try:
                result = first.preprocess([text], prompt=prefix, task="query" if query else "document",
                                          processing_kwargs={"text": {"truncation": False, "padding": False},
                                                             "common": {"return_tensors": None}})
            finally:
                if backend is not None:
                    backend.no_truncation() if truncation is None else backend.enable_truncation(**truncation)
                    backend.no_padding() if padding is None else backend.enable_padding(**padding)
        ids = result.get("input_ids")
        if not isinstance(ids, list) or len(ids) != 1 or not isinstance(ids[0], list):
            raise CapabilityError("requires_provider", "Embedding preprocessing returned an unsupported token layout")
        return len(ids[0])

    return EmbeddingWindow(count, limit, "native_tokenizer_reject_oversize")
