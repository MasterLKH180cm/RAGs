"""Pinned, text-only Sentence Transformers without dynamic checkpoint modules.

Only standard Transformer -> Pooling -> optional Normalize recipes are loaded.
Constructing the supported modules ourselves prevents a checkpoint's modules.json
from importing arbitrary Python classes, including from local model caches.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from threading import Lock

from rags.file_io import read_regular_bytes
from rags.model_profiles import strict_json
from rags.registry import CapabilityError

_LOAD_LOCK = Lock()
_MODULES = ["sentence_transformers.models.Transformer", "sentence_transformers.models.Pooling"]


def _metadata(model: str, revision: str, filename: str, cache_dir: str | None):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(model, filename, revision=revision, cache_dir=cache_dir, token=False)
    # The Hub deliberately uses symlinks to content-addressed cache blobs.
    return strict_json(read_regular_bytes(Path(path).resolve(strict=True), max_bytes=65_536))


def _assert_no_adapter(model: str, revision: str, cache_dir: str | None) -> None:
    """A pinned adapter can redirect to unpinned base weights; never follow it."""
    from huggingface_hub import _CACHED_NO_EXIST, hf_hub_download, try_to_load_from_cache
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
    cached = try_to_load_from_cache(model, "adapter_config.json", revision=revision, cache_dir=cache_dir)
    if cached is _CACHED_NO_EXIST:
        return
    try:
        hf_hub_download(model, "adapter_config.json", revision=revision, cache_dir=cache_dir, token=False)
    except EntryNotFoundError as exc:
        # A cache miss/network failure cannot prove the remote file is absent.
        if isinstance(exc, LocalEntryNotFoundError) or exc.response is None or exc.response.status_code != 404:
            raise
        return
    raise ValueError("PEFT adapter checkpoints require a separately pinned supported runtime")


def _assert_standard_checkpoint(model: str, revision: str, cache_dir: str | None) -> None:
    # Transformers accepts both Hub IDs and relative directories. Do not let a
    # same-named cwd directory replace the remote checkpoint we have validated.
    if Path(model).exists():
        raise ValueError("Hub model identifier is shadowed by a local filesystem path")
    _assert_no_adapter(model, revision, cache_dir)
    configuration = _metadata(model, revision, "config.json", cache_dir)
    if (not isinstance(configuration, dict) or configuration.get("auto_map") or configuration.get("configuration_files")
            or configuration.get("model_type") not in {"bert", "roberta", "xlm-roberta", "distilbert", "mpnet"}):
        raise ValueError("This adapter accepts standard text transformer architectures only")


def _recipe(model: str, revision: str, cache_dir: str | None) -> tuple[dict, dict, bool]:
    modules = _metadata(model, revision, "modules.json", cache_dir)
    if not isinstance(modules, list) or len(modules) not in {2, 3}:
        raise ValueError("Unsupported Sentence Transformers module recipe")
    expected = _MODULES + (["sentence_transformers.models.Normalize"] if len(modules) == 3 else [])
    paths = ["", "1_Pooling", "2_Normalize"]
    for index, (module, kind) in enumerate(zip(modules, expected, strict=True)):
        if (not isinstance(module, dict) or set(module) != {"idx", "name", "path", "type"}
                or type(module["idx"]) is not int or module["idx"] != index
                or module["name"] != str(index) or module["type"] != kind or module["path"] != paths[index]):
            raise ValueError("Unsupported Sentence Transformers module recipe")
    source = _metadata(model, revision, "sentence_bert_config.json", cache_dir)
    if (not isinstance(source, dict) or set(source) - {"max_seq_length", "do_lower_case"}
            or type(source.get("max_seq_length")) is not int or not 1 <= source["max_seq_length"] <= 100_000
            or type(source.get("do_lower_case", False)) is not bool):
        raise ValueError("Unsupported Sentence Transformers text preprocessing")
    pooling = _metadata(model, revision, "1_Pooling/config.json", cache_dir)
    allowed = {"word_embedding_dimension", "pooling_mode_cls_token", "pooling_mode_mean_tokens",
               "pooling_mode_max_tokens", "pooling_mode_mean_sqrt_len_tokens",
               "pooling_mode_weightedmean_tokens", "pooling_mode_lasttoken", "include_prompt"}
    if (not isinstance(pooling, dict) or set(pooling) - allowed
            or type(pooling.get("word_embedding_dimension")) is not int
            or not 1 <= pooling["word_embedding_dimension"] <= 4096
            or any(type(value) is not bool for key, value in pooling.items() if key != "word_embedding_dimension")
            or not pooling.get("include_prompt", True)):
        raise ValueError("Unsupported Sentence Transformers pooling configuration")
    return source, pooling, len(modules) == 3


class SentenceEncoder:
    def __init__(self, model, limit: int):
        self.model, self.limit = model, limit
        self._lock = Lock()

    def count(self, text: str) -> int:
        # The exact same preprocessing entry point is used for encoding below.
        # Disabling truncation does not mutate the shared native tokenizer.
        with self._lock:
            features = self.model[0].preprocess([text], processing_kwargs={
                "text": {"truncation": False, "padding": False}, "common": {"return_tensors": None},
            })
            ids = features.get("input_ids")
            if not isinstance(ids, list) or len(ids) != 1 or not isinstance(ids[0], list):
                raise CapabilityError("requires_provider", "Encoder does not expose an exact text token counter")
            return len(ids[0])

    def embed(self, text: str, *, normalize: bool, dimensions: int | None) -> list[float]:
        with self._lock:
            output = self.model.encode([text], batch_size=1, show_progress_bar=False, prompt="",
                                       normalize_embeddings=normalize, truncate_dim=dimensions,
                                       convert_to_numpy=True,
                                       processing_kwargs={"text": {"truncation": False}})
            if not hasattr(output, "shape") or len(output.shape) != 2 or output.shape[0] != 1:
                raise CapabilityError("invalid_embedding", "Encoder returned an unexpected embedding count", 502)
            return output[0].tolist()


@lru_cache(maxsize=2)
def _cached_encoder(model: str, revision: str, cache_dir: str | None):
    try:
        from sentence_transformers import SentenceTransformer, models
    except ImportError as exc:
        raise CapabilityError("requires_provider", "Install the sentence-transformers extra for this embedding runtime") from exc
    try:
        _assert_standard_checkpoint(model, revision, cache_dir)
        source, pooling, normalized = _recipe(model, revision, cache_dir)
        common = {"revision": revision, "trust_remote_code": False, "token": False}
        if cache_dir:
            common["cache_dir"] = cache_dir
        transformer = models.Transformer(
            model, max_seq_length=source["max_seq_length"], do_lower_case=source.get("do_lower_case", False),
            model_kwargs={**common, "use_safetensors": True}, processor_kwargs=dict(common),
            config_kwargs=dict(common), processing_kwargs={"text": {"truncation": False}},
        )
        if transformer.tokenizer is None or set(transformer.modality_config) != {"text"}:
            raise ValueError("Only ordinary text embedding encoders are supported")
        modes = {"pooling_mode_cls_token": "cls", "pooling_mode_max_tokens": "max",
                 "pooling_mode_mean_tokens": "mean", "pooling_mode_mean_sqrt_len_tokens": "mean_sqrt_len_tokens",
                 "pooling_mode_weightedmean_tokens": "weightedmean", "pooling_mode_lasttoken": "lasttoken"}
        selected = [mode for flag, mode in modes.items() if pooling.get(flag, False)]
        if not selected:
            raise ValueError("Pooling must select at least one supported mode")
        modules = [transformer, models.Pooling(embedding_dimension=pooling["word_embedding_dimension"],
                                               pooling_mode=selected, include_prompt=True)]
        if normalized:
            modules.append(models.Normalize())
        encoder = SentenceTransformer(modules=modules, device="cpu", trust_remote_code=False,
                                      prompts={"query": "", "document": ""})
        encoder.eval()
        return SentenceEncoder(encoder, source["max_seq_length"])
    except CapabilityError:
        raise
    except Exception as exc:
        raise CapabilityError("requires_provider", "Pinned Sentence Transformers checkpoint or supported recipe is unavailable") from exc


def sentence_encoder(model: str, revision: str, cache_dir: str | None = None) -> SentenceEncoder:
    with _LOAD_LOCK:
        return _cached_encoder(model, revision, cache_dir)


sentence_encoder.cache_clear = _cached_encoder.cache_clear
sentence_encoder.cache_info = _cached_encoder.cache_info
