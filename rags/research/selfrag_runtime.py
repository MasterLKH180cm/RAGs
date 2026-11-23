"""Bounded CPU inference for the pinned, trained Self-RAG checkpoint.

This is a deliberately small ctypes binding to one verified llama.cpp ABI. It
does not discover arbitrary native plugins, import model code, download weights,
or expose a caller-controlled model/runtime. Each research worker owns and closes
one model. Reflection scores use raw logits, never a truncated top-logprob list.
"""
from __future__ import annotations

import ctypes as C
import hashlib
import math
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from rags.registry import CapabilityError

CHECKPOINT_REPOSITORY = "selfrag/selfrag_llama2_7b"
CHECKPOINT_REVISION = "190261383b0779ff66d2f95a73c7ad267d94b820"
RUNTIME_REVISION = "e300919cc704ddde3a8a8346a8714f57eeeee71b"
RUNTIME_VERSION = "b12631"
QUANTIZATION = "Q4_K_M"
# Deterministically converted from both verified official shards, with the
# documented CodeLlama EOT-metadata correction before Q4_K_M quantization.
MODEL_SHA256 = "e6bcabadccb962ddda7f1899f1a684cb3f84dd98c07778f034d6b43d0f245e85"
MODEL_BYTES = 4081095936
VOCAB_SIZE = 32016
CONTEXT_TOKENS = 1792
MAX_INPUT_TOKENS = 1536
MAX_OUTPUT_TOKENS = 128
TOKEN_IDS = {
    "[No Retrieval]": 32000, "[Retrieval]": 32001,
    "[Continue to Use Evidence]": 32002, "[Irrelevant]": 32003,
    "[Relevant]": 32004, "<paragraph>": 32005, "</paragraph>": 32006,
    "[Utility:1]": 32007, "[Utility:2]": 32008, "[Utility:3]": 32009,
    "[Utility:4]": 32010, "[Utility:5]": 32011, "[Fully supported]": 32012,
    "[Partially supported]": 32013, "[No support / Contradictory]": 32014,
    "<pad>": 32015,
}
REFLECTION_IDS = tuple(TOKEN_IDS.values())
LIBRARY_HASHES = {
    "libllama.so.1.0.0": "ac2798b6ec4818da5339c579586c9ba54e9904d5412de0eb17e7eae83e4eef7d",
    "libggml.so.1.0.0": "405441c70127eb7fbf000ee9681a8ef9e4a540f151bed57d1475b8115b6b46d4",
    "libggml-base.so.1.0.0": "b03b6e322f476a7e2fca31ea07f95d91e439b2e1e6c8077e1864a47ea58df905",
    "libggml-cpu-alderlake.so": "2f977cbc5b1805788187cbaa25220b833aefd9fcb0e64c67a793a9a2214383fe",
    "libggml-cpu-cannonlake.so": "26b2825103049b5bfffa80be1a65ab6453152f7c32fb93fb9e9865ba60b0bad7",
    "libggml-cpu-cascadelake.so": "e8c485ed680844959eb76ec6b45af2b4ddfabb74aeda9093f373c956b2eabef5",
    "libggml-cpu-cooperlake.so": "f54049992facc29d01bca74c01323503095b9fd7d4eb355600e72d4cbfe49c8c",
    "libggml-cpu-haswell.so": "039e232dc8243dc4d9c7b6ab879bc414bd203f46b4a937f13e62b444520b713d",
    "libggml-cpu-icelake.so": "b444c3f789048ca3dc6dacf25d2f2f560e5810cd3a34d3f5e66b4941544f42ca",
    "libggml-cpu-ivybridge.so": "ec15d63385780ad6e8426003bf7a0f2e59119293ff417a416a5de1b1a2286293",
    "libggml-cpu-piledriver.so": "b2a5a39d5112d0b7ac79bd47bf68e3242446fbc76742c3a8e595af524512b598",
    "libggml-cpu-sandybridge.so": "b35b453b7063c1d0db9340c9dcce4cfa53d41d1a87be3c486eff8705117ab4e5",
    "libggml-cpu-sapphirerapids.so": "79747f60800ad66ed01ce0e839f49833aec55295a9e1f0f1b72d4f4741d61815",
    "libggml-cpu-skylakex.so": "a44e64ac9892ec05268fdb6f8e94bf58367a885d1b65da7e31413b7ec909a209",
    "libggml-cpu-sse42.so": "8103927f863313ef364097b00192bab39f3353088989c37452f4004a54719061",
    "libggml-cpu-x64.so": "3b8b675c08e2963e5dc1b0915175756ac0660b657be824305793da772a308341",
    "libggml-cpu-zen4.so": "de6d921ee1c28086c3d2193efb821d23cf9795152c5b3094acc450316d93054c",
}
_ACTIVE = Lock()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def verified_paths() -> tuple[Path, Path, int]:
    """Verify all executable artifacts before the first dlopen or model read."""
    model_value = os.getenv("RAGS_SELFRAG_MODEL_PATH", "")
    library_value = os.getenv("RAGS_SELFRAG_LLAMA_DIR", "")
    if not model_value or not library_value:
        raise CapabilityError("requires_backend", "Configure the pinned Self-RAG model and CPU runtime")
    if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise CapabilityError("requires_backend", "Pinned Self-RAG CPU runtime requires Linux x86_64")
    model, directory = Path(model_value), Path(library_value)
    try:
        threads = int(os.getenv("RAGS_SELFRAG_THREADS", "4"))
        if not 1 <= threads <= 8:
            raise ValueError("threads")
        if not model.is_absolute() or model.is_symlink() or not directory.is_absolute() or directory.is_symlink():
            raise ValueError("paths")
        if not model.is_file() or model.stat().st_size != MODEL_BYTES or file_digest(model) != MODEL_SHA256:
            raise ValueError("checkpoint")
        for filename, digest in LIBRARY_HASHES.items():
            artifact = directory / filename
            if artifact.is_symlink() or not artifact.is_file() or file_digest(artifact) != digest:
                raise ValueError("runtime")
        # The loader's versioned dependencies resolve through these shipped links.
        for short, target in (("libggml.so.1", "libggml.so.1.0.0"),
                              ("libggml-base.so.1", "libggml-base.so.1.0.0")):
            if (directory / short).resolve() != directory / target:
                raise ValueError("runtime link")
    except (OSError, ValueError) as exc:
        raise CapabilityError("requires_backend", "Self-RAG checkpoint or pinned CPU runtime verification failed") from exc
    return model, directory, threads


# ABI at llama.cpp e300919/include/llama.h. No guessed struct layouts or support
# for arbitrary installed versions: verified_paths pins every loaded library.
class _ModelParams(C.Structure):
    _fields_ = [("devices", C.c_void_p), ("tensor_buft_overrides", C.c_void_p),
                ("n_gpu_layers", C.c_int32), ("split_mode", C.c_int), ("load_mode", C.c_int),
                ("lazy_mode", C.c_int), ("main_gpu", C.c_int32), ("tensor_split", C.c_void_p),
                ("progress_callback", C.c_void_p), ("progress_callback_user_data", C.c_void_p),
                ("kv_overrides", C.c_void_p), ("vocab_only", C.c_bool), ("check_tensors", C.c_bool),
                ("use_extra_bufts", C.c_bool), ("no_host", C.c_bool), ("no_alloc", C.c_bool),
                ("load_mtp", C.c_bool)]


class _ContextParams(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in (
        "n_ctx", "n_batch", "n_ubatch", "n_seq_max", "n_rs_seq", "n_outputs_max", "n_outputs_max_per_seq")]
    _fields_ += [("n_threads", C.c_int32), ("n_threads_batch", C.c_int32)]
    _fields_ += [(name, C.c_int) for name in (
        "ctx_type", "rope_scaling_type", "pooling_type", "attention_type", "flash_attn_type")]
    _fields_ += [(name, C.c_float) for name in (
        "rope_freq_base", "rope_freq_scale", "yarn_ext_factor", "yarn_attn_factor", "yarn_beta_fast", "yarn_beta_slow")]
    _fields_ += [("yarn_orig_ctx", C.c_uint32), ("cb_eval", C.c_void_p), ("cb_eval_user_data", C.c_void_p),
                ("type_k", C.c_int), ("type_v", C.c_int), ("moe_cache_size", C.c_size_t),
                ("moe_cache_cpu_share", C.c_float), ("abort_callback", C.c_void_p), ("abort_callback_data", C.c_void_p)]
    _fields_ += [(name, C.c_bool) for name in (
        "embeddings", "offload_kqv", "no_perf", "op_offload", "swa_full", "kv_unified")]
    _fields_ += [("samplers", C.c_void_p), ("n_samplers", C.c_size_t), ("ctx_other", C.c_void_p)]


_Ints = C.POINTER(C.c_int32)


class _Batch(C.Structure):
    _fields_ = [("n_tokens", C.c_int32), ("token", _Ints), ("embd", C.POINTER(C.c_float)),
                ("pos", _Ints), ("n_seq_id", _Ints), ("seq_id", C.POINTER(_Ints)),
                ("logits", C.POINTER(C.c_int8))]


def _function(library, name, result, arguments):
    function = getattr(library, name)
    function.restype, function.argtypes = result, arguments
    return function


def _cpu_model_params(library):
    params = library.llama_model_default_params()
    # Pinned llama_load_mode NONE=0 and llama_lazy_mode OFF=0 load complete
    # tensors into owned CPU buffers. AUTO selects mmap, which caused repeated
    # major page faults per token and HTTP deadlines on the measured filesystem.
    # Keep the same verified weights and native kernels; never skip verification.
    params.load_mode, params.lazy_mode = 0, 0
    params.n_gpu_layers, params.use_extra_bufts = 0, False
    return params


@dataclass(frozen=True)
class Generation:
    token_ids: tuple[int, ...]
    pieces: tuple[bytes, ...]
    reflection_logits: tuple[dict[int, float], ...]
    input_tokens: int
    ended: bool


class SelfRagRuntime:
    """One CPU model and one sequence; no sampling, fallback or prompt truncation."""

    def __init__(self):
        self.model = self.context = None
        self.vocab = None
        self.context_tokens = 0
        self._locked = False
        model, directory, threads = verified_paths()
        self.threads = threads
        if not _ACTIVE.acquire(blocking=False):
            raise CapabilityError("research_capacity", "Only one native Self-RAG model may run in this process", 429)
        self._locked = True
        try:
            self._libraries = [C.CDLL(str(directory / "libggml-base.so.1.0.0"), mode=C.RTLD_GLOBAL)]
            ggml = C.CDLL(str(directory / "libggml.so.1.0.0"), mode=C.RTLD_GLOBAL)
            self._libraries.append(ggml)
            candidates = []
            for name in LIBRARY_HASHES:
                if name.startswith("libggml-cpu-"):
                    library = C.CDLL(str(directory / name))
                    self._libraries.append(library)
                    score = _function(library, "ggml_backend_score", C.c_int, [])()
                    candidates.append((score, name))
            score, name = max(candidates)
            if score <= 0:
                raise ValueError("CPU runtime unavailable")
            backend = _function(ggml, "ggml_backend_load", C.c_void_p, [C.c_char_p])(os.fsencode(directory / name))
            if not backend:
                raise ValueError("CPU backend unavailable")
            self.cpu_backend = name
            self.lib = C.CDLL(str(directory / "libllama.so.1.0.0"))
            self._bind()
            self.lib.llama_backend_init()
            params = _cpu_model_params(self.lib)
            self.model = self.lib.llama_model_load_from_file(os.fsencode(model), params)
            if not self.model:
                raise ValueError("model load failed")
            self.vocab = self.lib.llama_model_get_vocab(self.model)
            if not self.vocab or self.lib.llama_vocab_n_tokens(self.vocab) != VOCAB_SIZE:
                raise ValueError("wrong checkpoint vocabulary")
            for text, identifier in TOKEN_IDS.items():
                if (self.lib.llama_vocab_get_text(self.vocab, identifier) != text.encode()
                        or self.tokenize(text, add_bos=False) != [identifier]
                        or self.lib.llama_vocab_is_eog(self.vocab, identifier)):
                    raise ValueError("wrong reflection token identity")
            if not self.lib.llama_vocab_is_eog(self.vocab, 2):
                raise ValueError("wrong end token")
        except Exception as exc:
            self.close()
            if isinstance(exc, CapabilityError):
                raise
            raise CapabilityError("requires_backend", "Pinned Self-RAG CPU runtime could not initialize") from exc

    def _bind(self):
        signatures = {
            "llama_backend_init": (None, []),
            "llama_model_default_params": (_ModelParams, []),
            "llama_context_default_params": (_ContextParams, []),
            "llama_model_load_from_file": (C.c_void_p, [C.c_char_p, _ModelParams]),
            "llama_model_get_vocab": (C.c_void_p, [C.c_void_p]),
            "llama_vocab_n_tokens": (C.c_int32, [C.c_void_p]),
            "llama_vocab_get_text": (C.c_char_p, [C.c_void_p, C.c_int32]),
            "llama_vocab_is_eog": (C.c_bool, [C.c_void_p, C.c_int32]),
            "llama_init_from_model": (C.c_void_p, [C.c_void_p, _ContextParams]),
            "llama_tokenize": (C.c_int32, [C.c_void_p, C.c_char_p, C.c_int32, _Ints, C.c_int32, C.c_bool, C.c_bool]),
            "llama_token_to_piece": (C.c_int32, [C.c_void_p, C.c_int32, C.c_void_p, C.c_int32, C.c_int32, C.c_bool]),
            "llama_decode": (C.c_int32, [C.c_void_p, _Batch]),
            "llama_get_logits_ith": (C.POINTER(C.c_float), [C.c_void_p, C.c_int32]),
            "llama_get_memory": (C.c_void_p, [C.c_void_p]),
            "llama_memory_clear": (None, [C.c_void_p, C.c_bool]),
            "llama_model_free": (None, [C.c_void_p]),
            "llama_free": (None, [C.c_void_p]),
        }
        for name, (result, arguments) in signatures.items():
            _function(self.lib, name, result, arguments)

    def tokenize(self, text: str, *, add_bos: bool = True) -> list[int]:
        if not self.model or not self.vocab:
            raise CapabilityError("requires_backend", "Self-RAG model is closed or unavailable")
        try:
            raw = text.encode("utf-8")
        except UnicodeError as exc:
            raise CapabilityError("invalid_research_input", "Self-RAG requires valid UTF-8 text", 422) from exc
        if not raw or len(raw) > 800_000:
            raise CapabilityError("invalid_research_input", "Self-RAG prompt byte limit exceeded", 422)
        capacity = len(raw) + len(TOKEN_IDS) + 4
        output = (C.c_int32 * capacity)()
        count = self.lib.llama_tokenize(self.vocab, raw, len(raw), output, capacity, add_bos, True)
        if count <= 0 or count > capacity:
            raise CapabilityError("invalid_model_input", "Self-RAG tokenizer failed", 502)
        return list(output[:count])

    def prepare_context(self, input_tokens: int) -> None:
        """Reserve the complete prompt plus output bound, without unused KV pages.

        The adapter calls this once after tokenizing its entire snapshot. Direct
        generate() callers get the same bound for their individual full prompt.
        A larger later prompt releases the previous context before allocating.
        """
        if type(input_tokens) is not int or not 1 <= input_tokens <= MAX_INPUT_TOKENS:
            raise CapabilityError("context_budget_exceeded", "Full Self-RAG source prompt exceeds its token bound", 422)
        capacity = ((input_tokens + MAX_OUTPUT_TOKENS + 255) // 256) * 256
        if capacity > CONTEXT_TOKENS:
            raise CapabilityError("context_budget_exceeded", "Self-RAG context capacity exceeded", 422)
        if not self.model:
            raise CapabilityError("requires_backend", "Self-RAG model is closed or unavailable")
        if self.context and self.context_tokens >= capacity:
            return
        if self.context:
            self.lib.llama_free(self.context)
            self.context = None
            self.context_tokens = 0
        try:
            params = self.lib.llama_context_default_params()
            params.n_ctx, params.n_batch, params.n_ubatch, params.n_seq_max = capacity, 256, 256, 1
            params.n_threads = params.n_threads_batch = self.threads
            params.offload_kqv, params.op_offload, params.no_perf = False, False, True
            self.context = self.lib.llama_init_from_model(self.model, params)
            if not self.context:
                raise ValueError("context allocation failed")
            self.context_tokens = capacity
        except Exception as exc:
            self.close()
            raise CapabilityError("requires_backend", "Self-RAG CPU context could not initialize") from exc

    def _decode(self, tokens: list[int], position: int) -> None:
        size = len(tokens)
        identifiers = (C.c_int32 * size)(*tokens)
        positions = (C.c_int32 * size)(*range(position, position + size))
        counts = (C.c_int32 * size)(*([1] * size))
        sequence = (C.c_int32 * 1)(0)
        sequences = (_Ints * size)(*([C.cast(sequence, _Ints)] * size))
        outputs = (C.c_int8 * size)(*([0] * (size - 1) + [1]))
        batch = _Batch(size, identifiers, None, positions, counts, sequences, outputs)
        if self.lib.llama_decode(self.context, batch) != 0:
            raise CapabilityError("upstream_failed", "Self-RAG CPU decoding failed", 502)

    def _piece(self, identifier: int) -> bytes:
        buffer = C.create_string_buffer(256)
        size = self.lib.llama_token_to_piece(self.vocab, identifier, buffer, len(buffer), 0, True)
        if size < 0 or size > len(buffer):
            raise CapabilityError("invalid_model_output", "Self-RAG token piece exceeds limit", 502)
        return buffer.raw[:size]

    def generate(self, prompt: str) -> Generation:
        tokens = self.tokenize(prompt)
        if len(tokens) > MAX_INPUT_TOKENS:
            raise CapabilityError("context_budget_exceeded", "Full Self-RAG source prompt exceeds 1536 model tokens", 422)
        self.prepare_context(len(tokens))
        memory = self.lib.llama_get_memory(self.context)
        if not memory:
            raise CapabilityError("requires_backend", "Self-RAG context memory is unavailable")
        self.lib.llama_memory_clear(memory, True)
        for offset in range(0, len(tokens), 256):
            self._decode(tokens[offset:offset + 256], offset)
        output, pieces, reflections = [], [], []
        for index in range(MAX_OUTPUT_TOKENS):
            pointer = self.lib.llama_get_logits_ith(self.context, -1)
            if not pointer:
                raise CapabilityError("invalid_model_output", "Self-RAG logits are unavailable", 502)
            logits = list(pointer[:VOCAB_SIZE])
            if not all(math.isfinite(value) for value in logits):
                raise CapabilityError("invalid_model_output", "Self-RAG produced non-finite logits", 502)
            identifier = max(range(VOCAB_SIZE), key=logits.__getitem__)
            output.append(identifier)
            pieces.append(self._piece(identifier))
            reflections.append({token: logits[token] for token in REFLECTION_IDS})
            if self.lib.llama_vocab_is_eog(self.vocab, identifier):
                return Generation(tuple(output), tuple(pieces), tuple(reflections), len(tokens), True)
            self._decode([identifier], len(tokens) + index)
        raise CapabilityError("invalid_model_output", "Self-RAG exhausted its output limit without completing", 502)

    def close(self):
        if self.context:
            self.lib.llama_free(self.context)
            self.context = None
        self.context_tokens = 0
        if self.model:
            self.lib.llama_model_free(self.model)
            self.model = None
        self.vocab = None
        if self._locked:
            self._locked = False
            _ACTIVE.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
