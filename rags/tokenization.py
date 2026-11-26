"""Explicit text counters, lossless source spans, and bounded prompt assembly.

Unicode mode counts code points, not model tokens. Tiktoken counts are exact for
the selected text encoding; provider message-envelope costs remain separate.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
import json
from threading import BoundedSemaphore, RLock
from typing import Callable, Iterator, Literal, Mapping, Protocol, Sequence

from rags.tokenizer_artifacts import (
    ARTIFACT_BACKENDS, SHA256, TOKENIZER_ALIAS, parse_tokenizer_artifacts, read_tokenizer_artifact,
)


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _text(value: str) -> None:
    if not isinstance(value, str):
        raise ValueError("Text must be a string")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError("Text contains an invalid Unicode surrogate") from exc


@dataclass(frozen=True)
class TokenizerConfig:
    backend: Literal["unicode", "tiktoken", "utf8_bytes", "huggingface", "sentencepiece", "embedding"] = "unicode"
    model: str | None = None
    encoding: str | None = None
    artifact_sha256: str | None = None
    embedding_identity: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or self.backend not in {
            "unicode", "tiktoken", "utf8_bytes", "huggingface", "sentencepiece", "embedding",
        }:
            raise ValueError("Unsupported tokenizer backend")
        for name in ("model", "encoding"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"Tokenizer {name} must be a non-empty string")
        if self.backend in {"unicode", "utf8_bytes"} and (self.model or self.encoding):
            raise ValueError("Fallback counting does not accept a model or encoding")
        if self.backend == "tiktoken" and not (self.model or self.encoding):
            raise ValueError("Tiktoken requires a model or explicit encoding")
        if self.backend in ARTIFACT_BACKENDS:
            if not isinstance(self.model, str) or not TOKENIZER_ALIAS.fullmatch(self.model) or self.encoding:
                raise ValueError("Local tokenizer requires a registered model alias and no encoding override")
        if self.artifact_sha256 is not None:
            if (self.backend not in ARTIFACT_BACKENDS or not isinstance(self.artifact_sha256, str)
                    or not SHA256.fullmatch(self.artifact_sha256)):
                raise ValueError("Artifact SHA256 is only valid for a registered local tokenizer")
        if self.embedding_identity is not None:
            if (self.backend != "embedding" or not isinstance(self.embedding_identity, str)
                    or not SHA256.fullmatch(self.embedding_identity)):
                raise ValueError("Embedding tokenizer identity must be its bound SHA256")
        if self.backend == "embedding" and (self.model is not None or self.encoding is not None):
            raise ValueError("Embedding tokenizer selects only the configured provider, without model overrides")


@dataclass(frozen=True)
class TokenSpan:
    start_char: int
    end_char: int
    text: str
    token_count: int

    @property
    def start(self) -> int:
        return self.start_char

    @property
    def end(self) -> int:
        return self.end_char


class Tokenizer(Protocol):
    id: str
    unit: str
    count_kind: str
    profile: dict

    def encode(self, text: str) -> list[int]: ...
    def decode(self, tokens: Sequence[int]) -> str: ...
    def count(self, text: str) -> int: ...
    def truncate(self, text: str, budget: int) -> str: ...
    def suffix(self, text: str, budget: int) -> str: ...
    def iter_spans(self, text: str, max_tokens: int, overlap_tokens: int = 0) -> Iterator[TokenSpan]:
        """Produce source windows lazily for callers that enforce allocation limits."""
        raise NotImplementedError
    def spans(self, text: str, max_tokens: int, overlap_tokens: int = 0) -> list[TokenSpan]: ...


class _SpanTokenizer:
    def count(self, text: str) -> int:
        return len(self.encode(text))

    def spans(self, text: str, max_tokens: int, overlap_tokens: int = 0) -> list[TokenSpan]:
        """Compatibility list API; bounded ingestion consumes iter_spans instead."""
        return list(self.iter_spans(text, max_tokens, overlap_tokens))

    def iter_spans(self, text: str, max_tokens: int, overlap_tokens: int = 0) -> Iterator[TokenSpan]:
        """Yield one source window at a time without retaining overlapping copies."""
        _text(text)
        _integer(max_tokens, "max_tokens", 1)
        _integer(overlap_tokens, "overlap_tokens")
        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        previous_end = 0
        start = 0
        while start < len(text):
            if previous_end:
                # Removing a leading character can increase a BPE suffix count.
                # Revalidate overlap after any progress adjustment below.
                while start < previous_end and self.count(text[start:previous_end]) > overlap_tokens:
                    start += 1
            value = self.truncate(text[start:], max_tokens)
            if not value:
                raise ValueError("Token budget is too small for the next Unicode character")
            end = start + len(value)
            if previous_end and end <= previous_end:
                # Prefer less overlap to a window that adds no new source text.
                start += 1
                continue
            yield TokenSpan(start, end, value, self.count(value))
            previous_end = end
            if end == len(text):
                break
            overlap = self.suffix(value, overlap_tokens)
            # A tiny valid chunk may itself fit the requested overlap. Always advance.
            start = max(start + 1, end - len(overlap))


class UnicodeTokenizer(_SpanTokenizer):
    id = "unicode-codepoints-v1"
    unit = "unicode_codepoints"
    count_kind = "fallback_units"

    @property
    def profile(self) -> dict:
        return {"backend": "unicode", "id": self.id, "unit": self.unit,
                "count_kind": self.count_kind, "version": 1}

    def encode(self, text: str) -> list[int]:
        _text(text)
        return [ord(character) for character in text]

    def decode(self, tokens: Sequence[int]) -> str:
        try:
            result = "".join(chr(token) for token in tokens)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid Unicode code points") from exc
        _text(result)
        return result

    def count(self, text: str) -> int:
        _text(text)
        return len(text)

    def truncate(self, text: str, budget: int) -> str:
        _text(text)
        _integer(budget, "budget")
        return text[:budget]

    def suffix(self, text: str, budget: int) -> str:
        _text(text)
        _integer(budget, "budget")
        return text[-budget:] if budget else ""


class UTF8ByteTokenizer(_SpanTokenizer):
    """Conservative UTF-8 text budget, explicitly not exact model tokenization."""
    id = "utf8-bytes-v1"
    unit = "utf8_bytes"
    count_kind = "conservative_bytes"

    @property
    def profile(self) -> dict:
        return {"backend": "utf8_bytes", "id": self.id, "unit": self.unit,
                "count_kind": self.count_kind, "version": 1}

    def encode(self, text: str) -> list[int]:
        _text(text)
        return list(text.encode("utf-8"))

    def decode(self, tokens: Sequence[int]) -> str:
        try:
            if any(type(token) is not int for token in tokens):
                raise ValueError("Expected byte integers")
            return bytes(tokens).decode("utf-8", errors="strict")
        except (ValueError, TypeError, UnicodeError) as exc:
            raise ValueError("Byte sequence does not decode to valid UTF-8") from exc

    def count(self, text: str) -> int:
        _text(text)
        return len(text.encode("utf-8"))

    def truncate(self, text: str, budget: int) -> str:
        _text(text)
        _integer(budget, "budget")
        return text.encode("utf-8")[:budget].decode("utf-8", errors="ignore")

    def suffix(self, text: str, budget: int) -> str:
        _text(text)
        _integer(budget, "budget")
        return text.encode("utf-8")[-budget:].decode("utf-8", errors="ignore") if budget else ""


MAX_EXACT_TEXT_CHARACTERS = 1_000_000
MAX_SOURCE_WINDOW_CHARACTERS = 16_384
MAX_NATIVE_TOKENIZERS = 4
MAX_NATIVE_ARTIFACT_CACHE_BYTES = 128 * 1024 * 1024
_NATIVE_CACHE: OrderedDict[tuple[str, str, str], tuple[Callable, Callable, int]] = OrderedDict()
_NATIVE_CACHE_LOCK = RLock()
_NATIVE_CACHE_BYTES = 0
# Bound simultaneously retained file buffers as well as the loaded-object cache.
_NATIVE_LOAD_SLOTS = BoundedSemaphore(2)


class ExactEncodingTokenizer(_SpanTokenizer):
    """An exact counter with source-preserving, bounded character-window slices.

    A provider may supply just its complete-input counter. Token IDs and native
    decoding are available only when callbacks are supplied. Decoding can be
    lossy; it is never used to construct source spans. A fitting slice is verified
    rather than assumed maximal, because token counts need not be monotone.
    """
    unit = "model_tokens"
    count_kind = "exact_encoding"

    def __init__(self, *, tokenizer_id: str, profile: Mapping,
                 count: Callable[[str], int] | None = None,
                 encode: Callable[[str], Sequence[int]] | None = None,
                 decode: Callable[[Sequence[int]], str] | None = None):
        if not isinstance(tokenizer_id, str) or not 1 <= len(tokenizer_id) <= 240:
            raise ValueError("An exact tokenizer requires a bounded identity")
        if not isinstance(profile, Mapping) or not isinstance(profile.get("backend"), str):
            raise ValueError("An exact tokenizer requires an explicit backend profile")
        if count is None and encode is None:
            raise ValueError("An exact tokenizer requires a count or encode callback")
        if any(callback is not None and not callable(callback) for callback in (count, encode, decode)):
            raise ValueError("Tokenizer callbacks must be callable")
        try:
            self._profile = json.loads(json.dumps(dict(profile), allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError("Tokenizer profile must contain JSON-safe values") from exc
        self._profile.update(id=tokenizer_id, unit=self.unit, count_kind=self.count_kind,
                             source_window="original-character-slices-v1",
                             source_window_characters=MAX_SOURCE_WINDOW_CHARACTERS)
        self.id = tokenizer_id
        self._count, self._encode, self._decode = count, encode, decode

    @property
    def profile(self) -> dict:
        return deepcopy(self._profile)

    @staticmethod
    def _validate_text(text: str) -> None:
        _text(text)
        if len(text) > MAX_EXACT_TEXT_CHARACTERS:
            raise ValueError("Tokenizer text exceeds its character budget")

    @staticmethod
    def _validate_ids(tokens: Sequence[int]) -> list[int]:
        try:
            values = list(tokens)
        except TypeError as exc:
            raise ValueError("Expected a sequence of token integers") from exc
        if any(type(token) is not int or token < 0 for token in values):
            raise ValueError("Expected nonnegative token integers")
        return values

    def encode(self, text: str) -> list[int]:
        self._validate_text(text)
        if self._encode is None:
            raise ValueError("This bound tokenizer exposes counts, not token IDs")
        return self._validate_ids(self._encode(text))

    def decode(self, tokens: Sequence[int]) -> str:
        values = self._validate_ids(tokens)
        if self._decode is None:
            raise ValueError("This bound tokenizer does not expose native decoding")
        result = self._decode(values)
        _text(result)
        return result

    def count(self, text: str) -> int:
        self._validate_text(text)
        result = self._count(text) if self._count is not None else len(self.encode(text))
        _integer(result, "token count")
        return result

    def _slice(self, text: str, budget: int, *, from_end: bool) -> str:
        self._validate_text(text)
        _integer(budget, "budget")
        if not budget or not text:
            return ""
        candidate = (text[-MAX_SOURCE_WINDOW_CHARACTERS:] if from_end
                     else text[:MAX_SOURCE_WINDOW_CHARACTERS])
        if self.count(candidate) <= budget:
            return candidate
        low, high, fitted = 1, len(candidate) - 1, ""
        while low <= high:
            size = (low + high) // 2
            value = candidate[-size:] if from_end else candidate[:size]
            if self.count(value) <= budget:
                fitted = value
                low = size + 1
            else:
                high = size - 1
        if fitted and self.count(fitted) > budget:
            raise ValueError("Tokenizer count changed while selecting a source slice")
        return fitted

    def truncate(self, text: str, budget: int) -> str:
        return self._slice(text, budget, from_end=False)

    def suffix(self, text: str, budget: int) -> str:
        return self._slice(text, budget, from_end=True)


def _load_native_callbacks(backend: str, raw: bytes) -> tuple[Callable, Callable]:
    if backend == "huggingface":
        try:
            from tokenizers import Tokenizer as NativeTokenizer
        except ImportError as exc:
            raise ValueError("Hugging Face tokenizer requires the optional tokenizers package") from exc
        try:
            native = NativeTokenizer.from_buffer(raw)
            if getattr(native.model, "dropout", None) not in (None, 0):
                raise ValueError("Stochastic tokenizer dropout is unsupported")
            native.no_truncation()
            native.no_padding()
            # The BPE word cache otherwise retains source-derived data. Supported
            # 0.22/0.23 runtimes expose this bounded-cache control on the model.
            resize = getattr(native.model, "_resize_cache", None)
            if resize is not None:
                resize(0)
        except Exception as exc:
            raise ValueError("Unable to load a deterministic Hugging Face tokenizer artifact") from exc

        def encode(text):
            return native.encode(text, add_special_tokens=False).ids

        def decode(tokens):
            if any(native.id_to_token(token) is None for token in tokens):
                raise ValueError("Unknown tokenizer token ID")
            return native.decode(tokens, skip_special_tokens=False)

        return encode, decode
    try:
        import sentencepiece
    except ImportError as exc:
        raise ValueError("SentencePiece tokenizer requires the optional sentencepiece package") from exc
    try:
        native = sentencepiece.SentencePieceProcessor(model_proto=raw)
    except Exception as exc:
        raise ValueError("Unable to load the registered SentencePiece artifact") from exc

    def encode(text):
        return native.encode(text, out_type=int, enable_sampling=False, nbest_size=0,
                             add_bos=False, add_eos=False, reverse=False, num_threads=1)

    def decode(tokens):
        if any(token >= native.get_piece_size() for token in tokens):
            raise ValueError("Unknown tokenizer token ID")
        return native.decode(tokens)

    return encode, decode


def _cached_native_callbacks(backend: str, entry: Mapping):
    global _NATIVE_CACHE_BYTES
    with _NATIVE_LOAD_SLOTS:
        # Verify on every resolution, including cache hits. Replacing a file
        # cannot continue under an old registration; parsing uses these bytes.
        raw = read_tokenizer_artifact(entry)
        package = "tokenizers" if backend == "huggingface" else "sentencepiece"
        try:
            runtime_version = version(package)
        except PackageNotFoundError as exc:
            raise ValueError(f"Tokenizer backend requires the optional {package} package") from exc
        key = (backend, entry["sha256"], runtime_version)
        with _NATIVE_CACHE_LOCK:
            cached = _NATIVE_CACHE.get(key)
            if cached is None:
                encode, decode = _load_native_callbacks(backend, raw)
                cached = (encode, decode, len(raw))
                if len(raw) <= MAX_NATIVE_ARTIFACT_CACHE_BYTES:
                    while _NATIVE_CACHE and (len(_NATIVE_CACHE) >= MAX_NATIVE_TOKENIZERS
                                            or _NATIVE_CACHE_BYTES + len(raw) > MAX_NATIVE_ARTIFACT_CACHE_BYTES):
                        _, evicted = _NATIVE_CACHE.popitem(last=False)
                        _NATIVE_CACHE_BYTES -= evicted[2]
                    _NATIVE_CACHE[key] = cached
                    _NATIVE_CACHE_BYTES += len(raw)
            else:
                _NATIVE_CACHE.move_to_end(key)
        return cached, runtime_version


def _artifact_tokenizer(config: TokenizerConfig, artifacts: Mapping | None) -> ExactEncodingTokenizer:
    registry = parse_tokenizer_artifacts(artifacts)
    entry = registry.get(config.model)
    if entry is None or entry["backend"] != config.backend:
        raise ValueError("Tokenizer alias is not registered for the selected backend")
    if config.artifact_sha256 is not None and config.artifact_sha256 != entry["sha256"]:
        raise ValueError("Tokenizer artifact identity changed; select new processing settings explicitly")
    cached, runtime_version = _cached_native_callbacks(config.backend, entry)
    return ExactEncodingTokenizer(
        tokenizer_id=config.backend + ":" + entry["sha256"], encode=cached[0], decode=cached[1],
        profile={"backend": config.backend, "model": config.model,
                 "artifact_sha256": entry["sha256"], "version": runtime_version,
                 "special_tokens": False, "padding": False, "truncation": False,
                 "sampling": False},
    )


class TiktokenTokenizer(_SpanTokenizer):
    unit = "model_tokens"
    count_kind = "exact_encoding"

    def __init__(self, config: TokenizerConfig):
        try:
            import tiktoken
        except ImportError as exc:
            raise ValueError("Tiktoken backend requires the tiktoken package") from exc
        if config.model and config.encoding:
            try:
                known_encoding = tiktoken.encoding_name_for_model(config.model)
            except KeyError:
                known_encoding = None  # Explicit encodings support custom model names.
            if known_encoding is not None and known_encoding != config.encoding:
                raise ValueError("Explicit encoding conflicts with the known model tokenizer")
        try:
            self._encoding = (tiktoken.get_encoding(config.encoding) if config.encoding
                              else tiktoken.encoding_for_model(config.model))
        except KeyError as exc:
            raise ValueError("Unknown tokenizer model; configure an explicit encoding") from exc
        except Exception as exc:
            raise ValueError("Unable to load tiktoken encoding; check its name and local cache") from exc
        self.id = "tiktoken:" + self._encoding.name
        self._config = config
        self._version = version("tiktoken")
        self._max_bytes = max(map(len, self._encoding.token_byte_values()))

    @property
    def profile(self) -> dict:
        # New optional selectors must not change historical profile fingerprints.
        return {"backend": self._config.backend, "model": self._config.model,
                "encoding": self._encoding.name, "id": self.id,
                "unit": self.unit, "count_kind": self.count_kind, "version": self._version}

    def encode(self, text: str) -> list[int]:
        _text(text)
        # Documents containing special-token spellings remain ordinary source text.
        return self._encoding.encode(text, disallowed_special=())

    def decode(self, tokens: Sequence[int]) -> str:
        try:
            return self._encoding.decode(list(tokens), errors="strict")
        except (KeyError, ValueError, UnicodeDecodeError, TypeError) as exc:
            raise ValueError("Token sequence does not decode to valid Unicode text") from exc

    def truncate(self, text: str, budget: int) -> str:
        _text(text)
        _integer(budget, "budget")
        if not budget or not text:
            return ""
        # A token cannot span more characters than its maximum byte length. This
        # bounds work for every window without repeatedly encoding a whole document.
        candidate = text[:budget * self._max_bytes]
        tokens = self.encode(candidate)
        if len(tokens) <= budget:
            return candidate
        raw = b"".join(self._encoding.decode_single_token_bytes(t) for t in tokens[:budget])
        prefix = raw.decode("utf-8", errors="ignore")
        while prefix and self.count(prefix) > budget:
            prefix = prefix[:-1]
        if not text.startswith(prefix):
            raise ValueError("Tokenizer failed to preserve the original source prefix")
        return prefix

    def suffix(self, text: str, budget: int) -> str:
        _text(text)
        _integer(budget, "budget")
        if not budget or not text:
            return ""
        candidate = text[-budget * self._max_bytes:]
        tokens = self.encode(candidate)
        if len(tokens) <= budget:
            return candidate
        raw = b"".join(self._encoding.decode_single_token_bytes(t) for t in tokens[-budget:])
        suffix = raw.decode("utf-8", errors="ignore")
        while suffix and self.count(suffix) > budget:
            suffix = suffix[1:]
        if not text.endswith(suffix):
            raise ValueError("Tokenizer failed to preserve the original source suffix")
        return suffix


def resolve_tokenizer(config: TokenizerConfig | Mapping | None = None, *,
                      artifacts: Mapping | None = None,
                      embedding_tokenizer: ExactEncodingTokenizer | None = None) -> Tokenizer:
    if config is None:
        config = TokenizerConfig()
    if isinstance(config, Mapping):
        try:
            config = TokenizerConfig(**config)
        except TypeError as exc:
            raise ValueError("Unknown tokenizer configuration field") from exc
    if not isinstance(config, TokenizerConfig):
        raise ValueError("Expected TokenizerConfig or a configuration object")
    if config.backend == "unicode":
        return UnicodeTokenizer()
    if config.backend == "utf8_bytes":
        return UTF8ByteTokenizer()
    if config.backend in ARTIFACT_BACKENDS:
        return _artifact_tokenizer(config, artifacts)
    if config.backend == "embedding":
        if not isinstance(embedding_tokenizer, ExactEncodingTokenizer):
            raise ValueError("Embedding tokenizer requires an exact configured-provider binding")
        profile = embedding_tokenizer.profile
        identity = profile.get("embedding_identity")
        if (profile.get("backend") != "embedding" or not isinstance(identity, str)
                or not SHA256.fullmatch(identity)):
            raise ValueError("Embedding tokenizer has an invalid provider binding")
        if config.embedding_identity is not None and config.embedding_identity != identity:
            raise ValueError("Embedding tokenizer identity changed; select new processing settings explicitly")
        return embedding_tokenizer
    return TiktokenTokenizer(config)


@dataclass(frozen=True)
class ContextItem:
    id: str
    text: str
    header: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("Context item requires a non-empty ID")
        _text(self.text)
        _text(self.header)


@dataclass(frozen=True)
class PackedContext:
    context: str
    text: str
    items: list[ContextItem]
    included_ids: list[str]
    dropped_ids: list[str]
    truncated_ids: list[str]
    token_count: int
    tokenizer_id: str
    unit: str
    count_kind: str
    counter_id: str | None = None
    metadata: dict = field(default_factory=dict)


def pack_context(
    items: Sequence[ContextItem | Mapping], *, tokenizer: Tokenizer, input_budget: int,
    fixed_prefix: str = "", fixed_suffix: str = "", separator: str = "\n\n",
    counter: Callable[[str], int] | None = None, counter_id: str | None = None,
) -> PackedContext:
    """Fit evidence into a fully rendered prompt, preserving its fixed text.

    The caller reserves output tokens and message-envelope overhead. A callback
    may count a provider's complete request around the supplied text; callback
    accuracy is its own contract and is reported as provider_count, not inferred.
    """
    _integer(input_budget, "input_budget", 1)
    for value in (fixed_prefix, fixed_suffix, separator):
        _text(value)
    if counter is not None:
        if not callable(counter):
            raise ValueError("Custom counter must be callable")
        if not isinstance(counter_id, str) or not counter_id.strip():
            raise ValueError("A custom counter requires a counter_id")

    def count(value: str) -> int:
        result = counter(value) if counter else tokenizer.count(value)
        _integer(result, "counter result")
        return result

    normalized = []
    for item in items:
        if isinstance(item, Mapping):
            item = ContextItem(item.get("id", item.get("chunk_id", "")),
                               item.get("text", item.get("content", "")), item.get("header", ""))
        if not isinstance(item, ContextItem):
            raise ValueError("Expected ContextItem or a context item object")
        normalized.append(item)
    if len({item.id for item in normalized}) != len(normalized):
        raise ValueError("Context item IDs must be unique")
    if count(fixed_prefix + fixed_suffix) > input_budget:
        raise ValueError("Question and fixed prompt exceed the input budget")
    accepted: list[ContextItem] = []
    blocks: list[str] = []
    dropped, truncated = [], []

    def render(extra: str | None = None) -> str:
        values = blocks if extra is None else [*blocks, extra]
        return fixed_prefix + separator.join(values) + fixed_suffix

    for item in normalized:
        header = item.header or f"[{item.id}]"
        value = item.text
        block = header + "\n" + value
        if not value or count(render(block)) > input_budget:
            # Search code-point boundaries and retain only a verified fitting
            # prefix. Token counts need not be strictly monotone; maximal packing
            # is not required, but the returned prompt must always fit.
            low, high, fit = 1, len(value), ""
            while low <= high:
                middle = (low + high) // 2
                candidate = value[:middle]
                if count(render(header + "\n" + candidate)) <= input_budget:
                    fit = candidate
                    low = middle + 1
                else:
                    high = middle - 1
            value = fit
            if not value:
                dropped.append(item.id)
                continue
            truncated.append(item.id)
            block = header + "\n" + value
        accepted.append(ContextItem(item.id, value, header))
        blocks.append(block)
    rendered = render()
    measured = count(rendered)
    if measured > input_budget:
        raise ValueError("Counter changed during packing; prompt budget cannot be guaranteed")
    return PackedContext(
        context=separator.join(blocks), text=rendered, items=accepted,
        included_ids=[item.id for item in accepted], dropped_ids=dropped,
        truncated_ids=truncated, token_count=measured, tokenizer_id=tokenizer.id,
        unit="model_tokens" if counter else tokenizer.unit,
        count_kind="provider_count" if counter else tokenizer.count_kind,
        counter_id=counter_id, metadata={"input_budget": input_budget},
    )
