"""Explicit text counters, lossless source spans, and bounded prompt assembly.

Unicode mode counts code points, not model tokens. Tiktoken counts are exact for
the selected text encoding; provider message-envelope costs remain separate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from importlib.metadata import version
from typing import Callable, Literal, Mapping, Protocol, Sequence


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
    backend: Literal["unicode", "tiktoken"] = "unicode"
    model: str | None = None
    encoding: str | None = None

    def __post_init__(self) -> None:
        if self.backend not in {"unicode", "tiktoken"}:
            raise ValueError("Tokenizer backend must be unicode or tiktoken")
        for name in ("model", "encoding"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"Tokenizer {name} must be a non-empty string")
        if self.backend == "unicode" and (self.model or self.encoding):
            raise ValueError("Unicode counting does not accept a model or encoding")
        if self.backend == "tiktoken" and not (self.model or self.encoding):
            raise ValueError("Tiktoken requires a model or explicit encoding")


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
    def spans(self, text: str, max_tokens: int, overlap_tokens: int = 0) -> list[TokenSpan]: ...


class _SpanTokenizer:
    def count(self, text: str) -> int:
        return len(self.encode(text))

    def spans(self, text: str, max_tokens: int, overlap_tokens: int = 0) -> list[TokenSpan]:
        _text(text)
        _integer(max_tokens, "max_tokens", 1)
        _integer(overlap_tokens, "overlap_tokens")
        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        result = []
        start = 0
        while start < len(text):
            if result:
                # Removing a leading character can increase a BPE suffix count.
                # Revalidate overlap after any progress adjustment below.
                while start < result[-1].end_char and self.count(text[start:result[-1].end_char]) > overlap_tokens:
                    start += 1
            value = self.truncate(text[start:], max_tokens)
            if not value:
                raise ValueError("Token budget is too small for the next Unicode character")
            end = start + len(value)
            if result and end <= result[-1].end_char:
                # Prefer less overlap to a window that adds no new source text.
                start += 1
                continue
            result.append(TokenSpan(start, end, value, self.count(value)))
            if end == len(text):
                break
            overlap = self.suffix(value, overlap_tokens)
            # A tiny valid chunk may itself fit the requested overlap. Always advance.
            start = max(start + 1, end - len(overlap))
        return result


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
        return {**asdict(self._config), "encoding": self._encoding.name, "id": self.id,
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


def resolve_tokenizer(config: TokenizerConfig | Mapping | None = None) -> Tokenizer:
    if config is None:
        config = TokenizerConfig()
    if isinstance(config, Mapping):
        try:
            config = TokenizerConfig(**config)
        except TypeError as exc:
            raise ValueError("Unknown tokenizer configuration field") from exc
    if not isinstance(config, TokenizerConfig):
        raise ValueError("Expected TokenizerConfig or a configuration object")
    return UnicodeTokenizer() if config.backend == "unicode" else TiktokenTokenizer(config)


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
