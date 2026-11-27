"""Optional text tokenizers from bounded, operator-selected local artifacts.

The model may normalize input or encode unknown words, so decoded token IDs are
not a source-preservation mechanism. Windows always slice the original Unicode
text and recount that exact slice. Counts exclude automatic BOS/EOS and provider
message templates; matching an artifact to a deployment is an operator contract.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from importlib.metadata import version

from rags.file_io import read_regular_bytes
from rags.tokenization import TokenizerConfig, _integer, _SpanTokenizer, _text

MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_WINDOW_CHARS = 65_536


def _window_chars(budget: int) -> int:
    return min(MAX_WINDOW_CHARS, max(256, budget * 64))


class LocalArtifactTokenizer(_SpanTokenizer):
    """Hugging Face tokenizer.json or SentencePiece .model, with no network I/O."""

    unit = "model_tokens"
    count_kind = "exact_encoding"

    def __init__(self, config: TokenizerConfig):
        if config.backend not in {"huggingface", "sentencepiece"}:
            raise ValueError("Expected a local artifact tokenizer backend")
        try:
            raw = read_regular_bytes(config.path, max_bytes=MAX_ARTIFACT_BYTES)
        except (OSError, ValueError) as exc:
            raise ValueError("Tokenizer artifact must be a readable regular local file of at most 64 MiB") from exc
        if not raw:
            raise ValueError("Tokenizer artifact must not be empty")
        self._backend = config.backend
        self._digest = hashlib.sha256(raw).hexdigest()
        if self._backend == "huggingface":
            try:
                from tokenizers import Tokenizer
            except ImportError as exc:
                raise ValueError("Hugging Face tokenizer requires the tokenizers extra: uv sync --extra tokenizers") from exc
            try:
                self._tokenizer = Tokenizer.from_str(raw.decode("utf-8", errors="strict"))
                # A saved training/export configuration must never silently trim
                # source text or charge padded tokens as the text's exact count.
                self._tokenizer.no_truncation()
                self._tokenizer.no_padding()
                self._vocab_size = self._tokenizer.get_vocab_size(with_added_tokens=True)
                self._version = version("tokenizers")
            except Exception as exc:
                raise ValueError("Invalid Hugging Face tokenizer.json artifact") from exc
        else:
            try:
                import sentencepiece
            except ImportError as exc:
                raise ValueError("SentencePiece tokenizer requires the tokenizers extra: uv sync --extra tokenizers") from exc
            try:
                self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=raw)
                self._vocab_size = self._tokenizer.get_piece_size()
                self._version = version("sentencepiece")
            except Exception as exc:
                raise ValueError("Invalid SentencePiece model artifact") from exc
        if self._vocab_size < 1:
            raise ValueError("Tokenizer artifact has no vocabulary")
        self.id = self._backend + ":sha256:" + self._digest

    @property
    def profile(self) -> dict:
        # A host path is neither a portable identity nor safe public metadata.
        return {"backend": self._backend, "id": self.id, "artifact_sha256": self._digest,
                "unit": self.unit, "count_kind": self.count_kind, "version": self._version,
                "add_special_tokens": False, "source_window_policy": "unicode-slices-v1"}

    def encode(self, text: str) -> list[int]:
        _text(text)
        try:
            if self._backend == "huggingface":
                return self._tokenizer.encode(text, add_special_tokens=False).ids
            return self._tokenizer.encode(text, out_type=int, add_bos=False, add_eos=False,
                                          enable_sampling=False)
        except Exception as exc:
            raise ValueError("Local tokenizer could not encode the input text") from exc

    def decode(self, tokens: Sequence[int]) -> str:
        try:
            if isinstance(tokens, (str, bytes, bytearray)):
                raise ValueError("Expected token IDs")
            values = list(tokens)
            if any(type(token) is not int or token < 0 for token in values):
                raise ValueError("Expected valid integer token IDs")
            if self._backend == "huggingface":
                # The library permits vocabularies with gaps in their IDs. A
                # vocabulary's size is not necessarily its maximum token ID.
                if any(token > 2 ** 32 - 1 or self._tokenizer.id_to_token(token) is None for token in values):
                    raise ValueError("Expected an ID from the configured vocabulary")
                result = self._tokenizer.decode(values, skip_special_tokens=False)
            else:
                if any(token >= self._vocab_size for token in values):
                    raise ValueError("Expected an ID from the configured vocabulary")
                result = self._tokenizer.decode(values)
            _text(result)
            return result
        except Exception as exc:
            raise ValueError("Invalid local tokenizer token sequence") from exc

    def _source_window(self, text: str, budget: int, *, suffix: bool) -> str:
        _text(text)
        _integer(budget, "budget")
        if not budget or not text:
            return ""
        # Arbitrary artifacts may normalize a very long source into one token.
        # Bound each search's encoded source independently of that compression.
        # Maximal packing is not promised: the returned original slice must fit.
        size = min(len(text), _window_chars(budget))
        candidate = text[-size:] if suffix else text[:size]
        if self.count(candidate) <= budget:
            return candidate
        low, high, fit = 1, len(candidate) - 1, ""
        while low <= high:
            middle = (low + high) // 2
            piece = candidate[-middle:] if suffix else candidate[:middle]
            if self.count(piece) <= budget:
                fit = piece
                low = middle + 1
            else:
                high = middle - 1
        # Counts can decrease at later BPE boundaries. Binary search is an
        # optimization for a verified fit, never a monotonicity assumption used
        # to assert that a larger slice necessarily cannot fit.
        return fit

    def _truncate_from(self, text: str, start: int, budget: int) -> str:
        # iter_spans validates the full source once. Bound later copies and
        # Unicode validation as well as encode probes, independently of its size.
        return self.truncate(text[start:start + _window_chars(budget)], budget)

    def truncate(self, text: str, budget: int) -> str:
        return self._source_window(text, budget, suffix=False)

    def suffix(self, text: str, budget: int) -> str:
        return self._source_window(text, budget, suffix=True)
