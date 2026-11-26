"""Pure, bounded operator embedding contracts; no model or credential loading."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from urllib.parse import urlsplit

from rags.model_profiles import strict_json, validate_local_no_auth_endpoint, validate_model_auth_mode
from rags.tokenization import TokenizerConfig


PROVIDERS = {"local", "fastembed", "sentence_transformers", "ollama", "openai", "router"}
PROFILE_ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,59}")


def _utf8(value: str) -> None:
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError("Embedding profile values must be valid UTF-8") from exc


def contract_digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class EmbeddingProfile:
    provider: str
    model: str
    revision: str = ""
    base_url: str | None = None
    api_key_env: str | None = None
    auth_mode: str = "required"
    dimensions: int | None = None
    query_prefix: str = ""
    document_prefix: str = ""
    normalize: bool | None = None
    input_policy: str = "auto"
    tokenizer: dict | None = None
    token_limit: int | None = None
    special_tokens: int | None = None
    space_id: str = ""

    def __post_init__(self):
        if (not isinstance(self.provider, str) or self.provider not in PROVIDERS
                or not isinstance(self.model, str) or not self.model or len(self.model) > 150
                or any(ord(c) <= 32 or ord(c) == 127 for c in self.model)):
            raise ValueError("Embedding profiles require a supported provider and bounded model identifier")
        _utf8(self.model)
        if not isinstance(self.revision, str) or (self.revision and not re.fullmatch(r"[0-9a-f]{40}", self.revision)):
            raise ValueError("Embedding revision must be an immutable 40-character commit SHA")
        if self.provider == "sentence_transformers" and (
            not self.revision or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", self.model)
        ):
            raise ValueError("Sentence Transformers requires a Hub repository and immutable revision")
        if self.revision and self.provider != "sentence_transformers":
            raise ValueError("Pinned revisions are supported by the Sentence Transformers adapter")
        if self.dimensions is not None and (type(self.dimensions) is not int or not 1 <= self.dimensions <= 4096):
            raise ValueError("Embedding dimensions must be an integer from 1 to 4096")
        if self.provider in {"local", "fastembed", "router"} and self.dimensions is not None:
            raise ValueError("This embedding adapter does not accept dimension overrides")
        for prefix in (self.query_prefix, self.document_prefix):
            if not isinstance(prefix, str) or len(prefix) > 2048 or "\x00" in prefix:
                raise ValueError("Embedding input prefixes must be bounded text")
            _utf8(prefix)
        if self.normalize is not None and type(self.normalize) is not bool:
            raise ValueError("Embedding normalization must be boolean")
        if not isinstance(self.input_policy, str) or self.input_policy not in {"auto", "exact", "server_reject"}:
            raise ValueError("Embedding input policy must be auto, exact or server_reject")
        if self.input_policy == "server_reject" and self.provider != "ollama":
            raise ValueError("Server overflow rejection is supported by the native Ollama adapter")
        if self.provider == "ollama" and self.input_policy == "exact" and self.tokenizer is None:
            raise ValueError("Exact Ollama counting requires an explicit tokenizer and token limit")
        if (self.tokenizer is None) != (self.token_limit is None):
            raise ValueError("Custom embedding counters require both tokenizer and token limit")
        if self.tokenizer is None and self.special_tokens is not None:
            raise ValueError("Special-token overhead belongs to an explicit embedding counter")
        if self.tokenizer is not None:
            if not isinstance(self.tokenizer, dict):
                raise ValueError("Embedding tokenizer must be a configuration object")
            try:
                config = TokenizerConfig(**self.tokenizer)
            except TypeError as exc:
                raise ValueError("Invalid embedding tokenizer fields") from exc
            if config.backend not in {"tiktoken", "huggingface", "sentencepiece"}:
                raise ValueError("Embedding counter requires an exact model tokenizer")
            if self.provider not in {"openai", "router", "ollama"}:
                raise ValueError("Local trained encoders use their own native input tokenizer")
            if type(self.token_limit) is not int or not 1 <= self.token_limit <= 100_000:
                raise ValueError("Embedding token limit must be an integer from 1 to 100000")
            if type(self.special_tokens) is not int or not 0 <= self.special_tokens <= 1024:
                raise ValueError("An explicit embedding counter requires verified special_tokens from 0 to 1024")
            object.__setattr__(self, "tokenizer", dict(self.tokenizer))
        validate_model_auth_mode(self.auth_mode)
        if self.base_url is not None:
            if not isinstance(self.base_url, str) or not self.base_url or len(self.base_url) > 2048:
                raise ValueError("Embedding endpoint must be bounded HTTP(S)")
            _utf8(self.base_url)
            parsed = urlsplit(self.base_url)
            if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username is not None
                    or parsed.password is not None or "?" in self.base_url or "#" in self.base_url
                    or any(ord(c) <= 32 or ord(c) == 127 for c in self.base_url)):
                raise ValueError("Embedding endpoint must be clean HTTP(S)")
            try:
                if parsed.port == 0:
                    raise ValueError("Invalid embedding port")
            except ValueError as exc:
                raise ValueError("Invalid embedding port") from exc
            object.__setattr__(self, "base_url", self.base_url.rstrip("/"))
        if self.base_url is not None and self.provider not in {"openai", "ollama"}:
            raise ValueError("Only direct HTTP embedding profiles accept an endpoint")
        if self.provider == "ollama" and self.base_url is None:
            raise ValueError("Native Ollama requires an explicit endpoint")
        if self.api_key_env is not None and (
            not isinstance(self.api_key_env, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]{0,119}", self.api_key_env)
        ):
            raise ValueError("Embedding credential must name a bounded environment variable")
        if self.api_key_env is not None and self.provider not in {"openai", "ollama"}:
            raise ValueError("This provider does not accept a direct credential")
        if self.auth_mode == "none":
            if self.provider not in {"openai", "ollama"} or self.api_key_env is not None:
                raise ValueError("Unauthenticated embeddings require a direct loopback provider without credentials")
            validate_local_no_auth_endpoint(self.base_url)
        if (not isinstance(self.space_id, str) or (self.space_id and not
                re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,159}", self.space_id))):
            raise ValueError("Embedding space ID must be a bounded deployment/version identifier")

    def public(self) -> dict:
        """Discovery intentionally omits endpoints and credential environment names."""
        return {key: value for key, value in asdict(self).items()
                if key not in {"base_url", "api_key_env", "tokenizer"}}


def parse_embedding_profiles(value) -> dict[str, EmbeddingProfile]:
    if isinstance(value, str):
        if len(value.encode("utf-8")) > 65_536:
            raise ValueError("Embedding profile registry exceeds 64 KiB")
        value = strict_json(value)
    if not isinstance(value, dict) or len(value) > 16:
        raise ValueError("Embedding profiles require at most 16 named entries")
    result = {}
    for alias, profile in value.items():
        if not isinstance(alias, str) or not PROFILE_ID.fullmatch(alias):
            raise ValueError("Invalid embedding profile alias")
        if isinstance(profile, EmbeddingProfile):
            result[alias] = profile
        else:
            if not isinstance(profile, dict):
                raise ValueError("Embedding profile must be an object")
            try:
                result[alias] = EmbeddingProfile(**profile)
            except TypeError as exc:
                raise ValueError("Unknown embedding profile field") from exc
    return result
