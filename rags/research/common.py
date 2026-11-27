"""Request-local validation and bounded provider calls for upstream adapters."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
from pathlib import Path
from typing import Protocol

from rags.features import FeatureFlags, strategy_features
from rags.registry import CapabilityError
from rags.tokenization import resolve_tokenizer


class ProviderProtocol(Protocol):
    settings: object
    has_generation: bool
    embedding_model: str
    model_identity: str

    def embedding(self, text: str) -> list[float]: ...

    def query_embedding(self, text: str) -> list[float]: ...

    def complete(self, system: str, user: str) -> str | None: ...


def validate_request(question: str, passages: list[dict], top_k: int) -> dict[str, str]:
    if not isinstance(question, str) or not question.strip() or len(question) > 2000:
        raise CapabilityError("invalid_research_input", "Invalid research question", 422)
    if type(top_k) is not int or not 1 <= top_k <= 64:
        raise CapabilityError("invalid_research_input", "Research top_k must be 1..64", 422)
    if not isinstance(passages, list) or not 1 <= len(passages) <= 64:
        raise CapabilityError("invalid_research_input", "Research requires 1..64 passages", 422)
    sources = {}
    total = 0
    for passage in passages:
        if not isinstance(passage, dict):
            raise CapabilityError("invalid_research_input", "Invalid research passage", 422)
        identifier, content = passage.get("id"), passage.get("text")
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,200}", identifier):
            raise CapabilityError("invalid_research_input", "Invalid research passage ID", 422)
        if identifier in sources or not isinstance(content, str) or not content.strip():
            raise CapabilityError("invalid_research_input", "Duplicate or empty research passage", 422)
        try:
            content.encode("utf-8")
        except UnicodeError as exc:
            raise CapabilityError("invalid_research_input", "Invalid passage encoding", 422) from exc
        total += len(content)
        if total > 200_000:
            raise CapabilityError("invalid_research_input", "Research source character limit exceeded", 422)
        sources[identifier] = content
    return sources


def private_directory(workdir: Path, name: str) -> Path:
    root = Path(workdir)
    if not root.is_dir():
        raise CapabilityError("invalid_research_input", "Research work directory is unavailable", 422)
    target = root / name
    try:
        target.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise CapabilityError("invalid_research_input", "Research index directory must be fresh", 422) from exc
    return target


def verify_source(package: str, expected: dict[str, str]) -> Path:
    """Verify pinned algorithm files before importing an optional upstream package."""
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.origin:
        raise CapabilityError("requires_backend", "Pinned upstream package is not installed")
    root = Path(spec.origin).parent
    try:
        if any(hashlib.sha256((root / name).read_bytes()).hexdigest() != digest
               for name, digest in expected.items()):
            raise CapabilityError("requires_backend", "Upstream source differs from the pinned revision")
    except OSError as exc:
        raise CapabilityError("requires_backend", "Pinned upstream source files are unavailable") from exc
    return root


def disable_encoder_telemetry() -> None:
    """Compatibility facade for the shared encoder lifecycle policy."""
    from rags.embedding_runtime import disable_encoder_telemetry as disable
    disable()


def fastembed_token_window(encoder):
    """Compatibility facade for research callers; use the shared runtime contract."""
    from rags.embedding_runtime import fastembed_window
    window = fastembed_window(encoder)
    return window.count, window.limit


class ProviderBridge:
    """Per-request counters/caches; never stores source data outside this instance."""

    def __init__(self, provider: ProviderProtocol, *, engine: str | None = None):
        self.provider = provider
        settings = getattr(provider, "settings", None)
        configured_features = getattr(settings, "features", None)
        self.features = configured_features if configured_features is not None else FeatureFlags.from_env()
        if not isinstance(self.features, FeatureFlags):
            raise ValueError("Research features must be validated operator permissions")
        self.features.require("research", "model_generation")
        if engine is not None:
            self.features.require(*strategy_features(engine))
        self.embedding_model = getattr(provider, "embedding_model", "")
        self.generation_provider = getattr(provider, "model_identity", "")
        embedding_provider = getattr(settings, "embedding_provider", "local")
        if (embedding_provider not in {"fastembed", "openai", "router", "sentence_transformers", "ollama"}
                or not isinstance(self.embedding_model, str) or not self.embedding_model
                or len(self.embedding_model) > 200 or "hash" in self.embedding_model.lower()):
            raise CapabilityError("requires_provider", "Research requires a configured trained embedding provider")
        self.features.require("embedding." + embedding_provider)
        tokenizer_config = getattr(settings, "generation_tokenizer_config", {"backend": "unicode"})
        self.features.require("tokenizer." + tokenizer_config.get("backend", "unicode"))
        if (getattr(provider, "has_generation", False) is not True
                or not isinstance(self.generation_provider, str) or not self.generation_provider
                or len(self.generation_provider) > 200 or self.generation_provider.startswith("extractive")):
            raise CapabilityError("requires_provider", "Research requires a configured generation provider")
        self.tokenizer = resolve_tokenizer(tokenizer_config)
        self.input_budget = (getattr(settings, "context_budget", 16000)
                             - getattr(settings, "output_budget", 1024)
                             - getattr(settings, "envelope_budget", 256))
        self.embedding_calls = 0
        self.query_embedding_calls = 0
        self.generation_calls = 0
        self.dimension: int | None = None
        self._embeddings: dict[tuple[bool, str], list[float]] = {}
        self._embedding_counter = None
        self.embedding_token_limit = None
        self.embedding_window_policy = "uninitialized"
        if getattr(settings, "embedding_provider", None) == "fastembed":
            disable_encoder_telemetry()

    def _prepare_embedding_counter(self):
        # The optional explicit callback is used by injected providers, including
        # tests. The production worker constructs only the built-in Providers.
        counter = getattr(self.provider, "embedding_token_count", None)
        if callable(counter):
            limit = getattr(self.provider, "embedding_token_limit", None)
            if type(limit) is not int or not 1 <= limit <= 100_000:
                raise CapabilityError("requires_provider", "Embedding token limit is unavailable")
            self._embedding_counter, self.embedding_token_limit = counter, limit
            self.embedding_window_policy = getattr(self.provider, "embedding_window_policy", "provider_exact_counter_reject_oversize")
        elif self.provider.settings.embedding_provider == "fastembed":
            self.features.require("embedding.fastembed")
            from rags.providers import _semantic_encoder
            name = getattr(self.provider.settings, "fastembed_model", None)
            if not isinstance(name, str) or not name:
                raise CapabilityError("requires_provider", "Embedding model configuration is unavailable")
            encoder = _semantic_encoder(name, os.getenv("RAGS_MODEL_CACHE") or None)
            self._embedding_counter, self.embedding_token_limit = fastembed_token_window(encoder)
            self.embedding_window_policy = "native_tokenizer_reject_oversize"
        else:
            self.features.require("tokenizer.tiktoken")
            name = getattr(self.provider.settings, "embedding_model", None)
            if self.provider.settings.embedding_provider == "router":
                router = self.provider.model_router
                resolved = router.config.models.get(name, "")
                # The operator's explicit provider/model mapping is the token
                # window contract. Generic gateway aliases cannot establish it.
                name = resolved.removeprefix("openai/") if resolved.startswith("openai/") else ""
            if name not in {"text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"}:
                raise CapabilityError("requires_provider", "Research needs an exact token counter for this embedding model")
            tokenizer = resolve_tokenizer({"backend": "tiktoken", "model": name})
            self._embedding_counter = lambda text, query=False: tokenizer.count(text)
            self.embedding_token_limit = 8191
            self.embedding_window_policy = "cl100k_8191_reject_oversize"

    def embed(self, text: str, *, query: bool = False) -> list[float]:
        if not isinstance(text, str) or not text.strip() or len(text) > 200_000:
            raise CapabilityError("invalid_model_input", "Invalid research embedding input", 422)
        key = (query, text)
        if key in self._embeddings:
            return self._embeddings[key]
        if self._embedding_counter is None:
            self._prepare_embedding_counter()
        count = self._embedding_counter(text, query=query)
        if type(count) is not int or count < 0:
            raise CapabilityError("invalid_model_output", "Invalid embedding token count", 502)
        if count > self.embedding_token_limit:
            raise CapabilityError("context_budget_exceeded", "Research embedding input exceeds the model token window", 422)
        if self.embedding_calls >= 4096:
            raise CapabilityError("research_limit_exceeded", "Research embedding call limit reached", 422)
        self.embedding_calls += 1
        self.query_embedding_calls += int(query)
        value = self.provider.query_embedding(text) if query else self.provider.embedding(text)
        if (not isinstance(value, (list, tuple)) or not 1 <= len(value) <= 16384
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)):
            raise CapabilityError("invalid_embedding", "Research provider returned invalid vectors", 502)
        if self.dimension is not None and len(value) != self.dimension:
            raise CapabilityError("invalid_embedding", "Research embedding dimensions changed", 502)
        scale = max(abs(v) for v in value)
        if scale == 0:
            raise CapabilityError("invalid_embedding", "Research provider returned a zero vector", 502)
        scaled = [v / scale for v in value]
        norm = math.sqrt(sum(v * v for v in scaled))
        normalized = [v / norm for v in scaled]
        self.dimension = len(value)
        self._embeddings[key] = normalized
        return normalized

    def complete(self, prompt: str, *, system: str = "", history: list | None = None) -> str:
        if not isinstance(prompt, str) or not isinstance(system, str):
            raise CapabilityError("invalid_model_input", "Invalid research generation input", 422)
        if history:
            # Preserve complete history objects; never slice serialized JSON.
            prompt = "Previous conversation data:\n" + json.dumps(history, ensure_ascii=False) + "\nCurrent request:\n" + prompt
        if len(system) + len(prompt) > 200_000 or self.tokenizer.count(system + prompt) > self.input_budget:
            raise CapabilityError("context_budget_exceeded", "Research model input exceeds its configured budget", 422)
        if self.generation_calls >= 256:
            raise CapabilityError("research_limit_exceeded", "Research generation call limit reached", 422)
        self.generation_calls += 1
        result = self.provider.complete(system, prompt)
        if not isinstance(result, str) or not result.strip() or len(result) > 60_000:
            raise CapabilityError("invalid_model_output", "Research generation returned invalid text", 502)
        return result

    def diagnostics(self) -> dict:
        return {"embedding_model": self.embedding_model, "generation_provider": self.generation_provider,
                "embedding_dimensions": self.dimension, "embedding_calls": self.embedding_calls,
                "embedding_token_limit": self.embedding_token_limit,
                "embedding_window_policy": self.embedding_window_policy,
                "query_embedding_calls": self.query_embedding_calls,
                "generation_calls": self.generation_calls}
