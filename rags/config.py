"""Explicit settings; production fails closed without authentication."""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
import re
from urllib.parse import urlparse


@dataclass(frozen=True)
class Settings:
    db_path: str
    keys: dict[str, str]
    embedding_provider: str
    model_provider: str
    openai_key: str
    openai_base_url: str
    embedding_model: str
    chat_model: str
    anthropic_key: str
    anthropic_model: str
    tokenizer_config: dict = field(default_factory=lambda: {"backend": "unicode"})
    chunking_config: dict = field(default_factory=lambda: {"strategy": "structure", "max_tokens": 850, "overlap_tokens": 120})
    context_budget: int = 16_000
    output_budget: int = 1024
    envelope_budget: int = 256
    max_iterations: int = 3
    federated_sources: dict = field(default_factory=dict)
    fastembed_model: str = "BAAI/bge-small-en-v1.5"
    generation_tokenizer_config: dict = field(default_factory=lambda: {"backend": "unicode"})
    admission_policy: str = "compatibility"
    admission_profile: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.getenv("RAGS_ENV", "production").lower()
        try:
            keys = json.loads(os.getenv("RAGS_KEYS_JSON", "{}"))
        except json.JSONDecodeError as exc:
            raise ValueError("RAGS_KEYS_JSON must be a JSON token-to-tenant mapping") from exc
        if not isinstance(keys, dict) or any(
            not isinstance(k, str) or not k or not isinstance(v, str) or not v
            for k, v in keys.items()
        ):
            raise ValueError("RAGS_KEYS_JSON must map non-empty API keys to tenant names")
        if env in {"development", "test"} and not keys:
            keys = {os.getenv("RAGS_API_KEY", "local-development-key"): "local"}
        if not keys:
            raise ValueError("Configure RAGS_KEYS_JSON before starting in production")
        if env == "production" and any(len(key) < 24 for key in keys):
            raise ValueError("Production API keys must have at least 24 characters")
        embedding_provider = os.getenv("RAGS_EMBEDDING_PROVIDER", "local").lower()
        model_provider = os.getenv("RAGS_MODEL_PROVIDER", "extractive").lower()
        if embedding_provider not in {"local", "openai", "fastembed"}:
            raise ValueError("Unsupported embedding provider")
        if model_provider not in {"extractive", "openai", "anthropic"}:
            raise ValueError("Unsupported model provider")
        db_path = os.getenv("RAGS_DB_PATH", str(Path("data") / "rags.db"))
        openai_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        parsed_base = urlparse(openai_base)
        if parsed_base.scheme not in {"http", "https"} or not parsed_base.netloc or parsed_base.username or parsed_base.password or parsed_base.query or parsed_base.fragment:
            raise ValueError("OPENAI_BASE_URL must be an HTTP(S) endpoint without credentials or query parameters")
        def object_env(name: str, fallback: str) -> dict:
            try:
                value = json.loads(os.getenv(name, fallback))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{name} must be a JSON object") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{name} must be a JSON object")
            return value

        tokenizer = object_env("RAGS_TOKENIZER_JSON", '{"backend":"unicode"}')
        generation_default = json.dumps({"backend": "tiktoken", "model": os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")}) if model_provider == "openai" else '{"backend":"unicode"}'
        generation_tokenizer = object_env("RAGS_GENERATION_TOKENIZER_JSON", generation_default)
        chunking = object_env("RAGS_CHUNKING_JSON", '{"strategy":"structure","max_tokens":850,"overlap_tokens":120}')
        sources = object_env("RAGS_SOURCES_JSON", "{}")
        if len(sources) > 8:
            raise ValueError("Configure at most eight federated sources")
        for source_id, source in sources.items():
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,59}", source_id) or not isinstance(source, dict):
                raise ValueError("Federation requires named source objects")
            parsed = urlparse(source.get("url", ""))
            if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
                raise ValueError("Federated source requires an operator-configured HTTP(S) URL")
            tokens = source.get("tenant_keys", {})
            if not isinstance(tokens, dict) or not tokens or any(
                not isinstance(t, str) or not t or not isinstance(k, str) or not k
                for t, k in tokens.items()
            ):
                raise ValueError("Federated sources require tenant_keys mappings")
            if len(set(tokens.values())) != len(tokens):
                raise ValueError("Federated sources cannot share credentials between tenants")
        context_budget = int(os.getenv("RAGS_CONTEXT_BUDGET", "16000"))
        output_budget = int(os.getenv("RAGS_OUTPUT_BUDGET", "1024"))
        envelope_budget = int(os.getenv("RAGS_ENVELOPE_BUDGET", "256"))
        if not 512 <= context_budget <= 200_000 or not 1 <= output_budget <= 16_000:
            raise ValueError("Invalid context or output budget")
        if not 0 <= envelope_budget <= 4096 or output_budget + envelope_budget >= context_budget:
            raise ValueError("Output and envelope reserves must fit the context budget")
        iterations = int(os.getenv("RAGS_MAX_ITERATIONS", "3"))
        if not 1 <= iterations <= 3:
            raise ValueError("RAGS_MAX_ITERATIONS must be 1..3")
        admission_policy = os.getenv("RAGS_ADMISSION_POLICY", "compatibility")
        if admission_policy not in {"compatibility", "calibrated_required"}:
            raise ValueError("RAGS_ADMISSION_POLICY must be compatibility or calibrated_required")
        return cls(
            db_path=db_path,
            keys=keys,
            embedding_provider=embedding_provider,
            model_provider=model_provider,
            openai_key=os.getenv("OPENAI_API_KEY", ""),
            openai_base_url=openai_base,
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
            anthropic_key=os.getenv("ANTHROPIC_API_KEY", ""),
            anthropic_model=os.getenv("ANTHROPIC_CHAT_MODEL", "claude-sonnet-4-5"),
            tokenizer_config=tokenizer,
            chunking_config=chunking,
            context_budget=context_budget,
            output_budget=output_budget,
            envelope_budget=envelope_budget,
            max_iterations=iterations,
            federated_sources=sources,
            fastembed_model=os.getenv("RAGS_FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5"),
            generation_tokenizer_config=generation_tokenizer,
            admission_policy=admission_policy,
            admission_profile=os.getenv("RAGS_ADMISSION_PROFILE") or None,
        )
