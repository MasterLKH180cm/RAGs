"""Explicit settings; production fails closed without authentication."""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
import re
from urllib.parse import urlparse

from rags.auth import OAuth2Config, ROLES, TENANT
from rags.chunking import ChunkConfig
from rags.features import FeatureFlags
from rags.tokenization import TokenizerConfig


def secret(name: str) -> str:
    value = os.getenv(name, "")
    filename = os.getenv(name + "_FILE")
    if filename:
        if value:
            raise ValueError(f"Choose {name} or {name}_FILE, not both")
        value = Path(filename).read_text(encoding="utf-8").strip()
    return value


def endpoint(value: str, name: str) -> str:
    value = value.rstrip("/")
    parsed = urlparse(value)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment):
        raise ValueError(f"{name} must be an HTTP(S) endpoint without credentials or query parameters")
    return value


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
    features: FeatureFlags = field(default_factory=FeatureFlags)
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    generation_base_url: str | None = None
    generation_api_key: str | None = None
    auth_mode: str = "api_key"
    key_roles: dict[str, tuple[str, ...]] = field(default_factory=dict)
    oauth2: OAuth2Config | None = None

    @property
    def embedding_endpoint(self) -> str:
        return (self.embedding_base_url or self.openai_base_url) if self.embedding_provider == "openai" else "local"

    @property
    def embedding_credential(self) -> str:
        return self.embedding_api_key if self.embedding_api_key is not None else self.openai_key

    @property
    def generation_endpoint(self) -> str:
        return self.generation_base_url or (self.openai_base_url if self.model_provider == "openai"
                                           else "https://api.anthropic.com/v1")

    @property
    def generation_credential(self) -> str:
        return self.generation_api_key if self.generation_api_key is not None else (
            self.openai_key if self.model_provider == "openai" else self.anthropic_key)

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.getenv("RAGS_ENV", "production").lower()
        try:
            keys = json.loads(os.getenv("RAGS_KEYS_JSON") or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError("RAGS_KEYS_JSON must be a JSON token-to-tenant mapping") from exc
        if not isinstance(keys, dict) or any(
            not isinstance(k, str) or not k or not isinstance(v, str) or not TENANT.fullmatch(v)
            for k, v in keys.items()
        ):
            raise ValueError("RAGS_KEYS_JSON requires API keys and bounded tenant identifiers")
        auth_mode = os.getenv("RAGS_AUTH_MODE", "api_key").lower()
        if auth_mode not in {"api_key", "oauth2_introspection"}:
            raise ValueError("RAGS_AUTH_MODE must be api_key or oauth2_introspection")
        key_roles = {}
        roles_json = os.getenv("RAGS_KEY_ROLES_JSON", "")
        if roles_json:
            try:
                parsed_roles = json.loads(roles_json)
            except json.JSONDecodeError as exc:
                raise ValueError("RAGS_KEY_ROLES_JSON must be a JSON role mapping") from exc
            if (not isinstance(parsed_roles, dict) or set(parsed_roles) != set(keys)
                    or any(not isinstance(v, list) or not v or len(v) > len(ROLES)
                           or any(not isinstance(role, str) or role not in ROLES for role in v)
                           or len(set(v)) != len(v) for v in parsed_roles.values())):
                raise ValueError("RAGS_KEY_ROLES_JSON must cover all configured keys with known roles")
            key_roles = {key: tuple(roles) for key, roles in parsed_roles.items()}
        oauth2 = None
        if auth_mode == "api_key":
            if env in {"development", "test"} and not keys:
                keys = {os.getenv("RAGS_API_KEY", "local-development-key"): "local"}
            if not keys:
                raise ValueError("Configure RAGS_KEYS_JSON before starting in production")
            if env == "production" and any(len(key) < 24 for key in keys):
                raise ValueError("Production API keys must have at least 24 characters")
        else:
            if keys or roles_json:
                raise ValueError("OAuth2 mode cannot use static API keys or key roles")
            oauth2 = OAuth2Config(
                introspection_url=os.getenv("RAGS_OAUTH2_INTROSPECTION_URL", ""),
                issuer=os.getenv("RAGS_OAUTH2_ISSUER", ""),
                audience=os.getenv("RAGS_OAUTH2_AUDIENCE", ""),
                client_id=os.getenv("RAGS_OAUTH2_CLIENT_ID", ""),
                client_secret=secret("RAGS_OAUTH2_CLIENT_SECRET"),
                tenant_claim=os.getenv("RAGS_OAUTH2_TENANT_CLAIM", "tenant_id"),
                timeout_seconds=float(os.getenv("RAGS_OAUTH2_TIMEOUT_SECONDS", "3")),
            )
        embedding_provider = os.getenv("RAGS_EMBEDDING_PROVIDER", "local").lower()
        model_provider = os.getenv("RAGS_MODEL_PROVIDER", "extractive").lower()
        if embedding_provider not in {"local", "openai", "fastembed"}:
            raise ValueError("Unsupported embedding provider")
        if model_provider not in {"extractive", "openai", "anthropic"}:
            raise ValueError("Unsupported model provider")
        database_url = os.getenv("RAGS_DATABASE_URL")
        if database_url and not database_url.startswith(("postgresql://", "postgres://")):
            raise ValueError("RAGS_DATABASE_URL must select PostgreSQL")
        db_path = database_url or os.getenv("RAGS_DB_PATH") or str(Path("data") / "rags.db")
        openai_base = endpoint(os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1", "OPENAI_BASE_URL")
        embedding_base = os.getenv("RAGS_EMBEDDING_BASE_URL")
        generation_base = os.getenv("RAGS_GENERATION_BASE_URL") or (
            os.getenv("ANTHROPIC_BASE_URL") if model_provider == "anthropic" else None)
        if embedding_base:
            embedding_base = endpoint(embedding_base, "RAGS_EMBEDDING_BASE_URL")
        if generation_base:
            generation_base = endpoint(generation_base, "RAGS_GENERATION_BASE_URL")
        chat_model = os.getenv("RAGS_GENERATION_MODEL") or os.getenv("OPENAI_CHAT_MODEL") or "gpt-4.1-mini"
        features = FeatureFlags.from_env()
        def object_env(name: str, fallback: str) -> dict:
            try:
                value = json.loads(os.getenv(name) or fallback)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{name} must be a JSON object") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{name} must be a JSON object")
            return value

        tokenizer = object_env("RAGS_TOKENIZER_JSON", '{"backend":"unicode"}')
        generation_default = json.dumps({"backend": "tiktoken", "model": chat_model}) if model_provider == "openai" else '{"backend":"unicode"}'
        generation_tokenizer = object_env("RAGS_GENERATION_TOKENIZER_JSON", generation_default)
        chunking = object_env("RAGS_CHUNKING_JSON", '{"strategy":"structure","max_tokens":850,"overlap_tokens":120}')
        try:
            TokenizerConfig(**tokenizer)
            TokenizerConfig(**generation_tokenizer)
            ChunkConfig(**chunking)
        except TypeError as exc:
            raise ValueError("Unknown tokenizer or chunking configuration field") from exc
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
        if admission_policy == "calibrated_required" and not features.enabled("admission_scoring"):
            raise ValueError("Required evidence policy needs admission_scoring")
        return cls(
            db_path=db_path,
            keys=keys,
            embedding_provider=embedding_provider,
            model_provider=model_provider,
            openai_key=secret("OPENAI_API_KEY"),
            openai_base_url=openai_base,
            embedding_model=os.getenv("RAGS_EMBEDDING_MODEL") or os.getenv("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small",
            chat_model=chat_model,
            anthropic_key=secret("ANTHROPIC_API_KEY"),
            anthropic_model=os.getenv("RAGS_GENERATION_MODEL") or os.getenv("ANTHROPIC_CHAT_MODEL") or "claude-sonnet-4-5",
            tokenizer_config=tokenizer,
            chunking_config=chunking,
            context_budget=context_budget,
            output_budget=output_budget,
            envelope_budget=envelope_budget,
            max_iterations=iterations,
            federated_sources=sources,
            fastembed_model=os.getenv("RAGS_EMBEDDING_MODEL") or os.getenv("RAGS_FASTEMBED_MODEL") or "BAAI/bge-small-en-v1.5",
            generation_tokenizer_config=generation_tokenizer,
            admission_policy=admission_policy,
            admission_profile=os.getenv("RAGS_ADMISSION_PROFILE") or None,
            features=features,
            embedding_base_url=embedding_base,
            embedding_api_key=secret("RAGS_EMBEDDING_API_KEY") or None,
            generation_base_url=generation_base,
            generation_api_key=secret("RAGS_GENERATION_API_KEY") or None,
            auth_mode=auth_mode,
            key_roles=key_roles,
            oauth2=oauth2,
        )
