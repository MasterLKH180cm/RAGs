"""Explicit settings; production fails closed without authentication."""
import json
import os
from dataclasses import dataclass
from pathlib import Path


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
        if embedding_provider not in {"local", "openai"}:
            raise ValueError("Unsupported embedding provider")
        if model_provider not in {"extractive", "openai", "anthropic"}:
            raise ValueError("Unsupported model provider")
        db_path = os.getenv("RAGS_DB_PATH", str(Path("data") / "rags.db"))
        return cls(
            db_path=db_path,
            keys=keys,
            embedding_provider=embedding_provider,
            model_provider=model_provider,
            openai_key=os.getenv("OPENAI_API_KEY", ""),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
            anthropic_key=os.getenv("ANTHROPIC_API_KEY", ""),
            anthropic_model=os.getenv("ANTHROPIC_CHAT_MODEL", "claude-sonnet-4-5"),
        )
