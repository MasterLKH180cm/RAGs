"""Default-on Agent Platform; unconfigured external Gateway fails closed."""
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
from urllib.parse import urlsplit


ALIAS = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")
MODEL = re.compile(r"[a-z][a-z0-9_-]{0,39}/[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,159}\Z")


def _boolean(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "true" if default else "false").lower()
    if raw not in {"true", "false"}:
        raise ValueError(f"{name} must be true or false")
    return raw == "true"


def _integer(name: str, default: int, lower: int, upper: int) -> int:
    try:
        raw = os.getenv(name, str(default))
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be {lower}..{upper}")
    return value


def _secret(name: str) -> str:
    inline = os.getenv(name, "")
    filename = os.getenv(name + "_FILE", "")
    if inline and filename:
        raise ValueError(f"Use {name} or {name}_FILE, not both")
    if filename:
        path = Path(filename)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{name}_FILE must be an existing regular secret file")
        if path.stat().st_size > 8192:
            raise ValueError(f"{name}_FILE is too large")
        return path.read_text("utf-8").strip()
    return inline


@dataclass(frozen=True)
class AgentPlatformConfig:
    router_enabled: bool = True
    agent_enabled: bool = True
    memory_enabled: bool = True
    gateway_url: str = ""
    gateway_key: str = field(default="", repr=False)
    models: dict[str, str] = field(default_factory=dict)
    allow_http: bool = False
    max_steps: int = 6
    max_tool_calls: int = 12
    max_runtime_seconds: int = 60
    max_output_tokens: int = 1024
    max_history_turns: int = 12
    memory_ttl_days: int = 30

    def __post_init__(self) -> None:
        if self.agent_enabled and not self.router_enabled:
            raise ValueError("Agent engine requires an enabled model router")
        if not isinstance(self.models, dict) or len(self.models) > 32 or any(
            not isinstance(k, str) or not ALIAS.fullmatch(k)
            or not isinstance(v, str) or not MODEL.fullmatch(v)
            for k, v in self.models.items()
        ):
            raise ValueError("Router models require operator-approved alias to provider/model mapping")
        # An empty operator configuration is a normal first-run state: keep
        # endpoints available but do not attempt inference or invent credentials.
        supplied = bool(self.gateway_url or self.gateway_key or self.models)
        if self.router_enabled and supplied:
            url = urlsplit(self.gateway_url)
            if (url.scheme not in {"http", "https"} or not url.hostname
                    or url.username or url.password or url.query or url.fragment
                    or (url.scheme == "http" and not self.allow_http)):
                raise ValueError("Router gateway must use HTTPS without credentials or URL query")
            if not self.gateway_key or len(self.gateway_key) > 4096 or not self.models:
                raise ValueError("Router requires gateway credentials and approved model aliases")
        for name, number, low, high in (
            ("max_steps", self.max_steps, 1, 8),
            ("max_tool_calls", self.max_tool_calls, 1, 16),
            ("max_runtime_seconds", self.max_runtime_seconds, 5, 90),
            ("max_output_tokens", self.max_output_tokens, 1, 4096),
            ("max_history_turns", self.max_history_turns, 1, 20),
            ("memory_ttl_days", self.memory_ttl_days, 1, 365),
        ):
            if type(number) is not int or not low <= number <= high:
                raise ValueError(f"Agent {name} must be {low}..{high}")

    @property
    def router_ready(self) -> bool:
        return bool(self.router_enabled and self.gateway_url and self.gateway_key and self.models)

    @classmethod
    def from_env(cls) -> "AgentPlatformConfig":
        enabled = _boolean("RAGS_ROUTER_ENABLED", True)
        agent = _boolean("RAGS_AGENT_ENABLED", True)
        memory = _boolean("RAGS_MEMORY_ENABLED", True)
        allow_http = _boolean("RAGS_ROUTER_ALLOW_HTTP")
        if allow_http and os.getenv("RAGS_ENV", "production") not in {"test", "development"}:
            raise ValueError("Plaintext router connections only allowed in development/test")
        mapping = os.getenv("RAGS_ROUTER_MODELS_JSON") or "{}"
        if len(mapping.encode("utf-8")) > 8192:
            raise ValueError("Router models config is too large")
        try:
            models = json.loads(mapping)
        except ValueError as exc:
            raise ValueError("RAGS_ROUTER_MODELS_JSON requires a JSON object") from exc
        if not isinstance(models, dict):
            raise ValueError("RAGS_ROUTER_MODELS_JSON requires a JSON object")
        return cls(
            router_enabled=enabled, agent_enabled=agent, memory_enabled=memory,
            gateway_url=os.getenv("RAGS_ROUTER_GATEWAY_URL", "").rstrip("/"),
            gateway_key=_secret("RAGS_ROUTER_API_KEY") if enabled else "",
            models=models, allow_http=allow_http,
            max_steps=_integer("RAGS_AGENT_MAX_STEPS", 6, 1, 8),
            max_tool_calls=_integer("RAGS_AGENT_MAX_TOOL_CALLS", 12, 1, 16),
            max_runtime_seconds=_integer("RAGS_AGENT_MAX_RUNTIME_SECONDS", 60, 5, 90),
            max_output_tokens=_integer("RAGS_ROUTER_MAX_OUTPUT_TOKENS", 1024, 1, 4096),
            max_history_turns=_integer("RAGS_AGENT_MAX_HISTORY_TURNS", 12, 1, 20),
            memory_ttl_days=_integer("RAGS_MEMORY_TTL_DAYS", 30, 1, 365),
        )
