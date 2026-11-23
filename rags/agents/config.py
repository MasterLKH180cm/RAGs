"""Agent, memory and router are default enabled; external calls require Gateway setup."""
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import re
import ssl
import stat
from urllib.parse import urlsplit

from rags.model_profiles import (
    chat_profile, strict_json, validate_local_no_auth_endpoint, validate_model_auth_mode,
)


ALIAS = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")
MODEL = re.compile(r"[a-z][a-z0-9_-]{0,39}/[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,159}\Z")
MAX_CA_BYTES = 256 * 1024


def _gateway_ca(filename: str) -> str:
    if not isinstance(filename, str) or len(filename) > 4096:
        raise ValueError("Router CA file must be a bounded path")
    if not filename:
        return ""
    try:
        # Open without following a final symlink, and without blocking on a
        # special file; validate the same descriptor that supplies the bytes.
        descriptor = os.open(filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as handle:
            metadata = os.fstat(handle.fileno())
            if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= MAX_CA_BYTES:
                raise ValueError("Router CA file must be a bounded, nonempty regular PEM file")
            content = handle.read(MAX_CA_BYTES + 1)
        if not content or len(content) > MAX_CA_BYTES:
            raise ValueError("Router CA file must be a bounded, nonempty regular PEM file")
        pem = content.decode("ascii")
        ssl.create_default_context(cadata=pem)
    except (OSError, UnicodeError) as exc:
        raise ValueError("Router CA file must be an existing regular nonsymlink PEM file") from exc
    return pem


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
    model_options: dict[str, dict] = field(default_factory=dict)
    gateway_auth_mode: str = "required"
    gateway_ca_file: str = ""
    gateway_ca_pem: str = field(default="", init=False, repr=False)
    gateway_ca_sha256: str = field(default="", init=False)

    def __post_init__(self) -> None:
        validate_model_auth_mode(self.gateway_auth_mode)
        ca_pem = _gateway_ca(self.gateway_ca_file)
        if ca_pem and urlsplit(self.gateway_url).scheme != "https":
            raise ValueError("Router CA file requires an explicit HTTPS gateway")
        object.__setattr__(self, "gateway_ca_pem", ca_pem)
        object.__setattr__(self, "gateway_ca_sha256",
                           hashlib.sha256(ca_pem.encode("ascii")).hexdigest() if ca_pem else "")
        if not isinstance(self.gateway_key, str) or len(self.gateway_key) > 4096:
            raise ValueError("Router gateway credential must be a bounded string")
        if self.agent_enabled and not self.router_enabled:
            raise ValueError("Agent engine requires an enabled model router")
        if not isinstance(self.models, dict) or len(self.models) > 32 or any(
            not isinstance(k, str) or not ALIAS.fullmatch(k)
            or not isinstance(v, str) or not MODEL.fullmatch(v)
            for k, v in self.models.items()
        ):
            raise ValueError("Router models require operator-approved alias to provider/model mapping")
        if not isinstance(self.model_options, dict) or set(self.model_options) - set(self.models):
            raise ValueError("Model options must name approved model aliases")
        for alias, options in self.model_options.items():
            if not isinstance(options, dict) or set(options) - {"protocol", "reasoning_effort"}:
                raise ValueError("Model options support protocol and reasoning_effort only")
            profile = chat_profile(self.models[alias], options.get("protocol", "auto"))
            profile.parameters(max_output_tokens=1, effort=options.get("reasoning_effort"))
        if self.gateway_auth_mode == "none":
            validate_local_no_auth_endpoint(self.gateway_url)
            if self.gateway_key:
                raise ValueError("Unauthenticated router cannot configure a gateway credential")
            if not self.models or any(
                self.model_options.get(alias, {}).get("protocol") != "openai"
                or self.model_options.get(alias, {}).get("reasoning_effort") is not None
                for alias in self.models
            ):
                raise ValueError("Unauthenticated router requires explicit OpenAI protocol without reasoning effort")
        # A default-on interface is not permission to invent an external model
        # endpoint or secret. Until configured, Gateway calls fail closed.
        supplied = bool(self.gateway_url or self.gateway_key or self.models)
        if self.router_enabled and supplied:
            url = urlsplit(self.gateway_url)
            if (url.scheme not in {"http", "https"} or not url.hostname
                    or url.username or url.password or url.query or url.fragment
                    or (url.scheme == "http" and not self.allow_http)):
                raise ValueError("Router gateway must use HTTPS without credentials or URL query")
            if (self.gateway_auth_mode == "required" and not self.gateway_key) or len(self.gateway_key) > 4096 or not self.models:
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
        return bool(self.router_enabled and self.gateway_url and self.models
                    and (self.gateway_auth_mode == "none" or self.gateway_key))

    @classmethod
    def from_env(cls) -> "AgentPlatformConfig":
        enabled = _boolean("RAGS_ROUTER_ENABLED", True)
        agent = _boolean("RAGS_AGENT_ENABLED", True)
        memory = _boolean("RAGS_MEMORY_ENABLED", True)
        auth_mode = validate_model_auth_mode(os.getenv("RAGS_ROUTER_AUTH_MODE", "required"))
        if auth_mode == "none" and any(os.getenv(name) for name in (
            "RAGS_ROUTER_API_KEY", "RAGS_ROUTER_API_KEY_FILE",
        )):
            raise ValueError("Unauthenticated router cannot configure a gateway credential")
        allow_http = _boolean("RAGS_ROUTER_ALLOW_HTTP")
        if allow_http and os.getenv("RAGS_ENV", "production") not in {"test", "development"}:
            raise ValueError("Plaintext router connections only allowed in development/test")
        mapping = os.getenv("RAGS_ROUTER_MODELS_JSON") or "{}"
        if len(mapping.encode("utf-8")) > 8192:
            raise ValueError("Router models config is too large")
        try:
            models = strict_json(mapping)
        except ValueError as exc:
            raise ValueError("RAGS_ROUTER_MODELS_JSON requires a JSON object") from exc
        if not isinstance(models, dict):
            raise ValueError("RAGS_ROUTER_MODELS_JSON requires a JSON object")
        options = os.getenv("RAGS_ROUTER_MODEL_OPTIONS_JSON") or "{}"
        if len(options.encode("utf-8")) > 8192:
            raise ValueError("Router model options config is too large")
        try:
            model_options = strict_json(options)
        except ValueError as exc:
            raise ValueError("RAGS_ROUTER_MODEL_OPTIONS_JSON requires a JSON object") from exc
        return cls(
            router_enabled=enabled, agent_enabled=agent, memory_enabled=memory,
            gateway_url=os.getenv("RAGS_ROUTER_GATEWAY_URL", "").rstrip("/"),
            gateway_key=_secret("RAGS_ROUTER_API_KEY") if enabled and auth_mode == "required" else "",
            gateway_auth_mode=auth_mode,
            gateway_ca_file=os.getenv("RAGS_ROUTER_CA_FILE", ""),
            models=models, model_options=model_options, allow_http=allow_http,
            max_steps=_integer("RAGS_AGENT_MAX_STEPS", 6, 1, 8),
            max_tool_calls=_integer("RAGS_AGENT_MAX_TOOL_CALLS", 12, 1, 16),
            max_runtime_seconds=_integer("RAGS_AGENT_MAX_RUNTIME_SECONDS", 60, 5, 90),
            max_output_tokens=_integer("RAGS_ROUTER_MAX_OUTPUT_TOKENS", 1024, 1, 4096),
            max_history_turns=_integer("RAGS_AGENT_MAX_HISTORY_TURNS", 12, 1, 20),
            memory_ttl_days=_integer("RAGS_MEMORY_TTL_DAYS", 30, 1, 365),
        )
