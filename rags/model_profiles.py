"""Operator-selected chat protocols and pure local authentication validation.

An OpenAI-compatible wire format does not imply identical request semantics.
Keep provider capabilities explicit and validate structured output locally.
"""
from dataclasses import dataclass
from ipaddress import ip_address
import json
import math
from urllib.parse import urlsplit


INKLING_MODEL = "thinkingmachines/Inkling-2-Mini"
TINKER_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}


def validate_model_auth_mode(value: str) -> str:
    if not isinstance(value, str) or value not in {"required", "none"}:
        raise ValueError("Model authentication must be required or none")
    return value


def validate_local_no_auth_endpoint(value: str) -> None:
    """No-auth inference is an explicit literal-loopback operation, without DNS."""
    if (not isinstance(value, str) or not value or len(value) > 2048
            or any(ord(char) <= 32 or ord(char) == 127 for char in value)):
        raise ValueError("Unauthenticated inference requires an explicit clean loopback endpoint")
    try:
        parsed = urlsplit(value)
        host, port = parsed.hostname, parsed.port
        address = ip_address(host or "")
        local = bool("%" not in (host or "") and address.is_loopback
                     and (address.version == 4 or str(address) == "::1"))
    except ValueError as exc:
        raise ValueError("Unauthenticated inference requires a literal loopback IP") from exc
    if (parsed.scheme not in {"http", "https"} or not local
            or parsed.username is not None or parsed.password is not None
            or "?" in value or "#" in value or port == 0):
        raise ValueError("Unauthenticated inference requires an explicit clean loopback endpoint")


def reasoning_effort(value):
    if value is None:
        return None
    if isinstance(value, str) and value in EFFORTS:
        return value
    if type(value) in (int, float) and 0 <= value <= .99 and math.isfinite(value):
        return value
    raise ValueError("Reasoning effort must be a supported name or a number from 0 to 0.99")


def effort_from_env(raw: str | None):
    if raw is None or not raw.strip():
        return None
    if raw in EFFORTS:
        return raw
    try:
        return reasoning_effort(float(raw))
    except (ValueError, OverflowError) as exc:
        raise ValueError("Invalid configured reasoning effort") from exc


def _reject_constant(_value):
    raise ValueError("Nonfinite JSON numbers are not permitted")


def _finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Nonfinite JSON numbers are not permitted")
    return result


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON fields are not permitted")
        result[key] = value
    return result


def strict_json(raw: str | bytes):
    """Reject ambiguous duplicate fields, NaN and infinities from untrusted JSON."""
    return json.loads(raw, parse_constant=_reject_constant, parse_float=_finite_float,
                      object_pairs_hook=_unique_object)


@dataclass(frozen=True)
class ChatProfile:
    protocol: str
    token_limit_field: str
    supports_embeddings: bool
    supports_json_schema: bool

    def parameters(self, *, max_output_tokens: int, effort=None) -> dict:
        if type(max_output_tokens) is not int or not 1 <= max_output_tokens <= 16_000:
            raise ValueError("Invalid model output token limit")
        result = {self.token_limit_field: max_output_tokens, "temperature": 0}
        selected = reasoning_effort(effort)
        if selected is not None:
            if self.protocol != "tinker":
                raise ValueError("Reasoning effort requires an explicitly supported protocol")
            result["reasoning_effort"] = selected
        return result


def chat_profile(model: str, protocol: str = "auto") -> ChatProfile:
    if not isinstance(protocol, str) or protocol not in {"auto", "openai", "tinker"}:
        raise ValueError("Chat protocol must be auto, openai or tinker")
    selected = "tinker" if protocol == "auto" and model == INKLING_MODEL else protocol
    if selected == "tinker":
        return ChatProfile("tinker", "max_completion_tokens", False, False)
    # Existing compatible endpoints retain the previously supported field.
    # JSON-schema capability is unknown unless a future profile verifies it.
    return ChatProfile("openai", "max_tokens", model != INKLING_MODEL, False)


def normalized_usage(payload: dict) -> dict[str, int]:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return {}
    result = {}
    for key in ("input_tokens", "output_tokens", "prompt_tokens", "completion_tokens", "total_tokens",
                "cache_creation_input_tokens", "cache_read_input_tokens", "cached_prompt_tokens", "reasoning_tokens"):
        value = usage.get(key)
        if type(value) is int and 0 <= value <= 10**12:
            result[key] = value
    for outer, inner, target in (("prompt_tokens_details", "cached_tokens", "cached_prompt_tokens"),
                                  ("completion_tokens_details", "reasoning_tokens", "reasoning_tokens")):
        details = usage.get(outer)
        value = details.get(inner) if isinstance(details, dict) else None
        if type(value) is int and 0 <= value <= 10**12:
            result[target] = value
    return result
