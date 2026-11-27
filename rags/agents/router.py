"""Typed, provider-neutral chat/tool interface through an operator LiteLLM Gateway.

The Gateway supports >40 upstream providers; this adapter only claims an
OpenAI-compatible HTTP contract. Exact model aliases are operator-approved.
"""
import json
import math
import re
import ssl
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from rags.agents.config import AgentPlatformConfig
from rags.model_profiles import chat_profile, normalized_usage, strict_json
from rags.provider_transport import ResponseRejected, bounded_post
from rags.registry import CapabilityError


TOOL_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,63}\Z")
CALL_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
MAX_RESPONSE_BYTES = 512 * 1024
MAX_CONTEXT_BYTES = 120 * 1024


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: str
    content: str = Field(default="", max_length=30000)
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict] | None = None
    # Upstream reasoning is replayed only to the model, never to a user or log.
    reasoning_content: str | None = Field(default=None, max_length=30000, repr=False)

    @field_validator("role")
    @classmethod
    def valid_role(cls, value):
        if value not in {"system", "user", "assistant", "tool"}:
            raise ValueError("Invalid chat role")
        return value


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    name: str
    arguments: dict[str, Any]

    @field_validator("id")
    @classmethod
    def call_id(cls, value):
        if not CALL_ID.fullmatch(value):
            raise ValueError("Invalid tool call ID")
        return value

    @field_validator("name")
    @classmethod
    def tool_name(cls, value):
        if not TOOL_NAME.fullmatch(value):
            raise ValueError("Invalid tool name")
        return value


class ModelReply(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: str = Field(default="", max_length=30000)
    tool_calls: list[ToolCall] = Field(default_factory=list, max_length=8)
    finish_reason: str
    model: str
    usage: dict[str, int] = Field(default_factory=dict)
    reasoning_content: str | None = Field(default=None, max_length=30000, repr=False, exclude=True)


class EmbeddingReply(BaseModel):
    model_config = ConfigDict(extra="forbid")
    vector: list[float] = Field(min_length=1, max_length=16000)
    model: str
    usage: dict[str, int] = Field(default_factory=dict)


class ModelRouter:
    def __init__(self, config: AgentPlatformConfig):
        if not config.router_enabled:
            raise CapabilityError("feature_disabled", "Model router is disabled", 403)
        self.config = config
        self._verify = (ssl.create_default_context(cadata=config.gateway_ca_pem)
                        if config.gateway_ca_pem else True)

    def _client(self, timeout_seconds: float = 30.) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds, connect=min(5., timeout_seconds)),
                                 follow_redirects=False, trust_env=False, verify=self._verify)

    def _post(self, path: str, payload: dict, timeout_seconds: float) -> httpx.Response:
        return bounded_post(
            self.config.gateway_url.rstrip("/") + path,
            headers=self._headers(), json=payload, timeout=timeout_seconds,
            max_request_bytes=MAX_CONTEXT_BYTES + 4096,
            max_response_bytes=MAX_RESPONSE_BYTES, trust_env=False,
            client_factory=lambda **options: self._client(options["timeout"]),
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "Accept-Encoding": "identity"}
        if self.config.gateway_auth_mode == "required":
            headers["Authorization"] = "Bearer " + self.config.gateway_key
        return headers

    def embedding(self, *, model: str, text: str, dimensions: int | None = None,
                  input_type: str | None = None, timeout_seconds: float = 30.) -> EmbeddingReply:
        """One OpenAI-compatible embeddings endpoint for approved Gateway models."""
        if not self.config.router_ready:
            raise CapabilityError("router_unconfigured", "Model gateway is not configured", 503)
        selected = self.config.models.get(model)
        if selected is None:
            raise CapabilityError("model_not_allowed", "Embedding model alias is not approved", 403)
        profile = chat_profile(selected, self.config.model_options.get(model, {}).get("protocol", "auto"))
        if not profile.supports_embeddings:
            raise CapabilityError("unsupported_model_operation", "Selected model does not provide embeddings", 422)
        if not isinstance(text, str) or not text or len(text.encode("utf-8")) > MAX_CONTEXT_BYTES:
            raise ValueError("Embedding input exceeds the configured length limit")
        if type(timeout_seconds) not in (int, float) or not 0.05 <= timeout_seconds <= 90.:
            raise ValueError("Embedding timeout must be 0.05..90 seconds")
        if dimensions is not None and (type(dimensions) is not int or not 1 <= dimensions <= 16_384):
            raise ValueError("Embedding dimensions must be an integer from 1 to 16384")
        if input_type is not None and (not isinstance(input_type, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./@+-]{0,159}", input_type)):
            raise ValueError("Embedding input_type must be a bounded identifier")
        body = {"model": selected, "input": text}
        if dimensions is not None:
            body["dimensions"] = dimensions
        if input_type is not None:
            body["input_type"] = input_type
        try:
            response = self._post("/embeddings", body, timeout_seconds)
            if response.status_code != 200:
                raise CapabilityError("embedding_gateway_failed", "Gateway rejected the embedding request", 502)
            payload = strict_json(response.content)
        except CapabilityError:
            raise
        except (TimeoutError, httpx.TimeoutException):
            raise CapabilityError("model_timeout", "Embedding deadline exceeded", 504) from None
        except ResponseRejected as exc:
            raise CapabilityError("invalid_embedding", str(exc), 502) from None
        except (httpx.HTTPError, ValueError, UnicodeError, RecursionError) as exc:
            raise CapabilityError("embedding_gateway_failed",
                                  "Gateway embedding request failed", 502) from exc
        try:
            data = payload["data"]
            if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
                raise ValueError("Expected one embedding result")
            vector = data[0]["embedding"]
            if not isinstance(vector, list) or not 1 <= len(vector) <= 16384 or any(
                type(value) not in (int, float) or not math.isfinite(value) for value in vector
            ):
                raise ValueError("Invalid embedding vector")
            if dimensions is not None and len(vector) != dimensions:
                raise ValueError("Embedding provider ignored requested dimensions")
            safe_usage = normalized_usage(payload)
            return EmbeddingReply(vector=vector, model=selected, usage=safe_usage)
        except (ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError) as exc:
            raise CapabilityError("invalid_embedding", "Invalid Gateway embedding result", 502) from exc

    def complete(self, *, model: str, messages: list[dict], tools: list[dict] | None = None,
                 max_output_tokens: int | None = None, timeout_seconds: float = 30.) -> ModelReply:
        if not self.config.router_ready:
            raise CapabilityError("router_unconfigured",
                                  "Model gateway is not yet configured; use agent doctor", 503)
        selected = self.config.models.get(model)
        if selected is None:
            raise CapabilityError("model_not_allowed", "Selected model alias is not configured", 403)
        # Supported agent bounds: initial system/goal + up to 20 history
        # turns + 8 assistant steps + up to 16 tool results, including one
        # observation context block. Reserve slack for interrupted runs.
        if not isinstance(messages, list) or not 1 <= len(messages) <= 64:
            raise ValueError("Messages must contain 1..64 turns")
        normalized = [Message.model_validate(msg).model_dump(exclude_none=True) for msg in messages]
        if len(json.dumps(normalized, ensure_ascii=False).encode("utf-8")) > MAX_CONTEXT_BYTES:
            raise CapabilityError("context_budget_exceeded", "Router message budget exceeded", 422)
        if tools is None:
            tools = []
        if not isinstance(tools, list) or len(tools) > 12:
            raise ValueError("Tool definitions must be a bounded list")
        names = set()
        for item in tools:
            if (not isinstance(item, dict) or item.get("type") != "function"
                    or not isinstance(item.get("function"), dict)
                    or not TOOL_NAME.fullmatch(str(item["function"].get("name", "")))):
                raise ValueError("Invalid tool definition")
            name = item["function"]["name"]
            if name in names:
                raise ValueError("Duplicate tool declaration")
            names.add(name)
        tokens = self.config.max_output_tokens if max_output_tokens is None else max_output_tokens
        if type(tokens) is not int or not 1 <= tokens <= self.config.max_output_tokens:
            raise ValueError("Invalid model output token limit")
        options = self.config.model_options.get(model, {})
        profile = chat_profile(selected, options.get("protocol", "auto"))
        request = {"model": selected, "messages": normalized,
                   **profile.parameters(max_output_tokens=tokens, effort=options.get("reasoning_effort"))}
        if tools:
            request["tools"] = tools
            request["tool_choice"] = "auto"
        try:
            request_size = len(json.dumps(request, ensure_ascii=False, allow_nan=False).encode("utf-8"))
        except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
            raise ValueError("Model request must contain valid JSON values") from exc
        if request_size > MAX_CONTEXT_BYTES:
            raise CapabilityError("context_budget_exceeded", "Router request budget exceeded", 422)
        if type(timeout_seconds) not in (int, float) or not 0.05 <= timeout_seconds <= 90.:
            raise ValueError("Model request timeout must be 0.05..90 seconds")
        try:
            response = self._post("/chat/completions", request, timeout_seconds)
            if response.status_code != 200:
                raise CapabilityError("model_gateway_failed", "Model gateway did not complete the request", 502)
            data = strict_json(response.content)
        except CapabilityError:
            raise
        except (TimeoutError, httpx.TimeoutException):
            raise CapabilityError("model_timeout", "Model deadline exceeded", 504) from None
        except ResponseRejected as exc:
            raise CapabilityError("invalid_model_output", str(exc), 502) from None
        except (httpx.HTTPError, UnicodeError, ValueError, RecursionError) as exc:
            raise CapabilityError("model_gateway_failed",
                                  "Model gateway is unavailable or returned invalid JSON", 502) from exc
        try:
            choices = data["choices"]
            if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
                raise ValueError("Expected one completion result")
            message = choices[0]["message"]
            if not isinstance(message, dict):
                raise ValueError("Gateway message must be an object")
            if message.get("role", "assistant") != "assistant":
                raise ValueError("Gateway reply must be an assistant message")
            raw_calls = message.get("tool_calls")
            if raw_calls is None:
                raw_calls = []
            if not isinstance(raw_calls, list) or len(raw_calls) > 8:
                raise ValueError("Unexpected tool calls")
            calls = []
            seen = set()
            for item in raw_calls:
                if not isinstance(item, dict) or item.get("type") != "function":
                    raise ValueError("Invalid function call")
                args = item["function"]["arguments"]
                if not isinstance(args, str) or len(args.encode("utf-8")) > 16_384:
                    raise ValueError("Invalid arguments length")
                parsed_args = strict_json(args)
                if not isinstance(parsed_args, dict):
                    raise ValueError("Tool arguments must be JSON object")
                call = ToolCall(id=item["id"], name=item["function"]["name"], arguments=parsed_args)
                if call.id in seen:
                    raise ValueError("Duplicate tool call")
                seen.add(call.id)
                calls.append(call)
            reason = choices[0].get("finish_reason", "stop")
            if reason not in {"stop", "tool_calls", "length"}:
                raise ValueError("Unsupported termination reason")
            if calls and reason != "tool_calls":
                raise ValueError("Partial or inconsistently terminated tool calls")
            if reason == "tool_calls" and not calls:
                raise ValueError("Tool termination without tool calls")
            declared = {item["function"]["name"] for item in tools}
            if any(call.name not in declared for call in calls):
                raise ValueError("Undeclared tool call")
            content = message.get("content")
            if content is None:
                content = ""
            if not isinstance(content, str) or len(content) > 30000:
                raise ValueError("Invalid message content")
            reasoning = message.get("reasoning_content")
            if reasoning is not None and (not isinstance(reasoning, str) or len(reasoning) > 30000):
                raise ValueError("Invalid reasoning continuation")
            report = normalized_usage(data)
            return ModelReply(content=content, tool_calls=calls, finish_reason=reason,
                              model=selected, usage=report, reasoning_content=reasoning)
        except (KeyError, IndexError, TypeError, ValueError, AttributeError, RecursionError) as exc:
            raise CapabilityError("invalid_model_output",
                                  "Model gateway response did not match tool schema", 502) from exc
