"""Typed, provider-neutral chat/tool interface through an operator LiteLLM Gateway.

The Gateway supports >40 upstream providers; this adapter only claims an
OpenAI-compatible HTTP contract. Exact model aliases are operator-approved.
"""
import json
import re
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from rags.agents.config import AgentPlatformConfig
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


class ModelRouter:
    def __init__(self, config: AgentPlatformConfig):
        if not config.router_enabled:
            raise CapabilityError("feature_disabled", "Model router is disabled", 403)
        self.config = config

    def _client(self) -> httpx.Client:
        return httpx.Client(timeout=httpx.Timeout(30., connect=5.), follow_redirects=False,
                            trust_env=False)

    def complete(self, *, model: str, messages: list[dict], tools: list[dict] | None = None,
                 max_output_tokens: int | None = None) -> ModelReply:
        selected = self.config.models.get(model)
        if selected is None:
            raise CapabilityError("model_not_allowed", "Selected model alias is not configured", 403)
        if not isinstance(messages, list) or not 1 <= len(messages) <= 42:
            raise ValueError("Messages must contain 1..42 turns")
        normalized = [Message.model_validate(msg).model_dump(exclude_none=True) for msg in messages]
        if len(json.dumps(normalized, ensure_ascii=False).encode("utf-8")) > MAX_CONTEXT_BYTES:
            raise CapabilityError("context_budget_exceeded", "Router message budget exceeded", 422)
        if tools is None:
            tools = []
        if not isinstance(tools, list) or len(tools) > 12:
            raise ValueError("Tool definitions must be a bounded list")
        for item in tools:
            if (not isinstance(item, dict) or item.get("type") != "function"
                    or not isinstance(item.get("function"), dict)
                    or not TOOL_NAME.fullmatch(str(item["function"].get("name", "")))):
                raise ValueError("Invalid tool definition")
        tokens = self.config.max_output_tokens if max_output_tokens is None else max_output_tokens
        if type(tokens) is not int or not 1 <= tokens <= self.config.max_output_tokens:
            raise ValueError("Invalid model output token limit")
        request = {"model": selected, "messages": normalized, "max_tokens": tokens,
                   "temperature": 0}
        if tools:
            request["tools"] = tools
            request["tool_choice"] = "auto"
        try:
            with self._client() as client:
                with client.stream(
                    "POST", self.config.gateway_url + "/chat/completions",
                    headers={"Authorization": "Bearer " + self.config.gateway_key,
                             "Accept": "application/json"},
                    json=request,
                ) as response:
                    if response.status_code != 200:
                        raise CapabilityError("model_gateway_failed",
                                              "Model gateway did not complete the request", 502)
                    payload, size = [], 0
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > MAX_RESPONSE_BYTES:
                            raise CapabilityError("invalid_model_output", "Model response exceeds limit", 502)
                        payload.append(chunk)
            data = json.loads(b"".join(payload))
        except CapabilityError:
            raise
        except (httpx.HTTPError, UnicodeError, ValueError) as exc:
            raise CapabilityError("model_gateway_failed",
                                  "Model gateway is unavailable or returned invalid JSON", 502) from exc
        try:
            choices = data["choices"]
            message = choices[0]["message"]
            raw_calls = message.get("tool_calls") or []
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
                parsed_args = json.loads(args)
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
            content = message.get("content") or ""
            if not isinstance(content, str) or len(content) > 30000:
                raise ValueError("Invalid message content")
            usage = data.get("usage") or {}
            report = {}
            for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = usage.get(name) if isinstance(usage, dict) else None
                if type(value) is int and 0 <= value <= 10**9:
                    report[name] = value
            return ModelReply(content=content, tool_calls=calls, finish_reason=reason,
                              model=selected, usage=report)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise CapabilityError("invalid_model_output",
                                  "Model gateway response did not match tool schema", 502) from exc
