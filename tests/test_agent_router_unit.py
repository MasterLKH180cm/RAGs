"""The unified LiteLLM Gateway contract is provider-neutral and fail-closed."""
import json
import os
from dataclasses import replace

import httpx
import pytest

from rags.agents.config import AgentPlatformConfig
from rags.agents.router import Message, ModelRouter
from rags.registry import CapabilityError

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def reset_agent_env(monkeypatch):
    for name in tuple(os.environ):
        if name.startswith(("RAGS_ROUTER_", "RAGS_AGENT_", "RAGS_MEMORY_")):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RAGS_ENV", "test")


@pytest.fixture
def config():
    return AgentPlatformConfig(
        router_enabled=True, agent_enabled=True,
        gateway_url="https://gateway.test/v1", gateway_key="super-secret",
        models={"openai": "openai/gpt-luna-latest",
                "anthropic": "anthropic/claude-sonnet-latest",
                "gemini": "gemini/gemini-2.5-pro",
                "xai": "xai/grok-latest"},
    )


@pytest.mark.parametrize(("alias", "target"), [
    ("openai", "openai/gpt-luna-latest"),
    ("anthropic", "anthropic/claude-sonnet-latest"),
    ("gemini", "gemini/gemini-2.5-pro"),
    ("xai", "xai/grok-latest"),
])
def test_one_router_contract_accepts_four_provider_aliases(config, monkeypatch, alias, target):
    calls = []
    def handler(request):
        calls.append(request)
        assert request.url == "https://gateway.test/v1/chat/completions"
        assert request.method == "POST"
        assert request.headers["authorization"] == "Bearer super-secret"
        sent = json.loads(request.content)
        assert sent["model"] == target
        assert sent["messages"][0]["role"] == "user"
        return httpx.Response(200, json={
            "model": target, "choices": [{"finish_reason": "stop",
                "message": {"role": "assistant", "content": "A normalized answer"}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 3},
        })
    router = ModelRouter(config)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(router, "_client",
                        lambda *_args: real_client(transport=httpx.MockTransport(handler)))
    result = router.complete(model=alias, messages=[{"role": "user", "content": "Hello"}])
    assert result.model == target and result.content == "A normalized answer"
    assert result.usage == {"prompt_tokens": 4, "completion_tokens": 3}
    assert result.tool_calls == [] and len(calls) == 1


def test_router_parses_typed_tool_call_and_forbids_provider_argument_trust(config, monkeypatch):
    def handler(request):
        payload = json.loads(request.content)
        assert payload["tools"][0]["function"]["name"] == "knowledge_search"
        return httpx.Response(200, json={
            "choices": [{"finish_reason": "tool_calls", "message": {"content": None,
                "tool_calls": [{"id": "call_1", "type": "function",
                    "function": {"name": "knowledge_search",
                                 "arguments": '{"question":"records policy","top_k":2}'}}]}}],
        })
    router = ModelRouter(config)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(router, "_client",
                        lambda *_args: real_client(transport=httpx.MockTransport(handler)))
    reply = router.complete(model="openai", messages=[{"role": "user", "content": "Search"}],
                            tools=[{"type": "function", "function": {
                                "name": "knowledge_search", "parameters": {"type": "object"}}}])
    assert reply.tool_calls[0].name == "knowledge_search"
    assert reply.tool_calls[0].arguments == {"question": "records policy", "top_k": 2}


@pytest.mark.parametrize("response", [
    {"choices": []},
    {"choices": [{"finish_reason": "stop", "message": None}]},
    {"choices": [{"finish_reason": "stop", "message": []}]},
    {"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [
        {"id": "duplicate", "type": "function", "function": {"name": "exec", "arguments": "{}"}},
        {"id": "duplicate", "type": "function", "function": {"name": "exec", "arguments": "{}"}}]}}]},
    {"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [
        {"id": "call", "type": "function", "function": {"name": "exec", "arguments": "[]"}}]}}]},
    {"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [
        {"id": "call", "type": "function", "function": {"name": "exec", "arguments": "{oops"}}]}}]},
])
def test_router_rejects_malformed_tool_envelopes_without_leaking_payload(config, monkeypatch, response):
    router = ModelRouter(config)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(router, "_client", lambda *_args: real_client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response))))
    with pytest.raises(CapabilityError) as err:
        router.complete(model="openai", messages=[{"role": "user", "content": "test"}])
    assert err.value.code == "invalid_model_output"


def test_router_unknown_model_does_not_make_network_request(config, monkeypatch):
    router = ModelRouter(config)
    monkeypatch.setattr(router, "_client", lambda *_args: pytest.fail("No unauthorized outbound call"))
    with pytest.raises(CapabilityError) as err:
        router.complete(model="untrusted_url", messages=[{"role": "user", "content": "hi"}])
    assert err.value.status_code == 403


def test_gateway_redirect_and_errors_fail_closed(config, monkeypatch):
    router = ModelRouter(config)
    real_client = httpx.AsyncClient
    for status in (302, 401, 429, 500):
        monkeypatch.setattr(router, "_client", lambda *_args: real_client(
            transport=httpx.MockTransport(lambda _: httpx.Response(
                status, text="PRIVATE credentials must not be surfaced"))))
        with pytest.raises(CapabilityError) as error:
            router.complete(model="openai", messages=[{"role": "user", "content": "hello"}])
        assert "PRIVATE" not in error.value.reason


def test_disabled_router_never_accepts_gateway_calls():
    config = AgentPlatformConfig()
    assert config.router_enabled and config.agent_enabled and config.memory_enabled
    assert not config.router_ready
    with pytest.raises(CapabilityError) as no_gateway:
        ModelRouter(config).complete(
            model="missing", messages=[{"role": "user", "content": "hi"}])
    assert no_gateway.value.code == "router_unconfigured"
    with pytest.raises(CapabilityError) as disabled:
        ModelRouter(replace(config, agent_enabled=False, router_enabled=False))
    assert disabled.value.code == "feature_disabled"


@pytest.mark.parametrize("changes", [
    {"gateway_url": "http://open-external.invalid/v1"},
    {"gateway_url": "https://secret:token@example.com"},
    {"gateway_url": "https://example.com/?access_token=secret"},
    {"gateway_key": ""},
    {"models": {"a": "not-a-provider-model"}},
    {"models": {"with space": "openai/gpt"}},
    {"models": {}},
    {"max_steps": 100},
    {"max_tool_calls": 0},
    {"max_runtime_seconds": 180},
    {"max_output_tokens": 0},
])
def test_config_fails_closed(config, changes):
    with pytest.raises(ValueError):
        replace(config, **changes)


def test_agent_runtime_requires_explicit_router_flag():
    with pytest.raises(ValueError):
        AgentPlatformConfig(agent_enabled=True, router_enabled=False)


def test_env_router_secret_settings_are_never_revealed(monkeypatch):
    for k, v in {
        "RAGS_ROUTER_ENABLED": "true",
        "RAGS_AGENT_ENABLED": "true",
        "RAGS_ROUTER_GATEWAY_URL": "https://gateway.example.com/v1",
        "RAGS_ROUTER_API_KEY": "super-private-secret",
        "RAGS_ROUTER_MODELS_JSON": '{"main":"xai/grok-latest"}',
    }.items():
        monkeypatch.setenv(k, v)
    config = AgentPlatformConfig.from_env()
    assert config.models == {"main": "xai/grok-latest"}
    assert "super-private-secret" not in repr(config)


def test_unrecognized_model_role_rejected():
    with pytest.raises(ValueError):
        Message.model_validate({"role": "admin", "content": "malicious"})


def test_gateway_accepts_maximum_supported_agent_history_envelope(config, monkeypatch):
    router = ModelRouter(config)
    def respond(req):
        body = json.loads(req.content)
        assert len(body["messages"]) == 47
        return httpx.Response(200, json={"choices": [
            {"message": {"content": "ok"}, "finish_reason": "stop"}]})
    actual = httpx.AsyncClient
    monkeypatch.setattr(router, "_client", lambda *_args:
                        actual(transport=httpx.MockTransport(respond)))
    bounded = [{"role": "user", "content": "x"} for _ in range(47)]
    assert router.complete(model="openai", messages=bounded).content == "ok"
    with pytest.raises(ValueError, match="64"):
        router.complete(model="openai", messages=bounded * 2)
