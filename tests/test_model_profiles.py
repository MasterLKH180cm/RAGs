"""Provider protocol boundaries, explicit budgets and valid structured output."""
from dataclasses import replace
import json
import os

import httpx
import pytest

from rags.agents.config import AgentPlatformConfig
from rags.agents.router import ModelRouter
from rags.config import Settings
from rags.model_profiles import (
    INKLING_MODEL, TINKER_BASE_URL, chat_profile, effort_from_env, normalized_usage,
    reasoning_effort, strict_json,
)
from rags.providers import Providers, response_text
from rags.registry import CapabilityError
from rags.tokenization import resolve_tokenizer

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for key in tuple(os.environ):
        if key.startswith(("RAGS_", "OPENAI_", "ANTHROPIC_")):
            monkeypatch.delenv(key)
    monkeypatch.setenv("RAGS_ENV", "test")


def config(**changes):
    return AgentPlatformConfig(
        gateway_url="https://gateway.example/v1", gateway_key="private-test-key",
        models={"inkling": INKLING_MODEL, "legacy": "openai/legacy",
                "embed": "openai/embedding"}, **changes)


def fake_router(monkeypatch, response, **options):
    router = ModelRouter(config(**options))
    actual_client = httpx.Client
    captured = []
    def send(request):
        captured.append(json.loads(request.content))
        assert request.headers["accept-encoding"] == "identity"
        return httpx.Response(200, json=response)
    monkeypatch.setattr(router, "_client", lambda *_: actual_client(
        transport=httpx.MockTransport(send)))
    return router, captured


def test_inkling_uses_exact_protocol_and_legacy_default_is_unchanged():
    inkling = chat_profile(INKLING_MODEL)
    assert inkling.protocol == "tinker"
    assert inkling.parameters(max_output_tokens=96, effort="low") == {
        "max_completion_tokens": 96, "temperature": 0, "reasoning_effort": "low"}
    assert inkling.supports_json_schema is False and inkling.supports_embeddings is False
    assert chat_profile("openai/legacy").parameters(max_output_tokens=96) == {
        "max_tokens": 96, "temperature": 0}
    assert chat_profile("operator-alias", "tinker").parameters(max_output_tokens=20,
                                                              effort=.2)["reasoning_effort"] == .2
    with pytest.raises(ValueError):
        chat_profile("operator-alias", "unsupported")
    with pytest.raises(ValueError):
        chat_profile("openai/legacy").parameters(max_output_tokens=96, effort="low")


@pytest.mark.parametrize("model,protocol", [(INKLING_MODEL, "auto"), ("approved-alias", "tinker")])
@pytest.mark.parametrize("generation_base", [None, TINKER_BASE_URL, "https://generation-fixture.invalid/v1"])
def test_tinker_never_inherits_embedding_or_ambient_provider_credentials(
    monkeypatch, model, protocol, generation_base,
):
    from rags.vision import describe_image

    monkeypatch.setenv("RAGS_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("RAGS_GENERATION_MODEL", model)
    monkeypatch.setenv("RAGS_GENERATION_PROTOCOL", protocol)
    monkeypatch.setenv("RAGS_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "embedding-credential")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://embedding-fixture.invalid/v1")
    monkeypatch.setenv("TINKER_API_KEY", "ambient-tinker-credential")
    if generation_base is not None:
        monkeypatch.setenv("RAGS_GENERATION_BASE_URL", generation_base)
    monkeypatch.setattr("rags.provider_transport.post", lambda *a, **kw: pytest.fail("Implicit credential was sent"))
    settings = Settings.from_env()
    assert settings.generation_endpoint == (generation_base or TINKER_BASE_URL)
    assert settings.generation_credential == ""
    assert settings.embedding_endpoint == "https://embedding-fixture.invalid/v1"
    assert settings.embedding_credential == "embedding-credential"
    provider = Providers(settings)
    assert provider.has_generation is False
    with pytest.raises(CapabilityError) as error:
        provider.complete("instructions", "question")
    assert error.value.code == "requires_provider"
    with pytest.raises(ValueError, match="required") as vision_error:
        describe_image(b"pixels", "image/png", settings)
    assert "OPENAI_API_KEY" not in str(vision_error.value)


@pytest.mark.parametrize("credential_source", ["value", "file"])
@pytest.mark.parametrize("generation_base", [None, "https://generation-fixture.invalid/v1"])
def test_tinker_explicit_generation_credentials_and_endpoint_preserve_embedding_isolation(
    monkeypatch, tmp_path, credential_source, generation_base,
):
    monkeypatch.setenv("RAGS_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("RAGS_GENERATION_MODEL", INKLING_MODEL)
    monkeypatch.setenv("RAGS_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "embedding-credential")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://embedding-fixture.invalid/v1")
    monkeypatch.setenv("TINKER_API_KEY", "ambient-tinker-credential")
    if generation_base is not None:
        monkeypatch.setenv("RAGS_GENERATION_BASE_URL", generation_base)
    if credential_source == "file":
        keyfile = tmp_path / "generation-key"
        keyfile.write_text("explicit-generation-credential\n")
        monkeypatch.setenv("RAGS_GENERATION_API_KEY_FILE", str(keyfile))
    else:
        monkeypatch.setenv("RAGS_GENERATION_API_KEY", "explicit-generation-credential")
    sent = []

    def post(url, **kwargs):
        sent.append((url, kwargs))
        return httpx.Response(200, request=httpx.Request("POST", url), json={
            "choices": [{"finish_reason": "stop", "message": {"content": "Answer"}}]})

    monkeypatch.setattr("rags.provider_transport.post", post)
    settings = Settings.from_env()
    assert settings.generation_credential == "explicit-generation-credential"
    assert settings.embedding_credential == "embedding-credential"
    assert settings.embedding_endpoint == "https://embedding-fixture.invalid/v1"
    assert Providers(settings).complete("instructions", "question") == "Answer"
    assert sent[0][0] == (generation_base or TINKER_BASE_URL) + "/chat/completions"
    assert sent[0][1]["headers"]["Authorization"] == "Bearer explicit-generation-credential"


def test_ordinary_openai_profile_keeps_legacy_endpoint_and_key(monkeypatch):
    monkeypatch.setenv("RAGS_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-openai-credential")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://legacy-fixture.invalid/v1")
    monkeypatch.setenv("TINKER_API_KEY", "ambient-tinker-credential")
    settings = Settings.from_env()
    assert settings.generation_endpoint == "https://legacy-fixture.invalid/v1"
    assert settings.generation_credential == "legacy-openai-credential"


@pytest.mark.parametrize("value", [True, False, -0.1, 1, float("inf"), float("nan"), "ultra", {}, []])
def test_invalid_reasoning_effort_is_rejected(value):
    with pytest.raises(ValueError):
        reasoning_effort(value)


@pytest.mark.parametrize("value", [None, 0, .99, "none", "minimal", "low", "medium", "high", "xhigh", "max"])
def test_supported_reasoning_effort_is_preserved(value):
    assert reasoning_effort(value) == value


def test_reasoning_effort_environment_preserves_explicit_zero_and_rejects_bad_values():
    assert effort_from_env("0") == 0
    assert effort_from_env("0.2") == .2
    assert effort_from_env("medium") == "medium"
    assert effort_from_env("") is None
    for value in ("secret", "Infinity", "NaN", "1", "-1"):
        with pytest.raises(ValueError):
            effort_from_env(value)


@pytest.mark.parametrize("raw", ['{"x":NaN}', '{"x":Infinity}', '{"x":1e999}',
                                 '{"x":1,"x":2}', '{"nested":{"x":1,"x":2}}'])
def test_strict_json_rejects_ambiguous_or_nonfinite_model_data(raw):
    with pytest.raises(ValueError):
        strict_json(raw)


def test_reported_usage_preserves_cache_without_inventing_reasoning():
    usage = normalized_usage({"usage": {"prompt_tokens": 40, "completion_tokens": 8,
                                        "total_tokens": 48,
                                        "prompt_tokens_details": {"cached_tokens": 20},
                                        "private": "never forward"}})
    assert usage == {"prompt_tokens": 40, "completion_tokens": 8,
                     "total_tokens": 48, "cached_prompt_tokens": 20}
    assert normalized_usage({"usage": {"prompt_tokens": True, "completion_tokens": -1,
        "total_tokens": 1e100, "prompt_tokens_details": {"cached_tokens": False}}}) == {}
    assert normalized_usage({"usage": []}) == {}
    assert normalized_usage({"usage": {"completion_tokens_details": {"reasoning_tokens": 7}}}) == {
        "reasoning_tokens": 7}


def test_router_inkling_request_and_reply_exclude_private_reasoning(monkeypatch):
    router, calls = fake_router(monkeypatch, {
        "choices": [{"finish_reason": "stop", "message": {"content": "A final answer",
                      "reasoning_content": "PRIVATE chain of thought"}}],
        "usage": {"prompt_tokens": 8, "completion_tokens": 7,
                  "prompt_tokens_details": {"cached_tokens": 3}},
    }, model_options={"inkling": {"protocol": "tinker", "reasoning_effort": .2}})
    result = router.complete(model="inkling", messages=[{"role": "user", "content": "hello"}],
                             max_output_tokens=64)
    assert calls[0]["max_completion_tokens"] == 64 and calls[0]["reasoning_effort"] == .2
    assert "max_tokens" not in calls[0] and "response_format" not in calls[0]
    assert result.content == "A final answer" and "PRIVATE" not in result.model_dump_json()
    assert result.usage["cached_prompt_tokens"] == 3


def test_router_unsupported_embedding_never_sends_network_request(monkeypatch):
    router = ModelRouter(config())
    monkeypatch.setattr(router, "_client", lambda *_: pytest.fail("Must fail before network"))
    with pytest.raises(CapabilityError) as error:
        router.embedding(model="inkling", text="do not send")
    assert error.value.code == "unsupported_model_operation"


@pytest.mark.parametrize("changes", [
    {"model_options": {"unknown": {"protocol": "tinker"}}},
    {"model_options": {"inkling": {"base_url": "https://arbitrary"}}},
    {"model_options": {"inkling": {"reasoning_effort": True}}},
    {"model_options": {"inkling": {"protocol": "other"}}},
    {"model_options": {"legacy": {"reasoning_effort": .5}}},
])
def test_model_options_are_bound_to_approved_aliases(changes):
    with pytest.raises(ValueError):
        config(**changes)


def test_model_options_load_with_bounded_strict_json(monkeypatch):
    monkeypatch.setenv("RAGS_ROUTER_GATEWAY_URL", "https://gateway.example/v1")
    monkeypatch.setenv("RAGS_ROUTER_API_KEY", "test-secret")
    monkeypatch.setenv("RAGS_ROUTER_MODELS_JSON", json.dumps({"main": INKLING_MODEL}))
    monkeypatch.setenv("RAGS_ROUTER_MODEL_OPTIONS_JSON", '{"main":{"reasoning_effort":0.2}}')
    assert AgentPlatformConfig.from_env().model_options == {"main": {"reasoning_effort": .2}}
    for raw in ('{"main":{"protocol":"tinker","protocol":"openai"}}', '[]', 'x' * 8193):
        monkeypatch.setenv("RAGS_ROUTER_MODEL_OPTIONS_JSON", raw)
        with pytest.raises(ValueError):
            AgentPlatformConfig.from_env()


@pytest.mark.parametrize("finish,arguments", [("length", "{}"), ("stop", "{}"),
    ("tool_calls", '{"x":NaN}'), ("tool_calls", '{"x":1,"x":2}')])
def test_router_rejects_truncated_or_ambiguous_tool_calls(monkeypatch, finish, arguments):
    router, _ = fake_router(monkeypatch, {"choices": [{"finish_reason": finish, "message": {
        "tool_calls": [{"id": "call_1", "type": "function", "function": {
            "name": "lookup", "arguments": arguments}}]}}]})
    with pytest.raises(CapabilityError) as error:
        router.complete(model="inkling", messages=[{"role": "user", "content": "Find data"}],
                        tools=[{"type": "function", "function": {"name": "lookup"}}])
    assert error.value.code == "invalid_model_output"


def test_router_bounds_tool_schema_bytes_before_network(monkeypatch):
    router = ModelRouter(config())
    monkeypatch.setattr(router, "_client", lambda *_: pytest.fail("Must fail before network"))
    with pytest.raises(CapabilityError) as error:
        router.complete(model="inkling", messages=[{"role": "user", "content": "Find data"}],
                        tools=[{"type": "function", "function": {
                            "name": "lookup", "description": "x" * (121 * 1024)}}])
    assert error.value.code == "context_budget_exceeded"


def test_tinker_text_truncation_is_returned_for_explicit_failure_classification(monkeypatch):
    router, _ = fake_router(monkeypatch, {"choices": [{"finish_reason": "length",
                                                     "message": {"content": "partial"}}]})
    assert router.complete(model="inkling", messages=[{"role": "user", "content": "hi"}]).finish_reason == "length"
    with pytest.raises(CapabilityError):
        response_text({"choices": [{"finish_reason": "length", "message": {"content": "partial"}}]}, "openai")


def test_inkling_configuration_uses_byte_budget_and_independent_embedding(monkeypatch):
    monkeypatch.setenv("RAGS_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("RAGS_GENERATION_MODEL", INKLING_MODEL)
    monkeypatch.setenv("RAGS_REASONING_EFFORT", "low")
    settings = Settings.from_env()
    assert settings.generation_tokenizer_config == {"backend": "utf8_bytes"}
    assert settings.embedding_provider == "local"
    assert settings.embedding_model == "text-embedding-3-small"
    tokenizer = resolve_tokenizer(settings.generation_tokenizer_config)
    assert tokenizer.count("中文😀") == 10
    assert tokenizer.count_kind == "conservative_bytes"
    assert tokenizer.profile["unit"] == "utf8_bytes"


def test_utf8_counter_preserves_prefix_suffix_and_never_splits_a_character():
    counter = resolve_tokenizer({"backend": "utf8_bytes"})
    text = "漢a😀字"
    assert counter.decode(counter.encode(text)) == text
    for budget in range(15):
        prefix, suffix = counter.truncate(text, budget), counter.suffix(text, budget)
        assert text.startswith(prefix) and text.endswith(suffix)
        assert counter.count(prefix) <= budget and counter.count(suffix) <= budget
        assert "\ufffd" not in prefix + suffix
    for invalid in ([True], [256], [0xFF], [0xE4], ["a"]):
        with pytest.raises(ValueError):
            counter.decode(invalid)


def test_direct_provider_uses_profile_and_rejects_truncated_generation(monkeypatch):
    monkeypatch.setenv("RAGS_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("RAGS_GENERATION_MODEL", INKLING_MODEL)
    monkeypatch.setenv("RAGS_GENERATION_API_KEY", "test-key")
    settings = replace(Settings.from_env(), reasoning_effort="low", output_budget=64)
    calls = []
    def send(url, **kwargs):
        calls.append(kwargs)
        return httpx.Response(200, request=httpx.Request("POST", url), json={
            "choices": [{"finish_reason": "length", "message": {"content": "partial"}}]})
    monkeypatch.setattr("rags.provider_transport.post", send)
    with pytest.raises(CapabilityError):
        Providers(settings).complete("instructions", "question")
    assert calls[0]["json"]["max_completion_tokens"] == 64
    assert calls[0]["json"]["reasoning_effort"] == "low"
    assert "max_tokens" not in calls[0]["json"]


def test_router_replays_bounded_reasoning_without_serializing_public_reply(monkeypatch):
    private = "Upstream continuation fixture"
    router, requests = fake_router(monkeypatch, {"choices": [{"finish_reason": "stop", "message": {
        "content": "The answer", "reasoning_content": private}}]})
    result = router.complete(model="inkling", messages=[
        {"role": "assistant", "content": "", "reasoning_content": private},
        {"role": "user", "content": "Continue"}])
    assert requests[0]["messages"][0]["reasoning_content"] == private
    assert result.reasoning_content == private
    assert private not in result.model_dump_json() and private not in repr(result)
    with pytest.raises(ValueError):
        router.complete(model="inkling", messages=[
            {"role": "assistant", "content": "", "reasoning_content": "x" * 30001}])


@pytest.mark.parametrize("value", [True, [], "x" * 30001])
def test_router_rejects_malformed_reasoning_continuation(monkeypatch, value):
    router, _ = fake_router(monkeypatch, {"choices": [{"finish_reason": "stop", "message": {
        "content": "A result", "reasoning_content": value}}]})
    with pytest.raises(CapabilityError) as error:
        router.complete(model="inkling", messages=[{"role": "user", "content": "hello"}])
    assert error.value.code == "invalid_model_output"
