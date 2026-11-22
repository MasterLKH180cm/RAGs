"""Integration: existing RAG providers use one LiteLLM gateway for chat and embeddings."""
import json
import os

import httpx
import pytest

from rags.agents.router import ModelRouter
from rags.config import Settings
from rags.providers import Providers
from rags.registry import CapabilityError

pytestmark = pytest.mark.integration


@pytest.fixture
def settings(tmp_path, monkeypatch):
    for key in tuple(os.environ):
        if key.startswith(("RAGS_", "OPENAI_", "ANTHROPIC_")):
            monkeypatch.delenv(key, raising=False)
    for key, value in {
        "RAGS_ENV": "test",
        "RAGS_DB_PATH": str(tmp_path / "router.db"),
        "RAGS_ROUTER_ENABLED": "true",
        "RAGS_AGENT_ENABLED": "true",
        "RAGS_MEMORY_ENABLED": "true",
        "RAGS_ROUTER_GATEWAY_URL": "https://lite.test/v1",
        "RAGS_ROUTER_API_KEY": "not-a-real-secret",
        "RAGS_ROUTER_MODELS_JSON": '{"chat":"anthropic/claude-sonnet-latest","embed":"openai/text-embedding-3-small"}',
        "RAGS_MODEL_PROVIDER": "router",
        "RAGS_GENERATION_MODEL": "chat",
        "RAGS_EMBEDDING_PROVIDER": "router",
        "RAGS_EMBEDDING_MODEL": "embed",
    }.items():
        monkeypatch.setenv(key, value)
    return Settings.from_env()


def test_router_for_rag_chat_and_embedding(settings, monkeypatch):
    calls = []
    def gateway(req):
        assert req.headers["authorization"] == "Bearer not-a-real-secret"
        body = json.loads(req.content)
        calls.append((req.url.path, body["model"]))
        if req.url.path.endswith("/embeddings"):
            return httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2, 0.3]}],
                                             "usage": {"prompt_tokens": 3}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "Routed answer"},
                                                     "finish_reason": "stop"}],
                                         "usage": {"completion_tokens": 5}})
    actual = httpx.Client
    monkeypatch.setattr(ModelRouter, "_client",
                        lambda self, *_args: actual(transport=httpx.MockTransport(gateway)))
    provider = Providers(settings)
    assert provider.has_generation
    assert provider.embedding_model == "router:openai/text-embedding-3-small"
    assert provider.model_identity == "router:anthropic/claude-sonnet-latest"
    provider.begin_usage()
    assert provider.embedding("Example") == [0.1, 0.2, 0.3]
    assert provider.complete("System", "Question") == "Routed answer"
    assert provider.usage()["calls"] == 2
    assert calls == [("/v1/embeddings", "openai/text-embedding-3-small"),
                     ("/v1/chat/completions", "anthropic/claude-sonnet-latest")]


def test_router_embedding_validation_never_accepts_malformed_vectors(settings, monkeypatch):
    actual = httpx.Client
    for vector in ([], [True, 0.1], ["invalid"]):
        monkeypatch.setattr(ModelRouter, "_client", lambda self, *_args:
                            actual(transport=httpx.MockTransport(
                                lambda _: httpx.Response(200, json={
                                    "data": [{"embedding": vector}]}))))
        with pytest.raises(CapabilityError) as error:
            Providers(settings).embedding("Expected vector")
        assert error.value.code == "invalid_embedding"


def test_unconfigured_gateway_does_not_fallback_to_another_vendor(settings, monkeypatch):
    from rags.agents.config import AgentPlatformConfig

    # A partial endpoint configuration must fail startup instead of accepting a
    # request or silently swapping to a different model provider.
    monkeypatch.setenv("RAGS_ROUTER_API_KEY", "")
    with pytest.raises(ValueError, match="gateway credentials"):
        AgentPlatformConfig.from_env()

    # A *completely* unconfigured Gateway remains safe to run locally,
    # reports unavailable generation, and cannot make a network request.
    monkeypatch.delenv("RAGS_ROUTER_GATEWAY_URL")
    monkeypatch.delenv("RAGS_ROUTER_MODELS_JSON")
    provider = Providers(settings)
    assert not provider.has_generation
    with pytest.raises(CapabilityError) as error:
        provider.embedding("Private content")
    assert error.value.code == "router_unconfigured"
