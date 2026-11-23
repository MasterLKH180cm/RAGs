"""Worker credential selection follows the public provider configuration syntax."""
import json

import pytest

from rags.research_service import ResearchSettings, worker_environment


pytestmark = pytest.mark.unit


@pytest.mark.parametrize("engine,generation,embedding,expected", [
    ("full_graphrag", "OPENAI", "FastEmbed", {"OPENAI_API_KEY"}),
    ("hipporag", "AnThRoPiC", "FASTEMBED", {"ANTHROPIC_API_KEY"}),
    ("full_graphrag", "ROUTER", "FastEmbed", {"RAGS_ROUTER_API_KEY"}),
    ("hipporag", "EXTRACTIVE", "OpenAI", {"RAGS_EMBEDDING_API_KEY"}),
    ("full_graphrag", "OPENAI", "ROUTER", {"OPENAI_API_KEY", "RAGS_ROUTER_API_KEY"}),
    ("self_rag", "ROUTER", "OPENAI", set()),
])
def test_normalized_providers_forward_only_the_selected_credentials(monkeypatch, engine, generation, embedding, expected):
    secret_names = {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "RAGS_GENERATION_API_KEY",
                    "RAGS_EMBEDDING_API_KEY", "RAGS_ROUTER_API_KEY", "RAGS_CRAG_SEARCH_KEY"}
    for name in secret_names:
        monkeypatch.delenv(name + "_FILE", raising=False)
        monkeypatch.setenv(name, "synthetic-credential-" + name)
    monkeypatch.delenv("RAGS_GENERATION_API_KEY")
    monkeypatch.setenv("RAGS_ENV", "test")
    monkeypatch.setenv("RAGS_MODEL_PROVIDER", generation)
    monkeypatch.setenv("RAGS_EMBEDDING_PROVIDER", embedding)
    monkeypatch.setenv("RAGS_GENERATION_AUTH_MODE", "required")
    monkeypatch.setenv("RAGS_ROUTER_AUTH_MODE", "required")
    monkeypatch.setenv("RAGS_ROUTER_GATEWAY_URL", "https://gateway.example/v1")
    monkeypatch.setenv("RAGS_ROUTER_MODELS_JSON", '{"main":"openai/gpt-4.1-mini"}')
    monkeypatch.setenv("RAGS_KEYS_JSON", '{"private-api-key":"tenant-a"}')
    monkeypatch.setenv("RAGS_RESEARCH_KEYS_JSON", '{"private-sidecar-key":"tenant-a"}')

    environment = worker_environment(ResearchSettings(engine, {"sidecar-key": "tenant-a"}))
    assert secret_names.intersection(environment) == expected
    if "RAGS_ROUTER_API_KEY" in expected:
        assert environment["RAGS_ROUTER_GATEWAY_URL"] == "https://gateway.example/v1"
        assert json.loads(environment["RAGS_ROUTER_MODELS_JSON"]) == {"main": "openai/gpt-4.1-mini"}
    else:
        assert "RAGS_ROUTER_GATEWAY_URL" not in environment
    assert "RAGS_KEYS_JSON" not in environment
    assert "RAGS_RESEARCH_KEYS_JSON" not in environment
