"""Operator configuration fails closed before a service or model is created."""
import json
import os

import pytest

from rags.config import Settings

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    for name in tuple(os.environ):
        if name.startswith(("RAGS_", "OPENAI_", "ANTHROPIC_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("RAGS_ENV", "test")


@pytest.mark.parametrize(("variables", "reason"), [
    ({"RAGS_KEYS_JSON": "not-json"}, "JSON token-to-tenant mapping"),
    ({"RAGS_KEYS_JSON": "[]"}, "non-empty API keys"),
    ({"RAGS_KEYS_JSON": '{"key":1}'}, "non-empty API keys"),
    ({"RAGS_ENV": "production", "RAGS_KEYS_JSON": "{}"}, "Configure RAGS_KEYS_JSON"),
    ({"RAGS_ENV": "production", "RAGS_KEYS_JSON": '{"short":"tenant"}'}, "at least 24 characters"),
    ({"RAGS_EMBEDDING_PROVIDER": "unknown"}, "Unsupported embedding provider"),
    ({"RAGS_MODEL_PROVIDER": "unknown"}, "Unsupported model provider"),
    ({"OPENAI_BASE_URL": "file:///private"}, "HTTP"),
    ({"OPENAI_BASE_URL": "https://user:password@provider.invalid/v1"}, "without credentials"),
    ({"OPENAI_BASE_URL": "https://provider.invalid/v1?key=private"}, "without credentials"),
    ({"RAGS_TOKENIZER_JSON": "{"}, "RAGS_TOKENIZER_JSON must be a JSON object"),
    ({"RAGS_TOKENIZER_JSON": "[]"}, "RAGS_TOKENIZER_JSON must be a JSON object"),
    ({"RAGS_GENERATION_TOKENIZER_JSON": "false"}, "RAGS_GENERATION_TOKENIZER_JSON must be a JSON object"),
    ({"RAGS_CHUNKING_JSON": "null"}, "RAGS_CHUNKING_JSON must be a JSON object"),
    ({"RAGS_CONTEXT_BUDGET": "511"}, "Invalid context or output budget"),
    ({"RAGS_CONTEXT_BUDGET": "200001"}, "Invalid context or output budget"),
    ({"RAGS_OUTPUT_BUDGET": "0"}, "Invalid context or output budget"),
    ({"RAGS_OUTPUT_BUDGET": "16001"}, "Invalid context or output budget"),
    ({"RAGS_ENVELOPE_BUDGET": "-1"}, "must fit the context budget"),
    ({"RAGS_ENVELOPE_BUDGET": "4097"}, "must fit the context budget"),
    ({"RAGS_CONTEXT_BUDGET": "512", "RAGS_OUTPUT_BUDGET": "256", "RAGS_ENVELOPE_BUDGET": "256"}, "must fit the context budget"),
    ({"RAGS_MAX_ITERATIONS": "0"}, "must be 1..3"),
    ({"RAGS_MAX_ITERATIONS": "4"}, "must be 1..3"),
    ({"RAGS_ADMISSION_POLICY": "silently-accept"}, "compatibility or calibrated_required"),
], ids=["invalid-key-json", "key-array", "invalid-tenant", "production-without-key", "short-production-key",
        "embedding-provider", "generation-provider", "url-scheme", "url-credentials", "url-query",
        "invalid-tokenizer-json", "tokenizer-array", "generation-tokenizer-object", "chunking-object",
        "context-minimum", "context-maximum", "output-minimum", "output-maximum", "envelope-minimum",
        "envelope-maximum", "no-input-budget", "iteration-minimum", "iteration-maximum", "admission-policy"])
def test_invalid_operator_settings_fail_closed(monkeypatch, variables, reason):
    for name, value in variables.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=reason):
        Settings.from_env()


@pytest.mark.parametrize(("sources", "reason"), [
    ({f"source{index}": {} for index in range(9)}, "at most eight"),
    ({"1-invalid": {}}, "named source objects"),
    ({"archive": []}, "named source objects"),
    ({"archive": {"url": "file:///archive", "tenant_keys": {"a": "key-a"}}}, "HTTP"),
    ({"archive": {"url": "https://user:password@archive.invalid", "tenant_keys": {"a": "key-a"}}}, "HTTP"),
    ({"archive": {"url": "https://archive.invalid"}}, "tenant_keys mappings"),
    ({"archive": {"url": "https://archive.invalid", "tenant_keys": {"a": ""}}}, "tenant_keys mappings"),
    ({"archive": {"url": "https://archive.invalid", "tenant_keys": {"a": "shared", "b": "shared"}}}, "share credentials"),
], ids=["source-count", "source-name", "source-type", "url-scheme", "url-credentials",
        "missing-tenant-keys", "empty-tenant-key", "cross-tenant-key-reuse"])
def test_federation_configuration_requires_named_sources_and_separate_tenant_credentials(monkeypatch, sources, reason):
    monkeypatch.setenv("RAGS_SOURCES_JSON", json.dumps(sources))
    with pytest.raises(ValueError, match=reason):
        Settings.from_env()


def test_valid_production_configuration_preserves_explicit_budgets_models_and_tenant_mapping(monkeypatch):
    sources = {"archive": {"url": "https://archive.invalid", "tenant_keys": {"a": "key-a", "b": "key-b"}}}
    variables = {
        "RAGS_ENV": "production", "RAGS_KEYS_JSON": json.dumps({"a" * 24: "a", "b" * 24: "b"}),
        "RAGS_DB_PATH": "operator-data.db", "RAGS_SOURCES_JSON": json.dumps(sources),
        "RAGS_EMBEDDING_PROVIDER": "OPENAI", "RAGS_MODEL_PROVIDER": "OPENAI",
        "OPENAI_BASE_URL": "https://provider.invalid/v1/", "OPENAI_EMBEDDING_MODEL": "custom-embedding",
        "OPENAI_CHAT_MODEL": "gpt-4o", "RAGS_CONTEXT_BUDGET": "512", "RAGS_OUTPUT_BUDGET": "128",
        "RAGS_ENVELOPE_BUDGET": "64", "RAGS_MAX_ITERATIONS": "1",
        "RAGS_ADMISSION_POLICY": "calibrated_required", "RAGS_ADMISSION_PROFILE": "operator-profile.json",
        "RAGS_TOKENIZER_JSON": '{"backend":"tiktoken","encoding":"cl100k_base"}',
        "RAGS_CHUNKING_JSON": '{"strategy":"fixed","max_tokens":64,"overlap_tokens":8}',
    }
    for name, value in variables.items():
        monkeypatch.setenv(name, value)
    settings = Settings.from_env()
    assert settings.keys == {"a" * 24: "a", "b" * 24: "b"}
    assert settings.db_path == "operator-data.db" and settings.federated_sources == sources
    assert settings.embedding_provider == settings.model_provider == "openai"
    assert settings.openai_base_url == "https://provider.invalid/v1"
    assert settings.embedding_model == "custom-embedding" and settings.chat_model == "gpt-4o"
    assert settings.generation_tokenizer_config == {"backend": "tiktoken", "model": "gpt-4o"}
    assert settings.tokenizer_config == {"backend": "tiktoken", "encoding": "cl100k_base"}
    assert settings.chunking_config == {"strategy": "fixed", "max_tokens": 64, "overlap_tokens": 8}
    assert (settings.context_budget, settings.output_budget, settings.envelope_budget, settings.max_iterations) == (512, 128, 64, 1)
    assert settings.admission_policy == "calibrated_required" and settings.admission_profile == "operator-profile.json"


def test_development_key_is_limited_to_explicit_nonproduction_environments(monkeypatch):
    monkeypatch.setenv("RAGS_ENV", "development")
    monkeypatch.setenv("RAGS_API_KEY", "explicit-development-key")
    settings = Settings.from_env()
    assert settings.keys == {"explicit-development-key": "local"}
    assert settings.model_provider == "extractive" and settings.embedding_provider == "local"
    assert settings.generation_tokenizer_config == {"backend": "unicode"}
    monkeypatch.setenv("RAGS_ENV", "production")
    with pytest.raises(ValueError, match="Configure RAGS_KEYS_JSON"):
        Settings.from_env()
