"""Configuration precedence and permissions fail before optional dependencies."""
from dataclasses import replace
import os

import pytest

from rags.config import Settings
from rags.backends import BackendManager
from rags.features import FeatureFlags, MANIFEST, strategy_features
from rags.providers import Providers
from rags.registry import CapabilityError, MODES
from rags.vision import describe_image

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    for name in tuple(os.environ):
        if name.startswith(("RAGS_", "OPENAI_", "ANTHROPIC_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("RAGS_ENV", "test")


def disabled(name):
    return FeatureFlags(((name, False),))


def test_file_then_environment_merges_per_key_with_immutable_defaults(tmp_path, monkeypatch):
    path = tmp_path / "features.json"
    path.write_text('{"query":false,"ingestion":false}', encoding="utf-8")
    monkeypatch.setenv("RAGS_FEATURES_FILE", str(path))
    monkeypatch.setenv("RAGS_FEATURES_JSON", '{"query":true,"api_docs":false}')
    flags = FeatureFlags.from_env()
    assert flags.enabled("query") and not flags.enabled("ingestion")
    assert flags.enabled("documents") and not flags.enabled("api_docs")
    resolved = flags.as_dict()
    assert set(resolved) == set(MANIFEST)
    resolved["documents"] = False
    assert flags.enabled("documents") and all(FeatureFlags().as_dict().values())


@pytest.mark.parametrize("raw", ["{", "[]", "null", '{"query":0}', '{"query":"false"}',
                                     '{"unknown":false}', '{"query":true,"query":false}', " " * 65537])
def test_invalid_or_oversized_flag_configuration_fails(monkeypatch, raw):
    monkeypatch.setenv("RAGS_FEATURES_JSON", raw)
    with pytest.raises(ValueError):
        Settings.from_env()


@pytest.mark.parametrize("contents", [None, b"\xff", b'{"query":true,"query":false}'])
def test_explicit_unreadable_or_invalid_flag_file_fails(tmp_path, monkeypatch, contents):
    path = tmp_path / "features.json"
    if contents is not None:
        path.write_bytes(contents)
    monkeypatch.setenv("RAGS_FEATURES_FILE", str(path))
    with pytest.raises(ValueError):
        FeatureFlags.from_env()


@pytest.mark.parametrize("overrides", [[("query", False)], (("query", 0),), (("query", False), ("query", True)),
                                            (([], False),), (("unknown", False),), (("query",),)])
def test_direct_feature_configuration_has_same_strict_contract(overrides):
    with pytest.raises(ValueError):
        FeatureFlags(overrides)


def test_every_manifest_permission_is_rejectable_without_mutating_other_flags():
    for name in MANIFEST:
        flags = disabled(name)
        with pytest.raises(CapabilityError) as error:
            flags.require(name)
        assert (error.value.code, error.value.status_code) == ("feature_disabled", 403)
        assert sum(not enabled for enabled in flags.as_dict().values()) == 1
    with pytest.raises(ValueError, match="Unknown feature"):
        FeatureFlags().enabled("unknown")
    with pytest.raises(ValueError, match="Unknown retrieval"):
        strategy_features("unknown")


def test_retrieval_dependencies_are_declared_before_any_work():
    for mode in MODES:
        assert {"query", "strategy." + mode} <= set(strategy_features(mode))
    assert {"strategy.dense", "strategy.sparse", "strategy.hybrid", "model_generation"} <= set(strategy_features("fusion"))
    assert {"research", "model_generation"} <= set(strategy_features("raptor"))
    assert disabled("strategy.dense").disabled_for("hyde") == ["strategy.dense"]


def test_database_url_is_explicit_and_has_priority(monkeypatch):
    monkeypatch.setenv("RAGS_DB_PATH", "old.db")
    monkeypatch.setenv("RAGS_DATABASE_URL", "postgresql://application:private@db/rags")
    assert Settings.from_env().db_path == "postgresql://application:private@db/rags"
    monkeypatch.setenv("RAGS_DATABASE_URL", "sqlite:///old.db")
    with pytest.raises(ValueError, match="PostgreSQL"):
        Settings.from_env()


def test_dedicated_provider_settings_override_legacy_values_and_read_secret_files(tmp_path, monkeypatch):
    key = tmp_path / "embedding.key"
    key.write_text("embedding-secret\n", encoding="utf-8")
    values = {"RAGS_EMBEDDING_PROVIDER": "openai", "RAGS_MODEL_PROVIDER": "openai",
              "OPENAI_BASE_URL": "https://legacy.invalid/v1", "OPENAI_API_KEY": "legacy-key",
              "OPENAI_CHAT_MODEL": "legacy-chat", "OPENAI_EMBEDDING_MODEL": "legacy-embedding",
              "RAGS_EMBEDDING_BASE_URL": "https://vectors.invalid/v1/", "RAGS_EMBEDDING_API_KEY_FILE": str(key),
              "RAGS_EMBEDDING_MODEL": "operator-vectors", "RAGS_GENERATION_BASE_URL": "https://chat.invalid/v1/",
              "RAGS_GENERATION_API_KEY": "chat-secret", "RAGS_GENERATION_MODEL": "operator-chat"}
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    settings = Settings.from_env()
    assert (settings.embedding_endpoint, settings.embedding_credential, settings.embedding_model) == (
        "https://vectors.invalid/v1", "embedding-secret", "operator-vectors")
    assert (settings.generation_endpoint, settings.generation_credential, settings.chat_model) == (
        "https://chat.invalid/v1", "chat-secret", "operator-chat")
    assert settings.generation_tokenizer_config == {"backend": "tiktoken", "model": "operator-chat"}
    monkeypatch.setenv("RAGS_EMBEDDING_API_KEY", "ambiguous-key")
    with pytest.raises(ValueError, match="Choose RAGS_EMBEDDING_API_KEY"):
        Settings.from_env()


def test_anthropic_and_fastembed_selected_model_overrides_and_blank_json_defaults(monkeypatch):
    monkeypatch.setenv("RAGS_MODEL_PROVIDER", "anthropic")
    monkeypatch.setenv("RAGS_EMBEDDING_PROVIDER", "fastembed")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://messages.invalid/v1/")
    monkeypatch.setenv("RAGS_GENERATION_MODEL", "messages-model")
    monkeypatch.setenv("RAGS_EMBEDDING_MODEL", "trained-encoder")
    for name in ("RAGS_FEATURES_JSON", "RAGS_GENERATION_TOKENIZER_JSON", "RAGS_TOKENIZER_JSON", "RAGS_CHUNKING_JSON"):
        monkeypatch.setenv(name, "")
    settings = Settings.from_env()
    assert settings.generation_endpoint == "https://messages.invalid/v1"
    assert settings.anthropic_model == "messages-model" and settings.fastembed_model == "trained-encoder"
    assert settings.generation_tokenizer_config == {"backend": "unicode"}
    assert settings.embedding_endpoint == "local"


@pytest.mark.parametrize("name", ["RAGS_GENERATION_BASE_URL", "RAGS_EMBEDDING_BASE_URL"])
@pytest.mark.parametrize("value", ["file:///private", "https://a:b@provider.invalid", "https://provider.invalid?secret=x", "https://provider.invalid#fragment"])
def test_new_provider_endpoints_have_the_same_validation(name, value, monkeypatch):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="HTTP"):
        Settings.from_env()


def test_required_evidence_policy_cannot_be_disabled(monkeypatch):
    monkeypatch.setenv("RAGS_ADMISSION_POLICY", "calibrated_required")
    monkeypatch.setenv("RAGS_FEATURES_JSON", '{"admission_scoring":false}')
    with pytest.raises(ValueError, match="Required evidence"):
        Settings.from_env()


@pytest.mark.parametrize("name,value", [
    ("RAGS_TOKENIZER_JSON", '{"backend":"unknown"}'),
    ("RAGS_GENERATION_TOKENIZER_JSON", '{"backend":"tiktoken"}'),
    ("RAGS_CHUNKING_JSON", '{"strategy":"fixed","max_tokens":4,"overlap_tokens":5}'),
    ("RAGS_CHUNKING_JSON", '{"unexpected":true}'),
])
def test_processing_configuration_is_validated_at_startup(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        Settings.from_env()


@pytest.mark.parametrize("provider_name", ["local", "openai", "fastembed"])
def test_embedding_permission_rejects_before_any_encoder_or_network(provider_name, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("disabled embedding reached an encoder or HTTP request")
    monkeypatch.setattr("rags.providers._semantic_encoder", forbidden)
    monkeypatch.setattr("rags.provider_transport.post", forbidden)
    monkeypatch.setattr("rags.providers.local_embedding", forbidden)
    settings = replace(Settings.from_env(), embedding_provider=provider_name, features=disabled("embedding." + provider_name))
    for method in (Providers(settings).embedding, Providers(settings).query_embedding):
        with pytest.raises(CapabilityError, match="Feature is disabled"):
            method("source")


@pytest.mark.parametrize("feature", ["model_generation", "tokenizer.unicode"])
def test_generation_permission_precedes_http_even_with_configured_credentials(feature, monkeypatch):
    monkeypatch.setattr("rags.provider_transport.post", lambda *a, **kw: pytest.fail("disabled generation called HTTP"))
    settings = replace(Settings.from_env(), model_provider="openai", generation_api_key="fixture-key", features=disabled(feature))
    with pytest.raises(CapabilityError) as error:
        Providers(settings).complete("instruction", "question")
    assert error.value.code == "feature_disabled"


@pytest.mark.parametrize("feature", ["image_captioning", "model_generation"])
def test_vision_permission_precedes_image_encoding_or_http(feature, monkeypatch):
    monkeypatch.setattr("rags.provider_transport.post", lambda *a, **kw: pytest.fail("disabled vision called HTTP"))
    settings = replace(Settings.from_env(), model_provider="openai", features=disabled(feature))
    with pytest.raises(CapabilityError, match="Feature is disabled"):
        describe_image(b"not-an-image", "image/png", settings)


@pytest.mark.parametrize("mode", ["colbert", "multivector", "multimodal_native"])
def test_external_encoder_adapters_require_research_permission(mode):
    manager = BackendManager(replace(Settings.from_env(), features=disabled("research")))
    with pytest.raises(CapabilityError, match="research"):
        manager.search(mode, tenant="a", question="source", chunks=[], top_k=1)
    assert {entry["id"]: entry for entry in manager.catalog()}[mode]["status"] == "feature_disabled"


def test_local_late_interaction_encoder_requires_embedding_permission(monkeypatch):
    monkeypatch.setenv("RAGS_LATE_INTERACTION_MODEL", "fixture-checkpoint")
    manager = BackendManager(replace(Settings.from_env(), features=disabled("embedding.fastembed")))
    with pytest.raises(CapabilityError, match="embedding.fastembed"):
        manager.search("colbert", tenant="a", question="source", chunks=[], top_k=1)
    with pytest.raises(CapabilityError, match="embedding.fastembed"):
        manager.encode_tokens("source")
