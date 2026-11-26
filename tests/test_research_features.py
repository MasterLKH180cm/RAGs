"""Research permissions precede tokenizers, model loads and temporary indexes."""
import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from rags.config import Settings
from rags.features import FeatureFlags
from rags.providers import Providers
from rags.registry import CapabilityError
from rags.research.common import ProviderBridge
from rags.research import lightrag_adapter, raptor_adapter


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    monkeypatch.setenv("RAGS_ENV", "test")
    monkeypatch.delenv("RAGS_FEATURES_FILE", raising=False)
    monkeypatch.delenv("RAGS_FEATURES_JSON", raising=False)
    # This is an injected provider boundary; no actual ONNX runtime is needed.
    monkeypatch.setattr("rags.research.common.disable_encoder_telemetry", lambda: None)


def provider(*, embedding="fastembed", features=None):
    calls = []
    settings = SimpleNamespace(embedding_provider=embedding, embedding_model="text-embedding-3-small",
                               generation_tokenizer_config={"backend": "unicode"})
    if features is not None:
        settings.features = features
    def embedded(text, *, query=False):
        calls.append(("query" if query else "document", text))
        return [3.0, 4.0]
    def generated(system, user):
        calls.append(("generation", user))
        return "bounded result"
    return SimpleNamespace(
        settings=settings, has_generation=True, embedding_model="fixture:trained", model_identity="fixture:generation",
        embedding=embedded, query_embedding=lambda text: embedded(text, query=True), complete=generated,
        embedding_token_count=lambda text, query=False: len(text), embedding_token_limit=512, calls=calls,
    )


def disabled(name):
    return FeatureFlags(((name, False),))


@pytest.mark.integration
@pytest.mark.parametrize("engine", ["raptor", "lightrag"])
@pytest.mark.parametrize("feature", ["research", "query", "model_generation", "embedding.fastembed",
                                      "tokenizer.unicode", "tokenizer.tiktoken", "engine"])
def test_disabled_adapter_dependencies_leave_no_tokenizer_encoder_http_or_index(
    tmp_path, monkeypatch, engine, feature,
):
    feature = "strategy." + engine if feature == "engine" else feature
    selected = provider(features=disabled(feature))
    def forbidden(*args, **kwargs):
        pytest.fail("Disabled research reached an optional dependency or provider")
    monkeypatch.setattr("rags.research.common.resolve_tokenizer", forbidden)
    monkeypatch.setattr("rags.providers._semantic_encoder", forbidden)
    monkeypatch.setattr("rags.provider_transport.post", forbidden)
    adapter = raptor_adapter if engine == "raptor" else lightrag_adapter
    monkeypatch.setattr(adapter, "verify_source", forbidden)
    with pytest.raises(CapabilityError) as error:
        result = adapter.search("Which policy applies?", [{"id": "source", "text": "Authorized policy."}],
                                1, tmp_path, selected)
        if engine == "lightrag":
            asyncio.run(result)
    assert error.value.code == "feature_disabled" and error.value.status_code == 403
    assert feature in error.value.reason
    assert selected.calls == [] and list(tmp_path.iterdir()) == []


@pytest.mark.unit
@pytest.mark.parametrize("feature", ["research", "model_generation", "embedding.fastembed", "tokenizer.unicode"])
def test_provider_without_feature_field_honors_operator_environment_before_loading(monkeypatch, feature):
    monkeypatch.setenv("RAGS_FEATURES_JSON", '{"' + feature + '":false}')
    monkeypatch.setattr("rags.research.common.resolve_tokenizer", lambda *_: pytest.fail("disabled tokenizer loaded"))
    selected = provider()
    assert not hasattr(selected.settings, "features")
    with pytest.raises(CapabilityError) as error:
        ProviderBridge(selected)
    assert error.value.code == "feature_disabled" and feature in error.value.reason
    assert selected.calls == []


@pytest.mark.unit
def test_invalid_explicit_permissions_are_not_replaced_with_environment_defaults():
    with pytest.raises(ValueError, match="validated operator"):
        ProviderBridge(provider(features={"research": False}))


@pytest.mark.unit
@pytest.mark.parametrize("embedding", ["fastembed", "openai", "sentence_transformers"])
def test_exact_injected_counter_remains_usable_without_native_tiktoken(embedding, monkeypatch):
    # An explicit Settings snapshot remains authoritative over ambient defaults.
    monkeypatch.setenv("RAGS_FEATURES_JSON", '{"research":false}')
    selected = provider(embedding=embedding, features=disabled("tokenizer.tiktoken"))
    bridge = ProviderBridge(selected)
    monkeypatch.setattr("rags.research.common.resolve_tokenizer", lambda *_: pytest.fail("unused native tokenizer loaded"))
    monkeypatch.setattr("rags.providers._semantic_encoder", lambda *_: pytest.fail("injected encoder reloaded"))
    assert bridge.embed("source") == pytest.approx([0.6, 0.8])
    assert bridge.embed("source") == pytest.approx([0.6, 0.8])
    assert bridge.embed("source", query=True) == pytest.approx([0.6, 0.8])
    assert bridge.complete("question", system="instruction") == "bounded result"
    assert selected.calls == [("document", "source"), ("query", "source"), ("generation", "question")]
    assert bridge.diagnostics()["embedding_window_policy"] == "provider_exact_counter_reject_oversize"
    assert bridge.embedding_calls == 2 and bridge.query_embedding_calls == 1


@pytest.mark.unit
def test_native_openai_counter_rejects_disabled_tiktoken_before_loading_or_http(monkeypatch):
    configured = replace(Settings.from_env(), embedding_provider="openai", model_provider="openai",
                         openai_key="local-fixture-only", generation_tokenizer_config={"backend": "unicode"},
                         features=disabled("tokenizer.tiktoken"))
    bridge = ProviderBridge(Providers(configured))
    monkeypatch.setattr("rags.research.common.resolve_tokenizer", lambda *_: pytest.fail("disabled tiktoken loaded"))
    monkeypatch.setattr("rags.provider_transport.post", lambda *a, **kw: pytest.fail("disabled research called HTTP"))
    with pytest.raises(CapabilityError) as error:
        bridge.embed("Authorized source text")
    assert error.value.code == "feature_disabled" and "tokenizer.tiktoken" in error.value.reason
    assert bridge.embedding_calls == 0 and bridge.generation_calls == 0


@pytest.mark.unit
def test_native_fastembed_load_rechecks_permission_at_lazy_counter_boundary(monkeypatch):
    selected = provider(features=FeatureFlags())
    del selected.embedding_token_count
    bridge = ProviderBridge(selected)
    bridge.features = disabled("embedding.fastembed")
    monkeypatch.setattr("rags.providers._semantic_encoder", lambda *_: pytest.fail("disabled encoder loaded"))
    with pytest.raises(CapabilityError) as error:
        bridge.embed("Authorized source text")
    assert error.value.code == "feature_disabled" and "embedding.fastembed" in error.value.reason
    assert bridge.embedding_calls == 0 and selected.calls == []
