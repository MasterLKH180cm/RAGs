"""Embedding choices and window contracts without downloaded checkpoints."""
from dataclasses import replace
import json
import sys
from types import SimpleNamespace

import httpx
import pytest

import rags.embedding_runtime as runtime
import rags.providers as provider_module
from rags.agents.config import AgentPlatformConfig
from rags.agents.router import ModelRouter
from rags.config import Settings
from rags.embedding_runtime import EmbeddingWindow
from rags.providers import Providers
from rags.registry import CapabilityError

pytestmark = pytest.mark.unit


@pytest.fixture
def settings(tmp_path):
    return Settings(db_path=str(tmp_path / "embedding.db"), keys={"fixture-key": "tenant"},
                    embedding_provider="openai", model_provider="extractive", openai_key="ambient-do-not-send",
                    openai_base_url="https://provider.invalid/v1", embedding_model="custom-embedding",
                    chat_model="unused", anthropic_key="", anthropic_model="unused",
                    embedding_tokenizer_config={"backend": "utf8_bytes"}, embedding_max_tokens=64)


def response(monkeypatch, payload):
    calls = []
    def post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        return httpx.Response(200, request=httpx.Request("POST", url), json=payload)
    monkeypatch.setattr(provider_module.provider_transport, "post", post)
    monkeypatch.setattr(provider_module.provider_transport, "bounded_post", post)
    return calls


@pytest.mark.parametrize("change", [
    {"embedding_auth_mode": "false"},
    {"embedding_auth_mode": "none", "embedding_base_url": "http://localhost:11434"},
    {"embedding_auth_mode": "none", "embedding_base_url": "http://192.0.2.1:11434"},
    {"embedding_auth_mode": "none", "embedding_base_url": "http://127.0.0.1:11434?x=1"},
    {"embedding_auth_mode": "none", "embedding_base_url": "http://127.0.0.1:11434", "embedding_api_key": "credential"},
    {"embedding_provider": "fastembed", "embedding_auth_mode": "none", "embedding_base_url": "http://127.0.0.1:1"},
    {"embedding_dimensions": True}, {"embedding_dimensions": 0}, {"embedding_dimensions": 16385},
    {"embedding_dimensions": "384"}, {"embedding_provider": "fastembed", "embedding_dimensions": 384},
    {"embedding_revision": "main"}, {"embedding_query_prefix": None},
    {"embedding_document_prefix": "x" * 1025}, {"embedding_query_prefix": "\ud800"},
    {"embedding_query_input_type": "query\nAuthorization"},
    {"embedding_provider": "local", "embedding_document_input_type": "document"},
    {"embedding_tokenizer_config": None}, {"embedding_max_tokens": None},
    {"embedding_max_tokens": True}, {"embedding_max_tokens": 0}, {"embedding_max_tokens": 100001},
    {"embedding_tokenizer_config": {"backend": "unicode"}},
    {"embedding_tokenizer_config": {"backend": "utf8_bytes", "unknown": True}},
])
def test_embedding_settings_reject_ambiguous_or_unsafe_choices(settings, change):
    with pytest.raises(ValueError):
        replace(settings, **change)


def test_explicit_custom_window_counts_both_role_prefixes_and_unicode_bytes(settings, monkeypatch):
    calls = response(monkeypatch, {"data": [{"embedding": [1.0, 0.0]}]})
    provider = Providers(replace(settings, embedding_query_prefix="query: ", embedding_document_prefix="doc: ",
                                 embedding_max_tokens=8))
    assert provider.embedding("中") == [1.0, 0.0]
    assert calls[0]["json"]["input"] == "doc: 中"
    with pytest.raises(CapabilityError) as error:
        provider.query_embedding("中")
    assert error.value.code == "context_budget_exceeded" and len(calls) == 1
    assert provider.embedding_window_policy == "operator_byte_budget_reject_oversize"


def test_unknown_model_requires_declared_window_before_network(settings, monkeypatch):
    calls = response(monkeypatch, {"data": [{"embedding": [1.0]}]})
    provider = Providers(replace(settings, embedding_tokenizer_config=None, embedding_max_tokens=None))
    with pytest.raises(CapabilityError) as error:
        provider.embedding("source")
    assert error.value.code == "requires_provider" and calls == []


@pytest.mark.parametrize("payload", [
    {"data": [{"embedding": [1.0]}, {"embedding": [2.0]}]},
    {"data": [{"embedding": [1.0], "index": 1}]},
    {"data": [{"embedding": [1.0], "index": False}]},
])
def test_openai_single_input_rejects_ambiguous_result_count_and_index(settings, monkeypatch, payload):
    response(monkeypatch, payload)
    with pytest.raises(CapabilityError, match="invalid embedding"):
        Providers(settings).embedding("source")


def test_embedding_dimensions_are_requested_checked_and_stable(settings, monkeypatch):
    payload = {"data": [{"embedding": [1.0, 0.0]}]}
    calls = response(monkeypatch, payload)
    provider = Providers(replace(settings, embedding_dimensions=2))
    assert provider.embedding("source") == [1.0, 0.0]
    assert calls[0]["json"]["dimensions"] == 2
    payload["data"][0]["embedding"] = [1.0]
    with pytest.raises(CapabilityError, match="dimensions"):
        provider.query_embedding("question")
    native = Providers(settings)
    assert native.embedding("source") == [1.0]
    payload["data"][0]["embedding"] = [1.0, 0.0]
    with pytest.raises(CapabilityError, match="dimensions"):
        native.embedding("new source")


@pytest.mark.parametrize("change", [
    {"embedding_query_prefix": "query: "}, {"embedding_document_prefix": "passage: "},
    {"embedding_query_input_type": "search_query"}, {"embedding_document_input_type": "search_document"},
    {"embedding_dimensions": 256}, {"embedding_max_tokens": 32},
])
def test_vector_affecting_choices_have_distinct_bounded_public_identities(settings, change):
    old = Providers(settings).embedding_model
    new = Providers(replace(settings, **change)).embedding_model
    assert old != new and len(new) <= 160
    assert "query: " not in new and "passage: " not in new


def test_sentence_revision_changes_identity_without_loading_runtime(settings, monkeypatch):
    monkeypatch.setattr(provider_module, "sentence_encoder", lambda *a: pytest.fail("identity must not download a model"))
    first = replace(settings, embedding_provider="sentence_transformers", embedding_model="BAAI/bge-m3",
                    embedding_revision="a" * 40)
    assert Providers(first).embedding_model != Providers(replace(first, embedding_revision="b" * 40)).embedding_model
    long = Providers(replace(settings, embedding_model="x" * 160)).embedding_model
    assert len(long) <= 160


@pytest.mark.parametrize("provider", ["openai", "ollama"])
def test_no_auth_embedding_cannot_send_ambient_credentials_or_use_proxies(settings, monkeypatch, provider):
    payload = {"embeddings": [[1.0, 0.0]], "prompt_eval_count": 7} if provider == "ollama" else {"data": [{"embedding": [1.0, 0.0]}]}
    calls = response(monkeypatch, payload)
    configured = replace(settings, embedding_provider=provider, embedding_auth_mode="none",
                         embedding_base_url="http://127.0.0.1:11434", embedding_query_prefix="query: ",
                         embedding_tokenizer_config=None, embedding_max_tokens=None,
                         embedding_model="text-embedding-3-small" if provider == "openai" else "nomic-embed-text")
    adapter = Providers(configured)
    adapter.begin_usage()
    assert adapter.query_embedding("source") == [1.0, 0.0]
    assert calls[0]["headers"] == {} and calls[0]["trust_env"] is False
    assert "ambient-do-not-send" not in json.dumps(calls)
    if provider == "ollama":
        assert calls[0]["url"].endswith("/api/embed")
        assert calls[0]["json"] == {"model": "nomic-embed-text", "input": "query: source", "truncate": False}
        assert adapter.usage()["provider_reports"][0]["usage"]["prompt_tokens"] == 7
        assert adapter.embedding_window_policy == "server_native_reject_oversize_with_byte_ceiling"


@pytest.mark.parametrize("payload", [
    {"embeddings": []}, {"embeddings": [[1.0], [2.0]]}, {"embeddings": [[True]]},
    {"embeddings": [[1.0]], "prompt_eval_count": True},
    {"embeddings": [[1.0]], "prompt_eval_count": -1},
])
def test_ollama_rejects_invalid_vectors_and_usage(settings, monkeypatch, payload):
    response(monkeypatch, payload)
    configured = replace(settings, embedding_provider="ollama", embedding_auth_mode="none",
                         embedding_base_url="http://127.0.0.1:11434")
    with pytest.raises(CapabilityError) as error:
        Providers(configured).embedding("source")
    assert error.value.code == "invalid_embedding"


def test_ollama_never_falls_back_to_ambient_openai_credential(settings, monkeypatch):
    calls = response(monkeypatch, {"embeddings": [[1.0]]})
    with pytest.raises(ValueError, match="RAGS_EMBEDDING_API_KEY"):
        Providers(replace(settings, embedding_provider="ollama")).embedding("source")
    assert calls == []


def test_sentence_adapter_uses_native_role_prompt_and_dimension_contract(settings, monkeypatch):
    calls = []
    def encode(role, text, **kwargs):
        calls.append((role, text, kwargs))
        return SimpleNamespace(tolist=lambda: [1.0, 0.0])
    encoder = SimpleNamespace(prompts={"query": "search: ", "document": "passage: "}, default_prompt_name=None,
                              encode_query=lambda text, **kw: encode("query", text, **kw),
                              encode_document=lambda text, **kw: encode("document", text, **kw))
    monkeypatch.setattr(provider_module, "sentence_encoder", lambda *a: encoder)
    monkeypatch.setattr(provider_module, "sentence_window", lambda *a: EmbeddingWindow(
        lambda text, query=False: len(text) + 10, 64, "native_tokenizer_reject_oversize"))
    adapter = Providers(replace(settings, embedding_provider="sentence_transformers", embedding_dimensions=2,
                                embedding_query_prefix="", embedding_document_prefix="doc: "))
    adapter.query_embedding("query source")
    adapter.embedding("document source")
    assert calls[0][0:2] == ("query", "query source") and calls[0][2]["prompt"] == "search: "
    assert calls[1][0:2] == ("document", "document source") and calls[1][2]["prompt"] == "doc: "
    assert all(call[2]["normalize_embeddings"] is True and call[2]["truncate_dim"] == 2 for call in calls)
    with pytest.raises(CapabilityError, match="window"):
        adapter.embedding("x" * 60)
    assert len(calls) == 2


def test_sentence_native_counter_counts_prefix_specials_and_whitespace(monkeypatch):
    seen = []
    class Transformer:
        modality_config = {"text": {}}
        can_flatten_inputs = False
        processing_kwargs = {}
        def preprocess(self, inputs, **kwargs):
            value = kwargs["prompt"] + inputs[0]
            seen.append((value, kwargs))
            return {"input_ids": [[0, *range(len(value)), 1]]}
    monkeypatch.setitem(sys.modules, "sentence_transformers.sentence_transformer.modules", SimpleNamespace(Transformer=Transformer))
    encoder = SimpleNamespace(_first_module=lambda: Transformer(), max_seq_length=16, tokenizer=SimpleNamespace(),
                              prompts={"query": "Query: "}, default_prompt_name=None)
    window = runtime.sentence_window(encoder, "", "")
    assert window.count(" Hi ", query=True) == len("Query:  Hi ") + 2
    assert seen[-1][0] == "Query:  Hi "
    assert seen[-1][1]["processing_kwargs"] == {"text": {"truncation": False, "padding": False},
                                                "common": {"return_tensors": None}}
    with pytest.raises(CapabilityError, match="window"):
        window.validate("x" * 16)


def test_sentence_loader_is_optional_and_disallows_remote_code(settings, monkeypatch):
    runtime.sentence_encoder.cache_clear()
    calls = []
    expected = object()
    def construct(*args, **kwargs):
        calls.append((args, kwargs))
        return expected
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=construct))
    try:
        assert runtime.sentence_encoder("model", "revision", "cache") is expected
        assert runtime.sentence_encoder("model", "revision", "cache") is expected
        assert len(calls) == 1
        assert calls[0][1] == {"revision": "revision", "cache_folder": "cache", "trust_remote_code": False,
                              "device": "cpu", "model_kwargs": {"use_safetensors": True}}
    finally:
        runtime.sentence_encoder.cache_clear()


@pytest.mark.parametrize("dependency", [None, SimpleNamespace(SentenceTransformer=lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("private")))])
def test_sentence_loader_sanitizes_dependency_and_checkpoint_failures(monkeypatch, dependency):
    runtime.sentence_encoder.cache_clear()
    monkeypatch.setitem(sys.modules, "sentence_transformers", dependency)
    try:
        with pytest.raises(CapabilityError) as error:
            runtime.sentence_encoder("unavailable")
        assert error.value.code == "requires_provider" and "private" not in str(error.value)
    finally:
        runtime.sentence_encoder.cache_clear()


def test_router_forwards_role_and_dimensions_with_approved_alias(monkeypatch):
    monkeypatch.setenv("RAGS_ROUTER_GATEWAY_URL", "https://gateway.invalid/v1")
    monkeypatch.setenv("RAGS_ROUTER_API_KEY", "gateway-fixture-key")
    monkeypatch.setenv("RAGS_ROUTER_MODELS_JSON", '{"embed":"cohere/embed-english-v3.0"}')
    router = ModelRouter(AgentPlatformConfig.from_env())
    calls = []
    def post(path, body, timeout):
        calls.append(body)
        return httpx.Response(200, json={"data": [{"embedding": [1.0, 0.0]}]})
    monkeypatch.setattr(router, "_post", post)
    reply = router.embedding(model="embed", text="source", dimensions=2, input_type="search_document")
    assert reply.vector == [1.0, 0.0]
    assert calls == [{"model": "cohere/embed-english-v3.0", "input": "source", "dimensions": 2, "input_type": "search_document"}]
    with pytest.raises(ValueError):
        router.embedding(model="embed", text="source", dimensions=True)
    with pytest.raises(ValueError):
        router.embedding(model="embed", text="source", input_type="\nsecret")
    assert len(calls) == 1


def test_router_numeric_overflow_is_a_sanitized_vector_error(monkeypatch):
    monkeypatch.setenv("RAGS_ROUTER_GATEWAY_URL", "https://gateway.invalid/v1")
    monkeypatch.setenv("RAGS_ROUTER_API_KEY", "gateway-fixture-key")
    monkeypatch.setenv("RAGS_ROUTER_MODELS_JSON", '{"embed":"openai/custom"}')
    router = ModelRouter(AgentPlatformConfig.from_env())
    monkeypatch.setattr(router, "_post", lambda *args: httpx.Response(200, json={"data": [{"embedding": [10 ** 400]}]}))
    with pytest.raises(CapabilityError) as error:
        router.embedding(model="embed", text="source")
    assert error.value.code == "invalid_embedding"


def test_loaded_tokenizer_snapshot_is_shared_by_identity_and_count(settings, monkeypatch):
    calls = []
    def resolve(config):
        serial = len(calls)
        calls.append(config)
        return SimpleNamespace(profile={"backend": "utf8_bytes", "version": serial},
                               count_kind="conservative_bytes", count=lambda text: len(text) + serial)
    monkeypatch.setattr("rags.tokenization.resolve_tokenizer", resolve)
    provider = Providers(settings)
    identity = provider.embedding_model
    assert provider.embedding_token_count("abc") == 3
    assert provider.embedding_model == identity and len(calls) == 1
    fresh = Providers(settings)
    assert fresh.embedding_model != identity and fresh.embedding_token_count("abc") == 4
    assert len(calls) == 2


def test_no_auth_only_embedding_configuration_does_not_read_hosted_secret_files(monkeypatch, tmp_path):
    monkeypatch.setenv("RAGS_ENV", "test")
    monkeypatch.setenv("RAGS_EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("RAGS_EMBEDDING_AUTH_MODE", "none")
    monkeypatch.setenv("RAGS_EMBEDDING_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("RAGS_MODEL_PROVIDER", "extractive")
    monkeypatch.setenv("OPENAI_API_KEY_FILE", str(tmp_path / "unused-openai-secret"))
    monkeypatch.setenv("ANTHROPIC_API_KEY_FILE", str(tmp_path / "unused-anthropic-secret"))
    configured = Settings.from_env()
    assert configured.embedding_credential == "" and configured.openai_key == "" and configured.anthropic_key == ""
