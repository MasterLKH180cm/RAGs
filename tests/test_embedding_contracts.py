"""Embedding selection and native-runtime contracts without downloads or paid APIs."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import json
import sys
from threading import Barrier, Event, Lock
from types import ModuleType, SimpleNamespace

import httpx
import pytest

from rags.agents.config import AgentPlatformConfig
from rags.agents.router import ModelRouter
from rags.config import Settings
from rags.embedding_profiles import EmbeddingProfile, parse_embedding_profiles
from rags.features import FeatureFlags
from rags import model_discovery, providers, sentence_embeddings
from rags.providers import Providers
from rags.registry import CapabilityError

pytestmark = pytest.mark.unit
REVISION = "a" * 40
COUNTER = {"backend": "tiktoken", "encoding": "cl100k_base"}


@pytest.fixture
def settings():
    return Settings(
        db_path="unused-contract.db", keys={"fixture": "tenant"}, embedding_provider="local",
        model_provider="extractive", openai_key="", openai_base_url="https://provider.invalid/v1",
        embedding_model="text-embedding-3-small", chat_model="fixture-chat",
        anthropic_key="", anthropic_model="fixture-chat",
    )


@pytest.mark.parametrize("change", [
    {"provider": "unknown"}, {"provider": []}, {"model": ""}, {"model": "bad model"},
    {"model": "x" * 151}, {"revision": "main"}, {"revision": REVISION},
    {"dimensions": True}, {"dimensions": 0}, {"dimensions": 4097},
    {"query_prefix": None}, {"document_prefix": "\x00"}, {"query_prefix": "x" * 2049},
    {"normalize": 1}, {"input_policy": "truncate"}, {"input_policy": "server_reject"},
    {"base_url": "file:///private/model"}, {"base_url": "https://user:password@provider.invalid"},
    {"base_url": "https://provider.invalid/?secret=1"}, {"base_url": "https://provider.invalid/#fragment"},
    {"base_url": "http://provider.invalid:0"}, {"base_url": "http://provider.invalid:99999"},
    {"base_url": "https://provider.invalid/\n"}, {"api_key_env": "lowercase-secret"},
    {"api_key_env": "KEY/FILE"}, {"auth_mode": "none", "base_url": "https://remote.invalid"},
    {"auth_mode": "none", "base_url": "http://127.0.0.1:11434", "api_key_env": "KEY"},
    {"space_id": "contains spaces"}, {"special_tokens": 0},
    {"tokenizer": COUNTER}, {"token_limit": 10},
    {"tokenizer": COUNTER, "token_limit": 10},
    {"tokenizer": COUNTER, "token_limit": 10, "special_tokens": True},
    {"tokenizer": COUNTER, "token_limit": 10, "special_tokens": -1},
    {"tokenizer": COUNTER, "token_limit": 10, "special_tokens": 1025},
    {"tokenizer": COUNTER, "token_limit": True, "special_tokens": 0},
    {"tokenizer": {"backend": "unicode"}, "token_limit": 10, "special_tokens": 0},
    {"tokenizer": {"backend": "tiktoken", "host_path": "/private"}, "token_limit": 10, "special_tokens": 0},
])
def test_profile_rejects_invalid_or_unsafe_configuration(change):
    with pytest.raises(ValueError):
        EmbeddingProfile(**({"provider": "openai", "model": "text-embedding-3-small"} | change))


@pytest.mark.parametrize("change", [{"model": "model\ud800"}, {"query_prefix": "prefix\udfff"},
                                    {"document_prefix": "prefix\ud800"}])
def test_profile_rejects_unpaired_surrogates_before_identity_or_model_loading(change):
    with pytest.raises(ValueError):
        EmbeddingProfile(**({"provider": "openai", "model": "text-embedding-3-small"} | change))


@pytest.mark.parametrize("profile", [
    {"provider": "sentence_transformers", "model": "org/model"},
    {"provider": "sentence_transformers", "model": "/private/checkpoint", "revision": REVISION},
    {"provider": "fastembed", "model": "org/model", "dimensions": 20},
    {"provider": "router", "model": "approved", "base_url": "https://provider.invalid"},
    {"provider": "local", "model": "demo", "api_key_env": "KEY"},
    {"provider": "fastembed", "model": "org/model", "tokenizer": COUNTER,
     "token_limit": 10, "special_tokens": 0},
    {"provider": "ollama", "model": "fixture"},
    {"provider": "ollama", "model": "fixture", "base_url": "http://127.0.0.1:11434", "input_policy": "exact"},
])
def test_profiles_require_runtime_specific_contracts(profile):
    with pytest.raises(ValueError):
        EmbeddingProfile(**profile)


@pytest.mark.parametrize("value", [
    "[]", '{"same":{"provider":"local","model":"demo"},"same":{}}',
    '{"same":{"provider":"local","provider":"openai","model":"demo"}}',
    '{"same":{"provider":"local","model":NaN}}', " " * 65_537,
    {str(index): {"provider": "local", "model": "demo"} for index in range(17)},
    {"../path": {"provider": "local", "model": "demo"}}, {"named": None},
    {"named": {"provider": "local", "model": "demo", "api_key": "private"}},
])
def test_profile_registry_rejects_duplicate_unknown_oversized_and_invalid_entries(value):
    with pytest.raises(ValueError):
        parse_embedding_profiles(value)


def test_registry_copies_inputs_and_public_profiles_omit_operator_credentials_and_paths():
    raw = {"approved": {"provider": "openai", "model": "custom", "base_url": "https://private.invalid/v1",
                        "api_key_env": "PRIVATE_EMBEDDING_SECRET", "tokenizer": dict(COUNTER),
                        "token_limit": 25, "special_tokens": 2}}
    profiles = parse_embedding_profiles(raw)
    raw["approved"]["tokenizer"]["encoding"] = "changed"
    assert profiles["approved"].tokenizer["encoding"] == "cl100k_base"
    public = json.dumps(profiles["approved"].public())
    assert "private.invalid" not in public and "PRIVATE_EMBEDDING_SECRET" not in public
    assert "tokenizer" not in profiles["approved"].public()


def test_discovery_is_pure_reports_effective_default_and_does_not_claim_runtime_validation(settings, monkeypatch):
    profile = EmbeddingProfile(provider="openai", model="custom", base_url="https://private.invalid/v1",
                               api_key_env="PRIVATE_EMBEDDING_SECRET")
    selected = replace(settings, embedding_profiles={"approved": profile},
                       tokenizer_artifacts={"registered": {"backend": "huggingface",
                                                           "path": "/private/tokenizer.json", "sha256": "a" * 64}})

    def unavailable(name):
        raise model_discovery.PackageNotFoundError(name)

    monkeypatch.setattr(model_discovery, "version", unavailable)
    monkeypatch.setattr(providers, "_semantic_encoder", lambda *_a, **_k: pytest.fail("Discovery loaded a model"))
    monkeypatch.setattr(sentence_embeddings, "sentence_encoder", lambda *_a, **_k: pytest.fail("Discovery loaded a model"))
    monkeypatch.setattr("rags.config.secret", lambda *_a: pytest.fail("Discovery loaded a credential"))
    result = model_discovery.model_discovery(selected)
    assert result["configured"]["embedding"]["model"] == "offline-hash-v1"
    assert result["configured"]["embedding"]["readiness"]["runtime_validation"] == "not_run"
    assert all(item["runtime_validation"] == "not_run" for item in result["embeddings"])
    assert not next(item for item in result["embeddings"] if item["provider"] == "fastembed")["dependency"]["installed"]
    encoded = json.dumps(result)
    assert "/private" not in encoded and "PRIVATE_EMBEDDING_SECRET" not in encoded and "private.invalid" not in encoded
    assert model_discovery.configured_models(replace(settings, embedding_provider="router"))["embedding"]["input_policy"] == "resolve_alias_at_runtime"


@pytest.mark.parametrize("provider,expected", [
    ("local", "offline-hash-v1"), ("fastembed", "fastembed:BAAI/bge-small-en-v1.5"),
    ("openai", "openai:text-embedding-3-small"),
])
def test_legacy_identity_stays_exact_and_generation_counter_does_not_reembed(settings, provider, expected):
    configured = replace(settings, embedding_provider=provider)
    assert Providers(configured).embedding_model == expected
    assert Providers(replace(configured, generation_tokenizer_config={"backend": "utf8_bytes"})).embedding_model == expected


@pytest.mark.parametrize("change", [
    {"embedding_query_prefix": "query: "}, {"embedding_document_prefix": "passage: "},
    {"embedding_dimensions": 2}, {"embedding_normalize": False},
    {"embedding_tokenizer_config": COUNTER, "embedding_token_limit_config": 7, "embedding_special_tokens": 0},
])
def test_new_vector_affecting_options_are_bound_to_the_embedding_identity(settings, change):
    configured = replace(settings, embedding_provider="openai")
    current = Providers(configured).embedding_model
    modified = Providers(replace(configured, **change)).embedding_model
    assert modified.startswith(current + "@") and modified != current
    assert len(modified) <= 155


def test_native_artifact_digest_and_special_token_contract_affect_identity_without_reading_files(settings):
    counter = {"backend": "huggingface", "model": "approved"}
    registered = {"approved": {"backend": "huggingface", "path": "/nonexistent/tokenizer.json", "sha256": "a" * 64}}
    configured = replace(settings, embedding_provider="openai", embedding_tokenizer_config=counter,
                         embedding_token_limit_config=20, embedding_special_tokens=2, tokenizer_artifacts=registered)
    original = Providers(configured).embedding_model
    changed_artifact = deepcopy(registered)
    changed_artifact["approved"]["sha256"] = "b" * 64
    assert Providers(replace(configured, tokenizer_artifacts=changed_artifact)).embedding_model != original
    assert Providers(replace(configured, embedding_special_tokens=3)).embedding_model != original
    changed_path = deepcopy(registered)
    changed_path["approved"]["path"] = "/another/operator/path.json"
    assert Providers(replace(configured, tokenizer_artifacts=changed_path)).embedding_model == original
    assert "/nonexistent" not in json.dumps(Providers(configured).embedding_contract)


def test_disabled_unknown_profiles_and_missing_credentials_fail_before_loading(settings, monkeypatch):
    approved = EmbeddingProfile(provider="fastembed", model="fixture/model")
    configured = replace(settings, embedding_profiles={"approved": approved},
                         features=FeatureFlags((("embedding.fastembed", False),)))
    monkeypatch.setattr(providers, "_semantic_encoder", lambda *_a, **_k: pytest.fail("Forbidden model loading"))
    monkeypatch.setattr("rags.config.secret", lambda *_a: pytest.fail("Forbidden credential loading"))
    service = Providers(configured)
    for alias, code in (("missing", "unknown_embedding_profile"), ("approved", "feature_disabled")):
        with pytest.raises(CapabilityError) as error:
            service.for_embedding_profile(alias)
        assert error.value.code == code
    monkeypatch.setattr("rags.tokenization.resolve_tokenizer", lambda *_a, **_k: pytest.fail("Unauthenticated counter loading"))
    with pytest.raises(CapabilityError) as error:
        Providers(replace(settings, embedding_provider="openai")).embedding("source")
    assert error.value.code == "requires_provider"


def test_profile_selection_is_request_local_and_credentials_do_not_bleed(settings, monkeypatch):
    profiles = {alias: EmbeddingProfile(provider="openai", model="text-embedding-3-small",
                                        api_key_env="PROFILE_" + alias.upper(), query_prefix=alias + ":")
                for alias in ("first", "second")}
    configured = replace(settings, embedding_profiles=profiles)
    parent = Providers(configured)
    monkeypatch.setenv("PROFILE_FIRST", "first-private-key")
    monkeypatch.setenv("PROFILE_SECOND", "second-private-key")
    monkeypatch.setattr("rags.tokenization.resolve_tokenizer", lambda *_a, **_k: SimpleNamespace(count=len))
    observed = []
    lock = Lock()

    def post(url, **kwargs):
        with lock:
            observed.append((kwargs["headers"]["Authorization"], kwargs["json"]["input"]))
        return httpx.Response(200, request=httpx.Request("POST", url), json={"data": [{"embedding": [1.0, 0.0]}]})

    monkeypatch.setattr(providers.provider_transport, "post", post)
    with ThreadPoolExecutor(max_workers=2) as pool:
        result = list(pool.map(lambda alias: parent.for_embedding_profile(alias).query_embedding("text"), profiles))
    assert result == [[1.0, 0.0], [1.0, 0.0]]
    assert sorted(observed) == [("Bearer first-private-key", "first:text"), ("Bearer second-private-key", "second:text")]
    assert parent.settings is configured and parent.embedding_model == "offline-hash-v1"
    assert parent.settings.embedding_query_prefix == "" and parent.settings.embedding_api_key is None


def test_custom_counter_includes_prefix_and_verified_special_tokens_before_network(settings, monkeypatch):
    selected = replace(settings, embedding_provider="openai", embedding_model="custom",
                       embedding_api_key="fixture", embedding_document_prefix="d:",
                       embedding_tokenizer_config=COUNTER, embedding_token_limit_config=6,
                       embedding_special_tokens=2)
    monkeypatch.setattr("rags.tokenization.resolve_tokenizer", lambda *_a, **_k: SimpleNamespace(count=len))
    sent = []

    def post(url, **kwargs):
        sent.append(kwargs["json"]["input"])
        return httpx.Response(200, request=httpx.Request("POST", url), json={"data": [{"embedding": [1.0]}]})

    monkeypatch.setattr(providers.provider_transport, "post", post)
    provider = Providers(selected)
    assert provider.embedding_token_count("ab") == 6
    assert provider.embedding("ab") == [1.0]
    with pytest.raises(CapabilityError) as error:
        provider.embedding("abc")
    assert error.value.code == "context_budget_exceeded" and sent == ["d:ab"]


@pytest.mark.parametrize("provider,model", [("openai", "unknown-model"), ("router", "approved")])
def test_unknown_remote_window_fails_before_model_call(settings, monkeypatch, provider, model):
    selected = Providers(replace(settings, embedding_provider=provider, embedding_model=model, embedding_api_key="fixture"))
    if provider == "router":
        selected.__dict__["model_router"] = SimpleNamespace(
            config=SimpleNamespace(models={"approved": "ollama/custom"}, gateway_url="https://gateway.invalid", router_ready=True),
            embedding=lambda **_k: pytest.fail("Unknown window reached gateway"),
        )
    monkeypatch.setattr(providers.provider_transport, "post", lambda *_a, **_k: pytest.fail("Unknown window reached provider"))
    with pytest.raises(CapabilityError) as error:
        selected.embedding("source")
    assert error.value.code == "requires_provider"


class Vector:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return list(self.values)


def test_fastembed_role_preprocessing_window_and_batch_preflight_never_truncate(settings, monkeypatch):
    selected = replace(settings, embedding_provider="fastembed", embedding_document_prefix="d:", embedding_query_prefix="q:")
    encoded = []

    def operation(value, *, query=False):
        encoded.append((value, query))
        yield Vector([1.0, 0.0])

    encoder = SimpleNamespace(embed=operation, query_embed=lambda text: operation(text, query=True))
    monkeypatch.setattr(providers, "_semantic_encoder", lambda *_a: encoder)
    monkeypatch.setattr(providers, "fastembed_token_window", lambda _encoder: (lambda text, query=False: len(text) + (2 if query else 1), 8))
    service = Providers(selected)
    assert service.embedding_token_count("x") == 4
    assert service.embedding_token_count("x", query=True) == 5
    assert service.embedding("abcde") == service.query_embedding("x") == [1.0, 0.0]
    before = list(encoded)
    for operation in (lambda: service.query_embedding("abcde"), lambda: service.preflight_embeddings(["x", "abcdef"])):
        with pytest.raises(CapabilityError) as error:
            operation()
        assert error.value.code == "context_budget_exceeded"
    assert encoded == before == [(["d:abcde"], False), ("q:x", True)]


@pytest.mark.parametrize("text", [None, b"bytes", "a\ud800b", "a\udfffb", "x" * 1_000_001],
                         ids=["null", "bytes", "high-surrogate", "low-surrogate", "over-limit"])
def test_invalid_source_text_fails_before_native_counter_or_inference(settings, monkeypatch, text):
    monkeypatch.setattr(providers, "_semantic_encoder", lambda *_a: pytest.fail("Invalid input loaded a model"))
    with pytest.raises(CapabilityError) as error:
        Providers(replace(settings, embedding_provider="fastembed")).embedding(text)
    assert (error.value.code, error.value.status_code) == ("invalid_model_input", 422)


@pytest.mark.parametrize("count", [0, 2, 3])
def test_fastembed_requires_exactly_one_result_and_stops_after_second(settings, monkeypatch, count):
    consumed = []

    def encode(_texts):
        for index in range(count):
            consumed.append(index)
            yield Vector([1.0])

    monkeypatch.setattr(providers, "_semantic_encoder", lambda *_a: SimpleNamespace(embed=encode))
    monkeypatch.setattr(providers, "fastembed_token_window", lambda *_a: (lambda text, query=False: len(text), 100))
    with pytest.raises(CapabilityError) as error:
        Providers(replace(settings, embedding_provider="fastembed")).embedding("source")
    assert (error.value.code, error.value.status_code) == ("invalid_embedding", 502)
    assert len(consumed) == min(count, 2)


@pytest.mark.parametrize("vector", [[], [True], [float("nan")], [float("inf")], [10 ** 400], ["1"], [1.0] * 16_385])
def test_native_provider_vectors_have_the_same_strict_numeric_contract(settings, monkeypatch, vector):
    monkeypatch.setattr(providers, "_semantic_encoder", lambda *_a: SimpleNamespace(embed=lambda _texts: iter([Vector(vector)])))
    monkeypatch.setattr(providers, "fastembed_token_window", lambda *_a: (lambda text, query=False: len(text), 100))
    with pytest.raises(CapabilityError) as error:
        Providers(replace(settings, embedding_provider="fastembed")).embedding("source")
    assert (error.value.code, error.value.status_code) == ("invalid_embedding", 502)


def test_router_overflow_is_a_typed_embedding_failure(settings, monkeypatch):
    configured = replace(settings, embedding_provider="router", embedding_model="approved")
    selected = Providers(configured)
    router = ModelRouter(AgentPlatformConfig(gateway_url="https://gateway.invalid", gateway_key="fixture",
                                           models={"approved": "openai/text-embedding-3-small"}))
    selected.__dict__["model_router"] = router
    monkeypatch.setattr("rags.tokenization.resolve_tokenizer", lambda *_a, **_k: SimpleNamespace(count=len))
    monkeypatch.setattr(router, "_post", lambda *_a: httpx.Response(
        200, request=httpx.Request("POST", "https://gateway.invalid/embeddings"),
        content=json.dumps({"data": [{"embedding": [10 ** 400]}]}).encode(),
    ))
    with pytest.raises(CapabilityError) as error:
        selected.embedding("source")
    assert (error.value.code, error.value.status_code) == ("invalid_embedding", 502)


def ollama_settings(settings, **changes):
    return replace(settings, embedding_provider="ollama", embedding_model="fixture-model",
                   embedding_base_url="http://127.0.0.1:11434", embedding_auth_mode="none", **changes)


def test_ollama_disables_truncation_and_proxy_inheritance_without_claiming_exact_local_count(settings, monkeypatch):
    selected = Providers(ollama_settings(settings, embedding_query_prefix="query:", embedding_dimensions=2))
    sent = []

    def post(url, **kwargs):
        sent.append((url, kwargs))
        return httpx.Response(200, request=httpx.Request("POST", url), json={"embeddings": [[1.0, 0.0]]})

    monkeypatch.setattr(providers.provider_transport, "bounded_post", post)
    assert selected.query_embedding("source") == [1.0, 0.0]
    assert selected.embedding_window_policy == "server_reject"
    with pytest.raises(CapabilityError, match="exact embedding tokenizer"):
        selected.embedding_tokenizer()
    assert sent == [("http://127.0.0.1:11434/api/embed", {
        "headers": {}, "json": {"model": "fixture-model", "input": ["query:source"],
                                 "truncate": False, "dimensions": 2},
        "timeout": 30, "trust_env": False, "accepted_statuses": (400,),
    })]


@pytest.mark.parametrize("payload", [
    {}, {"embeddings": []}, {"embeddings": [[], []]}, {"embeddings": [None]},
    {"embeddings": [[True]]}, {"embeddings": [[float("nan")]]}, {"embeddings": [[10 ** 400]]},
])
def test_ollama_malformed_successes_are_typed_sanitized_errors(settings, monkeypatch, payload):
    monkeypatch.setattr(providers.provider_transport, "bounded_post", lambda url, **_k: httpx.Response(
        200, request=httpx.Request("POST", url), content=json.dumps(payload).encode(),
    ))
    with pytest.raises(CapabilityError) as error:
        Providers(ollama_settings(settings)).embedding("source")
    assert (error.value.code, error.value.status_code) == ("invalid_embedding", 502)


@pytest.mark.parametrize("body,code,status", [
    ({"error": "input length exceeds maximum context length"}, "context_budget_exceeded", 422),
    ({"error": "private runtime credential and checkpoint path"}, "provider_failed", 502),
    ([], "provider_failed", 502),
])
def test_ollama_overflow_is_explicit_and_other_error_bodies_are_not_exposed(settings, monkeypatch, body, code, status):
    monkeypatch.setattr(providers.provider_transport, "bounded_post", lambda url, **_k: httpx.Response(
        400, request=httpx.Request("POST", url), json=body,
    ))
    with pytest.raises(CapabilityError) as error:
        Providers(ollama_settings(settings)).embedding("source")
    assert (error.value.code, error.value.status_code) == (code, status)
    assert "private" not in error.value.reason and "credential" not in error.value.reason


@pytest.mark.parametrize("dimensions,vector", [(2, [1.0]), (3, [1.0, 0.0])])
def test_requested_dimensions_cannot_accept_a_different_vector_size(settings, monkeypatch, dimensions, vector):
    monkeypatch.setattr(providers.provider_transport, "bounded_post", lambda url, **_k: httpx.Response(
        200, request=httpx.Request("POST", url), json={"embeddings": [vector]},
    ))
    with pytest.raises(CapabilityError) as error:
        Providers(ollama_settings(settings, embedding_dimensions=dimensions)).embedding("source")
    assert error.value.code == "invalid_embedding"


def recipe_files(*, normalize=True):
    kinds = ["Transformer", "Pooling"] + (["Normalize"] if normalize else [])
    return {
        "config.json": {"model_type": "bert"},
        "modules.json": [{"idx": index, "name": str(index), "path": "" if index == 0 else f"{index}_{kind}",
                          "type": "sentence_transformers.models." + kind} for index, kind in enumerate(kinds)],
        "sentence_bert_config.json": {"max_seq_length": 12, "do_lower_case": True},
        "1_Pooling/config.json": {"word_embedding_dimension": 3, "pooling_mode_mean_tokens": True,
                                   "pooling_mode_cls_token": False, "include_prompt": True},
    }


class NativeBatch:
    def __init__(self, values):
        self.values = values
        self.shape = (len(values), len(values[0]) if values else 0)

    def __getitem__(self, index):
        return Vector(self.values[index])


@pytest.fixture
def fake_sentence_runtime(monkeypatch):
    state = SimpleNamespace(constructions=[], preprocess=[], encodes=[], recipes=recipe_files(),
                            output=NativeBatch([[1.0, 2.0, 3.0]]), gate=None, failures=0)

    class Transformer:
        def __init__(self, model, **kwargs):
            state.constructions.append((model, kwargs))
            if state.failures:
                state.failures -= 1
                raise ValueError("private checkpoint path")
            if state.gate is not None:
                entered, release = state.gate
                entered.set()
                assert release.wait(timeout=5)
            self.tokenizer = object()
            self.modality_config = {"text": {}}

        def preprocess(self, texts, **kwargs):
            state.preprocess.append((texts, kwargs))
            return {"input_ids": [list(range(len(texts[0]) + 2))]}

    class Pooling:
        def __init__(self, **kwargs):
            self.configuration = kwargs

    class Normalize:
        pass

    class SentenceTransformer:
        def __init__(self, **kwargs):
            self.configuration = kwargs
            self.evaluated = False

        def __getitem__(self, index):
            return self.configuration["modules"][index]

        def eval(self):
            self.evaluated = True

        def encode(self, texts, **kwargs):
            state.encodes.append((texts, kwargs))
            return state.output

    module = ModuleType("sentence_transformers")
    module.SentenceTransformer = SentenceTransformer
    module.models = SimpleNamespace(Transformer=Transformer, Pooling=Pooling, Normalize=Normalize)
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr(sentence_embeddings, "_metadata", lambda _m, _r, filename, _c: deepcopy(state.recipes[filename]))
    monkeypatch.setattr(sentence_embeddings, "_assert_no_adapter", lambda *_a: None)
    sentence_embeddings.sentence_encoder.cache_clear()
    yield state
    sentence_embeddings.sentence_encoder.cache_clear()


def test_sentence_runtime_pins_safe_text_recipe_and_disables_native_truncation(fake_sentence_runtime):
    state = fake_sentence_runtime
    encoder = sentence_embeddings.sentence_encoder("org/model", REVISION, "/operator/cache")
    assert encoder.limit == 12 and encoder.count("full text") == 11
    assert encoder.embed("full text", normalize=False, dimensions=3) == [1.0, 2.0, 3.0]
    assert state.preprocess == [(["full text"], {"processing_kwargs": {
        "text": {"truncation": False, "padding": False}, "common": {"return_tensors": None},
    }})]
    assert state.encodes[0][1]["processing_kwargs"] == {"text": {"truncation": False}}
    assert state.encodes[0][1]["prompt"] == "" and state.encodes[0][1]["normalize_embeddings"] is False
    _, kwargs = state.constructions[0]
    for name in ("model_kwargs", "processor_kwargs", "config_kwargs"):
        assert kwargs[name]["revision"] == REVISION
        assert kwargs[name]["trust_remote_code"] is False and kwargs[name]["token"] is False
    assert kwargs["model_kwargs"]["use_safetensors"] is True
    assert kwargs["max_seq_length"] == 12 and kwargs["do_lower_case"] is True
    assert encoder.model.configuration["device"] == "cpu"
    assert encoder.model.configuration["trust_remote_code"] is False
    assert encoder.model.configuration["prompts"] == {"query": "", "document": ""}
    assert encoder.model.evaluated is True
    assert len(encoder.model.configuration["modules"]) == 3


def test_sentence_runtime_uses_single_cold_construction_and_bounded_lru(fake_sentence_runtime):
    state = fake_sentence_runtime
    start = Barrier(9)
    entered, release = Event(), Event()
    state.gate = entered, release

    def load():
        start.wait(timeout=5)
        return sentence_embeddings.sentence_encoder("org/one", REVISION)

    with ThreadPoolExecutor(max_workers=8) as pool:
        pending = [pool.submit(load) for _ in range(8)]
        start.wait(timeout=5)
        assert entered.wait(timeout=5)
        release.set()
        encoders = [future.result(timeout=5) for future in pending]
    assert len(state.constructions) == 1 and all(encoder is encoders[0] for encoder in encoders)
    state.gate = None
    sentence_embeddings.sentence_encoder("org/two", REVISION)
    sentence_embeddings.sentence_encoder("org/three", REVISION)
    assert sentence_embeddings.sentence_encoder.cache_info().currsize == 2
    assert sentence_embeddings.sentence_encoder("org/one", REVISION) is not encoders[0]
    assert len(state.constructions) == 4


def test_sentence_runtime_failure_releases_load_lock_and_does_not_cache_errors(fake_sentence_runtime):
    state = fake_sentence_runtime
    state.failures = 1
    with pytest.raises(CapabilityError) as error:
        sentence_embeddings.sentence_encoder("org/model", REVISION)
    assert error.value.code == "requires_provider" and "private" not in error.value.reason
    assert sentence_embeddings.sentence_encoder.cache_info().currsize == 0
    assert sentence_embeddings.sentence_encoder("org/model", REVISION).limit == 12
    assert len(state.constructions) == 2


@pytest.mark.parametrize("filename,mutate", [
    ("config.json", lambda value: dict(value, auto_map={"AutoModel": "private_arbitrary.Class"})),
    ("config.json", lambda value: dict(value, configuration_files=["config.4.0.0.json"])),
    ("config.json", lambda value: dict(value, model_type="unsupported")),
    ("config.json", lambda value: {}),
    ("modules.json", lambda value: value + [dict(value[-1])]),
    ("modules.json", lambda value: [dict(value[0], type="private_arbitrary.Class"), *value[1:]]),
    ("modules.json", lambda value: [dict(value[0], path="../private"), *value[1:]]),
    ("modules.json", lambda value: [dict(value[0], idx=False), *value[1:]]),
    ("modules.json", lambda value: [dict(value[0], name="unexpected"), *value[1:]]),
    ("sentence_bert_config.json", lambda value: dict(value, max_seq_length=True)),
    ("sentence_bert_config.json", lambda value: dict(value, max_seq_length=100_001)),
    ("sentence_bert_config.json", lambda value: dict(value, arbitrary_preprocessing=True)),
    ("1_Pooling/config.json", lambda value: dict(value, include_prompt=False)),
    ("1_Pooling/config.json", lambda value: dict(value, word_embedding_dimension=0)),
    ("1_Pooling/config.json", lambda value: dict(value, pooling_mode_mean_tokens="true")),
    ("1_Pooling/config.json", lambda value: dict(value, arbitrary_module="private")),
])
def test_sentence_runtime_rejects_unsupported_recipe_before_weight_construction(fake_sentence_runtime, filename, mutate):
    state = fake_sentence_runtime
    state.recipes[filename] = mutate(state.recipes[filename])
    with pytest.raises(CapabilityError) as error:
        sentence_embeddings.sentence_encoder("org/model", REVISION)
    assert error.value.code == "requires_provider" and "private" not in error.value.reason
    assert state.constructions == []


@pytest.mark.parametrize("kind", ["directory", "file"])
def test_sentence_hub_identifier_cannot_resolve_to_a_same_named_local_path(fake_sentence_runtime, monkeypatch, tmp_path, kind):
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "org"
    folder.mkdir()
    if kind == "directory":
        (folder / "model").mkdir()
    else:
        (folder / "model").write_text("local checkpoint substitute")
    monkeypatch.setattr(sentence_embeddings, "_metadata", lambda *_a: pytest.fail("Local collision reached Hub metadata"))
    with pytest.raises(CapabilityError) as error:
        sentence_embeddings.sentence_encoder("org/model", REVISION)
    assert error.value.code == "requires_provider"
    assert fake_sentence_runtime.constructions == []


@pytest.mark.parametrize("output", [NativeBatch([]), NativeBatch([[1.0], [2.0]]), [], None])
def test_sentence_encoder_requires_a_single_two_dimensional_native_batch(fake_sentence_runtime, output):
    fake_sentence_runtime.output = output
    encoder = sentence_embeddings.sentence_encoder("org/model", REVISION)
    with pytest.raises(CapabilityError) as error:
        encoder.embed("source", normalize=True, dimensions=None)
    assert (error.value.code, error.value.status_code) == ("invalid_embedding", 502)


def test_sentence_provider_binds_prefix_revision_and_native_special_tokens_before_inference(settings, fake_sentence_runtime):
    selected = replace(settings, embedding_provider="sentence_transformers", embedding_model="org/model",
                       embedding_revision=REVISION, embedding_document_prefix="passage:", embedding_query_prefix="q:")
    provider = Providers(selected)
    assert provider.embedding_token_limit == 12
    assert provider.embedding_token_count("ab") == 12
    assert provider.embedding("ab") == [1.0, 2.0, 3.0]
    assert provider.query_embedding("text") == [1.0, 2.0, 3.0]
    with pytest.raises(CapabilityError) as error:
        provider.embedding("abc")
    assert error.value.code == "context_budget_exceeded"
    assert [row[0] for row in fake_sentence_runtime.encodes] == [["passage:ab"], ["q:text"]]
    assert Providers(replace(selected, embedding_revision="b" * 40)).embedding_model != provider.embedding_model


@pytest.fixture
def fake_hub_adapter_lookup(monkeypatch):
    class EntryNotFoundError(Exception):
        def __init__(self, status=None):
            self.response = SimpleNamespace(status_code=status) if status is not None else None

    class LocalEntryNotFoundError(EntryNotFoundError):
        pass

    sentinel = object()
    state = SimpleNamespace(cached=None, error=None, downloads=[], lookups=[], sentinel=sentinel,
                            remote_error=EntryNotFoundError, local_error=LocalEntryNotFoundError)

    def cached(model, filename, **kwargs):
        state.lookups.append((model, filename, kwargs))
        return state.cached

    def download(model, filename, **kwargs):
        state.downloads.append((model, filename, kwargs))
        if state.error is not None:
            raise state.error
        return "/operator/pinned/adapter_config.json"

    hub = ModuleType("huggingface_hub")
    hub._CACHED_NO_EXIST = sentinel
    hub.try_to_load_from_cache = cached
    hub.hf_hub_download = download
    errors = ModuleType("huggingface_hub.errors")
    errors.EntryNotFoundError = EntryNotFoundError
    errors.LocalEntryNotFoundError = LocalEntryNotFoundError
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    return state


def test_no_adapter_guard_uses_only_pinned_negative_cache_or_explicit_remote_404(fake_hub_adapter_lookup):
    state = fake_hub_adapter_lookup
    state.cached = state.sentinel
    sentence_embeddings._assert_no_adapter("org/model", REVISION, "/operator/cache")
    assert state.downloads == []
    assert state.lookups == [("org/model", "adapter_config.json", {"revision": REVISION, "cache_dir": "/operator/cache"})]
    state.cached = None
    state.error = state.remote_error(404)
    sentence_embeddings._assert_no_adapter("org/model", REVISION, "/operator/cache")
    assert state.downloads == [("org/model", "adapter_config.json", {
        "revision": REVISION, "cache_dir": "/operator/cache", "token": False,
    })]


@pytest.mark.parametrize("kind", ["adapter", "local404", "no-response", "unauthorized", "unavailable", "network"])
def test_no_adapter_guard_never_treats_presence_cache_miss_or_remote_failure_as_absence(fake_hub_adapter_lookup, kind):
    state = fake_hub_adapter_lookup
    state.error = {"adapter": None, "local404": state.local_error(404), "no-response": state.remote_error(),
                   "unauthorized": state.remote_error(403), "unavailable": state.remote_error(503),
                   "network": OSError("private offline failure")}[kind]
    expected = ValueError if state.error is None else type(state.error)
    with pytest.raises(expected):
        sentence_embeddings._assert_no_adapter("org/model", REVISION, None)
    assert len(state.downloads) == 1
