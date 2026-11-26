"""Tokenizer selection crosses real HTTP and service boundaries without downloads."""
from dataclasses import replace
import os

from fastapi.testclient import TestClient
import pytest

from rags.api import create_app
from rags.config import Settings
from rags.features import FeatureFlags
from rags.providers import Providers
from rags.retrieval import RAGService
from rags.storage import Store
from rags.tokenization import ExactEncodingTokenizer


pytestmark = pytest.mark.integration
AUTH = {"Authorization": "Bearer tokenizer-operator"}


@pytest.fixture
def settings(tmp_path, monkeypatch):
    for name in tuple(os.environ):
        if name.startswith(("RAGS_", "OPENAI_", "ANTHROPIC_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("RAGS_ENV", "test")
    return replace(Settings.from_env(), db_path=str(tmp_path / "tokenizers.db"),
                   keys={"tokenizer-operator": "tenant"}, key_roles={"tokenizer-operator": ("admin",)})


@pytest.mark.parametrize("backend", ["huggingface", "sentencepiece"])
def test_health_listing_and_invalid_request_paths_never_load_registered_files(settings, monkeypatch, tmp_path, backend):
    artifacts = {"missing-local": {"backend": backend, "path": str(tmp_path / "not-present"), "sha256": "a" * 64}}
    settings = replace(settings, tokenizer_config={"backend": backend, "model": "missing-local"},
                       tokenizer_artifacts=artifacts)
    monkeypatch.setattr("rags.tokenization.read_tokenizer_artifact",
                        lambda *_: pytest.fail("Listing or invalid requests must not read tokenizer artifacts"))
    monkeypatch.setattr("rags.tokenization._load_native_callbacks",
                        lambda *_: pytest.fail("Listing must not load optional tokenizer runtimes"))
    with TestClient(create_app(settings)) as client:
        for path in ("/health", "/capabilities", "/strategies"):
            response = client.get(path, headers=AUTH)
            assert response.status_code == 200, response.text
            assert str(tmp_path) not in response.text
        invalid = client.post("/documents", headers=AUTH, json={
            "title": "Source", "text": "Original source text", "source": "fixture",
            "tokenizer": {"backend": backend, "model": "missing-local", "path": "/host/private"},
        })
        assert invalid.status_code == 422


@pytest.mark.parametrize("backend", ["huggingface", "sentencepiece"])
def test_disabled_local_tokenizer_rejects_before_files_or_document_mutation(settings, monkeypatch, tmp_path, backend):
    artifacts = {"missing-local": {"backend": backend, "path": str(tmp_path / "not-present"), "sha256": "a" * 64}}
    settings = replace(settings, tokenizer_config={"backend": backend, "model": "missing-local"},
                       tokenizer_artifacts=artifacts, features=FeatureFlags((("tokenizer." + backend, False),)))
    monkeypatch.setattr("rags.tokenization.read_tokenizer_artifact",
                        lambda *_: pytest.fail("Disabled tokenizer read an artifact"))
    with TestClient(create_app(settings)) as client:
        response = client.post("/documents", headers=AUTH, json={
            "title": "Source", "text": "Original source text", "source": "fixture",
        })
        assert response.status_code == 403 and response.json()["detail"]["code"] == "feature_disabled"
        assert client.get("/documents", headers=AUTH).json() == []


def test_provider_bound_counter_overflow_leaves_original_revision_and_source_intact(settings, monkeypatch):
    settings = replace(settings, tokenizer_config={"backend": "embedding"})
    provider = Providers(settings)
    tokenizer = ExactEncodingTokenizer(
        tokenizer_id="embedding:fixture", count=lambda text: len(text) + 2,
        profile={"backend": "embedding", "embedding_identity": "a" * 64,
                 "input_role": "document", "special_tokens": "encoder_native"},
    )
    monkeypatch.setattr(provider, "embedding_tokenizer", lambda: tokenizer)
    service = RAGService(Store(settings.db_path), provider)
    source = "Original 中文 e\u0301🙂 evidence"
    first = service.ingest("tenant", "Original", source, "fixture://source", document_id="document",
                           chunking={"strategy": "fixed", "max_tokens": 10, "overlap_tokens": 0})
    document = service.store.get_document("tenant", "document")
    assert document["index_profile"]["tokenizer"]["embedding_identity"] == "a" * 64
    for chunk in service.store.scope("tenant"):
        assert chunk["content"] == source[chunk["start_char"]:chunk["end_char"]]
        assert chunk["token_count"] == len(chunk["content"]) + 2 <= 10
    monkeypatch.setattr(provider, "embedding", lambda *_: pytest.fail("Rejected source plan reached embedding"))
    with pytest.raises(ValueError, match="too small"):
        service.reindex("tenant", "document", first["revision"],
                        chunking={"strategy": "fixed", "max_tokens": 2, "overlap_tokens": 0})
    assert service.store.get_document("tenant", "document") == document
