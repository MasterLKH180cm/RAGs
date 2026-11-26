"""Actual pretrained encoders; explicitly provisioned, never replaced by hashes."""
from dataclasses import replace
import math

import pytest

from rags.config import Settings
from rags.model_catalog import get_embedding_preset
from rags.providers import Providers
from rags.registry import CapabilityError
from rags.retrieval import RAGService
from rags.storage import Store

pytestmark = [pytest.mark.integration, pytest.mark.external, pytest.mark.model_selection]


def selected_settings(identifier, path):
    preset = get_embedding_preset(identifier)
    env = preset["env"]
    return Settings(db_path=str(path), keys={"test-only": "tenant"}, embedding_provider=preset["provider"],
                    model_provider="extractive", openai_key="", openai_base_url="https://unused.invalid/v1",
                    embedding_model=preset["model"], fastembed_model=preset["model"],
                    chat_model="", anthropic_key="", anthropic_model="",
                    embedding_revision=preset.get("revision") or "",
                    embedding_query_prefix=env.get("RAGS_EMBEDDING_QUERY_PREFIX", ""),
                    embedding_document_prefix=env.get("RAGS_EMBEDDING_DOCUMENT_PREFIX", ""),
                    embedding_normalize=True if env.get("RAGS_EMBEDDING_NORMALIZE") == "true" else None,
                    tokenizer_config={"backend": "embedding"},
                    chunking_config={"strategy": "fixed", "max_tokens": 64, "overlap_tokens": 8})


@pytest.fixture(scope="module", params=["bge-small-en", "bge-small-zh", "fastembed-minilm", "st-minilm", "st-e5-small"])
def encoder(request, tmp_path_factory):
    settings = selected_settings(request.param, tmp_path_factory.mktemp(request.param) / "native.db")
    return request.param, Providers(settings)


def test_actual_encoder_contract_and_complete_input_rejection(encoder, monkeypatch):
    identifier, provider = encoder
    preset = get_embedding_preset(identifier)
    assert provider.embedding_token_limit == preset["max_input_tokens"]
    document = "The Moon orbits the Earth. 月亮繞著地球運行。"
    vector = provider.embedding(document)
    query = provider.query_embedding("What orbits Earth?")
    assert len(vector) == len(query) == preset["dimensions"]
    assert all(math.isfinite(value) for value in vector + query)
    assert sum(value * value for value in vector) > 0
    full = " context " * (provider.embedding_token_limit + 20)
    assert provider.embedding_token_count(full) > provider.embedding_token_limit
    with pytest.raises(CapabilityError) as error:
        provider.embedding(full)
    assert error.value.code == "context_budget_exceeded"
    with pytest.raises(CapabilityError) as error:
        provider.query_embedding(full)
    assert error.value.code == "context_budget_exceeded"
    if identifier == "st-e5-small":
        assert provider.embedding_token_count("") > 0
        assert vector != provider.query_embedding(document)


def test_actual_auto_source_counter_ingest_reindex_and_original_citations(encoder):
    _, provider = encoder
    rag = RAGService(Store(provider.settings.db_path), provider)
    source = ("Exact original evidence: CAFÉ é 👩🏽‍💻\r\n"
              "月亮繞著地球運行，這是繁體中文來源。 The Moon orbits the Earth.\n") * 4
    created = rag.ingest("tenant", "Original", source, "fixture://original", document_id="source")
    document = rag.store.get_document("tenant", created["id"])
    assert document["body"] == source
    assert document["index_profile"]["tokenizer"]["backend"] == "embedding"
    candidates = rag.store.scope("tenant")
    assert candidates
    for chunk in candidates:
        assert chunk["content"] == source[chunk["start_char"]:chunk["end_char"]]
        assert provider.embedding_token_count(chunk["content"]) <= 64
    result = rag.query("tenant", "What orbits Earth?", mode="dense", generate=False, top_k=2)
    assert result["citations"]
    for citation in result["citations"]:
        assert citation["text"] in source
    reindexed = rag.reindex("tenant", "source", created["revision"])
    assert reindexed["revision"] != created["revision"]
    assert rag.store.get_document("tenant", "source")["body"] == source
    changed = replace(provider.settings, embedding_document_prefix="changed: ")
    other = RAGService(rag.store, Providers(changed))
    with pytest.raises((CapabilityError, ValueError)):
        other.reindex("tenant", "source", reindexed["revision"])
    assert rag.store.get_document("tenant", "source")["revision"] == reindexed["revision"]
