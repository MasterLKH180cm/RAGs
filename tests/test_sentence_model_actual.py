"""Pinned trained MiniLM smoke; model download is explicit and opt-in."""
from dataclasses import replace
import math
import os

import pytest

from rags.auth import PERMISSIONS, Principal
from rags.collection_service import CollectionService
from rags.config import Settings
from rags.providers import Providers
from rags.registry import CapabilityError
from rags.retrieval import RAGService
from rags.storage import Store

pytestmark = [pytest.mark.integration, pytest.mark.external, pytest.mark.encoders,
              pytest.mark.skipif(os.getenv("RAGS_TEST_SENTENCE_TRANSFORMERS") != "1",
                                 reason="Set RAGS_TEST_SENTENCE_TRANSFORMERS=1 for trained MiniLM inference")]

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Publisher repository revision verified through the official Hugging Face API.
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def test_actual_pinned_minilm_document_collection_and_complete_input_window(tmp_path):
    settings = Settings(db_path=str(tmp_path / "minilm.db"), keys={},
                        embedding_provider="sentence_transformers", embedding_model=MODEL,
                        embedding_revision=REVISION, model_provider="extractive", openai_key="",
                        openai_base_url="https://unused.invalid/v1", chat_model="unused", anthropic_key="",
                        anthropic_model="unused")
    provider = Providers(settings)
    identity = provider.embedding_model
    vector = provider.embedding("Saturn has rings made of ice and rock.")
    assert len(vector) == 384 and all(math.isfinite(value) for value in vector)
    assert math.isclose(sum(value * value for value in vector), 1.0, rel_tol=1e-5)
    assert provider.embedding_model == identity
    assert provider.embedding_window_policy == "native_tokenizer_reject_oversize"
    store = Store(settings.db_path)
    documents = RAGService(store, provider)
    documents.ingest("tenant", "Astronomy", "Saturn has rings made of ice and rock.", "manual", document_id="saturn")
    documents.ingest("tenant", "Cooking", "Bread is baked until its crust turns golden.", "manual", document_id="bread")
    result = documents.query("tenant", "Which planet is surrounded by icy rings?", "dense", top_k=1, generate=False)
    assert result["citations"][0]["document_id"] == "saturn"
    principal = Principal("fixture", "tenant", PERMISSIONS, "api_key")
    collections = CollectionService(store, provider)
    info = collections.dispatch(principal, "create", data={"name": "science"})
    collections.dispatch(principal, "add", info["id"], data={"ids": ["saturn", "bread"],
        "documents": ["Saturn has rings made of ice and rock.", "Bread is baked until its crust turns golden."]})
    result = collections.dispatch(principal, "search", info["id"], data={"searches": [
        {"mode": "dense", "query_text": "Which planet has icy rings?", "limit": 1}]})
    assert result["results"][0]["items"][0]["id"] == "saturn"
    oversized = "astronomy " * 1000
    assert provider.embedding_token_count(oversized) > provider.embedding_token_limit
    with pytest.raises(CapabilityError) as error:
        collections.dispatch(principal, "add", info["id"], data={"ids": ["oversized"], "documents": [oversized]})
    assert error.value.code == "context_budget_exceeded"
    assert collections.dispatch(principal, "count", info["id"])["count"] == 2
    changed = Providers(replace(settings, embedding_query_prefix="query: "))
    assert changed.embedding_model != identity
    with pytest.raises(CapabilityError) as error:
        RAGService(store, changed).query("tenant", "planet", "dense", generate=False)
    assert error.value.code == "requires_reindex"
