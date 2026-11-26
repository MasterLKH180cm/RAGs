"""Actual trained BGE inference and full-input token-window enforcement."""
from dataclasses import replace
import os

import pytest

from rags.auth import PERMISSIONS, Principal
from rags.collection_service import CollectionService
from rags.config import Settings
from rags.providers import Providers
from rags.registry import CapabilityError
from rags.storage import Store

pytestmark = [pytest.mark.integration, pytest.mark.external, pytest.mark.encoders,
              pytest.mark.skipif(os.getenv("RAGS_TEST_ENCODERS") != "1",
                                 reason="Set RAGS_TEST_ENCODERS=1 for actual BGE inference")]


def test_real_bge_collection_lifecycle_preserves_space_and_rejects_truncation(tmp_path, monkeypatch):
    settings = replace(Settings.from_env(), db_path=str(tmp_path / "trained-collection.db"),
                       embedding_provider="fastembed", fastembed_model="BAAI/bge-small-en-v1.5",
                       model_provider="extractive")
    provider = Providers(settings)
    service = CollectionService(Store(settings.db_path), provider)
    principal = Principal("trained-collection-fixture", "test", PERMISSIONS, "api_key")
    info = service.dispatch(principal, "create", data={"name": "science"})
    cid = info["id"]
    service.dispatch(principal, "add", cid, {
        "ids": ["astronomy", "cooking"],
        "documents": ["Saturn has rings made of ice and rock.", "Bake bread until the crust turns golden."],
    })
    current = service.dispatch(principal, "get", cid)
    assert current["configuration"]["embedding"]["dimensions"] == 384
    assert current["configuration"]["embedding"]["model"] == "fastembed:BAAI/bge-small-en-v1.5"
    ranked = service.dispatch(principal, "search", cid, {"searches": [
        {"mode": "dense", "query_text": "Which planet is surrounded by icy rings?", "limit": 1},
        {"mode": "hybrid", "query_text": "Bake bread with a golden crust", "limit": 1},
    ]})
    assert [item["items"][0]["id"] for item in ranked["results"]] == ["astronomy", "cooking"]
    fork = service.dispatch(principal, "fork", cid, {"name": "science-experiment"})
    assert fork["configuration"] == current["configuration"]
    calls = []
    original = provider.embedding
    def counted(text):
        calls.append(text)
        return original(text)
    monkeypatch.setattr(provider, "embedding", counted)
    with pytest.raises(CapabilityError) as error:
        service.dispatch(principal, "add", cid, {
            "ids": ["small", "oversized"], "documents": ["A valid short source.", "science " * 1000],
        })
    assert error.value.code == "context_budget_exceeded" and calls == []
    assert service.dispatch(principal, "get", cid)["generation"] == current["generation"]
    assert service.dispatch(principal, "count", cid)["count"] == 2
    service.dispatch(principal, "delete_collection", cid)
    answer = service.dispatch(principal, "ask", fork["id"], {
        "question": "Which planet has ice rings?", "mode": "dense", "top_k": 1, "generate": False,
    })
    assert answer["citations"][0]["record_id"] == "astronomy"
    assert answer["citations"][0]["text"] == "Saturn has rings made of ice and rock."
