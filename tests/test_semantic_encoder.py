"""Opt-in verification of semantic chunking with an actual trained encoder."""
import os

import pytest

from rags.config import Settings
from rags.providers import Providers
from rags.retrieval import RAGService
from rags.storage import Store


@pytest.mark.skipif(os.getenv("RAGS_TEST_ENCODERS") != "1", reason="Set RAGS_TEST_ENCODERS=1 for actual BGE inference")
def test_real_bge_semantic_boundaries_and_dense_retrieval(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGS_ENV", "test")
    monkeypatch.setenv("RAGS_DB_PATH", str(tmp_path / "semantic.db"))
    monkeypatch.setenv("RAGS_EMBEDDING_PROVIDER", "fastembed")
    monkeypatch.setenv("RAGS_FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")
    monkeypatch.setenv("RAGS_MODEL_PROVIDER", "extractive")
    config = Settings.from_env()
    service = RAGService(Store(config.db_path), Providers(config))
    text = ("Jupiter is the largest planet in the solar system. Saturn has rings made of ice and rock. "
            "Pancakes require flour, eggs and milk. Bake bread in a hot oven until the crust turns golden.")
    result = service.ingest("fixture", "Astronomy and cooking", text, "synthetic",
                            document_id="topics", tokenizer={"backend": "tiktoken", "encoding": "cl100k_base"},
                            chunking={"strategy": "semantic", "max_tokens": 128, "overlap_tokens": 0,
                                      "semantic_threshold": .35})
    chunks = service.store.scope("fixture")
    assert result["chunks"] >= 2
    profile = service.store.get_document("fixture", "topics")["index_profile"]
    assert profile["embedding_model"] == "fastembed:BAAI/bge-small-en-v1.5"
    assert profile["embedding_dimensions"] == 384
    assert all(c["content"] == text[c["start_char"]:c["end_char"]] and c["token_count"] <= 128 for c in chunks)
    answer = service.query("fixture", "Which planet has rings made of ice?", "dense", 1, False)
    assert "Saturn" in answer["citations"][0]["text"]
