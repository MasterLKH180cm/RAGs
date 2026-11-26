"""Strategy inventory and the shared authorized-snapshot/result contract."""
from dataclasses import replace

import pytest

from rags.config import Settings
from rags.providers import Providers
from rags.registry import CapabilityError, MODES, strategy
from rags.retrieval import RAGService
from rags.storage import Store


@pytest.mark.unit
def test_every_advertised_strategy_has_exactly_one_owned_callable():
    ranked = RAGService._RANKED_HANDLERS
    complete = RAGService._COMPLETE_HANDLERS
    assert not set(ranked) & set(complete)
    assert set(ranked) | set(complete) == set(MODES)
    assert all(callable(getattr(RAGService, handler)) for handler in (*ranked.values(), *complete.values()))


@pytest.mark.unit
def test_unknown_strategy_fails_before_any_service_dependencies_are_read():
    service = object.__new__(RAGService)
    with pytest.raises(CapabilityError) as error:
        service._retrieve("tenant", "query", "not-advertised", 5, {})
    assert (error.value.code, error.value.status_code) == ("unknown_strategy", 422)


@pytest.mark.integration
@pytest.mark.parametrize("mode,score_kind", [
    ("sparse", "reciprocal_bm25_rank"), ("dense", "reciprocal_cosine_rank"),
    ("naive", "reciprocal_cosine_rank"), ("hybrid", "rrf"),
    ("structured", "rrf"), ("graph", "inverse_hop_distance"), ("temporal", "rrf_plus_recency"),
])
def test_ranked_handlers_keep_one_filtered_source_snapshot_and_common_provenance(tmp_path, monkeypatch, mode, score_kind):
    settings = replace(Settings.from_env(), db_path=str(tmp_path / "dispatch.db"),
                       embedding_provider="local", model_provider="extractive")
    service = RAGService(Store(settings.db_path), Providers(settings))
    service.ingest("tenant", "Archive", "Alpha -> Archived", "archive", document_id="policy",
                   metadata={"visible": False}, valid_from="2024-01-01T00:00:00Z")
    current = service.ingest("tenant", "Current", "Alpha -> Beta", "manual", document_id="policy",
                             metadata={"visible": True}, valid_from="2024-06-01T00:00:00Z")
    service.ingest("tenant", "Excluded", "Alpha -> Private", "manual", document_id="excluded",
                   metadata={"visible": False}, valid_from="2024-01-01T00:00:00Z")
    service.ingest("foreign", "Foreign", "Alpha -> Foreign", "manual", document_id="policy",
                   metadata={"visible": True}, valid_from="2024-01-01T00:00:00Z")
    scope = service.store.scope
    observed = []

    def inspect(tenant, **kwargs):
        rows = scope(tenant, **kwargs)
        observed.append((tenant, kwargs, rows))
        return rows

    monkeypatch.setattr(service.store, "scope", inspect)
    options = {"as_of": "2024-09-01T00:00:00Z", "filters": {"metadata": {"visible": True}}}
    result = service._retrieve("tenant", "Alpha", mode, 1, options)
    assert len(observed) == 1
    assert observed[0][:2] == ("tenant", {**options, "include_embeddings": False})
    assert len(result["chunks"]) == 1
    chunk = result["chunks"][0]
    assert chunk["document_id"] == "policy" and chunk["revision"] == current["revision"]
    assert chunk["content"] == "Alpha -> Beta" and chunk["source"] == "manual"
    assert "embedding" not in chunk
    assert result["mode_used"] == mode and result["score_kind"] == score_kind
    assert result["engine"] == strategy(mode).implementation
    assert result["trace"][0] == {"action": "retrieve", "mode": mode,
                                  "candidates": len(observed[0][2]), "encoder": "offline-hash-v1"}
