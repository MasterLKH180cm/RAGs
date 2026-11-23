"""Cross-layer regressions for source validity, router identity and federation."""
from dataclasses import replace
import json
import socket
from threading import Event, Thread
import time

import httpx
import pytest

from rags.agents.config import AgentPlatformConfig
from rags.agents.router import ModelRouter
from rags.config import Settings
from rags.providers import Providers
from rags.registry import CapabilityError
from rags.retrieval import RAGService
import rags.retrieval as retrieval_module
from rags.storage import Store, timestamp


pytestmark = pytest.mark.integration


def settings_for(path):
    return Settings(db_path=str(path), keys={"synthetic-key": "tenant"}, embedding_provider="local",
                    model_provider="extractive", openai_key="", openai_base_url="https://unused.invalid/v1",
                    embedding_model="unused", chat_model="unused", anthropic_key="", anthropic_model="unused")


def assert_reindex_validity(store, rebuilds, *, legacy=False):
    service = RAGService(store, Providers(settings_for(store.path)))
    original = service.ingest("tenant", "Policy", "The retired fee was ten units.", "manual",
                              document_id="policy", valid_from="2024-01-01T00:00:00Z")
    previous = original
    for _ in range(rebuilds):
        previous = service.reindex("tenant", "policy", previous["revision"])
    replacement = service.ingest("tenant", "Policy", "The replacement fee was twenty units.", "manual",
                                 document_id="policy", valid_from="2024-06-01T00:00:00Z",
                                 valid_to="2024-09-01T00:00:00Z", expected_revision=previous["revision"])
    revisions = store.revisions("tenant", "policy")
    assert all(row["valid_to"] == timestamp("2024-06-01T00:00:00Z") for row in revisions
               if row["revision"] != replacement["revision"])
    if legacy:
        assert rebuilds
        # Reproduce an already-written pre-correction interval without rewriting
        # current state. Read-time revision selection must still protect it.
        with store.connect() as connection:
            connection.execute("UPDATE document_revisions SET valid_to=NULL WHERE tenant=? AND id=? AND revision=?",
                               ("tenant", "policy", original["revision"]))
    for at, revision in (("2023-12-31T23:59:59Z", None), ("2024-01-01T00:00:00Z", previous["revision"]),
                         ("2024-05-31T23:59:59Z", previous["revision"]),
                         ("2024-06-01T00:00:00Z", replacement["revision"]),
                         ("2024-09-01T00:00:00Z", None), ("2024-10-01T00:00:00Z", None)):
        result = service.query("tenant", "policy fee", "hybrid", generate=False, as_of=at)
        assert {citation["revision"] for citation in result["citations"]} == ({revision} if revision else set())
        assert (result["outcome"] == "success") == bool(revision)
        assert not store.scope("foreign", as_of=at)


@pytest.mark.parametrize("rebuilds,legacy", [(0, False), (1, False), (3, False), (3, True)])
def test_reindex_does_not_resurrect_expired_source_intervals(tmp_path, rebuilds, legacy):
    assert_reindex_validity(Store(str(tmp_path / "temporal.db")), rebuilds, legacy=legacy)


def test_same_start_source_correction_does_not_restore_expired_older_copy(tmp_path):
    settings = settings_for(tmp_path / "correction.db")
    service = RAGService(Store(settings.db_path), Providers(settings))
    first = service.ingest("tenant", "Policy", "Original policy", "manual", document_id="policy",
                           valid_from="2024-01-01T00:00:00Z")
    service.ingest("tenant", "Policy", "Corrected policy", "manual", document_id="policy",
                   valid_from="2024-01-01T00:00:00Z", valid_to="2024-03-01T00:00:00Z",
                   expected_revision=first["revision"])
    assert service.store.scope("tenant", as_of="2024-03-01T00:00:00Z") == []


def routed_service(store, host, calls, *, space_id="", key="synthetic-key"):
    settings = replace(settings_for(store.path), embedding_provider="router", embedding_model="embed",
                       embedding_space_id=space_id)

    def gateway(request):
        body = json.loads(request.content)
        calls.append((request.url.host, body["input"]))
        apple = "apple" in body["input"]
        first_dimension = apple == (request.url.host == "gateway-a.invalid")
        return httpx.Response(200, json={"data": [{"embedding": [1.0, 0.0] if first_dimension else [0.0, 1.0]}]})

    provider = Providers(settings)
    provider.model_router = ModelRouter(AgentPlatformConfig(gateway_url=host, gateway_key=key,
                                                           models={"embed": "openai/synthetic-embedding"}))
    provider.model_router._client = lambda *_: httpx.AsyncClient(transport=httpx.MockTransport(gateway))
    return RAGService(store, provider)


@pytest.mark.parametrize("change", ["endpoint", "space"])
def test_router_space_change_rejects_old_vectors_and_rebuilds_unchanged_sources(tmp_path, change):
    calls = []
    store = Store(str(tmp_path / "identity.db"))
    original = routed_service(store, "https://gateway-a.invalid/v1", calls, space_id="space-v1")
    initial = {name: original.ingest("tenant", name, name + " source", "manual", document_id=name)
               for name in ("apple", "banana")}
    changed = routed_service(store, "https://gateway-b.invalid/v1" if change == "endpoint" else "https://gateway-a.invalid/v1",
                             calls, space_id="space-v2" if change == "space" else "space-v1")
    before = len(calls)
    with pytest.raises(CapabilityError) as error:
        changed.query("tenant", "apple question", "dense", top_k=1, generate=False)
    assert error.value.code == "requires_reindex" and len(calls) == before
    for name in ("apple", "banana"):
        result = changed.ingest("tenant", name, name + " source", "manual", document_id=name)
        assert result["unchanged"] is False and result["profile_id"] != initial[name]["profile_id"]
        assert result["index_profile"]["embedding_endpoint"] == changed.provider.embedding_endpoint
    assert len(calls) == before + 2
    result = changed.query("tenant", "apple question", "dense", top_k=1, generate=False)
    assert result["citations"][0]["document_id"] == "apple"
    assert not changed.query("foreign", "apple", "dense", generate=False)["citations"]


def test_router_identity_normalizes_equivalent_url_and_excludes_rotated_credentials(tmp_path):
    calls = []
    store = Store(str(tmp_path / "normalization.db"))
    original = routed_service(store, "https://GATEWAY-A.invalid:443/v1/", calls)
    first = original.ingest("tenant", "apple", "apple source", "manual", document_id="apple")
    changed = routed_service(store, "https://gateway-a.invalid/v1", calls, key="rotated-synthetic-key")
    before = len(calls)
    repeated = changed.ingest("tenant", "apple", "apple source", "manual", document_id="apple")
    assert repeated["unchanged"] and repeated["profile_id"] == first["profile_id"] and len(calls) == before
    assert first["index_profile"]["embedding_endpoint"] == "https://gateway-a.invalid/v1"
    assert "synthetic-key" not in json.dumps(first["index_profile"])


@pytest.mark.parametrize("space", [None, 1, "contains spaces", "x" * 161, "token?secret", "new\nline"])
def test_invalid_embedding_space_identity_is_rejected(tmp_path, space):
    with pytest.raises(ValueError, match="deployment/version"):
        replace(settings_for(tmp_path / "unused.db"), embedding_space_id=space)


def test_ingestion_rejects_pathological_plan_before_embedding(tmp_path, monkeypatch):
    settings = settings_for(tmp_path / "bounded.db")
    provider = Providers(settings)
    monkeypatch.setattr(provider, "embedding", lambda *_: pytest.fail("Rejected plan cannot call embeddings"))
    service = RAGService(Store(settings.db_path), provider)
    with pytest.raises(ValueError, match="byte budget"):
        service.ingest("tenant", "Bounded", "x" * 34_768, "manual",
                       chunking={"strategy": "fixed", "max_tokens": 32768, "overlap_tokens": 32767})
    assert not service.store.list_documents("tenant")


def test_federation_whole_deadline_cancels_real_slow_response_headers(tmp_path, monkeypatch):
    accepted, release = Event(), Event()
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(2)

    def serve():
        with listener:
            try:
                connection, _ = listener.accept()
            except TimeoutError:
                return
            with connection:
                connection.settimeout(2)
                connection.recv(65536)
                accepted.set()
                try:
                    connection.sendall(b"HTTP/1.1 200 OK\r\nX-Slow: ")
                    while not release.wait(0.02):
                        connection.sendall(b"x")
                except OSError:
                    pass

    settings = replace(settings_for(tmp_path / "federation.db"), federated_sources={"remote": {
        "url": f"http://127.0.0.1:{listener.getsockname()[1]}", "tenant_keys": {"tenant": "synthetic-key"}}})
    service = RAGService(Store(settings.db_path), Providers(settings))
    monkeypatch.setattr(retrieval_module, "FEDERATED_TIMEOUT_SECONDS", 0.3)
    thread = Thread(target=serve, daemon=True)
    thread.start()
    started = time.monotonic()
    try:
        with pytest.raises(CapabilityError) as error:
            service.query("tenant", "policy", "federated", generate=False)
        elapsed = time.monotonic() - started
        assert error.value.code == "federated_source_failed"
        assert accepted.is_set() and elapsed < 1.2
    finally:
        release.set()
        thread.join(timeout=3)
    assert not thread.is_alive()
