"""Federation preserves one request's deadline and isolated provider accounting."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
from time import monotonic

import httpx
import pytest

from rags.agents.deadline import execution_deadline, remaining_timeout
from rags.config import Settings
from rags.providers import Providers, local_embedding
from rags.registry import CapabilityError
from rags.retrieval import RAGService
from rags.storage import Store

pytestmark = pytest.mark.integration


@pytest.fixture
def service(tmp_path):
    settings = Settings(
        db_path=str(tmp_path / "federated-context.db"), keys={"key-a": "a", "key-b": "b"},
        embedding_provider="local", model_provider="extractive", openai_key="",
        openai_base_url="https://provider.invalid/v1", embedding_model="fixture-embedding",
        chat_model="fixture-chat", anthropic_key="", anthropic_model="fixture-chat",
        federated_sources={"archive": {"url": "https://archive.invalid", "tenant_keys": {"a": "remote-a", "b": "remote-b"}}},
    )
    return RAGService(Store(settings.db_path), Providers(settings))


def remote_response(question):
    return httpx.Response(200, json={"citations": [{
        "chunk_id": question, "document_id": question, "title": "Remote evidence",
        "source": "archive", "section": "Policy", "text": question + " authorized evidence",
    }]})


def test_concurrent_federated_requests_keep_separate_deadlines_and_usage(service, monkeypatch):
    for tenant in ("a", "b"):
        service.ingest(tenant, "Policy", "Authorized policy evidence", "policy", document_id="policy")
    workers_ready = Barrier(4)
    observed, lock = {}, Lock()
    counts = {"request-a": 7, "request-b": 11}

    def embedding(question):
        workers_ready.wait(timeout=5)
        with lock:
            observed[question, "local"] = (execution_deadline.get(), remaining_timeout(30))
        service.provider.record("embedding", question, {"usage": {"total_tokens": counts[question]}})
        return local_embedding(question)

    def post(_url, **kwargs):
        workers_ready.wait(timeout=5)
        question = kwargs["json"]["question"]
        with lock:
            observed[question, "remote"] = (execution_deadline.get(), kwargs["timeout"])
        assert kwargs["headers"]["Authorization"] == "Bearer remote-" + question[-1]
        return remote_response(question)

    monkeypatch.setattr(service.provider, "query_embedding", embedding)
    monkeypatch.setattr("rags.retrieval.provider_transport.post", post)
    service.provider.begin_usage()
    deadlines = {"a": monotonic() + 5, "b": monotonic() + 8}

    def query(tenant):
        token = execution_deadline.set(deadlines[tenant])
        try:
            return service.query(tenant, "request-" + tenant, "federated", generate=False,
                                 sources=["local", "archive"])
        finally:
            execution_deadline.reset(token)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(query, ("a", "b")))
    for tenant, result in zip(("a", "b"), results, strict=True):
        question = "request-" + tenant
        for source in ("local", "remote"):
            deadline, timeout = observed[question, source]
            assert deadline == deadlines[tenant]
            assert 0 < timeout <= (5 if tenant == "a" else 8)
        assert result["usage"]["calls"] == 1
        assert result["usage"]["provider_reports"] == [{
            "kind": "embedding", "model": question, "usage": {"total_tokens": counts[question]},
        }]
        assert {citation["origin_source"] for citation in result["citations"]} == {"local", "archive"}
    assert service.provider.usage()["calls"] == 0


def test_exhausted_federation_deadline_prevents_remote_request(service, monkeypatch):
    calls = []

    def post(_url, **kwargs):
        calls.append(kwargs)
        return remote_response(kwargs["json"]["question"])

    monkeypatch.setattr("rags.retrieval.provider_transport.post", post)
    token = execution_deadline.set(monotonic() - 1)
    try:
        with pytest.raises(CapabilityError) as error:
            service.query("a", "request-a", "federated", generate=False, sources=["archive"])
    finally:
        execution_deadline.reset(token)
    assert error.value.code == "federated_source_failed"
    assert isinstance(error.value.__cause__, CapabilityError)
    assert error.value.__cause__.code == "tool_timeout"
    assert calls == []
