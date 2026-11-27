"""Actual loopback transport, atomic persistence and model-switch protection."""
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread

from fastapi.testclient import TestClient
import pytest

from rags.api import create_app
from rags.auth import PERMISSIONS, Principal
from rags.collection_service import CollectionService
from rags.config import Settings
from rags.providers import Providers
from rags.registry import CapabilityError
from rags.storage import Store

pytestmark = pytest.mark.integration


@contextmanager
def endpoint():
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append({"path": self.path, "body": body, "authorization": self.headers.get("Authorization")})
            status = 200
            if self.path == "/api/embed":
                if len(body["input"]) > 80:
                    status, payload = 400, {"error": "input length exceeds model context"}
                else:
                    payload = {"embeddings": [[1.0, 0.0]], "prompt_eval_count": 3}
            else:
                payload = {"data": [{"embedding": [1.0, 0.0]}], "usage": {"prompt_tokens": 3}}
            raw = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", calls
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
        assert not worker.is_alive()


def configuration(tmp_path, url, **kwargs):
    settings = Settings(db_path=str(tmp_path / "embedding.db"), keys={"fixture-key": "tenant"},
                        embedding_provider="openai", model_provider="extractive", openai_key="ambient-secret",
                        openai_base_url="https://must-not-be-contacted.invalid/v1", embedding_model="custom-embedding",
                        chat_model="unused", anthropic_key="", anthropic_model="unused",
                        embedding_base_url=url, embedding_auth_mode="none",
                        embedding_tokenizer_config={"backend": "utf8_bytes"}, embedding_max_tokens=64)
    return replace(settings, **kwargs)


def test_real_no_auth_transport_ignores_proxy_and_uses_role_specific_fields(tmp_path, monkeypatch):
    with endpoint() as (url, calls):
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
        monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:1")
        monkeypatch.setenv("NO_PROXY", "")
        settings = configuration(tmp_path, url, embedding_document_prefix="doc: ", embedding_query_prefix="query: ",
                                 embedding_document_input_type="document", embedding_query_input_type="query")
        provider = Providers(settings)
        assert provider.embedding("source") == [1.0, 0.0]
        assert provider.query_embedding("question") == [1.0, 0.0]
    assert calls == [
        {"path": "/embeddings", "authorization": None,
         "body": {"model": "custom-embedding", "input": "doc: source", "input_type": "document"}},
        {"path": "/embeddings", "authorization": None,
         "body": {"model": "custom-embedding", "input": "query: question", "input_type": "query"}},
    ]


def test_model_options_cannot_mix_existing_document_or_collection_vectors(tmp_path):
    with endpoint() as (url, calls):
        settings = configuration(tmp_path, url)
        auth = {"Authorization": "Bearer fixture-key"}
        with TestClient(create_app(settings)) as client:
            inserted = client.post("/documents", headers=auth, json={
                "document_id": "policy", "title": "Retention", "text": "Records are retained for thirty days."})
            assert inserted.status_code == 201, inserted.text
        principal = Principal("fixture", "tenant", PERMISSIONS, "api_key")
        service = CollectionService(Store(settings.db_path), Providers(settings))
        info = service.dispatch(principal, "create", data={"name": "records"})
        service.dispatch(principal, "add", info["id"], data={"ids": ["one"], "documents": ["Source text."]})
        before = len(calls)
        for change in ({"embedding_query_prefix": "query: "}, {"embedding_document_prefix": "passage: "},
                       {"embedding_dimensions": 2}, {"embedding_query_input_type": "query"}):
            changed = replace(settings, **change)
            with TestClient(create_app(changed)) as client:
                result = client.post("/query", headers=auth, json={"question": "retention", "mode": "dense", "generate": False})
                assert result.status_code == 409 and result.json()["detail"]["code"] == "requires_reindex"
            other = CollectionService(Store(settings.db_path), Providers(changed))
            with pytest.raises(CapabilityError) as error:
                other.dispatch(principal, "add", info["id"], data={"ids": ["two"], "documents": ["Another source."]})
            assert error.value.code == "embedding_space_mismatch"
        assert len(calls) == before
        assert service.dispatch(principal, "count", info["id"])["count"] == 1


def test_document_and_collection_oversize_preflight_do_not_write_or_call_model(tmp_path):
    with endpoint() as (url, calls):
        settings = configuration(tmp_path, url, embedding_max_tokens=16)
        with TestClient(create_app(settings)) as client:
            result = client.post("/documents", headers={"Authorization": "Bearer fixture-key"}, json={
                "document_id": "oversize", "title": "Oversize", "text": "This source is larger than sixteen bytes."})
            assert result.status_code == 422 and result.json()["detail"]["code"] == "context_budget_exceeded"
        assert Store(settings.db_path).get_document("tenant", "oversize") is None
        principal = Principal("fixture", "tenant", PERMISSIONS, "api_key")
        service = CollectionService(Store(settings.db_path), Providers(settings))
        info = service.dispatch(principal, "create", data={"name": "records"})
        with pytest.raises(CapabilityError) as error:
            service.dispatch(principal, "add", info["id"], data={"ids": ["small", "large"],
                "documents": ["short", "A second document exceeds the complete-input budget."]})
        assert error.value.code == "context_budget_exceeded"
        assert service.dispatch(principal, "count", info["id"])["count"] == 0
        assert calls == []


def test_native_ollama_rejects_server_window_without_persisting_truncated_text(tmp_path):
    with endpoint() as (url, calls):
        settings = configuration(tmp_path, url, embedding_provider="ollama", embedding_model="local-encoder",
                                 embedding_tokenizer_config=None, embedding_max_tokens=None)
        with TestClient(create_app(settings)) as client:
            good = client.post("/documents", headers={"Authorization": "Bearer fixture-key"}, json={
                "document_id": "good", "title": "Short", "text": "A source that fits."})
            assert good.status_code == 201
            rejected = client.post("/documents", headers={"Authorization": "Bearer fixture-key"}, json={
                "document_id": "large", "title": "Too long", "text": "word " * 30})
            assert rejected.status_code == 422
        assert all(call["body"]["truncate"] is False and call["authorization"] is None for call in calls)
        assert Store(settings.db_path).get_document("tenant", "large") is None
