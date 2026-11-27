"""Separate real local HTTP servers prove endpoint, key and model routing."""
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
from threading import Thread

from fastapi.testclient import TestClient
import pytest

from rags.api import create_app
from rags.config import Settings

pytestmark = pytest.mark.integration
AUTH = {"Authorization": "Bearer endpoints-key"}


@contextmanager
def provider_server():
    recorded = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            recorded.append({"path": self.path, "body": body, "authorization": self.headers.get("Authorization"),
                             "api_key": self.headers.get("x-api-key")})
            if self.path.endswith("/embeddings"):
                response = {"data": [{"embedding": [0.5, 0.5]}]}
            else:
                identifiers = re.findall(r"\[([a-f0-9]{32})\]", json.dumps(body))
                answer = "Telemetry is retained for thirty days. [" + identifiers[0] + "]"
                response = ({"content": [{"type": "text", "text": answer}]} if self.path.endswith("/messages")
                            else {"choices": [{"message": {"content": answer}}]})
            encoded = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", recorded
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.mark.parametrize("model_provider", ["openai", "anthropic"])
def test_separate_embedding_and_generation_endpoints_keys_models_and_index_identity(tmp_path, monkeypatch, model_provider):
    monkeypatch.setenv("RAGS_ENV", "test")
    with provider_server() as (embedding_url, embedding_calls), provider_server() as (generation_url, generation_calls):
        settings = replace(Settings.from_env(), db_path=str(tmp_path / "endpoints.db"), keys={"endpoints-key": "tenant"},
                           embedding_provider="openai", model_provider=model_provider,
                           openai_base_url="https://legacy-must-not-be-called.invalid/v1", openai_key="legacy-key",
                           embedding_base_url=embedding_url, embedding_api_key="embedding-only-key", embedding_model="embedding-v2",
                           embedding_tokenizer_config={"backend": "utf8_bytes"}, embedding_max_tokens=100000,
                           generation_base_url=generation_url, generation_api_key="generation-only-key",
                           chat_model="chat-v2", anthropic_model="messages-v2", generation_tokenizer_config={"backend": "unicode"})
        with TestClient(create_app(settings)) as client:
            response = client.post("/documents", headers=AUTH, json={
                "document_id": "policy", "title": "Retention", "text": "Telemetry is retained for thirty days."})
            assert response.status_code == 201, response.text
            assert response.json()["index_profile"]["embedding_endpoint"] == embedding_url
            response = client.post("/query", headers=AUTH, json={"question": "Telemetry retention", "mode": "dense"})
            assert response.status_code == 200, response.text
            assert response.json()["answer"].startswith("Telemetry is retained")
            assert len(response.json()["citations"]) == 1
        assert len(embedding_calls) == 2 and len(generation_calls) == 1
        assert all(call["path"] == "/v1/embeddings" and call["authorization"] == "Bearer embedding-only-key"
                   and call["body"]["model"] == "embedding-v2" for call in embedding_calls)
        generation = generation_calls[0]
        if model_provider == "openai":
            assert generation["path"] == "/v1/chat/completions"
            assert generation["authorization"] == "Bearer generation-only-key" and generation["body"]["model"] == "chat-v2"
        else:
            assert generation["path"] == "/v1/messages"
            assert generation["api_key"] == "generation-only-key" and generation["body"]["model"] == "messages-v2"
        # Even identical model names cannot mix embeddings from a new endpoint.
        changed = replace(settings, embedding_base_url=generation_url)
        with TestClient(create_app(changed)) as client:
            response = client.post("/query", headers=AUTH, json={"question": "Telemetry retention", "mode": "dense", "generate": False})
            assert response.status_code == 409 and response.json()["detail"]["code"] == "requires_reindex"
        assert len(embedding_calls) == 2 and len(generation_calls) == 1
