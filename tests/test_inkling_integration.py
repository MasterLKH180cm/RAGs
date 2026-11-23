"""Real local HTTP compatibility fixture, never evidence of live model quality."""
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading

from fastapi.testclient import TestClient
import pytest

from rags.agents.config import AgentPlatformConfig
from rags.agents.router import ModelRouter
from rags.api import create_app
from rags.config import Settings
from rags.model_profiles import INKLING_MODEL

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    for key in tuple(os.environ):
        if key.startswith(("RAGS_", "OPENAI_", "ANTHROPIC_")):
            monkeypatch.delenv(key)
    monkeypatch.setenv("RAGS_ENV", "test")


@contextmanager
def gateway_fixture():
    requests = []
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body))
            assert self.headers["Authorization"] == "Bearer fixture-key"
            assert body["model"] == INKLING_MODEL
            assert "max_completion_tokens" in body and "max_tokens" not in body
            assert "response_format" not in body
            if body.get("tools"):
                message = {"content": None, "tool_calls": [{"id": "call_1", "type": "function",
                    "function": {"name": "lookup", "arguments": '{"query":"retention"}'}}]}
                reason = "tool_calls"
            else:
                content = body["messages"][-1]["content"]
                if "source text" in content:
                    message = {"content": "A validated final answer"}
                else:
                    import re
                    ids = re.findall(r"\[([^\]\n]+)\]", content)
                    message = {"content": "The retention period is 30 days [" + ids[-1] + "]"}
                reason = "stop"
            raw = json.dumps({"model": INKLING_MODEL, "choices": [
                {"finish_reason": reason, "message": message}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                          "total_tokens": 120, "prompt_tokens_details": {"cached_tokens": 10}}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_inkling_rag_query_keeps_authorized_evidence_and_independent_embeddings(tmp_path):
    with gateway_fixture() as (url, requests):
        config = replace(Settings.from_env(), db_path=str(tmp_path / "rag.db"),
            keys={"owner-key": "tenant-a", "other-key": "tenant-b"},
            embedding_provider="local", model_provider="openai", chat_model=INKLING_MODEL,
            generation_base_url=url, generation_api_key="fixture-key", generation_protocol="tinker",
            reasoning_effort="low", generation_tokenizer_config={"backend": "utf8_bytes"})
        with TestClient(create_app(config)) as client:
            owner = {"Authorization": "Bearer owner-key"}
            other = {"Authorization": "Bearer other-key"}
            record = client.post("/documents", headers=owner, json={
                "title": "Retention", "text": "The retention period is 30 days.", "document_id": "policy"})
            assert record.status_code == 201
            assert record.json()["index_profile"]["embedding_model"] == "offline-hash-v1"
            result = client.post("/query", headers=owner, json={
                "question": "What is the retention period?", "mode": "sparse", "generate": True})
            assert result.status_code == 200, result.text
            assert result.json()["citations"] and "30 days" in result.json()["answer"]
            assert len(requests) == 1 and requests[0][1]["reasoning_effort"] == "low"
            forbidden = client.post("/query", headers=other, json={
                "question": "What is the retention period?", "mode": "sparse", "generate": True})
            assert forbidden.status_code == 200 and forbidden.json()["citations"] == []
            assert len(requests) == 1  # Empty authorized scope never calls the model.


def test_inkling_native_tool_and_followup_envelopes_over_actual_http():
    with gateway_fixture() as (url, requests):
        config = AgentPlatformConfig(gateway_url=url, gateway_key="fixture-key", allow_http=True,
            models={"main": INKLING_MODEL}, model_options={"main": {"reasoning_effort": .2}})
        router = ModelRouter(config)
        messages = [{"role": "user", "content": "Look up the retention period"}]
        response = router.complete(model="main", messages=messages, tools=[{
            "type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}])
        assert response.tool_calls[0].arguments == {"query": "retention"}
        assert response.usage["cached_prompt_tokens"] == 10
        messages += [{"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_1", "type": "function", "function": {
                "name": "lookup", "arguments": '{"query":"retention"}'}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "Authorized source text"}]
        final = router.complete(model="main", messages=messages, max_output_tokens=100)
        assert final.content == "A validated final answer" and len(requests) == 2
        assert requests[1][1]["max_completion_tokens"] == 100
