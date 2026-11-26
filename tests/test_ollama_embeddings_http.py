"""Native Ollama HTTP contract against a separate deterministic local process.

This exercises real sockets, framing and transport bounds. It does not run an
Ollama model and provides no evidence of live-model quality or inference speed.
"""
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import multiprocessing
import os
from types import SimpleNamespace
import time

import pytest

from rags.agents.deadline import execution_deadline
from rags.config import Settings
from rags.provider_transport import MAX_RESPONSE_BYTES
from rags.providers import Providers
from rags.registry import CapabilityError


pytestmark = pytest.mark.integration
PRIVATE_BODY = "upstream-private-body-sentinel"
PROXY_NAMES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
               "http_proxy", "https_proxy", "all_proxy", "no_proxy")


def _serve_fixture(ready, received, options):
    """Child process owns the listener and every request handler."""
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            received.put({"pid": os.getpid(), "method": "POST", "path": self.path,
                          "headers": {key.lower(): value for key, value in self.headers.items()},
                          "body": json.loads(body)})
            if options.get("stall") == "headers":
                time.sleep(5)
            payload = options.get("body", b'{"embeddings":[[3.0,4.0,0.0]]}')
            self.send_response(options.get("status", 200))
            self.send_header("Content-Type", "application/json")
            chunked = options.get("chunked", False) or options.get("stall") == "body"
            if chunked:
                self.send_header("Transfer-Encoding", "chunked")
            else:
                self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            try:
                if options.get("stall") == "body":
                    self.wfile.write(b'1\r\n{\r\n')
                    self.wfile.flush()
                    received.put({"phase": "body_started"})
                    time.sleep(5)
                    return
                if chunked:
                    for offset in range(0, len(payload), 16 * 1024):
                        part = payload[offset:offset + 16 * 1024]
                        self.wfile.write(f"{len(part):x}\r\n".encode() + part + b"\r\n")
                    self.wfile.write(b"0\r\n\r\n")
                else:
                    self.wfile.write(payload)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                # The client is expected to close early on timeout/byte refusal.
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    ready.put((server.server_port, os.getpid()))
    try:
        server.serve_forever(poll_interval=0.05)
    finally:
        server.server_close()


@pytest.fixture
def settings(tmp_path, monkeypatch):
    for name in tuple(os.environ):
        if name.startswith(("RAGS_", "OPENAI_", "ANTHROPIC_")) or name in PROXY_NAMES:
            monkeypatch.delenv(name)
    monkeypatch.setenv("RAGS_ENV", "test")
    return replace(Settings.from_env(), db_path=str(tmp_path / "unused.db"))


@pytest.fixture
def server():
    context = multiprocessing.get_context("spawn")
    owned = []

    def start(**options):
        ready, received = context.Queue(), context.Queue()
        process = context.Process(target=_serve_fixture, args=(ready, received, options), daemon=True)
        process.start()
        owned.append((process, ready, received))
        port, pid = ready.get(timeout=10)
        assert pid == process.pid and pid != os.getpid()
        return SimpleNamespace(url=f"http://127.0.0.1:{port}", received=received, pid=pid)

    yield start
    for process, ready, received in reversed(owned):
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        assert not process.is_alive(), "Ollama HTTP fixture process did not terminate"
        for queue in (ready, received):
            queue.close()
            queue.join_thread()


def provider(settings, fixture, **changes):
    values = {"embedding_provider": "ollama", "embedding_model": "fixture-encoder:v1",
              "embedding_base_url": fixture.url, "embedding_auth_mode": "none",
              "embedding_api_key": "", "embedding_input_policy": "server_reject",
              "embedding_dimensions": 3}
    values.update(changes)
    return Providers(replace(settings, **values))


def test_native_ollama_uses_complete_role_inputs_and_server_rejection_without_proxy_or_auth(settings, server, monkeypatch):
    fixture = server()
    selected = provider(settings, fixture, embedding_document_prefix="document: ", embedding_query_prefix="query: ")
    for name in PROXY_NAMES:
        if name.lower() != "no_proxy":
            monkeypatch.setenv(name, "http://127.0.0.1:1")
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-credential-must-not-be-sent")
    assert selected.embedding("中文 source") == [3.0, 4.0, 0.0]
    assert selected.query_embedding("Where is it?") == [3.0, 4.0, 0.0]
    for expected in ("document: 中文 source", "query: Where is it?"):
        request = fixture.received.get(timeout=2)
        assert request["pid"] == fixture.pid
        assert request["path"] == "/api/embed" and request["method"] == "POST"
        assert request["body"] == {
            "model": "fixture-encoder:v1", "input": [expected], "truncate": False, "dimensions": 3,
        }
        assert request["headers"]["content-type"] == "application/json"
        assert request["headers"]["accept-encoding"] == "identity"
        assert "authorization" not in request["headers"]
    assert selected.embedding_window_policy == "server_reject"
    with pytest.raises(CapabilityError, match="exact embedding tokenizer"):
        selected.embedding_tokenizer()


def test_native_ollama_explicit_auth_and_normalization_cross_actual_http(settings, server):
    fixture = server()
    selected = provider(settings, fixture, embedding_auth_mode="required",
                        embedding_api_key="fixture-only-credential", embedding_normalize=True)
    assert selected.embedding("source") == pytest.approx([0.6, 0.8, 0.0])
    request = fixture.received.get(timeout=2)
    assert request["headers"]["authorization"] == "Bearer fixture-only-credential"
    assert request["body"]["truncate"] is False


@pytest.mark.parametrize("endpoint", ["https://models.invalid", "http://192.0.2.9:11434", "http://0.0.0.0:11434"])
def test_no_auth_ollama_refuses_remote_endpoints_before_any_connection(settings, monkeypatch, endpoint):
    monkeypatch.setattr("rags.provider_transport.bounded_post",
                        lambda *_args, **_kwargs: pytest.fail("Invalid no-auth endpoint reached transport"))
    with pytest.raises(ValueError, match="loopback"):
        provider(settings, SimpleNamespace(url=endpoint))


@pytest.mark.parametrize("upstream_error,expected_code,expected_status", [
    ("input length exceeds maximum context length", "context_budget_exceeded", 422),
    ("input length exceeds maximum context length: private details", "provider_failed", 502),
    ("model is unavailable " + PRIVATE_BODY, "provider_failed", 502),
])
def test_only_exact_native_ollama_overflow_maps_to_input_rejection(settings, server, upstream_error,
                                                                 expected_code, expected_status):
    fixture = server(status=400, body=json.dumps({"error": upstream_error, "private_detail": PRIVATE_BODY}).encode())
    with pytest.raises(CapabilityError) as error:
        provider(settings, fixture).embedding("private-source-sentinel")
    assert (error.value.code, error.value.status_code) == (expected_code, expected_status)
    assert PRIVATE_BODY not in str(error.value) and "private-source-sentinel" not in str(error.value)
    assert fixture.received.get(timeout=2)["body"]["truncate"] is False


@pytest.mark.parametrize("payload", [
    b'{"embeddings":', b'[]', b'{"embeddings":[]}',
    b'{"embeddings":[[1,2,3],[4,5,6]]}', b'{"embeddings":[[]]}',
    b'{"embeddings":[[1,2]]}', b'{"embeddings":[[true,2,3]]}',
    b'{"embeddings":[["1",2,3]]}', b'{"embeddings":[[NaN,2,3]]}',
    b'{"embeddings":[[1e400,2,3]]}', b'{"embeddings":{"unexpected":"shape"}}',
])
def test_malformed_counts_dimensions_and_nonfinite_vectors_fail_closed_over_http(settings, server, payload):
    fixture = server(body=payload)
    with pytest.raises(CapabilityError) as error:
        provider(settings, fixture).embedding("source")
    assert (error.value.code, error.value.status_code) == ("invalid_embedding", 502)
    assert fixture.received.get(timeout=2)["body"]["input"] == ["source"]


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("extra_byte", [0, 1])
def test_native_ollama_response_limit_applies_to_declared_and_streamed_valid_json(settings, server,
                                                                               chunked, extra_byte):
    envelope = {"embeddings": [[3.0, 4.0, 0.0]], "padding": ""}
    base_size = len(json.dumps(envelope, separators=(",", ":")).encode())
    envelope["padding"] = "x" * (MAX_RESPONSE_BYTES - base_size + extra_byte)
    body = json.dumps(envelope, separators=(",", ":")).encode()
    assert len(body) == MAX_RESPONSE_BYTES + extra_byte
    fixture = server(body=body, chunked=chunked)
    selected = provider(settings, fixture)
    if extra_byte:
        with pytest.raises(CapabilityError) as error:
            selected.embedding("source")
        assert (error.value.code, error.value.status_code) == ("provider_failed", 502)
    else:
        assert selected.embedding("source") == [3.0, 4.0, 0.0]
    assert fixture.received.get(timeout=2)["path"] == "/api/embed"


@pytest.mark.parametrize("phase", ["headers", "body"])
def test_native_ollama_deadline_bounds_slow_headers_and_body_in_real_process(settings, server, phase):
    fixture = server(stall=phase)
    selected = provider(settings, fixture)
    started = time.monotonic()
    marker = execution_deadline.set(started + 0.4)
    try:
        with pytest.raises(CapabilityError) as error:
            selected.embedding("source")
    finally:
        execution_deadline.reset(marker)
    assert (error.value.code, error.value.status_code) == ("provider_failed", 502)
    assert time.monotonic() - started < 2
    assert fixture.received.get(timeout=2)["path"] == "/api/embed"
    if phase == "body":
        assert fixture.received.get(timeout=2) == {"phase": "body_started"}
