"""Real transport, API concurrency and transactional agent completion regressions."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
import socket
import ssl
import subprocess
from threading import Barrier, Event, Lock, Thread
import time
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit
import uuid

import httpx
import pytest
import uvicorn

from rags.agents.config import AgentPlatformConfig
from rags.agents.engine import AgentEngine, ToolRegistry
from rags.agents.memory import ConversationStore, HistoryTurn, scoped_session
from rags.agents.router import ModelReply, ModelRouter
from rags.agents.state import StateStore
from rags.api import create_app
from rags.auth import PERMISSIONS, Principal
from rags.config import Settings
from rags.features import FeatureFlags
from rags.registry import CapabilityError
from rags.storage import Store

pytestmark = pytest.mark.integration


@contextmanager
def serve(handler, tls=None):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    if tls is not None:
        server.socket = tls.wrap_socket(server.socket, server_side=True)
    thread = Thread(target=lambda: server.serve_forever(poll_interval=0.02), daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
        assert not thread.is_alive()


@pytest.mark.parametrize("operation", ["chat", "embedding"])
@pytest.mark.parametrize("phase", ["headers", "body"])
def test_router_complete_deadline_cancels_actual_slow_drip(operation, phase):
    stopped, disconnected = Event(), Event()

    class Drip(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            prefix = (b"HTTP/1.1 200 OK\r\nX-Slow: " if phase == "headers" else
                      b"HTTP/1.1 200 OK\r\nContent-Length: 4096\r\n\r\n{")
            try:
                self.connection.sendall(prefix)
                for _ in range(200):
                    if stopped.wait(0.01):
                        break
                    self.connection.sendall(b" ")
            except OSError:
                disconnected.set()

        def log_message(self, *_):
            pass

    with serve(Drip) as server:
        router = ModelRouter(AgentPlatformConfig(
            gateway_url=f"http://127.0.0.1:{server.server_port}/v1", gateway_key="fixture",
            allow_http=True, models={"model": "openai/fixture"}))
        started = time.monotonic()
        try:
            with pytest.raises(CapabilityError) as error:
                if operation == "chat":
                    router.complete(model="model", messages=[{"role": "user", "content": "hello"}],
                                    timeout_seconds=0.1)
                else:
                    router.embedding(model="model", text="hello", timeout_seconds=0.1)
            assert (error.value.code, error.value.status_code) == ("model_timeout", 504)
            assert time.monotonic() - started < 0.75
            assert disconnected.wait(0.75), "The timed-out request must close its socket"
        finally:
            stopped.set()


@contextmanager
def live_app(app):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    thread = Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 3
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started
        yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    finally:
        server.should_exit = True
        thread.join(5)
        sock.close()
        assert not thread.is_alive()


@pytest.mark.parametrize("image", [False, True])
def test_multipart_provider_work_keeps_real_http_liveness_responsive(tmp_path, image):
    entered, release = Event(), Event()

    class Provider(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            entered.set()
            assert release.wait(3)
            payload = ({"choices": [{"message": {"content": "A synthetic diagram"}}]} if image else
                       {"data": [{"embedding": [0.2, 0.8]}]})
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    with serve(Provider) as provider:
        endpoint = f"http://127.0.0.1:{provider.server_port}/v1"
        settings = Settings(
            db_path=str(tmp_path / "api.db"), keys={"fixture": "tenant"},
            embedding_provider="local" if image else "openai",
            embedding_api_key="fixture", embedding_base_url=endpoint,
            embedding_model="fixture-embedding", model_provider="openai" if image else "extractive",
            embedding_tokenizer_config={"backend": "utf8_bytes"}, embedding_max_tokens=100000,
            openai_key="fixture", openai_base_url=endpoint, chat_model="fixture-caption",
            anthropic_key="", anthropic_model="fixture-unused")
        upload = ("source.txt", b"Synthetic source content")
        if image:
            from PIL import Image
            output = io.BytesIO()
            Image.new("RGB", (8, 8)).save(output, format="PNG")
            upload = ("source.png", output.getvalue())
        with live_app(create_app(settings)) as endpoint, ThreadPoolExecutor(max_workers=1) as pool:
            with httpx.Client(base_url=endpoint, timeout=3, trust_env=False) as client:
                pending = pool.submit(client.post, "/images" if image else "/documents/upload",
                                      headers={"Authorization": "Bearer fixture"}, files={"file": upload})
                try:
                    assert entered.wait(2)
                    started = time.monotonic()
                    assert client.get("/live", timeout=0.6).status_code == 200
                    assert time.monotonic() - started < 0.6
                finally:
                    release.set()
                response = pending.result(timeout=3)
                assert response.status_code == 201, response.text


def test_cancelled_uploads_keep_bounded_worker_slots_until_actual_completion(tmp_path, monkeypatch):
    app = create_app(Settings(db_path=str(tmp_path / "capacity.db"), keys={"fixture": "tenant"},
                              embedding_provider="local", model_provider="extractive", openai_key="",
                              openai_base_url="https://unused.invalid/v1", embedding_model="fixture",
                              chat_model="fixture", anthropic_key="", anthropic_model="fixture"))
    release, entered = Event(), Event()
    lock = Lock()
    started = []
    embedding = app.state.service.provider.embedding

    def blocked(text):
        with lock:
            started.append(text)
            if len(started) == 8:
                entered.set()
        assert release.wait(5)
        return embedding(text)

    monkeypatch.setattr(app.state.service.provider, "embedding", blocked)

    async def scenario():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                         base_url="http://api.test") as client:
                async def upload(index):
                    return await client.post("/documents/upload", headers={"Authorization": "Bearer fixture"},
                                             files={"file": (f"item-{index}.txt", f"Source {index}".encode())})
                tasks = [asyncio.create_task(upload(index)) for index in range(8)]
                try:
                    assert await asyncio.to_thread(entered.wait, 2)
                    assert (await client.get("/live")).status_code == 200
                    overflow = await upload(8)
                    assert overflow.status_code == 503
                    assert overflow.json()["detail"]["code"] == "ingestion_capacity"
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    assert (await upload(9)).status_code == 503
                    assert len(started) == 8
                finally:
                    release.set()
                    await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(scenario())
    assert len(app.state.service.store.list_documents("tenant")) == 8


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=[pytest.mark.external, pytest.mark.postgres])])
def completion_store(request, tmp_path):
    if request.param == "sqlite":
        yield StateStore(Store(str(tmp_path / "completion.db")))
        return
    from psycopg import connect, sql
    endpoint = os.getenv("RAGS_TEST_POSTGRES_URL")
    if not endpoint:
        pytest.fail("RAGS_TEST_POSTGRES_URL required for native conversation completion")
    parsed = urlsplit(endpoint)
    schema = "rags_completion_" + uuid.uuid4().hex
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["options"] = (query.get("options", "") + f" -c search_path={schema},public").strip()
    isolated = urlunsplit(parsed._replace(query=urlencode(query, quote_via=quote)))
    with connect(endpoint, autocommit=True, connect_timeout=5) as admin:
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            yield StateStore(Store(isolated))
        finally:
            admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def test_parallel_native_engines_commit_only_whole_turns(completion_store):
    state = completion_store
    gate = Barrier(2)
    principal = Principal("tenant", "owner", frozenset(PERMISSIONS), "api_key")
    config = AgentPlatformConfig(gateway_url="https://fixture.invalid/v1", gateway_key="fixture",
                                  models={"main": "openai/fixture"})

    class Router:
        def complete(self, **kwargs):
            goal = kwargs["messages"][-1]["content"]
            gate.wait(timeout=3)
            return ModelReply(content="Answer " + goal, finish_reason="stop", model="openai/fixture")

    def run(goal):
        engine = AgentEngine(config=config, state=state, router=Router(), features=FeatureFlags(),
                             registry=ToolRegistry())
        return engine.start(principal, goal=goal, model="main", session_id="same-session")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, goal) for goal in ("Question A", "Question B")]
        results = [future.result(timeout=5) for future in futures]
    assert all(result.status == "completed" for result in results)
    history = ConversationStore(state).last("tenant", scoped_session("owner", "same-session"))
    assert [message["role"] for message in history] == ["user", "assistant", "user", "assistant"]
    assert {(history[index]["content"], history[index + 1]["content"]) for index in (0, 2)} == {
        ("Question A", "Answer Question A"), ("Question B", "Answer Question B")}
    assert ConversationStore(state).last("other", scoped_session("owner", "same-session")) == []


def test_completion_failure_rolls_back_checkpoint_and_pair_and_retry_is_fenced(completion_store, monkeypatch):
    from rags.agents import memory
    state = completion_store
    checkpoint = state.create("tenant", kind="agent", fingerprint="a" * 64, state={"answer": None})
    history = ConversationStore(state)
    turn = HistoryTurn("session", "Question", "Answer [source]", ("source",))
    insert = memory._insert_message

    def fail_assistant(db, tenant, session_id, sequence, role, *remaining):
        if role == "assistant":
            raise RuntimeError("Injected failure after the user insert")
        return insert(db, tenant, session_id, sequence, role, *remaining)

    def complete():
        return state.update("tenant", checkpoint.run_id, expected_revision=0, fingerprint="a" * 64,
                            status="completed", state={"answer": "Answer [source]"}, history_turn=turn)

    monkeypatch.setattr(memory, "_insert_message", fail_assistant)
    with pytest.raises(RuntimeError, match="Injected"):
        complete()
    preserved = state.get("tenant", checkpoint.run_id)
    assert preserved.revision == 0 and preserved.status == "running"
    assert history.last("tenant", "session") == []
    monkeypatch.setattr(memory, "_insert_message", insert)
    assert complete().status == "completed"
    assert history.last("tenant", "session", with_evidence=True) == [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer [source]", "evidence_ids": ["source"]}]
    with pytest.raises(CapabilityError) as stale:
        complete()
    assert stale.value.code == "revision_conflict"
    assert len(history.last("tenant", "session")) == 2


@pytest.fixture
def tls_material(tmp_path):
    cert, key = tmp_path / "ca.pem", tmp_path / "server.key"
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
        "-keyout", str(key), "-out", str(cert), "-subj", "/CN=loopback-fixture",
        "-addext", "subjectAltName=IP:127.0.0.1", "-addext", "basicConstraints=critical,CA:TRUE",
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    return cert, context


def test_router_requires_verified_explicit_ca_for_private_https_gateway(tls_material, monkeypatch):
    cert, context = tls_material

    class Gateway(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            body = b'{"choices":[{"message":{"content":"verified"},"finish_reason":"stop"}]}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    with serve(Gateway, context) as server:
        config = AgentPlatformConfig(gateway_url=f"https://127.0.0.1:{server.server_port}/v1",
                                      gateway_key="fixture", models={"main": "openai/fixture"})
        monkeypatch.setenv("SSL_CERT_FILE", str(cert))
        monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
        with pytest.raises(CapabilityError) as untrusted:
            ModelRouter(config).complete(model="main", messages=[{"role": "user", "content": "hello"}])
        assert untrusted.value.code == "model_gateway_failed"
        configured = replace(config, gateway_ca_file=str(cert))
        assert len(configured.gateway_ca_sha256) == 64
        assert "BEGIN CERTIFICATE" not in repr(configured)
        cert.unlink()  # Verified CA bytes are snapshotted with configuration.
        result = ModelRouter(configured).complete(model="main", messages=[{"role": "user", "content": "hello"}])
        assert result.content == "verified"


@pytest.mark.parametrize("kind", ["missing", "directory", "symlink", "empty", "oversize", "invalid", "fifo"])
def test_custom_gateway_ca_rejects_unsafe_or_invalid_files(tmp_path, kind):
    path = tmp_path / "ca.pem"
    if kind == "directory":
        path.mkdir()
    elif kind == "symlink":
        target = tmp_path / "target.pem"
        target.write_text("not a CA")
        path.symlink_to(target)
    elif kind == "fifo":
        os.mkfifo(path)
    elif kind != "missing":
        path.write_bytes(b"x" * (256 * 1024 + 1) if kind == "oversize" else
                         b"" if kind == "empty" else b"not a PEM")
    with pytest.raises(ValueError, match="Router CA"):
        AgentPlatformConfig(gateway_url="https://fixture.invalid/v1", gateway_key="fixture",
                            models={"main": "openai/fixture"}, gateway_ca_file=str(path))


def test_gateway_ca_trust_change_invalidates_approval_fingerprint(tmp_path, tls_material):
    cert, _ = tls_material
    state = StateStore(Store(str(tmp_path / "ca-state.db")))
    original = AgentPlatformConfig(gateway_url="https://fixture.invalid/v1", gateway_key="fixture",
                                     models={"main": "openai/fixture"})
    custom = replace(original, gateway_ca_file=str(cert))

    def fingerprint(config):
        return AgentEngine(config=config, router=object(), state=state,
                           features=FeatureFlags(), registry=ToolRegistry())._fingerprint

    assert fingerprint(original) != fingerprint(custom)
    moved = tmp_path / "same-cert.pem"
    moved.write_bytes(cert.read_bytes())
    assert fingerprint(custom) == fingerprint(replace(original, gateway_ca_file=str(moved)))
    with pytest.raises(ValueError, match="HTTPS"):
        replace(custom, gateway_url="http://127.0.0.1:8080/v1", allow_http=True)
