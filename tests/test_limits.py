"""Request intake regression tests, including disconnect-aware ASGI replay."""
import asyncio

import pytest
from fastapi.testclient import TestClient

from rags.limits import BodyLimitMiddleware


def exercise(frames, *, headers=(), delays=(), timeout=0.5, check_disconnect=False):
    sent, replayed = [], []
    state = {"handler_called": False, "reads": 0}

    async def receive():
        index = state["reads"]
        state["reads"] += 1
        if index < len(delays):
            await asyncio.sleep(delays[index])
        if index >= len(frames):
            await asyncio.Event().wait()
        return frames[index]

    async def send(message):
        sent.append(message)

    async def handler(scope, receive, send):
        state["handler_called"] = True
        replayed.append(await receive())
        if check_disconnect:
            replayed.append(await receive())
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    scope = {"type": "http", "http_version": "1.1", "method": "POST", "headers": headers}
    middleware = BodyLimitMiddleware(handler, max_body_bytes=8, timeout_seconds=timeout)
    asyncio.run(middleware(scope, receive, send))
    return state, sent, replayed


def request(body=b"", more=False):
    return {"type": "http.request", "body": body, "more_body": more}


def assert_rejected(result, status):
    state, sent, replayed = result
    assert not state["handler_called"] and not replayed
    assert [message["type"] for message in sent] == ["http.response.start", "http.response.body"]
    assert sent[0]["status"] == status


@pytest.mark.parametrize("headers", [(), [(b"content-length", b"8")]])
def test_complete_body_at_limit_reaches_handler_once(headers):
    state, sent, replayed = exercise([request(b"1234", True), request(b"5678")], headers=headers)
    assert state == {"handler_called": True, "reads": 2}
    assert sent[0]["status"] == 200
    assert replayed == [request(b"12345678")]


def test_chunked_body_without_content_length_stops_before_parser():
    result = exercise([request(b"12345", True), request(b"6789", True), request()])
    assert_rejected(result, 413)
    assert result[0]["reads"] == 2


@pytest.mark.parametrize("length", [b"9", b"9" * 5000])
def test_declared_oversize_does_not_read_body(length):
    result = exercise([], headers=[(b"content-length", length)])
    assert_rejected(result, 413)
    assert result[0]["reads"] == 0


@pytest.mark.parametrize("headers,frames", [
    ([(b"content-length", b"+1")], []),
    ([(b"content-length", b"1"), (b"content-length", b"2")], []),
    ([(b"content-length", b"2")], [request(b"x")]),
])
def test_invalid_or_conflicting_content_length_is_rejected(headers, frames):
    assert_rejected(exercise(frames, headers=headers), 400)


def test_stalled_body_returns_single_timeout_before_handler():
    assert_rejected(exercise([], timeout=0.01), 408)


def test_deadline_covers_total_intake_instead_of_resetting_each_chunk():
    result = exercise(
        [request(b"a", True), request(b"b", True), request(b"c")],
        delays=(0.03, 0.03, 0.03), timeout=0.07,
    )
    assert_rejected(result, 408)


def test_disconnect_during_intake_does_not_run_handler_or_send_response():
    state, sent, replayed = exercise([request(b"part", True), {"type": "http.disconnect"}])
    assert not state["handler_called"] and not sent and not replayed


def test_replay_preserves_real_disconnect_for_streaming_response():
    state, sent, replayed = exercise(
        [request(), {"type": "http.disconnect"}], check_disconnect=True,
    )
    assert state == {"handler_called": True, "reads": 2}
    assert sent[0]["status"] == 200
    assert replayed == [request(), {"type": "http.disconnect"}]


def test_lifespan_is_passed_through_without_body_reads():
    scope = {"type": "lifespan"}
    called = []

    async def receive():
        raise AssertionError("Middleware must not read lifespan messages")

    async def send(message):
        raise AssertionError("Middleware must not send HTTP responses for lifespan")

    async def handler(actual, actual_receive, actual_send):
        called.append(actual)
        assert actual_receive is receive and actual_send is send

    asyncio.run(BodyLimitMiddleware(handler, max_body_bytes=8)(scope, receive, send))
    assert called == [scope]


@pytest.fixture
def client(tmp_path, monkeypatch):
    from rags.api import create_app
    from rags.config import Settings

    monkeypatch.setenv("RAGS_ENV", "test")
    monkeypatch.setenv("RAGS_DB_PATH", str(tmp_path / "rag.db"))
    monkeypatch.setenv("RAGS_KEYS_JSON", '{"a-key":"a"}')
    monkeypatch.setenv("RAGS_MODEL_PROVIDER", "extractive")
    monkeypatch.setenv("RAGS_EMBEDDING_PROVIDER", "local")
    with TestClient(create_app(Settings.from_env())) as result:
        yield result


def test_api_installs_body_limit_before_json_or_multipart_parsing(client):
    assert client.get("/live").status_code == 200
    response = client.post("/documents", content=b"x" * 2_100_001,
                           headers={"content-type": "application/json"})
    assert response.status_code == 413
    assert response.json()["detail"] == "Request body exceeds limit"
    response = client.post("/documents/upload", files={
        "file": ("large.txt", b"x" * 2_100_001, "text/plain"),
    })
    assert response.status_code == 413
