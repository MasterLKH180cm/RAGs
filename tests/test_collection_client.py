"""Public SDK wire boundaries and import safety, without provider credentials."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import httpx
import pytest

from rags.client import ClientError, Collection, HttpClient, SearchResult


pytestmark = pytest.mark.unit
KEY = "scoped-client-test-credential"
INFO = {"id": "col_test", "name": "notes", "metadata": {"team": "platform"}, "generation": 0}


def mocked_client(monkeypatch, handler, **kwargs):
    transport = httpx.MockTransport(handler)
    opened = []
    original = httpx.AsyncClient

    def factory(**options):
        opened.append(options)
        return original(transport=transport, **options)

    monkeypatch.setattr("rags.client.httpx.AsyncClient", factory)
    return HttpClient("https://rag.example/base", KEY, **kwargs), opened


def test_public_import_has_no_configuration_database_or_model_side_effect(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RAGS_", "OPENAI_", "ANTHROPIC_"))}
    env.update(PYTHONPATH=str(Path(__file__).parents[1]), RAGS_KEYS_JSON="deliberately-invalid")
    code = (
        "import pathlib,sys; import rags.client; "
        "assert not any(n in sys.modules for n in "
        "('rags.api','rags.collection_service','torch','transformers','fastembed')); "
        "assert not list(pathlib.Path('.').iterdir()); print('clean import')"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=20, check=True)
    assert result.stdout.strip() == "clean import"


@pytest.mark.parametrize("url", [
    "", " https://example.com", "https://example.com/ ", "https://example.com\\other",
    "https://key:secret@example.com", "https://example.com?secret=1", "https://example.com#token",
    "file:///tmp/data", "http://example.com", "https://example.com:99999", "https://example.com:0",
    "https://example.com/../elsewhere", "https://example.com/%2e%2e", "https://example.com/\n",
])
def test_http_endpoint_is_explicit_and_credentials_cannot_hide_in_url(url):
    with pytest.raises(ClientError) as error:
        HttpClient(url, KEY)
    assert error.value.code == "invalid_endpoint"
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("options", [
    {"timeout": True}, {"timeout": 0}, {"timeout": float("nan")}, {"timeout": float("inf")},
    {"timeout": 10 ** 1000}, {"timeout": "30"}, {"max_response_bytes": 0},
    {"max_response_bytes": True}, {"max_response_bytes": 2 * 1024 * 1024 + 1},
    {"allow_insecure_http": "true"},
])
def test_client_limits_reject_nonfinite_bool_or_unbounded_options(options):
    with pytest.raises(ClientError, match="invalid_client_configuration"):
        HttpClient("https://example.com", KEY, **options)


@pytest.mark.parametrize("token", ["", "has spaces", "secret\nvalue", "秘密", "x" * 4097, None])
def test_bad_bearer_never_appears_in_error(token):
    with pytest.raises(ClientError) as error:
        HttpClient("https://example.com", token)
    assert error.value.code == "invalid_token"
    assert str(error.value) == "invalid_token (HTTP 401)"


def test_loopback_ipv6_and_explicit_insecure_override_are_allowed():
    with HttpClient("http://[::1]:8000", KEY) as local:
        assert KEY not in repr(local)
    with HttpClient("http://private.example:8000", KEY, allow_insecure_http=True):
        pass
    with pytest.raises(ClientError, match="client_closed"):
        local.list_collections()


def test_collection_sdk_routes_use_ids_and_preserve_wire_contract(monkeypatch):
    seen = []
    info = dict(INFO)

    def handler(request):
        nonlocal info
        assert request.headers["authorization"] == "Bearer " + KEY
        assert request.headers["accept-encoding"] == "identity"
        path = request.url.path
        data = json.loads(request.content) if request.content else dict(request.url.params)
        seen.append((request.method, path, data))
        if path == "/base/collections":
            if request.method == "GET":
                return httpx.Response(200, json={"items": [info], "total": 1, "limit": 1, "offset": 0})
            return httpx.Response(201, json=info)
        if path == "/base/collections/col_test":
            if request.method == "PATCH":
                info = {**info, **data, "generation": 1}
            if request.method == "DELETE":
                return httpx.Response(200, json={"collection_id": "col_test", "deleted": 1})
            return httpx.Response(200, json=info)
        if path.endswith("/fork"):
            return httpx.Response(201, json={**info, "id": "col_fork", "name": data["name"]})
        if path.endswith("/count"):
            return httpx.Response(200, json={"collection_id": "col_test", "count": 2, "generation": 1})
        if path.endswith("/indexing-status"):
            return httpx.Response(200, json={"status": "ready", "consistency": "transactional"})
        if path.endswith("/search"):
            return httpx.Response(200, json={"generation": 1, "results": [
                {"items": [{"id": "a", "version": "v1", "document": str(index)}]}
                for index in range(len(data["searches"]))]})
        if path.endswith("/ask"):
            return httpx.Response(200, json={"answer": "Evidence-backed answer", "citations": []})
        if path.endswith("/records/get"):
            return httpx.Response(200, json={"items": [], "total": 0, "generation": 1})
        return httpx.Response(200, json={"collection_id": "col_test", "generation": 1, "changed": 1})

    client, opened = mocked_client(monkeypatch, handler)
    with client:
        collection = client.get_or_create_collection("notes", metadata={"team": "platform"})
        assert seen[-1][2]["get_or_create"] is True
        assert client.create_collection("notes").id == "col_test"
        assert client.get_collection("notes").id == "col_test"
        assert client.get_collection(collection_id="col_test").name == "notes"
        assert client.list_collections()[0].id == collection.id
        assert client.list_collections_page()["total"] == 1
        for method in (collection.add, collection.upsert, collection.update):
            assert method(ids=["a"], documents=["text"])["changed"] == 1
        assert collection.get(ids=["a"], include=[])["total"] == 0
        assert collection.count() == 2
        assert collection.indexing_status()["consistency"] == "transactional"
        assert collection.get_indexing_status()["status"] == "ready"
        original = collection.metadata
        original["team"] = "changed-locally"
        assert collection.metadata == {"team": "platform"}
        collection.modify(name="renamed")
        assert collection.name == "renamed" and collection.id == "col_test"
        assert collection.refresh() is collection
        assert collection.fork("branch").id == "col_fork"
        result = collection.search(query_texts=["one", "two"], mode="lexical")
        assert result.rows(1)[0]["document"] == "1"
        result.rows()[0]["document"] = "not-in-result"
        assert result.rows()[0]["document"] == "0"
        assert collection.query(query_text="one", mode="lexical").rows()[0]["id"] == "a"
        collection.search(query_embeddings=[[1.0, 0.0]], embedding_space_id="external", mode="dense")
        collection.search(query_texts=["one"], query_embeddings=[[1.0, 0.0]],
                          embedding_space_id="external", mode="hybrid")
        assert collection.ask("What is the answer?", mode="lexical")["answer"]
        collection.delete(ids=["a"], expected_generation=1)
        client.delete_collection(collection_id=collection.id, expected_generation=2)
    assert all(item["trust_env"] is False and item["follow_redirects"] is False for item in opened)
    assert ("DELETE", "/base/collections/col_test", {"expected_generation": "2"}) in seen
    assert any(path.endswith("/records/update") for _, path, _ in seen)
    assert not any("renamed" in path for _, path, _ in seen)


@pytest.mark.parametrize("call", [
    lambda c: c.get_collection(),
    lambda c: c.get_collection("notes", collection_id="other"),
    lambda c: c.get_collection(collection_id="../escape"),
    lambda c: c.delete_collection(),
    lambda c: c.create_collection("invalid name"),
    lambda c: Collection(c, INFO).upsert(ids=["a", "a"], documents=["one", "two"]),
    lambda c: Collection(c, INFO).upsert(ids=["a"], documents=["one", "two"]),
    lambda c: Collection(c, INFO).upsert(ids=["a"], metadatas=[{"bad": float("nan")}]),
    lambda c: Collection(c, INFO).delete(),
    lambda c: Collection(c, INFO).search(query_texts="text", mode="lexical"),
    lambda c: Collection(c, INFO).search(query_texts=[], mode="lexical"),
    lambda c: Collection(c, INFO).search(query_texts=["a"], query_embeddings=[[1, 0], [0, 1]]),
    lambda c: Collection(c, INFO).search(query_texts=["a"], query_text="b"),
    lambda c: Collection(c, INFO).search(searches=[{"mode": "lexical", "query_text": "a"}], limit=1),
])
def test_invalid_entire_request_is_rejected_before_any_http(monkeypatch, call):
    client, opened = mocked_client(monkeypatch, lambda _: pytest.fail("No HTTP allowed"))
    with pytest.raises(ClientError) as error:
        call(client)
    assert error.value.status_code == 422
    assert opened == []


def test_total_batch_bytes_are_bounded_before_http(monkeypatch):
    client, opened = mocked_client(monkeypatch, lambda _: pytest.fail("No HTTP allowed"))
    with pytest.raises(ClientError, match="request_too_large"):
        Collection(client, INFO).upsert(ids=[str(i) for i in range(100)], documents=["字" * 20000] * 100)
    assert not opened


@pytest.mark.parametrize("status,payload,code", [
    (401, {"detail": {"code": "invalid_token", "reason": "provider-private-secret"}}, "invalid_token"),
    (409, {"code": "generation_conflict", "debug": KEY}, "generation_conflict"),
    (500, {"detail": {"code": "provider-private-secret", "reason": KEY}}, "request_failed"),
    (502, {"detail": {"code": [KEY]}}, "request_failed"),
    (403, {"detail": KEY}, "request_failed"),
])
def test_safe_typed_errors_never_echo_untrusted_body(monkeypatch, status, payload, code):
    client, _ = mocked_client(monkeypatch, lambda _: httpx.Response(status, json=payload))
    with pytest.raises(ClientError) as error:
        client.list_collections()
    assert (error.value.code, error.value.status_code) == (code, status)
    assert KEY not in str(error.value) and "provider-private-secret" not in str(error.value)


@pytest.mark.parametrize("body", [b'[]', b'not-json', b'{"items":[],"items":[]}', b'{"a":NaN}'])
def test_success_payload_requires_strict_json_object(monkeypatch, body):
    client, _ = mocked_client(monkeypatch, lambda _: httpx.Response(200, content=body))
    with pytest.raises(ClientError, match="invalid_response"):
        client.list_collections()


def test_nonjson_server_error_preserves_status_without_echo(monkeypatch):
    client, _ = mocked_client(monkeypatch, lambda _: httpx.Response(503, content=KEY.encode()))
    with pytest.raises(ClientError) as error:
        client.list_collections()
    assert (error.value.code, error.value.status_code) == ("request_failed", 503)


class BodyStream(httpx.AsyncByteStream):
    def __init__(self, data, delay=0):
        self.data, self.delay, self.closed, self.reads = data, delay, False, 0

    async def __aiter__(self):
        for part in self.data:
            self.reads += 1
            await asyncio.sleep(self.delay)
            yield part

    async def aclose(self):
        self.closed = True


@pytest.mark.parametrize("status,headers,expected", [
    (307, {"location": "https://attacker.example"}, "redirect_rejected"),
    (200, {"content-encoding": "gzip"}, "unsupported_response_encoding"),
    (200, {"content-length": "999999999"}, "response_too_large"),
    (200, {"content-length": "1,2"}, "invalid_response"),
])
def test_rejected_envelopes_do_not_read_body_or_follow_redirect(monkeypatch, status, headers, expected):
    stream = BodyStream([b"private-error-body"])
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, headers=headers, stream=stream)

    client, _ = mocked_client(monkeypatch, handler)
    with pytest.raises(ClientError, match=expected):
        client.list_collections()
    assert len(calls) == 1 and stream.reads == 0 and stream.closed


def test_streamed_size_and_declared_length_are_checked(monkeypatch):
    stream = BodyStream([b"x" * 129])
    client, _ = mocked_client(monkeypatch, lambda _: httpx.Response(200, stream=stream), max_response_bytes=128)
    with pytest.raises(ClientError, match="response_too_large"):
        client.list_collections()
    assert stream.closed


def test_buffered_response_without_declared_length_is_also_bounded(monkeypatch):
    def handler(_):
        response = httpx.Response(200, content=b"x" * 129)
        del response.headers["content-length"]
        return response

    client, _ = mocked_client(monkeypatch, handler, max_response_bytes=128)
    with pytest.raises(ClientError, match="response_too_large"):
        client.list_collections()


def test_inconsistent_content_length_fails_closed(monkeypatch):
    stream = BodyStream([b'{"items":[]}'])
    client, _ = mocked_client(monkeypatch, lambda _: httpx.Response(200, headers={"content-length": "1"}, stream=stream))
    with pytest.raises(ClientError, match="invalid_response"):
        client.list_collections()


def test_whole_deadline_cancels_slow_drip_body_and_joins_loop_thread(monkeypatch):
    stream = BodyStream([b" "] * 100, delay=0.01)
    client, _ = mocked_client(monkeypatch, lambda _: httpx.Response(200, stream=stream), timeout=0.04)

    async def from_existing_event_loop():
        with pytest.raises(ClientError, match="request_timeout"):
            client.list_collections()

    started = time.monotonic()
    asyncio.run(from_existing_event_loop())
    assert time.monotonic() - started < 1.0
    assert stream.closed and stream.reads < 100
    assert not any(t.name.startswith("rags-collection-client") for t in threading.enumerate())


def test_transport_errors_are_sanitized(monkeypatch):
    def fail(request):
        raise httpx.ConnectError(KEY, request=request)

    client, _ = mocked_client(monkeypatch, fail)
    with pytest.raises(ClientError, match="transport_failed") as error:
        client.list_collections()
    assert KEY not in str(error.value)


@pytest.mark.parametrize("response", [
    {"items": []}, {"items": [{**INFO, "name": "other"}]}, {"items": [INFO, INFO]}, {"items": "invalid"},
])
def test_name_resolution_is_exact_and_does_not_choose_arbitrary_collection(monkeypatch, response):
    client, _ = mocked_client(monkeypatch, lambda _: httpx.Response(200, json=response))
    with pytest.raises(ClientError) as error:
        client.get_collection("notes")
    assert error.value.code == ("collection_not_found" if response == {"items": []} else "invalid_response")


def test_id_lookup_and_refresh_reject_a_server_response_for_another_collection(monkeypatch):
    client, _ = mocked_client(monkeypatch, lambda _: httpx.Response(200, json={**INFO, "id": "other-id"}))
    collection = Collection(client, INFO)
    for call in (lambda: client.get_collection(collection_id="col_test"), collection.refresh):
        with pytest.raises(ClientError, match="invalid_response"):
            call()
    assert collection.id == "col_test" and collection.to_dict()["id"] == "col_test"


@pytest.mark.parametrize("info", [None, {}, {"id": "../escape", "name": "notes"}, {"id": "id", "name": "bad name"}])
def test_handles_reject_invalid_server_identifiers(info):
    with pytest.raises(ClientError, match="invalid_response"):
        Collection(HttpClient("https://rag.example", KEY), info)


def test_search_rows_have_explicit_bounds_and_validate_shape():
    result = SearchResult(results=[{"items": []}])
    assert result.rows() == []
    for index in (-1, True, "0", 1):
        with pytest.raises(ClientError, match="invalid_result_index"):
            result.rows(index)
    with pytest.raises(ClientError, match="invalid_response"):
        SearchResult(results=[{}]).rows()
