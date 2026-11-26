"""Direct provider transport bounds without external models or credentials."""
import asyncio
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from time import monotonic

import httpx
import pytest

from rags import provider_transport as transport
from rags.registry import CapabilityError


class TrackedStream(httpx.AsyncByteStream):
    def __init__(self, parts=(), *, delay=0):
        self.parts, self.delay = parts, delay
        self.reads, self.closed = 0, False

    async def __aiter__(self):
        for part in self.parts:
            if self.delay:
                await asyncio.sleep(self.delay)
            self.reads += 1
            yield part

    async def aclose(self):
        self.closed = True


def client_fixture(stream=None, *, headers=None, status=200, handler=None):
    clients, requests = [], []

    async def dispatch(request):
        requests.append(request)
        if handler is not None:
            return await handler(request)
        return httpx.Response(status, headers=headers or {}, stream=stream)

    def factory(**kwargs):
        assert kwargs["follow_redirects"] is False
        assert 0 < kwargs["timeout"] <= 2
        client = httpx.AsyncClient(transport=httpx.MockTransport(dispatch), **kwargs)
        clients.append(client)
        return client

    return factory, clients, requests


def request(factory, **changes):
    values = {"headers": {"Authorization": "Bearer private-credential"},
              "json": {"model": "approved", "input": "來源"}, "timeout": 2,
              "client_factory": factory}
    values.update(changes)
    return transport.post("https://provider.invalid/v1/generate", **values)


@pytest.mark.unit
class TestBoundedProviderTransport:
    def test_json_facade_retains_its_declared_media_type(self):
        stream = TrackedStream([b"{}"])
        factory, clients, requests = client_fixture(stream)
        request(factory, headers={"Content-Type": "text/plain"})
        assert requests[0].headers["content-type"] == "application/json"
        assert stream.closed and clients[0].is_closed

    @pytest.mark.parametrize("inside_async", [False, True])
    def test_form_posts_share_bounds_and_preserve_basic_auth(self, inside_async):
        from urllib.parse import parse_qs

        stream = TrackedStream([b'{"active":false}'])
        factory, clients, requests = client_fixture(stream)

        def send():
            return transport.bounded_form_post(
                "https://identity.invalid/introspect", headers={"Accept": "application/json"},
                data={"token": "opaque&=+ /漢", "token_type_hint": "access_token"},
                auth=httpx.BasicAuth("client", "test-secret"), timeout=2,
                client_factory=factory,
            )

        async def invoke():
            return send()

        response = asyncio.run(invoke()) if inside_async else send()
        assert response.json() == {"active": False}
        sent = requests[0]
        assert parse_qs(sent.content.decode()) == {
            "token": ["opaque&=+ /漢"], "token_type_hint": ["access_token"],
        }
        assert sent.headers["content-type"] == "application/x-www-form-urlencoded"
        assert sent.headers["authorization"] == "Basic Y2xpZW50OnRlc3Qtc2VjcmV0"
        assert sent.headers["accept-encoding"] == "identity"
        assert stream.closed and clients[0].is_closed

    @pytest.mark.parametrize("data", [{"token": "too-large"}, {"token": False}, ["not-a-form"]])
    def test_form_validation_precedes_network(self, data):
        factory, clients, requests = client_fixture()
        with pytest.raises(ValueError):
            transport.bounded_form_post(
                "https://identity.invalid/introspect", headers={}, data=data,
                auth=httpx.BasicAuth("client", "test-secret"), timeout=2,
                max_request_bytes=3, client_factory=factory,
            )
        assert clients == [] and requests == []

    @pytest.mark.parametrize("inside_async", [False, True])
    def test_success_uses_exact_utf8_json_and_closes_owned_resources(self, inside_async):
        stream = TrackedStream([b'{"value":', '"結果"}'.encode()])
        factory, clients, requests = client_fixture(stream)

        async def invoke():
            # Existing synchronous model providers are also called by async
            # image/research paths; they must not attempt a nested event loop.
            return request(factory)

        response = asyncio.run(invoke()) if inside_async else request(factory)
        assert response.json() == {"value": "結果"}
        sent = requests[0]
        assert sent.content == '{"model":"approved","input":"來源"}'.encode()
        assert sent.headers["content-type"] == "application/json"
        assert sent.headers["accept-encoding"] == "identity"
        assert int(sent.headers["content-length"]) == len(sent.content)
        assert stream.closed and all(client.is_closed for client in clients)

    @pytest.mark.parametrize("input_text", ["a", "漢😀", '\"\\\n'])
    def test_total_request_budget_includes_json_framing_before_any_client(self, input_text):
        payload = {"model": "route", "input": input_text}
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        stream = TrackedStream([b"{}"])
        factory, clients, requests = client_fixture(stream)
        with pytest.raises(CapabilityError) as error:
            request(factory, json=payload, max_request_bytes=len(raw) - 1)
        assert (error.value.code, error.value.status_code) == ("invalid_model_input", 422)
        assert clients == [] and requests == []
        assert request(factory, json=payload, max_request_bytes=len(raw)).content == b"{}"
        assert requests[0].content == raw and stream.closed and clients[0].is_closed

    @pytest.mark.parametrize("payload", [{"value": float("nan")}, {"value": float("inf")},
                                         {"value": {1, 2}}, {"value": "\ud800"}])
    def test_invalid_request_json_fails_without_network(self, payload):
        factory, clients, requests = client_fixture()
        with pytest.raises(CapabilityError) as error:
            request(factory, json=payload)
        assert error.value.code == "invalid_model_input" and error.value.status_code == 422
        assert clients == [] and requests == []

    @pytest.mark.parametrize("encoding", ["gzip", "br", "GZip", "identity, gzip", ""])
    def test_compressed_response_is_rejected_before_read_or_decompression(self, encoding):
        stream = TrackedStream([b"private-compressed-body"])
        factory, clients, _ = client_fixture(stream, headers={"Content-Encoding": encoding})
        with pytest.raises(CapabilityError, match="Compressed"):
            request(factory)
        assert stream.reads == 0 and stream.closed and clients[0].is_closed

    @pytest.mark.parametrize("length", [str(transport.MAX_RESPONSE_BYTES + 1), "-1", "invalid", "9" * 21, "１２"])
    def test_invalid_or_oversized_length_is_rejected_before_body_reads(self, length):
        # httpx headers require ASCII; a non-ASCII value cannot come from its
        # validated header mapping and is covered at the protocol parser instead.
        headers = [(b"content-length", length.encode("utf-8"))]
        stream = TrackedStream([b"private-body"])
        factory, clients, _ = client_fixture(stream, headers=headers)
        with pytest.raises(CapabilityError, match="byte length"):
            request(factory)
        assert stream.reads == 0 and stream.closed and clients[0].is_closed

    @pytest.mark.parametrize("declared", [False, True])
    def test_response_at_exact_limit_is_allowed_and_larger_chunked_body_is_not(self, declared):
        chunk = b"x" * (16 * 1024)
        count = transport.MAX_RESPONSE_BYTES // len(chunk)
        stream = TrackedStream([chunk] * count)
        headers = {"Content-Length": str(transport.MAX_RESPONSE_BYTES)} if declared else {}
        factory, clients, _ = client_fixture(stream, headers=headers)
        assert len(request(factory).content) == transport.MAX_RESPONSE_BYTES
        assert stream.closed and clients[0].is_closed
        oversized = TrackedStream([chunk] * (count + 3))
        factory, clients, _ = client_fixture(oversized)
        with pytest.raises(CapabilityError, match="exceeds the byte limit"):
            request(factory)
        assert oversized.reads == count + 1
        assert oversized.closed and clients[0].is_closed

    @pytest.mark.parametrize("status", [301, 302, 307, 308, 401, 429, 503])
    def test_redirects_and_error_bodies_are_not_followed_or_read(self, status):
        stream = TrackedStream([b"private-body private-credential"])
        factory, clients, requests = client_fixture(stream, status=status,
                                                    headers={"Location": "https://other.invalid"})
        with pytest.raises(CapabilityError) as error:
            request(factory)
        assert (error.value.code, error.value.status_code) == ("provider_failed", 502)
        assert "private" not in str(error.value) and "other.invalid" not in str(error.value)
        assert len(requests) == 1 and stream.reads == 0
        assert stream.closed and clients[0].is_closed

    def test_transport_errors_are_fixed_and_do_not_expose_private_details(self):
        async def fail(sent):
            raise httpx.ReadTimeout("private-credential private-body", request=sent)

        factory, clients, _ = client_fixture(handler=fail)
        with pytest.raises(CapabilityError) as error:
            request(factory)
        assert error.value.reason == "Configured provider request failed"
        assert error.value.__suppress_context__ and clients[0].is_closed

    @pytest.mark.parametrize("phase", ["headers", "body"])
    def test_one_deadline_cancels_stalls_and_closes_clients(self, phase):
        cancelled = []
        stream = TrackedStream([b"x"] * 100, delay=0.01)

        async def delayed(sent):
            if phase == "headers":
                try:
                    await asyncio.sleep(2)
                finally:
                    cancelled.append(True)
            return httpx.Response(200, stream=stream)

        factory, clients, _ = client_fixture(handler=delayed)
        started = monotonic()
        with pytest.raises(CapabilityError) as error:
            request(factory, timeout=0.08)
        assert monotonic() - started < 0.8
        assert error.value.code == "provider_failed" and clients[0].is_closed
        if phase == "headers":
            assert cancelled == [True]
        else:
            assert stream.closed and 0 < stream.reads < 100

    @pytest.mark.parametrize("timeout", [0, -1, True, float("inf"), float("nan"), 10**1000])
    def test_invalid_deadline_cannot_construct_client(self, timeout):
        factory, clients, _ = client_fixture()
        with pytest.raises(ValueError, match="timeout"):
            request(factory, timeout=timeout)
        assert clients == []

    @pytest.mark.parametrize("limit", [0, -1, True, transport.MAX_IMAGE_REQUEST_BYTES + 1])
    def test_invalid_request_limit_cannot_construct_client(self, limit):
        factory, clients, _ = client_fixture()
        with pytest.raises(ValueError, match="byte limit"):
            request(factory, max_request_bytes=limit)
        assert clients == []


@contextmanager
def slow_server(phase):
    stop = threading.Event()
    seen = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def do_POST(self):
            seen.append((self.headers.get("Accept-Encoding"),
                         self.rfile.read(int(self.headers["Content-Length"]))))
            if phase == "headers":
                stop.wait(2)
                return
            self.send_response(200)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                for _ in range(100):
                    if stop.wait(0.02):
                        return
                    self.wfile.write(b"1\r\nx\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/generate", seen
    finally:
        stop.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


@pytest.mark.integration
class TestRealProviderTransport:
    @pytest.mark.parametrize("phase", ["headers", "body"])
    def test_local_http_deadline_stops_stalled_headers_and_continuous_slow_drip(self, phase):
        with slow_server(phase) as (url, seen):
            started = monotonic()
            with pytest.raises(CapabilityError) as error:
                transport.post(url, headers={}, json={"input": "bounded"}, timeout=0.25)
            assert error.value.code == "provider_failed"
            assert monotonic() - started < 1.5
            assert seen == [("identity", b'{"input":"bounded"}')]
