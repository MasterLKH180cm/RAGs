"""CRAG acquisition boundaries, including real TLS/SNI with a local fixture."""
import asyncio
import hashlib
import json
import socket
import ssl
import subprocess
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import httpx
import pytest

from rags.registry import CapabilityError
from rags.research import crag_web
from rags.research_protocol import ResearchRequest, validate_external_sources

HOST = "approved.example.com"
PUBLIC_IP = "93.184.215.14"
HTML = b'<html><head><title>Public evidence</title><script>private instruction</script></head><body><h1>OrionLab</h1><p>The river sensor detects pollution.</p><style>secret</style><noscript><p>hidden</p></noscript></body></html>'


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("RAGS_CRAG_WEB_HOSTS_JSON", json.dumps([HOST]))
    monkeypatch.setenv("RAGS_CRAG_SEARCH_KEY", "synthetic-search-service-key")
    monkeypatch.delenv("RAGS_CRAG_SEARCH_KEY_FILE", raising=False)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [
        (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (PUBLIC_IP, 443)),
    ])
    return SimpleNamespace(complete=lambda prompt, **kwargs: '{"query":"river pollution sensor"}')


def mock_client(monkeypatch, handler):
    original = httpx.AsyncClient

    def factory(**kwargs):
        assert kwargs["trust_env"] is False
        assert kwargs["follow_redirects"] is False
        return original(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(crag_web.httpx, "AsyncClient", factory)


def search_response(request, urls=None):
    return httpx.Response(200, json={"organic": [{"link": url} for url in (urls if urls is not None else [f"https://{HOST}/evidence"])]}, request=request)


class TestCRAGWebUnits:
    pytestmark = pytest.mark.unit

    @pytest.mark.parametrize("url", [
        f"http://{HOST}/", f"https://{HOST}:444/", f"https://{HOST}./", f"https://{HOST.upper()}/",
        f"https://user@{HOST}/", f"https://{HOST}/#fragment", "https://127.0.0.1/", "https://169.254.169.254/",
        f"https://{HOST}.evil.example/", f"https://{HOST}\\@evil.example/", f"https://{HOST}/\nheader", "file:///etc/passwd",
    ])
    def test_unapproved_or_ambiguous_urls_fail_closed(self, url):
        with pytest.raises(CapabilityError) as error:
            crag_web.validate_url(url, [HOST])
        assert error.value.code == "external_source_forbidden"

    def test_approved_url_is_canonical_without_losing_path_or_query(self):
        assert crag_web.validate_url(f"https://{HOST}:443?a=1&b=%2F", [HOST]) == f"https://{HOST}/?a=1&b=%2F"

    @pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.1", "192.168.1.1", "169.254.169.254", "100.64.0.1", "0.0.0.0", "224.0.0.1", "::1", "fe80::1", "fc00::1", "::ffff:127.0.0.1", "2002:7f00:1::"])
    def test_mixed_public_and_private_dns_answers_are_rejected(self, configured, monkeypatch, address):
        family = socket.AF_INET6 if ":" in address else socket.AF_INET
        monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (PUBLIC_IP, 443)),
            (family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, 443)),
        ])
        with pytest.raises(CapabilityError) as error:
            asyncio.run(crag_web.public_addresses(HOST))
        assert error.value.code == "external_source_forbidden"

    @pytest.mark.parametrize("configured_hosts,requested", [
        ([], [HOST]), ([HOST], ["other.example.com"]), ([HOST], []), (["*.example.com"], [HOST]),
    ])
    def test_authorization_precedes_credentials_model_and_dns(self, monkeypatch, configured_hosts, requested):
        monkeypatch.setenv("RAGS_CRAG_WEB_HOSTS_JSON", json.dumps(configured_hosts))
        monkeypatch.delenv("RAGS_CRAG_SEARCH_KEY", raising=False)
        model = SimpleNamespace(complete=lambda *args, **kwargs: pytest.fail("Unauthorized model call"))
        monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: pytest.fail("Unauthorized DNS call"))
        with pytest.raises(CapabilityError) as error:
            crag_web.acquire("question", "tenant-a", requested, model)
        assert error.value.code == "external_source_forbidden"

    @pytest.mark.parametrize("kind", ["missing", "ambiguous", "oversize", "newline", "fifo", "symlink"])
    def test_search_credential_is_bounded_and_rejected_before_model(self, configured, monkeypatch, tmp_path, kind):
        monkeypatch.delenv("RAGS_CRAG_SEARCH_KEY", raising=False)
        path = tmp_path / "service-key"
        if kind == "ambiguous":
            path.write_text("key")
            monkeypatch.setenv("RAGS_CRAG_SEARCH_KEY", "another-key")
            monkeypatch.setenv("RAGS_CRAG_SEARCH_KEY_FILE", str(path))
        elif kind == "oversize":
            path.write_text("x" * 4097)
            monkeypatch.setenv("RAGS_CRAG_SEARCH_KEY_FILE", str(path))
        elif kind == "newline":
            monkeypatch.setenv("RAGS_CRAG_SEARCH_KEY", "private\nheader")
        elif kind == "fifo":
            import os
            os.mkfifo(path)
            monkeypatch.setenv("RAGS_CRAG_SEARCH_KEY_FILE", str(path))
        elif kind == "symlink":
            target = tmp_path / "target"
            target.write_text("key")
            path.symlink_to(target)
            monkeypatch.setenv("RAGS_CRAG_SEARCH_KEY_FILE", str(path))
        configured.complete = lambda *args, **kwargs: pytest.fail("Model called without credentials")
        with pytest.raises(CapabilityError) as error:
            crag_web.acquire("question", "tenant-a", [HOST], configured)
        assert error.value.code == "requires_search_credentials"
        assert "private" not in str(error.value)

    def test_search_key_file_and_async_facade_preserve_provenance(self, configured, monkeypatch, tmp_path):
        path = tmp_path / "service-key"
        path.write_text("synthetic-search-service-key\n")
        monkeypatch.delenv("RAGS_CRAG_SEARCH_KEY")
        monkeypatch.setenv("RAGS_CRAG_SEARCH_KEY_FILE", str(path))
        requests = []

        def handler(request):
            requests.append(request)
            assert request.url.host == PUBLIC_IP
            assert request.extensions["sni_hostname"] == request.headers["host"]
            if request.method == "POST":
                assert request.headers["host"] == "google.serper.dev"
                assert request.headers["x-api-key"] == "synthetic-search-service-key"
                assert json.loads(request.content)["num"] == 5
                assert "site:" + HOST in json.loads(request.content)["q"]
                response = search_response(request)
                response.headers["Set-Cookie"] = "credential=must-not-leak"
                return response
            assert request.headers["host"] == HOST
            assert not any(header in request.headers for header in ("authorization", "x-api-key", "cookie"))
            return httpx.Response(200, content=HTML, headers={"Content-Type": "text/html; charset=UTF-8"}, request=request)

        mock_client(monkeypatch, handler)

        async def inside_loop():
            return crag_web.acquire("question", "tenant-a", [HOST], configured)

        result = asyncio.run(inside_loop())
        assert len(result) == 1 and len(requests) == 2
        assert result[0]["text"] == "OrionLab The river sensor detects pollution."
        assert result[0]["title"] == "Public evidence"
        assert result[0]["content_sha256"] == hashlib.sha256(result[0]["text"].encode()).hexdigest()
        request = ResearchRequest(protocol="authorized-evidence-v2", engine="crag", namespace="tenant-a", question="question", top_k=1, passages=[], external_hosts=[HOST])
        assert validate_external_sources(result, request) == result
        request.namespace = "tenant-b"
        with pytest.raises(ValueError, match="identity"):
            validate_external_sources(result, request)

    @pytest.mark.parametrize("output", ['{"query":"a","query":"b"}', '{"query":5}', '{"query":""}', '{"query":"a","url":"https://evil.example"}', 'not-json'])
    def test_rewrite_must_be_strict_bounded_json(self, configured, monkeypatch, output):
        configured.complete = lambda *args, **kwargs: output
        mock_client(monkeypatch, lambda request: pytest.fail("Malformed rewrite performed HTTP"))
        with pytest.raises(CapabilityError) as error:
            crag_web.acquire("question", "tenant-a", [HOST], configured)
        assert error.value.code == "invalid_model_output"

    def test_provider_failure_remains_a_sanitized_execution_error(self, configured):
        def fail(*args, **kwargs):
            raise RuntimeError("private upstream credential")
        configured.complete = fail
        with pytest.raises(CapabilityError) as error:
            crag_web.acquire("question", "tenant-a", [HOST], configured)
        assert error.value.code == "external_source_failed"
        assert "private" not in str(error.value)

    @pytest.mark.parametrize("question,namespace", [("question", "\ud800"), ("question\ud800", "tenant-a"), ("question", " ")])
    def test_invalid_input_is_rejected_before_provider_calls(self, configured, question, namespace):
        configured.complete = lambda *args, **kwargs: pytest.fail("Invalid input reached provider")
        with pytest.raises(CapabilityError) as error:
            crag_web.acquire(question, namespace, [HOST], configured)
        assert error.value.code == "invalid_research_input"

    @pytest.mark.parametrize("kind", ["redirect", "compression", "declared_size", "stream_size", "bad_json", "foreign_host", "too_many", "wrong_media"])
    def test_search_and_fetch_reject_unsafe_response_envelopes(self, configured, monkeypatch, kind):
        calls = []

        def handler(request):
            calls.append(request)
            if kind == "redirect":
                return httpx.Response(302, headers={"Location": "https://127.0.0.1/"}, request=request)
            if kind == "compression":
                return httpx.Response(200, content=b"plain", headers={"Content-Encoding": "br"}, request=request)
            if kind == "declared_size":
                return httpx.Response(200, content=b"{}", headers={"Content-Length": str(crag_web.MAX_RESPONSE_BYTES + 1)}, request=request)
            if kind == "stream_size":
                return httpx.Response(200, content=b" " * (crag_web.MAX_RESPONSE_BYTES + 1), request=request)
            if kind == "bad_json":
                return httpx.Response(200, content=b'{"organic":[],"organic":[]}', headers={"Content-Type": "application/json"}, request=request)
            if kind == "foreign_host":
                return search_response(request, ["https://evil.example/"])
            if kind == "too_many":
                return search_response(request, [f"https://{HOST}/{i}" for i in range(6)])
            return httpx.Response(200, json={"organic": []}, headers={"Content-Type": "text/html"}, request=request)

        mock_client(monkeypatch, handler)
        with pytest.raises(CapabilityError) as error:
            crag_web.acquire("question", "tenant-a", [HOST], configured)
        assert error.value.code in {"external_source_failed", "external_source_forbidden"}
        assert len(calls) == 1

    @pytest.mark.parametrize("operation", ["rewrite", "dns", "headers", "body"])
    def test_one_wall_deadline_bounds_all_phases(self, configured, monkeypatch, operation):
        monkeypatch.setattr(crag_web, "MAX_SECONDS", .05)
        if operation == "rewrite":
            configured.complete = lambda *args, **kwargs: (time.sleep(.3), '{"query":"x"}')[1]
        elif operation == "dns":
            monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: (time.sleep(.3), [])[1])

        class SlowStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'{"organic":'
                await asyncio.sleep(.3)
                yield b'[]}'

        async def handler(request):
            if operation == "headers":
                await asyncio.sleep(.3)
            return httpx.Response(200, stream=SlowStream(), headers={"Content-Type": "application/json"}, request=request)

        mock_client(monkeypatch, handler)
        start = time.monotonic()
        with pytest.raises(CapabilityError) as error:
            crag_web.acquire("question", "tenant-a", [HOST], configured)
        elapsed = time.monotonic() - start
        assert error.value.code == "external_source_timeout"
        assert elapsed < .25

    def test_html_extraction_ignores_scripts_and_bounds_unicode_bytes(self):
        title, text = crag_web.extract_text(HTML, "text/html", 16_000)
        assert title == "Public evidence"
        assert "private" not in text and "hidden" not in text and "secret" not in text
        _, text = crag_web.extract_text(("<p>" + "繁體中文👩🏽‍💻" * 100 + "</p>").encode(), "text/html", 61)
        assert len(text.encode()) <= 61
        assert "�" not in text

    def test_streamed_source_byte_limit_closes_the_response(self, configured, monkeypatch):
        closed = []

        class OversizedStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                for _ in range(33):
                    yield b"x" * (16 * 1024)

            async def aclose(self):
                closed.append(True)

        def handler(request):
            if request.method == "POST":
                return search_response(request)
            return httpx.Response(200, stream=OversizedStream(), headers={"Content-Type": "text/plain"}, request=request)

        mock_client(monkeypatch, handler)
        with pytest.raises(CapabilityError) as error:
            crag_web.acquire("question", "tenant-a", [HOST], configured)
        assert error.value.code == "external_source_failed"
        assert closed == [True]

    def test_existing_execution_deadline_also_bounds_acquisition(self, configured, monkeypatch):
        async def handler(request):
            await asyncio.sleep(.2)
            return search_response(request)
        mock_client(monkeypatch, handler)
        token = crag_web.execution_deadline.set(time.monotonic() + .03)
        try:
            with pytest.raises(CapabilityError) as error:
                crag_web.acquire("question", "tenant-a", [HOST], configured)
            assert error.value.code == "external_source_timeout"
        finally:
            crag_web.execution_deadline.reset(token)

    @pytest.mark.parametrize("media,raw", [("image/png", b"bytes"), ("text/html;charset=latin1", b"<p>text</p>"), ("text/html", b"<p>\xff</p>"), ("text/html", b"<script>text</script>")])
    def test_unsupported_or_empty_source_is_not_evidence(self, media, raw):
        with pytest.raises(CapabilityError):
            crag_web.extract_text(raw, media, 100)

    def test_all_snapshots_fit_total_text_bytes_and_source_count(self, configured, monkeypatch):
        calls = []

        def handler(request):
            calls.append(request)
            if request.method == "POST":
                return search_response(request, [f"https://{HOST}/{i}" for i in range(5)])
            return httpx.Response(200, content=("字" * 20_000).encode(), headers={"Content-Type": "text/plain"}, request=request)

        mock_client(monkeypatch, handler)
        result = crag_web.acquire("question", "tenant-a", [HOST], configured)
        assert 1 <= len(result) <= 5
        assert sum(len(item["text"].encode()) for item in result) <= 48_000
        assert all(len(item["text"]) <= 16_000 for item in result)


@contextmanager
def tls_server(tmp_path):
    cert, key = tmp_path / "test-ca.pem", tmp_path / "test-server.key"
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1", "-keyout", str(key),
        "-out", str(cert), "-subj", "/CN=" + HOST,
        "-addext", "subjectAltName=DNS:" + HOST + ",DNS:google.serper.dev",
        "-addext", "basicConstraints=critical,CA:TRUE",
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    names, requests = [], []
    context.set_servername_callback(lambda connection, hostname, context: names.append(hostname))

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append((self.command, dict(self.headers)))
            self.rfile.read(int(self.headers["Content-Length"]))
            self.respond(json.dumps({"organic": [{"link": f"https://{HOST}/evidence"}]}).encode(), "application/json")

        def do_GET(self):
            requests.append((self.command, dict(self.headers)))
            self.respond(HTML, "text/html")

        def respond(self, content, media):
            self.send_response(200)
            self.send_header("Content-Type", media)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield server, cert, names, requests
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


@pytest.mark.integration
def test_real_tls_uses_original_sni_verified_certificate_and_pinned_destination(configured, monkeypatch, tmp_path):
    original_client = httpx.AsyncClient
    with tls_server(tmp_path) as (server, cert, names, requests):
        observed = []

        class LocalWireTransport(httpx.AsyncBaseTransport):
            def __init__(self, trust_fixture):
                verify = ssl.create_default_context(cafile=str(cert)) if trust_fixture else True
                self.inner = httpx.AsyncHTTPTransport(verify=verify, trust_env=False)

            async def handle_async_request(self, request):
                observed.append((request.url.host, request.headers["Host"], request.extensions["sni_hostname"]))
                assert request.url.host == PUBLIC_IP
                # This test-only transport maps the already-pinned public TCP
                # destination to its local server; TLS/SNI remain untouched.
                local = httpx.Request(request.method, request.url.copy_with(host="127.0.0.1", port=server.server_port),
                                      headers=request.headers, content=request.content, extensions=request.extensions)
                return await self.inner.handle_async_request(local)

            async def aclose(self):
                await self.inner.aclose()

        def factory(trust_fixture):
            return lambda **kwargs: original_client(transport=LocalWireTransport(trust_fixture), **kwargs)

        monkeypatch.setattr(crag_web.httpx, "AsyncClient", factory(False))
        with pytest.raises(CapabilityError) as untrusted:
            crag_web.acquire("question", "tenant-a", [HOST], configured)
        assert untrusted.value.code == "external_source_failed"
        assert not requests  # No HTTP body/credential crossed an unverified TLS connection.

        monkeypatch.setattr(crag_web.httpx, "AsyncClient", factory(True))
        result = crag_web.acquire("question", "tenant-a", [HOST], configured)
        assert result[0]["text"] == "OrionLab The river sensor detects pollution."
        assert names[-2:] == ["google.serper.dev", HOST]
        assert observed[-2:] == [(PUBLIC_IP, "google.serper.dev", "google.serper.dev"), (PUBLIC_IP, HOST, HOST)]
        assert requests[0][1]["X-API-KEY"] == "synthetic-search-service-key"
        assert "X-API-KEY" not in requests[1][1] and "Cookie" not in requests[1][1]
