"""Artifact download bounds and atomic publication, without large downloads."""
import asyncio
import hashlib
from time import monotonic

import httpx
import pytest

from scripts import crag_local

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("location", ["http://drive.google.com/file", "https://evil.example/file",
                                      "https://user@drive.google.com/file", "https://drive.google.com:8443/file",
                                      "https://drive.google.com/file#fragment"])
def test_artifact_redirect_cannot_change_trusted_download_hosts(location):
    with pytest.raises(ValueError):
        crag_local.approved_url("https://drive.usercontent.google.com/download?id=test", location)


@pytest.mark.parametrize("mutation", ["oversize", "hash", "compression", "redirect"])
def test_failed_download_never_publishes_or_leaves_partial_file(tmp_path, monkeypatch, mutation):
    content = b"valid artifact"
    monkeypatch.setitem(crag_local.ARTIFACTS, "config.json", (len(content), hashlib.sha256(content).hexdigest()))
    def serve(request):
        if mutation == "redirect":
            return httpx.Response(302, headers={"location": "https://evil.example/file"})
        if mutation == "compression":
            return httpx.Response(200, headers={"content-encoding": "br"})
        body = content + b"extra" if mutation == "oversize" else b"X" * len(content)
        return httpx.Response(200, stream=httpx.ByteStream(body))
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as client:
            await crag_local.download_one(tmp_path, "config.json", deadline=monotonic() + 1, client=client)
    with pytest.raises(ValueError):
        asyncio.run(run())
    assert list(tmp_path.iterdir()) == []


def test_verified_download_is_atomic_and_existing_artifacts_are_not_overwritten(tmp_path, monkeypatch):
    content = b"valid artifact"
    monkeypatch.setitem(crag_local.ARTIFACTS, "config.json", (len(content), hashlib.sha256(content).hexdigest()))
    count = 0
    def serve(request):
        nonlocal count
        count += 1
        return httpx.Response(200, stream=httpx.ByteStream(content))
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as client:
            await crag_local.download_one(tmp_path, "config.json", deadline=monotonic() + 1, client=client)
    asyncio.run(run())
    asyncio.run(run())
    assert count == 1
    assert (tmp_path / "config.json").read_bytes() == content
    (tmp_path / "config.json").write_bytes(b"other artifact")
    with pytest.raises(ValueError):
        asyncio.run(run())
    assert (tmp_path / "config.json").read_bytes() == b"other artifact"


def test_slow_stream_obeys_total_deadline_and_removes_partial_file(tmp_path, monkeypatch):
    content = b"abcdef"
    monkeypatch.setitem(crag_local.ARTIFACTS, "config.json", (len(content), hashlib.sha256(content).hexdigest()))
    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for value in content:
                await asyncio.sleep(0.03)
                yield bytes([value])
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=Stream()))) as client:
            await crag_local.download_one(tmp_path, "config.json", deadline=monotonic() + 0.07, client=client)
    with pytest.raises(TimeoutError):
        asyncio.run(run())
    assert list(tmp_path.iterdir()) == []
