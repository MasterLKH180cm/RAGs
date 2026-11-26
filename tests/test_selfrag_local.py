"""Provisioning invariants use tiny files, not synthetic model-success claims."""
import asyncio
import hashlib
import io
import json
import tarfile

import httpx
import pytest

from scripts import selfrag_local as tool

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("body", [b"official fixture", b"corrupt", b"official fixture plus unwanted bytes"])
def test_checkpoint_publication_requires_full_official_size_and_digest(monkeypatch, tmp_path, body):
    expected = b"official fixture"
    real_client = httpx.AsyncClient
    requests = []

    def respond(request):
        requests.append(request)
        assert request.url.host == "huggingface.co"
        assert tool.CHECKPOINT_REVISION in request.url.path
        assert "authorization" not in request.headers
        return httpx.Response(200, stream=httpx.ByteStream(body))

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs))
    digest = hashlib.sha256(expected).hexdigest()
    if body == expected:
        asyncio.run(tool.download_checkpoint(tmp_path, "test.bin", len(expected), digest))
        asyncio.run(tool.download_checkpoint(tmp_path, "test.bin", len(expected), digest))
        assert len(requests) == 1
        assert (tmp_path / "test.bin").read_bytes() == expected
    else:
        with pytest.raises(ValueError):
            asyncio.run(tool.download_checkpoint(tmp_path, "test.bin", len(expected), digest))
        assert not (tmp_path / "test.bin").exists()
    assert not list(tmp_path.glob(".selfrag-download-*"))


def test_checkpoint_symlink_is_never_followed_or_replaced(tmp_path):
    target = tmp_path / "owned"
    target.write_bytes(b"owned data")
    path = tmp_path / "weights.bin"
    path.symlink_to(target)
    with pytest.raises(ValueError, match="Existing"):
        asyncio.run(tool.download_checkpoint(tmp_path, "weights.bin", 10, "0" * 64))
    assert path.is_symlink() and target.read_bytes() == b"owned data"


def test_checkpoint_publication_does_not_overwrite_a_concurrent_target(monkeypatch, tmp_path):
    target = tmp_path / "weights.bin"

    async def stream(output, *_args, **_kwargs):
        output.write(b"verified source")
        target.write_bytes(b"concurrent owner")

    monkeypatch.setattr(tool, "stream_model", stream)
    with pytest.raises(FileExistsError):
        asyncio.run(tool.download_checkpoint(tmp_path, target.name, 15,
                                             hashlib.sha256(b"verified source").hexdigest()))
    assert target.read_bytes() == b"concurrent owner"
    assert not list(tmp_path.glob(".selfrag-download-*"))


def test_converter_publication_does_not_overwrite_a_concurrent_target(monkeypatch, tmp_path):
    body = b"verified archive fixture"
    target = tmp_path / "llama-source.tar.gz"
    client = httpx.AsyncClient

    def respond(_request):
        target.write_bytes(b"concurrent owner")
        return httpx.Response(200, stream=httpx.ByteStream(body))

    monkeypatch.setattr(tool, "SOURCE_ARCHIVE_BYTES", len(body))
    monkeypatch.setattr(tool, "SOURCE_ARCHIVE_SHA256", hashlib.sha256(body).hexdigest())
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs:
                        client(transport=httpx.MockTransport(respond), **kwargs))
    with pytest.raises(FileExistsError):
        asyncio.run(tool.download_converter(tmp_path))
    assert target.read_bytes() == b"concurrent owner"
    assert not list(tmp_path.glob(".selfrag-source-*"))


def test_existing_mismatched_final_model_is_preserved_before_preparation(monkeypatch, tmp_path):
    target = tmp_path / tool.MODEL_FILENAME
    target.write_bytes(b"existing operator artifact")
    monkeypatch.setattr(tool, "verify_runtime", lambda _: None)
    monkeypatch.setattr(tool, "verify_conversion_python",
                        lambda _: pytest.fail("A conflicting model must fail before conversion setup"))
    with pytest.raises(ValueError, match="Existing Self-RAG model differs"):
        tool.prepare(tmp_path, tmp_path / "runtime", "unused-python")
    assert target.read_bytes() == b"existing operator artifact"


@pytest.mark.parametrize("name,kind", [("../escape.py", "file"),
                                      ("llama.cpp-wrong/file.py", "file"),
                                      ("llama.cpp-" + tool.RUNTIME_REVISION + "/unsafe.py", "symlink")])
def test_converter_archive_rejects_traversal_wrong_source_and_symlinks(tmp_path, name, kind):
    archive = tmp_path / "archive.tar"
    with tarfile.open(archive, "w") as tar:
        entry = tarfile.TarInfo(name)
        if kind == "symlink":
            entry.type = tarfile.SYMTYPE
            entry.linkname = "/outside"
            tar.addfile(entry)
        else:
            entry.size = 1
            tar.addfile(entry, io.BytesIO(b"x"))
    with pytest.raises(ValueError):
        tool.extract_converter(archive, tmp_path / "extract")
    assert not (tmp_path / "escape.py").exists()


def test_conversion_environment_excludes_provider_credentials_and_dynamic_plugin_paths(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-provider-secret")
    monkeypatch.setenv("RAGS_KEYS_JSON", '{"tenant-secret":"tenant"}')
    monkeypatch.setenv("GGML_BACKEND_PATH", "/untrusted/plugin.so")
    monkeypatch.setenv("PYTHONPATH", "/untrusted/python")
    environment = tool.conversion_environment()
    assert not {"OPENAI_API_KEY", "RAGS_KEYS_JSON", "GGML_BACKEND_PATH", "PYTHONPATH"} & set(environment)
    assert environment["HF_HUB_OFFLINE"] == environment["TRANSFORMERS_OFFLINE"] == "1"


def test_actual_checkpoint_manifest_never_implies_inference(tmp_path):
    manifest = tool.manifest(tmp_path)
    assert manifest["model"]["verified"] is False
    assert manifest["model_inference_executed"] is False
    assert manifest["checkpoint_repository"] == "selfrag/selfrag_llama2_7b"
    assert manifest["reflection_token_ids"]["[Utility:4]"] == 32010
    assert "EOS 2" in manifest["conversion_correction"]


def test_wrong_conversion_package_versions_fail_before_conversion(monkeypatch):
    class Result:
        stdout = json.dumps({**tool.CONVERSION_VERSIONS, "torch": "unverified"})

    monkeypatch.setattr(tool.subprocess, "run", lambda *_args, **_kwargs: Result())
    with pytest.raises(ValueError, match="dependencies differ"):
        tool.verify_conversion_python("operator-python")
