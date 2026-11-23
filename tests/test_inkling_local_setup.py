"""Resource and artifact failure boundaries for the pinned local preparation tool."""
import asyncio
import hashlib
import importlib.util
import json
import subprocess
import time
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def setup_tool():
    source = Path(__file__).resolve().parents[1] / "scripts/inkling_local.py"
    spec = importlib.util.spec_from_file_location("inkling_local_setup", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resources(tool, **updates):
    values = {"system": "Linux", "architecture": "x86_64", "logical_cpus": 8,
              "memory_total_bytes": 32 * tool.GIB, "memory_available_bytes": 28 * tool.GIB,
              "disk_free_bytes": 64 * tool.GIB, "accelerator_policy": "cpu_only"}
    return values | updates


def test_insufficient_memory_prevents_any_download(setup_tool, monkeypatch, tmp_path):
    tool = setup_tool
    monkeypatch.setattr(tool, "host_resources", lambda _: resources(tool, memory_available_bytes=6 * tool.GIB))
    monkeypatch.setattr(tool.httpx, "AsyncClient", lambda *_a, **_k: pytest.fail("Must not access network"))
    with pytest.raises(ValueError, match="insufficient_available_memory"):
        tool.download_model(tmp_path / "cache")
    assert not (tmp_path / "cache").exists()


def test_blocked_report_is_not_inference_evidence(setup_tool, monkeypatch, tmp_path):
    tool = setup_tool
    monkeypatch.setattr(tool, "host_resources", lambda _: resources(tool, memory_available_bytes=6 * tool.GIB))
    monkeypatch.setattr(tool, "inspect_runtime", lambda _: {"status": "matched"})
    output = tmp_path / "readiness.json"
    assert tool.main(["download", "--cache-dir", str(tmp_path / "cache"), "--output", str(output)]) == 2
    report = json.loads(output.read_text())
    assert report["evidence_kind"] == "hardware_preflight"
    assert report["provider_requests"] == 0 and report["model_inference_executed"] is False
    assert report["block_reasons"] == ["insufficient_available_memory", "model_weights_missing"]
    with pytest.raises(ValueError, match="blocked"):
        tool.deployment_manifest(report, 4)


@pytest.mark.parametrize("changes,reason", [
    ({"system": "Darwin"}, "unsupported_reference_platform"),
    ({"memory_available_bytes": None}, "memory_capacity_unknown"),
    ({"disk_free_bytes": 0}, "insufficient_disk_space"),
])
def test_unknown_or_insufficient_resources_are_explicit(setup_tool, changes, reason):
    assert reason in setup_tool.resource_blockers(resources(setup_tool, **changes), weights_present=False)


def test_wrong_runtime_revision_cannot_be_marked_ready(setup_tool, monkeypatch, tmp_path):
    executable = tmp_path / "server"
    executable.write_bytes(b"fixture executable")
    monkeypatch.setattr(setup_tool.shutil, "which", lambda _: str(executable))
    monkeypatch.setattr(setup_tool.subprocess, "run", lambda *_a, **_k: subprocess.CompletedProcess(
        [], 0, "version: 1.0.0-dev (build 12600, commit 29b52d398)", ""))
    assert setup_tool.inspect_runtime("server")["status"] == "version_mismatch"


def test_corrupted_or_symlinked_weights_are_rejected(setup_tool, monkeypatch, tmp_path):
    expected = b"verified model fixture"
    monkeypatch.setattr(setup_tool, "MODEL_BYTES", len(expected))
    monkeypatch.setattr(setup_tool, "MODEL_SHA256", hashlib.sha256(expected).hexdigest())
    path = tmp_path / setup_tool.ARTIFACT
    path.write_bytes(b"x" * len(expected))
    assert not setup_tool.verify_model(path)
    with pytest.raises(ValueError, match="Existing model"):
        setup_tool.download_model(tmp_path)
    path.write_bytes(expected)
    assert setup_tool.verify_model(path)
    link = tmp_path / "linked.gguf"
    link.symlink_to(path)
    assert not setup_tool.verify_model(link)


def test_launch_uses_fixed_cpu_loopback_contract(setup_tool, tmp_path):
    command = setup_tool.server_command("llama-server", tmp_path, 8080, 4)
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--alias") + 1] == setup_tool.MODEL
    assert command[command.index("--parallel") + 1] == "1"
    assert command[command.index("--ctx-size") + 1] == "4096"
    assert command[command.index("--fit") + 1] == "off"
    assert command[command.index("--n-gpu-layers") + 1] == "0"
    assert command[command.index("--reasoning-format") + 1] == "deepseek"
    assert "--jinja" in command and "--no-webui" in command
    assert "--api-key" not in command
    with pytest.raises(ValueError):
        setup_tool.server_command("llama-server", tmp_path, 8080, 0)


@pytest.mark.parametrize("destination", [
    "http://huggingface.co/model", "https://127.0.0.1/model", "https://huggingface.co.evil.invalid/model",
    "https://user:password@huggingface.co/model", "https://hf.co:8443/model",
])
def test_download_rejects_unsafe_redirect(setup_tool, destination):
    with pytest.raises(ValueError, match="HTTPS"):
        setup_tool.download_url("https://huggingface.co/example", destination)


@pytest.mark.parametrize("body", [b"verified fixture", b"corrupt! fixture", b"too long fixture body"])
def test_download_verifies_before_publishing_and_removes_partial_files(
    setup_tool, monkeypatch, tmp_path, body,
):
    tool = setup_tool
    expected = b"verified fixture"
    monkeypatch.setattr(tool, "MODEL_BYTES", len(expected))
    monkeypatch.setattr(tool, "MODEL_SHA256", hashlib.sha256(expected).hexdigest())
    monkeypatch.setattr(tool, "host_resources", lambda _: resources(tool))

    def handle_request(request):
        assert str(request.url).endswith(f"/{tool.REVISION}/{tool.ARTIFACT}")
        assert "authorization" not in request.headers
        return httpx.Response(200, stream=httpx.ByteStream(body))

    real_client = httpx.AsyncClient
    monkeypatch.setattr(tool.httpx, "AsyncClient", lambda **kwargs: real_client(
        transport=httpx.MockTransport(handle_request), **kwargs))
    if body == expected:
        tool.download_model(tmp_path)
        assert (tmp_path / tool.ARTIFACT).read_bytes() == expected
    else:
        with pytest.raises(ValueError):
            tool.download_model(tmp_path)
        assert not (tmp_path / tool.ARTIFACT).exists()
    assert not list(tmp_path.glob(".inkling-download-*"))


def test_continuous_slow_download_hits_elapsed_deadline_and_cleans_partial_file(
    setup_tool, monkeypatch, tmp_path,
):
    tool = setup_tool
    closed = []

    class SlowBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            while True:
                await asyncio.sleep(0.005)
                yield b"x"

        async def aclose(self):
            closed.append(True)

    def handle_request(_request):
        return httpx.Response(200, stream=SlowBody())

    monkeypatch.setattr(tool, "host_resources", lambda _: resources(tool))
    real_client = httpx.AsyncClient
    monkeypatch.setattr(tool.httpx, "AsyncClient", lambda **kwargs: real_client(
        transport=httpx.MockTransport(handle_request), **kwargs))
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        tool.download_model(tmp_path, timeout_seconds=0.04)
    assert time.monotonic() - started < 1
    assert closed and not (tmp_path / tool.ARTIFACT).exists()
    assert not list(tmp_path.glob(".inkling-download-*"))


def test_serving_manifest_matches_benchmark_contract_and_clears_runtime_overrides(
    setup_tool, monkeypatch, tmp_path,
):
    from rags.model_benchmark import DeploymentManifest

    tool = setup_tool
    ready = {"status": "ready_for_server_start", "block_reasons": [], "host": resources(tool)}
    monkeypatch.setattr(tool, "readiness", lambda *_: ready)
    monkeypatch.setenv("LLAMA_ARG_CHAT_TEMPLATE", "an-unrecorded-template")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(tool.subprocess, "run", run)
    manifest_path = tmp_path / "manifest.json"
    assert tool.main(["serve", "--server", "llama-server", "--threads", "4",
                      "--cache-dir", str(tmp_path), "--manifest", str(manifest_path),
                      "--output", str(tmp_path / "readiness.json")]) == 0
    manifest = DeploymentManifest.model_validate_json(manifest_path.read_text())
    assert manifest.server_threads == 4 and manifest.source == "operator_supplied"
    assert manifest.model_sha256 == tool.MODEL_SHA256
    assert "LLAMA_ARG_CHAT_TEMPLATE" not in calls[0][1]["env"]
    assert calls[0][1].get("shell", False) is False
