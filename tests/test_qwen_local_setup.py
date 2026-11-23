"""Offline checks for explicit Qwen deployment identity and execution bounds."""
from dataclasses import replace
import hashlib
import json
import subprocess

import httpx
import pytest

from scripts import local_model as shared
from scripts import qwen_local as tool

pytestmark = pytest.mark.unit


def resources(**updates):
    return {"system": "Linux", "architecture": "x86_64", "logical_cpus": 8,
            "memory_total_bytes": 10 * shared.GIB, "memory_available_bytes": 7 * shared.GIB,
            "disk_free_bytes": 10 * shared.GIB, "accelerator_policy": "cpu_only"} | updates


def test_qwen_identity_is_explicit_and_does_not_mutate_inkling():
    from scripts import inkling_local

    assert tool.ARTIFACT.repository == "Qwen/Qwen3-1.7B-GGUF"
    assert tool.ARTIFACT.revision == "90862c4b9d2787eaed51d12237eafdfe7c5f6077"
    assert tool.ARTIFACT.filename == "Qwen3-1.7B-Q8_0.gguf"
    assert tool.ARTIFACT.size_bytes == 1834426016
    assert tool.ARTIFACT.sha256 == "061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a"
    assert shared.resource_blockers(resources(), tool.ARTIFACT, weights_present=False) == []
    assert inkling_local.resource_blockers(resources(), weights_present=False) == [
        "insufficient_available_memory", "insufficient_disk_space",
    ]


def test_insufficient_memory_stops_qwen_before_network(monkeypatch, tmp_path):
    monkeypatch.setattr(shared, "host_resources", lambda _: resources(memory_available_bytes=2 * shared.GIB))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_: pytest.fail("No download below memory reserve"))
    with pytest.raises(ValueError, match="insufficient_available_memory"):
        tool.download_model(tmp_path / "cache")
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("body", [b"verified fixture", b"corrupted", b"oversized fixture data"])
def test_qwen_download_verifies_its_own_artifact_before_publication(monkeypatch, tmp_path, body):
    expected = b"verified fixture"
    artifact = replace(tool.ARTIFACT, size_bytes=len(expected), sha256=hashlib.sha256(expected).hexdigest())
    monkeypatch.setattr(tool, "ARTIFACT", artifact)
    monkeypatch.setattr(shared, "host_resources", lambda _: resources())
    real_client = httpx.AsyncClient
    calls = []

    def respond(request):
        calls.append(request)
        assert str(request.url) == f"https://huggingface.co/{artifact.repository}/resolve/{artifact.revision}/{artifact.filename}"
        assert "authorization" not in request.headers
        return httpx.Response(200, stream=httpx.ByteStream(body))

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: real_client(
        transport=httpx.MockTransport(respond), **kwargs))
    if body == expected:
        tool.download_model(tmp_path)
        tool.download_model(tmp_path)
        assert len(calls) == 1
        assert (tmp_path / artifact.filename).read_bytes() == expected
    else:
        with pytest.raises(ValueError):
            tool.download_model(tmp_path)
        assert not (tmp_path / artifact.filename).exists()
    assert not list(tmp_path.glob(".qwen-download-*"))


def test_qwen_symlink_is_not_verified_or_overwritten(monkeypatch, tmp_path):
    data = b"fixture"
    artifact = replace(tool.ARTIFACT, size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(tool, "ARTIFACT", artifact)
    target = tmp_path / "blob"
    target.write_bytes(data)
    link = tmp_path / artifact.filename
    link.symlink_to(target)
    with pytest.raises(ValueError, match="Existing model file"):
        tool.download_model(tmp_path)
    assert link.is_symlink() and target.read_bytes() == data


@pytest.mark.parametrize("threads", [0, 9, True, 1.5])
def test_qwen_thread_bounds_reject_invalid_values(tmp_path, threads):
    with pytest.raises(ValueError, match="thread"):
        tool.server_command("llama-server", tmp_path, 8080, threads)


def test_qwen_serve_records_fixed_cache_sampling_and_model_contract(monkeypatch, tmp_path):
    from rags.model_benchmark import DeploymentManifest

    ready = {"status": "ready_for_server_start", "block_reasons": [], "host": resources(),
             "runtime": {"status": "matched", "executable_sha256": "a" * 64}}
    monkeypatch.setattr(tool, "readiness", lambda *_: ready)
    monkeypatch.setenv("LLAMA_ARG_REASONING", "on")
    monkeypatch.setenv("LLAMA_ARG_CACHE_RAM", "8192")
    monkeypatch.setenv("LLAMA_API_KEY", "private-unselected-server-key")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    deployment_path = tmp_path / "deployment.json"
    execution_path = tmp_path / "execution.json"
    assert tool.main(["serve", "--server", "llama-server", "--cache-dir", str(tmp_path), "--threads", "4",
                      "--output", str(tmp_path / "ready.json"), "--manifest", str(deployment_path),
                      "--execution-contract", str(execution_path)]) == 0
    manifest = DeploymentManifest.model_validate_json(deployment_path.read_text())
    assert manifest.model_repository == tool.ARTIFACT.repository and manifest.quantization == "Q8_0"
    assert manifest.server_threads == 4 and manifest.server_parallel_slots == 1
    execution = json.loads(execution_path.read_text())
    assert execution["deployment_sha256"] == hashlib.sha256(json.dumps(
        manifest.model_dump(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert execution["runtime_executable_sha256"] == "a" * 64 and not execution["model_inference_executed"]
    command, kwargs = calls[0]
    expected = {"--host": "127.0.0.1", "--alias": tool.MODEL, "--model": str(tmp_path / tool.ARTIFACT.filename),
                "--ctx-size": "4096", "--parallel": "1", "--threads": "4", "--fit": "off",
                "--n-gpu-layers": "0", "--reasoning": "off", "--cache-ram": "0", "--temp": "0", "--seed": "42"}
    for flag, value in expected.items():
        assert command[command.index(flag) + 1] == value
    assert {"--no-op-offload", "--jinja", "--no-cache-prompt", "--no-webui"} <= set(command)
    assert not any(name.startswith("LLAMA_ARG_") for name in kwargs["env"])
    assert "LLAMA_API_KEY" not in kwargs["env"]
    assert not kwargs.get("shell", False)
    assert execution["parameters"]["reasoning"] == "off" and not execution["parameters"]["prompt_cache"]
    assert str(tmp_path) not in execution_path.read_text()
    assert "private-unselected-server-key" not in execution_path.read_text()


def test_qwen_preflight_never_claims_inference(monkeypatch, tmp_path):
    monkeypatch.setattr(shared, "host_resources", lambda _: resources())
    monkeypatch.setattr(shared, "inspect_runtime", lambda _: {"status": "matched"})
    report = tool.readiness(tmp_path, "llama-server")
    assert report["status"] == "blocked" and report["block_reasons"] == ["model_weights_missing"]
    assert report["evidence_kind"] == "hardware_preflight" and report["provider_requests"] == 0
    assert report["model_inference_executed"] is False
    with pytest.raises(ValueError, match="blocked"):
        tool.deployment_manifest(report, 4)


def test_qwen_cli_reports_download_timeout_without_private_details(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(shared, "host_resources", lambda _: resources())
    def timeout(*_):
        raise TimeoutError("private artifact path")
    monkeypatch.setattr(tool, "download_model", timeout)
    assert tool.main(["download", "--cache-dir", str(tmp_path)]) == 1
    output = capsys.readouterr().out
    assert json.loads(output) == {"status": "failed", "error_type": "TimeoutError"}
    assert "private" not in output
