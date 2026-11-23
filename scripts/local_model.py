"""Shared bounded preparation for revision-pinned CPU GGUF deployments."""
import asyncio
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import urllib.parse
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

RUNTIME_REVISION = "e300919cc704ddde3a8a8346a8714f57eeeee71b"
RUNTIME_VERSION = "b12631"
GIB = 1024**3
MEMORY_RESERVE = 2 * GIB
DISK_RESERVE = 256 * 1024**2
CONTEXT_TOKENS = 4096
PARALLEL_SLOTS = 1


@dataclass(frozen=True)
class Artifact:
    repository: str
    revision: str
    filename: str
    size_bytes: int
    sha256: str
    quantization: str


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".model-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _positive_file(path: str) -> int | None:
    try:
        value = int(Path(path).read_text().strip())
        return value if 0 <= value < 2**60 else None
    except (OSError, ValueError):
        return None


def host_resources(cache: Path) -> dict:
    total = available = None
    try:
        entries = dict(re.findall(r"^(MemTotal|MemAvailable):\s+(\d+) kB$",
                                  Path("/proc/meminfo").read_text(), re.MULTILINE))
        total = int(entries["MemTotal"]) * 1024
        available = int(entries["MemAvailable"]) * 1024
    except (OSError, ValueError, KeyError):
        pass
    # A container limit can be lower than the host's /proc memory figures.
    for limit_name, current_name in (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ):
        limit, current = _positive_file(limit_name), _positive_file(current_name)
        if limit is not None and current is not None:
            total = min(total, limit) if total is not None else limit
            remaining = max(0, limit - current)
            available = min(available, remaining) if available is not None else remaining
    parent = cache.resolve()
    while not parent.exists():
        parent = parent.parent
    return {"system": platform.system(), "architecture": platform.machine(),
            "logical_cpus": os.cpu_count(), "memory_total_bytes": total,
            "memory_available_bytes": available, "disk_free_bytes": shutil.disk_usage(parent).free,
            "accelerator_policy": "cpu_only"}


def resource_blockers(resources: dict, artifact: Artifact, *, weights_present: bool) -> list[str]:
    result = []
    if resources["system"] != "Linux" or resources["architecture"].lower() not in {"x86_64", "amd64"}:
        result.append("unsupported_reference_platform")
    available = resources["memory_available_bytes"]
    if available is None:
        result.append("memory_capacity_unknown")
    elif available < artifact.size_bytes + MEMORY_RESERVE:
        result.append("insufficient_available_memory")
    required_disk = DISK_RESERVE + (0 if weights_present else artifact.size_bytes)
    if resources["disk_free_bytes"] < required_disk:
        result.append("insufficient_disk_space")
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(path: Path, artifact: Artifact) -> bool:
    return bool(not path.is_symlink() and path.is_file() and path.stat().st_size == artifact.size_bytes
                and file_sha256(path) == artifact.sha256)


def inspect_runtime(server: str) -> dict:
    executable = shutil.which(server)
    if not executable:
        return {"status": "missing", "expected_version": RUNTIME_VERSION}
    try:
        run = subprocess.run([executable, "--version"], capture_output=True, text=True,
                             timeout=10, check=False)
        output = run.stdout + run.stderr
        matched = bool(run.returncode == 0 and re.search(
            rf"build 12631, commit {RUNTIME_REVISION[:9]}\b", output))
        return {"status": "matched" if matched else "version_mismatch",
                "expected_version": RUNTIME_VERSION,
                "expected_revision": RUNTIME_REVISION,
                "executable_sha256": file_sha256(Path(executable))}
    except (OSError, subprocess.TimeoutExpired, UnicodeError):
        return {"status": "unavailable", "expected_version": RUNTIME_VERSION}


def readiness(cache: Path, artifact: Artifact, resources: dict, runtime: dict, *,
              schema_version: str, model_note: str) -> dict:
    model = cache / artifact.filename
    present = model.is_file() and not model.is_symlink()
    blockers = resource_blockers(resources, artifact, weights_present=present)
    if runtime["status"] != "matched":
        blockers.append("runtime_" + runtime["status"])
    if not present:
        weight_status = "missing"
        blockers.append("model_weights_missing")
    elif blockers:
        weight_status = "present_unverified"
    elif verify_model(model, artifact):
        weight_status = "sha256_verified"
    else:
        weight_status = "digest_mismatch"
        blockers.append("model_weights_mismatch")
    return {"schema_version": schema_version, "evidence_kind": "hardware_preflight",
            "created_at": datetime.now(UTC).isoformat(),
            "status": "blocked" if blockers else "ready_for_server_start",
            "block_reasons": blockers, "provider_requests": 0, "model_inference_executed": False,
            "host": resources, "runtime": runtime,
            "model": {"repository": artifact.repository, "revision": artifact.revision, "artifact": artifact.filename,
                      "bytes": artifact.size_bytes, "sha256": artifact.sha256, "quantization": artifact.quantization,
                      "status": weight_status},
            "policy": {"memory_reserve_bytes": MEMORY_RESERVE, "disk_reserve_bytes": DISK_RESERVE,
                       "context_tokens": CONTEXT_TOKENS, "server_parallel_slots": PARALLEL_SLOTS},
            "notes": ["This preflight is not a model latency, throughput or quality measurement.",
                      "The memory reserve is a conservative recipe policy, not a universal hardware minimum.",
                      model_note,
                      "This recipe uses CPU RAM only; it does not add GPU memory to available RAM."]}


def download_url(current: str, location: str) -> str:
    target = urllib.parse.urljoin(current, location)
    parsed = urllib.parse.urlsplit(target)
    host = parsed.hostname or ""
    if (parsed.scheme != "https" or parsed.username is not None or parsed.password is not None
            or parsed.port not in {None, 443} or parsed.fragment
            or not (host in {"huggingface.co", "hf.co"}
                    or host.endswith((".huggingface.co", ".hf.co")))):
        raise ValueError("Model download requires an HTTPS Hugging Face artifact URL")
    return target


async def stream_model(output, artifact: Artifact, *, timeout_seconds: float) -> None:
    """One elapsed deadline covers redirects, headers and a continuously slow body."""
    url = f"https://huggingface.co/{artifact.repository}/resolve/{artifact.revision}/{artifact.filename}"
    digest, size = hashlib.sha256(), 0
    if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool) or not 0 < timeout_seconds <= 3600:
        raise ValueError("Invalid model download deadline")
    async with asyncio.timeout(timeout_seconds), httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
        for _ in range(10):
            async with client.stream("GET", url, headers={"Accept-Encoding": "identity"}) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location:
                        raise ValueError("Model download redirect is missing its location")
                    url = download_url(url, location)
                    continue
                response.raise_for_status()
                if response.headers.get("content-encoding", "identity").strip().lower() != "identity":
                    raise ValueError("Unexpected compressed model download")
                async for chunk in response.aiter_raw():
                    size += len(chunk)
                    if size > artifact.size_bytes:
                        raise ValueError("Model download exceeded the pinned artifact size")
                    output.write(chunk)
                    digest.update(chunk)
                    await asyncio.sleep(0)
                if size != artifact.size_bytes or digest.hexdigest() != artifact.sha256:
                    raise ValueError("Downloaded model does not match the pinned artifact")
                return
        raise ValueError("Too many model download redirects")


def download_model(cache: Path, artifact: Artifact, resources: dict, *,
                   timeout_seconds: float = 1800, partial_prefix: str = ".model-download-") -> None:
    target = cache / artifact.filename
    if target.exists() or target.is_symlink():
        if verify_model(target, artifact):
            return
        raise ValueError("Existing model file does not match the pinned artifact")
    blockers = resource_blockers(resources, artifact, weights_present=False)
    if blockers:
        raise ValueError("Local preparation blocked: " + ", ".join(blockers))
    cache.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=partial_prefix, dir=cache)
    try:
        with os.fdopen(descriptor, "wb") as output:
            asyncio.run(stream_model(output, artifact, timeout_seconds=timeout_seconds))
        # Do not replace an existing file introduced by another process.
        os.link(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def deployment_manifest(report: dict, threads: int, artifact: Artifact, *, max_threads: int = 128) -> dict:
    if report["status"] != "ready_for_server_start":
        raise ValueError("Cannot create deployment metadata from a blocked preflight")
    if type(threads) is not int or not 1 <= threads <= max_threads:
        raise ValueError("CPU thread count outside supported bounds")
    return {"schema_version": "model-deployment-v1", "source": "operator_supplied",
            "runtime_name": "llama.cpp", "runtime_version": RUNTIME_VERSION,
            "runtime_revision": RUNTIME_REVISION,
            "model_repository": artifact.repository, "model_revision": artifact.revision, "model_artifact": artifact.filename,
            "model_sha256": artifact.sha256, "model_bytes": artifact.size_bytes, "quantization": artifact.quantization,
            "context_tokens": CONTEXT_TOKENS, "server_parallel_slots": PARALLEL_SLOTS,
            "server_threads": threads,
            "hardware_class": "cpu", "hardware_label": report["host"]["architecture"],
            "host_memory_bytes": report["host"]["memory_total_bytes"],
            "accelerator_count": 0, "accelerator_memory_bytes": 0}


def server_command(server: str, cache: Path, port: int, threads: int, artifact: Artifact,
                   model: str, *, max_threads: int = 128) -> list[str]:
    if type(port) is not int or not 1024 <= port <= 65535 or type(threads) is not int or not 1 <= threads <= max_threads:
        raise ValueError("Port/CPU thread count outside supported bounds")
    return [server, "--model", str(cache / artifact.filename), "--alias", model,
            "--host", "127.0.0.1", "--port", str(port), "--ctx-size", str(CONTEXT_TOKENS),
            "--parallel", str(PARALLEL_SLOTS), "--threads", str(threads), "--fit", "off",
            "--n-gpu-layers", "0", "--no-op-offload", "--jinja", "--reasoning-format", "deepseek",
            "--no-webui"]
