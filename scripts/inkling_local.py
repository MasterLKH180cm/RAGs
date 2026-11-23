"""Pinned CPU-only Inkling preparation; readiness is not inference evidence.

Linux/WSL x86_64 reference recipe. No hosted model credentials are used.
"""
import argparse
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
from datetime import UTC, datetime
from pathlib import Path

import httpx

MODEL = "thinkingmachines/Inkling-2-Mini"
REPOSITORY = "ggml-org/Inkling-2-Mini-GGUF"
REVISION = "e8a2cecf0e78914f07cc27b9be711a9b2c7561d9"
ARTIFACT = "Inkling-2-Mini-IQ4_NL.gguf"
MODEL_BYTES = 17_627_940_768
MODEL_SHA256 = "cbdf3fa9dc50a4124977980f37da4f1fa764b0cfe800d275cd43d26ce2a81af4"
RUNTIME_REVISION = "e300919cc704ddde3a8a8346a8714f57eeeee71b"
RUNTIME_VERSION = "b12631"
GIB = 1024**3
# Operator policy for this CPU recipe, not a measured universal minimum.
MEMORY_RESERVE = 2 * GIB
DISK_RESERVE = 256 * 1024**2
CONTEXT_TOKENS = 4096
PARALLEL_SLOTS = 1


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".inkling-", dir=path.parent)
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


def resource_blockers(resources: dict, *, weights_present: bool) -> list[str]:
    result = []
    if resources["system"] != "Linux" or resources["architecture"].lower() not in {"x86_64", "amd64"}:
        result.append("unsupported_reference_platform")
    available = resources["memory_available_bytes"]
    if available is None:
        result.append("memory_capacity_unknown")
    elif available < MODEL_BYTES + MEMORY_RESERVE:
        result.append("insufficient_available_memory")
    required_disk = DISK_RESERVE + (0 if weights_present else MODEL_BYTES)
    if resources["disk_free_bytes"] < required_disk:
        result.append("insufficient_disk_space")
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(path: Path) -> bool:
    return bool(not path.is_symlink() and path.is_file() and path.stat().st_size == MODEL_BYTES
                and file_sha256(path) == MODEL_SHA256)


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


def readiness(cache: Path, server: str) -> dict:
    model = cache / ARTIFACT
    present = model.is_file() and not model.is_symlink()
    resources = host_resources(cache)
    blockers = resource_blockers(resources, weights_present=present)
    runtime = inspect_runtime(server)
    if runtime["status"] != "matched":
        blockers.append("runtime_" + runtime["status"])
    if not present:
        weight_status = "missing"
        blockers.append("model_weights_missing")
    elif blockers:
        weight_status = "present_unverified"
    elif verify_model(model):
        weight_status = "sha256_verified"
    else:
        weight_status = "digest_mismatch"
        blockers.append("model_weights_mismatch")
    return {"schema_version": "inkling-local-readiness-v1", "evidence_kind": "hardware_preflight",
            "created_at": datetime.now(UTC).isoformat(),
            "status": "blocked" if blockers else "ready_for_server_start",
            "block_reasons": blockers, "provider_requests": 0, "model_inference_executed": False,
            "host": resources, "runtime": runtime,
            "model": {"repository": REPOSITORY, "revision": REVISION, "artifact": ARTIFACT,
                      "bytes": MODEL_BYTES, "sha256": MODEL_SHA256, "quantization": "IQ4_NL",
                      "status": weight_status},
            "policy": {"memory_reserve_bytes": MEMORY_RESERVE, "disk_reserve_bytes": DISK_RESERVE,
                       "context_tokens": CONTEXT_TOKENS, "server_parallel_slots": PARALLEL_SLOTS},
            "notes": ["This preflight is not a model latency, throughput or quality measurement.",
                      "The memory reserve is a conservative recipe policy, not a universal hardware minimum.",
                      "IQ4_NL is a quantized artifact, not the native NVFP4/MXFP8 checkpoint.",
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


async def stream_model(output, *, timeout_seconds: float) -> None:
    """One elapsed deadline covers redirects, headers and a continuously slow body."""
    url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{ARTIFACT}"
    digest, size = hashlib.sha256(), 0
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
                    if size > MODEL_BYTES:
                        raise ValueError("Model download exceeded the pinned artifact size")
                    output.write(chunk)
                    digest.update(chunk)
                    await asyncio.sleep(0)
                if size != MODEL_BYTES or digest.hexdigest() != MODEL_SHA256:
                    raise ValueError("Downloaded model does not match the pinned artifact")
                return
        raise ValueError("Too many model download redirects")


def download_model(cache: Path, *, timeout_seconds: float = 1800) -> None:
    target = cache / ARTIFACT
    if target.exists() or target.is_symlink():
        if verify_model(target):
            return
        raise ValueError("Existing model file does not match the pinned artifact")
    blockers = resource_blockers(host_resources(cache), weights_present=False)
    if blockers:
        raise ValueError("Local preparation blocked: " + ", ".join(blockers))
    cache.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".inkling-download-", dir=cache)
    try:
        with os.fdopen(descriptor, "wb") as output:
            asyncio.run(stream_model(output, timeout_seconds=timeout_seconds))
        # Do not replace an existing file introduced by another process.
        os.link(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def deployment_manifest(report: dict, threads: int) -> dict:
    if report["status"] != "ready_for_server_start":
        raise ValueError("Cannot create deployment metadata from a blocked preflight")
    if not 1 <= threads <= 128:
        raise ValueError("CPU thread count outside supported bounds")
    return {"schema_version": "model-deployment-v1", "source": "operator_supplied",
            "runtime_name": "llama.cpp", "runtime_version": RUNTIME_VERSION,
            "runtime_revision": RUNTIME_REVISION,
            "model_repository": REPOSITORY, "model_revision": REVISION, "model_artifact": ARTIFACT,
            "model_sha256": MODEL_SHA256, "model_bytes": MODEL_BYTES, "quantization": "IQ4_NL",
            "context_tokens": CONTEXT_TOKENS, "server_parallel_slots": PARALLEL_SLOTS,
            "server_threads": threads,
            "hardware_class": "cpu", "hardware_label": report["host"]["architecture"],
            "host_memory_bytes": report["host"]["memory_total_bytes"],
            "accelerator_count": 0, "accelerator_memory_bytes": 0}


def server_command(server: str, cache: Path, port: int, threads: int) -> list[str]:
    if not 1024 <= port <= 65535 or not 1 <= threads <= 128:
        raise ValueError("Port/CPU thread count outside supported bounds")
    return [server, "--model", str(cache / ARTIFACT), "--alias", MODEL,
            "--host", "127.0.0.1", "--port", str(port), "--ctx-size", str(CONTEXT_TOKENS),
            "--parallel", str(PARALLEL_SLOTS), "--threads", str(threads), "--fit", "off",
            "--n-gpu-layers", "0", "--no-op-offload", "--jinja", "--reasoning-format", "deepseek",
            "--no-webui"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "download", "serve"))
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache/rags/inkling-iq4")
    parser.add_argument("--server", default="llama-server")
    parser.add_argument("--output", type=Path, default=Path("reports/testing/inkling-local-readiness.json"))
    parser.add_argument("--manifest", type=Path, default=Path("reports/testing/inkling-local-deployment.json"))
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args(argv)
    try:
        # Validate launch arguments even when only preparing the deployment.
        command = server_command(args.server, args.cache_dir, args.port, args.threads)
        if args.command == "download":
            blockers = resource_blockers(host_resources(args.cache_dir), weights_present=False)
            if not blockers:
                download_model(args.cache_dir)
        report = readiness(args.cache_dir, args.server)
        write_json(args.output, report)
        print(json.dumps({"status": report["status"], "block_reasons": report["block_reasons"]}))
        if report["status"] != "ready_for_server_start":
            return 2
        if args.command == "serve":
            write_json(args.manifest, deployment_manifest(report, args.threads))
            # argv is a fixed command, never interpreted by a shell.
            environment = {key: value for key, value in os.environ.items()
                           if not key.startswith("LLAMA_ARG_")}
            return subprocess.run(command, check=False, env=environment).returncode
        return 0
    except (ImportError, OSError, ValueError, httpx.HTTPError) as exc:
        # Do not print endpoint URLs, paths or remote response bodies.
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
