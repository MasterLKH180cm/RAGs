"""Pinned CPU-only Inkling preparation; readiness is not inference evidence.

Linux/WSL x86_64 reference recipe. No hosted model credentials are used.
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

import httpx

try:
    from scripts import local_model as shared
except ModuleNotFoundError:
    import local_model as shared

MODEL = "thinkingmachines/Inkling-2-Mini"
REPOSITORY = "ggml-org/Inkling-2-Mini-GGUF"
REVISION = "e8a2cecf0e78914f07cc27b9be711a9b2c7561d9"
ARTIFACT = "Inkling-2-Mini-IQ4_NL.gguf"
MODEL_BYTES = 17_627_940_768
MODEL_SHA256 = "cbdf3fa9dc50a4124977980f37da4f1fa764b0cfe800d275cd43d26ce2a81af4"
RUNTIME_REVISION = shared.RUNTIME_REVISION
RUNTIME_VERSION = shared.RUNTIME_VERSION
GIB = shared.GIB
# Operator policy for this CPU recipe, not a measured universal minimum.
MEMORY_RESERVE = shared.MEMORY_RESERVE
DISK_RESERVE = shared.DISK_RESERVE
CONTEXT_TOKENS = shared.CONTEXT_TOKENS
PARALLEL_SLOTS = shared.PARALLEL_SLOTS


def _artifact() -> shared.Artifact:
    return shared.Artifact(REPOSITORY, REVISION, ARTIFACT, MODEL_BYTES, MODEL_SHA256, "IQ4_NL")


write_json = shared.write_json
host_resources = shared.host_resources
file_sha256 = shared.file_sha256
inspect_runtime = shared.inspect_runtime
download_url = shared.download_url
shutil = shared.shutil


def resource_blockers(resources: dict, *, weights_present: bool) -> list[str]:
    return shared.resource_blockers(resources, _artifact(), weights_present=weights_present)


def verify_model(path: Path) -> bool:
    return shared.verify_model(path, _artifact())


def readiness(cache: Path, server: str) -> dict:
    return shared.readiness(cache, _artifact(), host_resources(cache), inspect_runtime(server),
                            schema_version="inkling-local-readiness-v1",
                            model_note="IQ4_NL is a quantized artifact, not the native NVFP4/MXFP8 checkpoint.")


async def stream_model(output, *, timeout_seconds: float) -> None:
    await shared.stream_model(output, _artifact(), timeout_seconds=timeout_seconds)


def download_model(cache: Path, *, timeout_seconds: float = 1800) -> None:
    shared.download_model(cache, _artifact(), host_resources(cache), timeout_seconds=timeout_seconds,
                          partial_prefix=".inkling-download-")


def deployment_manifest(report: dict, threads: int) -> dict:
    return shared.deployment_manifest(report, threads, _artifact())


def server_command(server: str, cache: Path, port: int, threads: int) -> list[str]:
    return shared.server_command(server, cache, port, threads, _artifact(), MODEL)


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
