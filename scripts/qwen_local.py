"""Pinned Qwen3-1.7B Q8_0 CPU generation; preparation is not inference evidence."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import httpx

try:
    from scripts import local_model as shared
except ModuleNotFoundError:
    import local_model as shared

MODEL = "qwen/Qwen3-1.7B"
ARTIFACT = shared.Artifact(
    repository="Qwen/Qwen3-1.7B-GGUF",
    revision="90862c4b9d2787eaed51d12237eafdfe7c5f6077",
    filename="Qwen3-1.7B-Q8_0.gguf",
    size_bytes=1_834_426_016,
    sha256="061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a",
    quantization="Q8_0",
)


def readiness(cache: Path, server: str) -> dict:
    return shared.readiness(
        cache, ARTIFACT, shared.host_resources(cache), shared.inspect_runtime(server),
        schema_version="qwen-local-readiness-v1",
        model_note="Q8_0 generation is distinct from the frozen bfloat16 answerability scorer.",
    )


def download_model(cache: Path, *, timeout_seconds: float = 1800) -> None:
    shared.download_model(cache, ARTIFACT, shared.host_resources(cache), timeout_seconds=timeout_seconds,
                          partial_prefix=".qwen-download-")


def deployment_manifest(report: dict, threads: int) -> dict:
    return shared.deployment_manifest(report, threads, ARTIFACT, max_threads=8)


def server_command(server: str, cache: Path, port: int, threads: int) -> list[str]:
    return shared.server_command(server, cache, port, threads, ARTIFACT, MODEL, max_threads=8) + [
        "--reasoning", "off", "--cache-ram", "0", "--no-cache-prompt", "--temp", "0", "--seed", "42",
    ]


def execution_contract(report: dict, threads: int) -> dict:
    manifest = deployment_manifest(report, threads)
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {
        "schema_version": "local-generation-execution-v1", "source": "operator_supplied",
        "model_alias": MODEL, "deployment_sha256": hashlib.sha256(encoded).hexdigest(),
        "runtime_executable_sha256": report["runtime"]["executable_sha256"],
        "parameters": {"hardware_class": "cpu", "context_tokens": shared.CONTEXT_TOKENS,
                       "parallel_slots": shared.PARALLEL_SLOTS, "threads": threads,
                       "template": "model_embedded_jinja", "reasoning": "off",
                       "reasoning_format": "deepseek", "cache_ram_mib": 0, "prompt_cache": False,
                       "temperature": 0, "seed": 42},
        "model_inference_executed": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "download", "serve"))
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache/rags/qwen3-1.7b-q8_0")
    parser.add_argument("--server", default="llama-server")
    parser.add_argument("--output", type=Path, default=Path("reports/testing/qwen-local-readiness.json"))
    parser.add_argument("--manifest", type=Path, default=Path("reports/testing/qwen-local-deployment.json"))
    parser.add_argument("--execution-contract", type=Path, default=Path("reports/testing/qwen-local-execution.json"))
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args(argv)
    try:
        command = server_command(args.server, args.cache_dir, args.port, args.threads)
        if args.command == "download":
            blockers = shared.resource_blockers(shared.host_resources(args.cache_dir), ARTIFACT, weights_present=False)
            if not blockers:
                download_model(args.cache_dir)
        report = readiness(args.cache_dir, args.server)
        shared.write_json(args.output, report)
        print(json.dumps({"status": report["status"], "block_reasons": report["block_reasons"]}))
        if report["status"] != "ready_for_server_start":
            return 2
        if args.command == "serve":
            shared.write_json(args.manifest, deployment_manifest(report, args.threads))
            shared.write_json(args.execution_contract, execution_contract(report, args.threads))
            environment = {key: value for key, value in os.environ.items()
                           if not key.startswith("LLAMA_ARG_") and key != "LLAMA_API_KEY"}
            return subprocess.run(command, check=False, env=environment).returncode
        return 0
    except (ImportError, OSError, ValueError, httpx.HTTPError) as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
