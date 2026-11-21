"""Fixed subprocess entrypoint. No caller-selected imports, paths or commands."""
import asyncio
from contextlib import redirect_stdout
import inspect
import json
import os
from pathlib import Path
import secrets
import sys

from rags.research_protocol import MAX_BODY_BYTES, MAX_RESPONSE_BYTES, UPSTREAMS, ResearchRequest, validate_result


def provider_settings():
    # Providers reuse Settings, but this subprocess exposes no RAG HTTP API.
    # An ephemeral internal key satisfies that separate settings invariant;
    # real service/tenant credentials never enter the worker environment.
    from rags.config import Settings
    os.environ["RAGS_KEYS_JSON"] = json.dumps({secrets.token_urlsafe(32): "research-worker"})
    settings = Settings.from_env()
    if settings.embedding_provider not in {"fastembed", "openai"}:
        raise ValueError("requires_provider")
    return settings


async def execute(request: ResearchRequest) -> dict:
    if request.engine != os.getenv("RAGS_RESEARCH_ENGINE"):
        raise ValueError("engine_mismatch")
    if not request.passages:
        return {"hits": [], "implementation": UPSTREAMS[request.engine],
                "diagnostics": {"source_count": 0, "engine_executed": False}, "usage": None}
    from rags.providers import Providers
    provider = Providers(provider_settings())
    provider.require_generation()
    provider.begin_usage()
    if request.engine == "raptor":
        from rags.research.raptor_adapter import search
    else:
        from rags.research.lightrag_adapter import search
    result = search(request.question, [p.model_dump() for p in request.passages], request.top_k, Path.cwd(), provider)
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, dict):
        raise ValueError("invalid_worker_output")
    result.setdefault("diagnostics", {}).update(source_count=len(request.passages), engine_executed=True)
    reports = provider.usage()
    entries = reports.get("provider_reports", [])
    usage = None
    if entries and len(entries) == reports.get("calls"):
        input_values = [e["usage"].get("input_tokens", e["usage"].get("prompt_tokens")) for e in entries]
        output_values = [e["usage"].get("output_tokens", e["usage"].get("completion_tokens")) for e in entries]
        if all(type(n) is int and n >= 0 for n in input_values + output_values):
            usage = {"input_tokens": sum(input_values), "output_tokens": sum(output_values)}
    result["usage"] = usage
    return validate_result(result, request)


def main():
    exit_code = 0
    try:
        raw = sys.stdin.buffer.read(MAX_BODY_BYTES + 1)
        if len(raw) > MAX_BODY_BYTES:
            raise ValueError("invalid_request")
        request = ResearchRequest.model_validate_json(raw)
        # Third-party progress output is private; stdout is one protocol object.
        with redirect_stdout(sys.stderr):
            result = asyncio.run(execute(request))
        encoded = json.dumps(result, ensure_ascii=False, allow_nan=False).encode()
        if len(encoded) > MAX_RESPONSE_BYTES:
            raise ValueError("invalid_worker_output")
    except Exception as error:
        from rags.registry import CapabilityError
        if isinstance(error, (ImportError, ModuleNotFoundError)):
            code = "requires_dependency"
        elif isinstance(error, CapabilityError) and error.code == "requires_provider":
            code = "requires_provider"
        elif isinstance(error, CapabilityError) and error.code == "requires_backend":
            code = "requires_dependency"
        elif isinstance(error, ValueError) and str(error) in {"requires_provider", "requires_source"}:
            code = str(error)
        else:
            code = "upstream_failed"
        encoded = json.dumps({"error": code}).encode()
        exit_code = 1
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
