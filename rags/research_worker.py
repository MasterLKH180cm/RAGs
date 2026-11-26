"""Fixed subprocess entrypoint. No caller-selected imports, paths or commands."""
import asyncio
from contextlib import redirect_stdout
import inspect
from importlib import import_module
import json
import os
from pathlib import Path
import secrets
import sys

from rags.research_protocol import MAX_BODY_BYTES, MAX_RESPONSE_BYTES, UPSTREAMS, ResearchRequest, validate_result
from rags.features import FeatureFlags, strategy_features


ADAPTERS = {
    "raptor": "rags.research.raptor_adapter",
    "lightrag": "rags.research.lightrag_adapter",
    "full_graphrag": "rags.research.graphrag_adapter",
    "hipporag": "rags.research.hipporag_adapter",
    "self_rag": "rags.research.selfrag_adapter",
    "crag": "rags.research.crag_adapter",
}


def provider_settings(*, require_embeddings=True):
    # Providers reuse Settings, but this subprocess exposes no RAG HTTP API.
    # An ephemeral internal key satisfies that separate settings invariant;
    # real service/tenant credentials never enter the worker environment.
    from rags.config import Settings
    os.environ["RAGS_KEYS_JSON"] = json.dumps({secrets.token_urlsafe(32): "research-worker"})
    settings = Settings.from_env()
    if require_embeddings and settings.embedding_provider not in {"fastembed", "openai", "router", "sentence_transformers"}:
        raise ValueError("requires_provider")
    return settings


async def execute(request: ResearchRequest) -> dict:
    if request.engine != os.getenv("RAGS_RESEARCH_ENGINE"):
        raise ValueError("engine_mismatch")
    FeatureFlags.from_env().require(*strategy_features(request.engine))
    if not request.passages and not request.external_hosts:
        return validate_result({"hits": [], "implementation": UPSTREAMS[request.engine],
                                "diagnostics": {"source_count": 0, "engine_executed": False}, "usage": None}, request)
    search = import_module(ADAPTERS[request.engine]).search
    if request.engine == "self_rag":
        result = search(request.question, [p.model_dump() for p in request.passages], request.top_k, Path.cwd())
        result.setdefault("diagnostics", {}).update(source_count=len(request.passages), engine_executed=True)
        return validate_result(result, request)
    from rags.providers import Providers
    provider = Providers(provider_settings(require_embeddings=request.engine != "crag"))
    provider.require_generation()
    provider.begin_usage()
    keywords = {"namespace": request.namespace, "external_hosts": request.external_hosts} if request.engine == "crag" else {}
    result = search(request.question, [p.model_dump() for p in request.passages], request.top_k,
                    Path.cwd(), provider, **keywords)
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
        elif isinstance(error, CapabilityError) and error.code in {
            "requires_provider", "requires_source", "requires_external_source", "feature_disabled",
            "context_budget_exceeded", "research_limit_exceeded", "insufficient_graph_evidence",
            "requires_search_credentials", "external_source_forbidden", "external_source_timeout",
            "invalid_research_input",
        }:
            code = error.code
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
