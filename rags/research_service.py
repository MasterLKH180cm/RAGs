"""Authenticated research sidecar; one fixed worker process and index per request."""
import asyncio
from dataclasses import dataclass, field
import hmac
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from rags.limits import BodyLimitMiddleware
from rags.features import FeatureFlags, strategy_features
from rags.registry import CapabilityError
from rags.research_protocol import (
    MAX_BODY_BYTES, MAX_RESPONSE_BYTES, ResearchRequest, UPSTREAMS, validate_external_hosts, validate_result,
)


@dataclass(frozen=True)
class ResearchSettings:
    engine: str
    keys: dict[str, str]
    timeout_seconds: float = 120
    concurrency: int = 1
    work_dir: str | None = None
    raptor_source: str | None = None
    features: FeatureFlags = field(default_factory=FeatureFlags)
    external_hosts: tuple[str, ...] = ()
    external_tenants: tuple[str, ...] = ()

    def __post_init__(self):
        if self.engine not in UPSTREAMS:
            raise ValueError("Configure one supported RAGS_RESEARCH_ENGINE")
        if not isinstance(self.keys, dict) or not 1 <= len(self.keys) <= 64 or any(
            not isinstance(k, str) or not k or not isinstance(v, str) or not 1 <= len(v) <= 128
            for k, v in self.keys.items()
        ):
            raise ValueError("Research keys must map credentials to tenants")
        if os.getenv("RAGS_ENV", "production") == "production" and any(len(k) < 24 for k in self.keys):
            raise ValueError("Production research credentials need at least 24 characters")
        if type(self.timeout_seconds) not in (int, float) or not math.isfinite(self.timeout_seconds) or not 0.1 <= self.timeout_seconds <= 300:
            raise ValueError("Research timeout must be between 0.1 and 300 seconds")
        if type(self.concurrency) is not int or not 1 <= self.concurrency <= 4:
            raise ValueError("Research concurrency must be 1..4 per service process")
        if self.raptor_source and (not Path(self.raptor_source).is_absolute() or os.pathsep in self.raptor_source):
            raise ValueError("RAPTOR source must be an absolute operator-selected path")
        validate_external_hosts(list(self.external_hosts))
        if (not isinstance(self.external_hosts, tuple) or not isinstance(self.external_tenants, tuple)
                or any(not isinstance(t, str) or t not in self.keys.values() for t in self.external_tenants)
                or len(set(self.external_tenants)) != len(self.external_tenants)
                or bool(self.external_hosts) != bool(self.external_tenants)
                or (self.external_hosts and self.engine != "crag")):
            raise ValueError("CRAG external evidence requires explicit hosts and authorized service tenants")

    @classmethod
    def from_env(cls):
        external_hosts = json.loads(os.getenv("RAGS_CRAG_WEB_HOSTS_JSON", "[]"))
        external_tenants = json.loads(os.getenv("RAGS_CRAG_WEB_TENANTS_JSON", "[]"))
        if not isinstance(external_hosts, list) or not isinstance(external_tenants, list):
            raise ValueError("CRAG external hosts and tenants must be JSON lists")
        return cls(
            engine=os.getenv("RAGS_RESEARCH_ENGINE", ""),
            keys=json.loads(os.getenv("RAGS_RESEARCH_KEYS_JSON", "{}")),
            timeout_seconds=float(os.getenv("RAGS_RESEARCH_TIMEOUT_SECONDS", "120")),
            concurrency=int(os.getenv("RAGS_RESEARCH_CONCURRENCY", "1")),
            work_dir=os.getenv("RAGS_RESEARCH_WORK_DIR") or None,
            raptor_source=os.getenv("RAGS_RAPTOR_SOURCE") or None,
            features=FeatureFlags.from_env(),
            external_hosts=tuple(external_hosts),
            external_tenants=tuple(external_tenants),
        )


class ResearchFailure(Exception):
    def __init__(self, code: str, status: int):
        self.code, self.status = code, status
        super().__init__(code)


def worker_environment(config: ResearchSettings) -> dict:
    # Only runtime setup and explicit provider settings cross into the child.
    # Caller credentials, other tenant keys and unrelated service secrets do not.
    allowed = {
        "PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "LANG", "LC_ALL", "TZ", "SYSTEMROOT",
        "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy",
        "RAGS_ENV", "RAGS_MODEL_PROVIDER", "RAGS_EMBEDDING_PROVIDER", "RAGS_FASTEMBED_MODEL", "RAGS_MODEL_CACHE",
        "TIKTOKEN_CACHE_DIR",
        "RAGS_GENERATION_TOKENIZER_JSON", "RAGS_CONTEXT_BUDGET", "RAGS_OUTPUT_BUDGET", "RAGS_ENVELOPE_BUDGET",
        "OPENAI_BASE_URL", "OPENAI_CHAT_MODEL", "OPENAI_EMBEDDING_MODEL",
        "ANTHROPIC_CHAT_MODEL", "ANTHROPIC_BASE_URL",
        "RAGS_EMBEDDING_BASE_URL", "RAGS_EMBEDDING_MODEL", "RAGS_EMBEDDING_SPACE_ID",
        "RAGS_GENERATION_BASE_URL", "RAGS_GENERATION_MODEL",
        "RAGS_GENERATION_PROTOCOL", "RAGS_REASONING_EFFORT", "RAGS_GENERATION_AUTH_MODE",
    }
    generation = os.getenv("RAGS_MODEL_PROVIDER", "extractive").lower()
    embedding = os.getenv("RAGS_EMBEDDING_PROVIDER", "local").lower()

    def include_secret(name):
        allowed.update((name, name + "_FILE"))

    if config.engine != "self_rag":
        if generation in {"openai", "anthropic"}:
            include_secret("RAGS_GENERATION_API_KEY")
            if (os.getenv("RAGS_GENERATION_AUTH_MODE", "required") != "none"
                    and not any(os.getenv(k) for k in ("RAGS_GENERATION_API_KEY", "RAGS_GENERATION_API_KEY_FILE"))):
                include_secret("OPENAI_API_KEY" if generation == "openai" else "ANTHROPIC_API_KEY")
        if embedding == "openai" and config.engine != "crag":
            include_secret("RAGS_EMBEDDING_API_KEY")
            if not any(os.getenv(k) for k in ("RAGS_EMBEDDING_API_KEY", "RAGS_EMBEDDING_API_KEY_FILE")):
                include_secret("OPENAI_API_KEY")
        if generation == "router" or (embedding == "router" and config.engine != "crag"):
            allowed.update({"RAGS_ROUTER_ENABLED", "RAGS_ROUTER_GATEWAY_URL", "RAGS_ROUTER_MODELS_JSON",
                            "RAGS_ROUTER_MODEL_OPTIONS_JSON", "RAGS_ROUTER_ALLOW_HTTP", "RAGS_ROUTER_AUTH_MODE",
                            "RAGS_ROUTER_CA_FILE", "RAGS_ROUTER_MAX_OUTPUT_TOKENS"})
            include_secret("RAGS_ROUTER_API_KEY")
    if config.engine == "self_rag":
        allowed.update({"RAGS_SELFRAG_MODEL_PATH", "RAGS_SELFRAG_LLAMA_DIR", "RAGS_SELFRAG_THREADS"})
    if config.engine == "crag":
        allowed.add("RAGS_CRAG_MODEL_PATH")
        if config.external_hosts:
            include_secret("RAGS_CRAG_SEARCH_KEY")
    environment = {key: value for key, value in os.environ.items() if key in allowed}
    environment.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", NUMBA_NUM_THREADS="2",
                       TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
                       ORT_DISABLE_TELEMETRY="1", HF_HUB_DISABLE_TELEMETRY="1",
                       LITELLM_LOCAL_MODEL_COST_MAP="True",
                       RAGS_RESEARCH_ENGINE=config.engine, RAGS_ADMISSION_POLICY="compatibility",
                       RAGS_AGENT_ENABLED="false", RAGS_MEMORY_ENABLED="false")
    environment["RAGS_CRAG_WEB_HOSTS_JSON"] = json.dumps(list(config.external_hosts))
    environment["RAGS_FEATURES_JSON"] = json.dumps(config.features.as_dict())
    for name in ("OPENAI_API_KEY_FILE", "ANTHROPIC_API_KEY_FILE", "RAGS_EMBEDDING_API_KEY_FILE",
                 "RAGS_GENERATION_API_KEY_FILE", "RAGS_ROUTER_API_KEY_FILE", "RAGS_ROUTER_CA_FILE",
                 "RAGS_SELFRAG_MODEL_PATH", "RAGS_SELFRAG_LLAMA_DIR", "RAGS_CRAG_MODEL_PATH",
                 "RAGS_CRAG_SEARCH_KEY_FILE"):
        if environment.get(name):
            environment[name] = str(Path(environment[name]).resolve())
    if config.raptor_source:
        environment["RAGS_RAPTOR_SOURCE"] = config.raptor_source
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, [config.raptor_source, environment.get("PYTHONPATH")]))
    for name in ("RAGS_MODEL_CACHE", "TIKTOKEN_CACHE_DIR"):
        if environment.get(name):
            environment[name] = str(Path(environment[name]).resolve())
    return environment


def authorize_external(config: ResearchSettings, request: ResearchRequest) -> None:
    if request.external_hosts and (request.namespace not in config.external_tenants
                                   or not set(request.external_hosts) <= set(config.external_hosts)):
        raise CapabilityError("external_source_forbidden", "External source acquisition is not authorized", 403)


class ProcessRunner:
    def __init__(self, config: ResearchSettings):
        self.config = config
        self.environment = worker_environment(config)
        if config.work_dir:
            Path(config.work_dir).mkdir(parents=True, exist_ok=True)

    async def run(self, request: ResearchRequest) -> dict:
        self.config.features.require(*strategy_features(request.engine))
        authorize_external(self.config, request)
        encoded = request.model_dump_json().encode()
        with tempfile.TemporaryDirectory(prefix="rags-research-", dir=self.config.work_dir) as directory:
            child = None
            try:
                child = await asyncio.create_subprocess_exec(
                    sys.executable, "-m", "rags.research_worker",
                    stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL, cwd=directory, env=self.environment,
                    start_new_session=True,
                )
                raw = await asyncio.wait_for(self._exchange(child, encoded), timeout=self.config.timeout_seconds)
                payload = json.loads(raw)
                if child.returncode != 0:
                    # Never return arbitrary subprocess/provider error text.
                    code = payload.get("error") if isinstance(payload, dict) else None
                    if code == "feature_disabled":
                        raise ResearchFailure(code, 403)
                    if code in {"requires_provider", "requires_dependency", "requires_source"}:
                        raise ResearchFailure(code, 503)
                    if code in {"requires_external_source", "requires_search_credentials"}:
                        raise ResearchFailure(code, 503)
                    if code == "external_source_forbidden":
                        raise ResearchFailure(code, 403)
                    if code == "external_source_timeout":
                        raise ResearchFailure(code, 504)
                    if code in {"context_budget_exceeded", "research_limit_exceeded", "insufficient_graph_evidence",
                                "invalid_research_input"}:
                        raise ResearchFailure(code, 422)
                    raise ResearchFailure("upstream_failed", 502)
                return validate_result(payload, request)
            except TimeoutError as exc:
                raise ResearchFailure("upstream_timeout", 504) from exc
            except (OSError, ValueError, TypeError) as exc:
                raise ResearchFailure("invalid_worker_output", 502) from exc
            finally:
                if child is not None:
                    # A library may leave descendants after its leader exits.
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    # A killed process can still have buffered stdout above the
                    # StreamReader high-water mark. wait() alone may then wait
                    # forever for pipe closure; drain the bounded remainder after
                    # killing every writer in the process group.
                    await child.communicate()

    @staticmethod
    async def _exchange(child, encoded: bytes) -> bytes:
        async def send():
            child.stdin.write(encoded)
            await child.stdin.drain()
            child.stdin.close()
            await child.stdin.wait_closed()

        writer = asyncio.create_task(send())
        try:
            output = bytearray()
            while part := await child.stdout.read(16_384):
                if len(output) + len(part) > MAX_RESPONSE_BYTES:
                    raise ResearchFailure("invalid_worker_output", 502)
                output.extend(part)
            await writer
            await child.wait()
            return bytes(output)
        finally:
            if not writer.done():
                writer.cancel()
            await asyncio.gather(writer, return_exceptions=True)


def create_app(config: ResearchSettings | None = None, *, runner=None) -> FastAPI:
    settings = config or ResearchSettings.from_env()
    worker = runner or ProcessRunner(settings)
    slots = asyncio.Semaphore(settings.concurrency)
    app = FastAPI(title="RAGs research service", version="0.3.0", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(BodyLimitMiddleware, max_body_bytes=MAX_BODY_BYTES)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, error):
        return JSONResponse(status_code=422, content={"detail": {"code": "invalid_request"}})

    @app.exception_handler(ResearchFailure)
    async def failed_request(request, error):
        return JSONResponse(status_code=error.status, content={"detail": {"code": error.code}})

    @app.exception_handler(CapabilityError)
    async def disabled_request(request, error):
        return JSONResponse(status_code=error.status_code, content={"detail": {"code": error.code, "reason": error.reason}})

    def tenant(authorization: str | None = Header(default=None)) -> str:
        credential = authorization[7:] if authorization and authorization.startswith("Bearer ") else ""
        for key, name in settings.keys.items():
            if hmac.compare_digest(key.encode("utf-8"), credential.encode("utf-8")):
                return name
        raise HTTPException(401, detail={"code": "invalid_credentials"})

    @app.get("/live")
    def live():
        return {"status": "ok"}

    @app.get("/health")
    def health(namespace: str = Depends(tenant)):
        return {"status": "configured", "engine": settings.engine, "execution": "request_local_process",
                "timeout_seconds": settings.timeout_seconds, "max_passages": 64,
                "features": settings.features.as_dict(),
                "verification": "upstream_not_exercised_by_health_probe"}

    @app.post("/retrieve")
    async def retrieve(body: ResearchRequest, namespace: str = Depends(tenant)):
        if body.namespace != namespace:
            raise HTTPException(403, detail={"code": "tenant_mismatch"})
        if body.engine != settings.engine:
            raise HTTPException(422, detail={"code": "engine_mismatch"})
        settings.features.require(*strategy_features(body.engine))
        authorize_external(settings, body)
        if slots.locked():
            raise HTTPException(429, detail={"code": "research_capacity"}, headers={"Retry-After": "1"})
        async with slots:
            result = await worker.run(body)
        try:
            return validate_result(result, body)
        except (ValueError, TypeError) as exc:
            raise ResearchFailure("invalid_worker_output", 502) from exc

    return app
