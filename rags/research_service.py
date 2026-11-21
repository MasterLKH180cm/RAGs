"""Authenticated research sidecar; one fixed worker process and index per request."""
import asyncio
from dataclasses import dataclass
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
from rags.research_protocol import MAX_BODY_BYTES, MAX_RESPONSE_BYTES, ResearchRequest, validate_result


@dataclass(frozen=True)
class ResearchSettings:
    engine: str
    keys: dict[str, str]
    timeout_seconds: float = 120
    concurrency: int = 1
    work_dir: str | None = None
    raptor_source: str | None = None

    def __post_init__(self):
        if self.engine not in {"raptor", "lightrag"}:
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

    @classmethod
    def from_env(cls):
        return cls(
            engine=os.getenv("RAGS_RESEARCH_ENGINE", ""),
            keys=json.loads(os.getenv("RAGS_RESEARCH_KEYS_JSON", "{}")),
            timeout_seconds=float(os.getenv("RAGS_RESEARCH_TIMEOUT_SECONDS", "120")),
            concurrency=int(os.getenv("RAGS_RESEARCH_CONCURRENCY", "1")),
            work_dir=os.getenv("RAGS_RESEARCH_WORK_DIR") or None,
            raptor_source=os.getenv("RAGS_RAPTOR_SOURCE") or None,
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
        "RAGS_GENERATION_TOKENIZER_JSON", "RAGS_CONTEXT_BUDGET", "RAGS_OUTPUT_BUDGET", "RAGS_ENVELOPE_BUDGET",
        "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_CHAT_MODEL", "OPENAI_EMBEDDING_MODEL",
        "ANTHROPIC_API_KEY", "ANTHROPIC_CHAT_MODEL",
    }
    environment = {key: value for key, value in os.environ.items() if key in allowed}
    environment.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", NUMBA_NUM_THREADS="2",
                       TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
                       ORT_DISABLE_TELEMETRY="1", HF_HUB_DISABLE_TELEMETRY="1",
                       RAGS_RESEARCH_ENGINE=config.engine, RAGS_ADMISSION_POLICY="compatibility")
    if config.raptor_source:
        environment["RAGS_RAPTOR_SOURCE"] = config.raptor_source
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, [config.raptor_source, environment.get("PYTHONPATH")]))
    if environment.get("RAGS_MODEL_CACHE"):
        environment["RAGS_MODEL_CACHE"] = str(Path(environment["RAGS_MODEL_CACHE"]).resolve())
    return environment


class ProcessRunner:
    def __init__(self, config: ResearchSettings):
        self.config = config
        self.environment = worker_environment(config)
        if config.work_dir:
            Path(config.work_dir).mkdir(parents=True, exist_ok=True)

    async def run(self, request: ResearchRequest) -> dict:
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
                    if code in {"requires_provider", "requires_dependency", "requires_source"}:
                        raise ResearchFailure(code, 503)
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
                "verification": "upstream_not_exercised_by_health_probe"}

    @app.post("/retrieve")
    async def retrieve(body: ResearchRequest, namespace: str = Depends(tenant)):
        if body.namespace != namespace:
            raise HTTPException(403, detail={"code": "tenant_mismatch"})
        if body.engine != settings.engine:
            raise HTTPException(422, detail={"code": "engine_mismatch"})
        if slots.locked():
            raise HTTPException(429, detail={"code": "research_capacity"}, headers={"Retry-After": "1"})
        async with slots:
            result = await worker.run(body)
        try:
            return validate_result(result, body)
        except (ValueError, TypeError) as exc:
            raise ResearchFailure("invalid_worker_output", 502) from exc

    return app
