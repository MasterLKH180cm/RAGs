"""Service boundaries and real subprocess lifecycle; upstream quality is separate."""
import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

from fastapi.testclient import TestClient
import httpx
import pytest

from rags.research_protocol import MAX_BODY_BYTES, ResearchRequest, UPSTREAMS
from rags.research_service import ProcessRunner, ResearchFailure, ResearchSettings, create_app, worker_environment


def request_body(**updates):
    return {"protocol": "authorized-evidence-v1", "engine": "lightrag", "namespace": "tenant-a",
            "question": "Where is the office?", "top_k": 2,
            "passages": [{"id": "original-1", "text": "The office is in Taipei."}], **updates}


def result_for(request):
    return {"hits": [{"chunk_id": p.id, "score": 1 / (i + 1)} for i, p in enumerate(request.passages[:request.top_k])],
            "implementation": dict(UPSTREAMS[request.engine]),
            "diagnostics": {"source_count": len(request.passages), "engine_executed": bool(request.passages)},
            "usage": None}


class Runner:
    def __init__(self):
        self.calls = []
        self.result = None

    async def run(self, request):
        self.calls.append(request)
        return deepcopy(self.result) if self.result is not None else result_for(request)


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setenv("RAGS_ENV", "test")
    runner = Runner()
    app = create_app(ResearchSettings("lightrag", {"key-a": "tenant-a", "key-b": "tenant-b"}), runner=runner)
    with TestClient(app) as client:
        yield client, runner


def test_authenticated_tenant_and_engine_before_worker(service):
    client, runner = service
    assert client.get("/live").status_code == 200
    assert client.get("/health").status_code == 401
    health = client.get("/health", headers={"Authorization": "Bearer key-a"}).json()
    assert health["verification"] == "upstream_not_exercised_by_health_probe"
    for credential in ("", "Bearer wrong", "Bearer 非法憑證"):
        # HTTP header bytes are ASCII on the wire; exercise non-ASCII credential
        # handling through a Latin-1 byte representation accepted by the client.
        if not credential.isascii():
            credential = credential.encode().decode("latin-1")
            headers = [(b"authorization", credential.encode("latin-1"))]
        else:
            headers = {"Authorization": credential}
        assert client.post("/retrieve", json=request_body(), headers=headers).status_code == 401
    assert client.post("/retrieve", json=request_body(), headers={"Authorization": "Bearer key-b"}).status_code == 403
    assert client.post("/retrieve", json=request_body(engine="raptor"), headers={"Authorization": "Bearer key-a"}).status_code == 422
    assert not runner.calls
    response = client.post("/retrieve", json=request_body(), headers={"Authorization": "Bearer key-a"})
    assert response.status_code == 200
    assert response.json()["hits"][0]["chunk_id"] == "original-1"
    assert len(runner.calls) == 1


@pytest.mark.parametrize("updates", [
    {"top_k": True}, {"top_k": 21}, {"question": "  "},
    {"options": {"url": "https://attacker.invalid"}},
    {"passages": [{"id": "same", "text": "a"}, {"id": "same", "text": "b"}]},
    {"passages": [{"id": "bad path/../", "text": "sensitive-source-sentinel"}]},
    {"passages": [{"id": str(i), "text": "x"} for i in range(65)]},
    {"passages": [{"id": "a", "text": "x" * 100_001}, {"id": "b", "text": "y" * 100_000}]},
])
def test_invalid_request_never_reaches_worker_or_echoes_source(service, updates):
    client, runner = service
    response = client.post("/retrieve", json=request_body(**updates), headers={"Authorization": "Bearer key-a"})
    assert response.status_code == 422
    assert response.json() == {"detail": {"code": "invalid_request"}}
    assert not runner.calls


def test_request_body_limit_before_parsing_or_work(service):
    client, runner = service
    response = client.post("/retrieve", content=b"x" * (MAX_BODY_BYTES + 1), headers={"Authorization": "Bearer key-a"})
    assert response.status_code == 413
    assert not runner.calls


@pytest.mark.parametrize("change", [
    {"hits": [{"chunk_id": "outside-authorization", "score": 1}]},
    {"hits": [{"chunk_id": "original-1", "score": True}]},
    {"hits": [{"chunk_id": "original-1", "score": float("inf")}]},
    {"hits": [{"chunk_id": "original-1", "score": 10**1000}]},
    {"hits": [{"chunk_id": "original-1", "score": 1}] * 2},
    {"implementation": {**UPSTREAMS["lightrag"], "source_commit": "0" * 40}},
    {"diagnostics": {"source_count": 1, "engine_executed": True, "prompt": "sensitive-source-sentinel"}},
    {"diagnostics": {"source_count": 1, "engine_executed": False}},
    {"diagnostics": {"source_count": 2, "engine_executed": True}},
    {"usage": {"input_tokens": True}},
    {"trace": "sensitive-source-sentinel"},
])
def test_worker_response_is_validated_before_exposure(service, change):
    client, runner = service
    runner.result = result_for(ResearchRequest.model_validate(request_body())) | change
    response = client.post("/retrieve", json=request_body(), headers={"Authorization": "Bearer key-a"})
    assert response.status_code == 502
    assert response.json() == {"detail": {"code": "invalid_worker_output"}}
    assert "sentinel" not in response.text


def test_capacity_rejects_without_queueing_extra_work(monkeypatch):
    monkeypatch.setenv("RAGS_ENV", "test")

    async def exercise():
        started, release = asyncio.Event(), asyncio.Event()

        class SlowRunner:
            calls = 0

            async def run(self, request):
                self.calls += 1
                started.set()
                await release.wait()
                return result_for(request)

        runner = SlowRunner()
        app = create_app(ResearchSettings("lightrag", {"key-a": "tenant-a"}), runner=runner)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            first = asyncio.create_task(client.post("/retrieve", json=request_body(), headers={"Authorization": "Bearer key-a"}))
            await asyncio.wait_for(started.wait(), 2)
            second = await client.post("/retrieve", json=request_body(), headers={"Authorization": "Bearer key-a"})
            assert second.status_code == 429 and second.headers["retry-after"] == "1"
            assert runner.calls == 1
            release.set()
            assert (await first).status_code == 200

    asyncio.run(exercise())


def test_child_environment_does_not_receive_service_keys_or_unrelated_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("RAGS_ENV", "test")
    for key in ("RAGS_KEYS_JSON", "RAGS_RESEARCH_KEYS_JSON", "AGENT_KEYS_JSON", "UNRELATED_SECRET", "RAGS_ADMISSION_PROFILE"):
        monkeypatch.setenv(key, "credential-sentinel")
    monkeypatch.setenv("RAGS_MODEL_CACHE", "relative-cache")
    monkeypatch.setenv("RAGS_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "explicit-generation-credential")
    registry = '{"selected":{"backend":"huggingface","path":"/models/tokenizer.json","sha256":"' + "a" * 64 + '"}}'
    monkeypatch.setenv("RAGS_TOKENIZER_ARTIFACTS_JSON", registry)
    environment = worker_environment(ResearchSettings("raptor", {"key": "a"}, raptor_source=str(tmp_path)))
    assert not any("credential-sentinel" in value for value in environment.values())
    assert environment["OPENAI_API_KEY"] == "explicit-generation-credential"
    assert Path(environment["RAGS_MODEL_CACHE"]).is_absolute()
    assert environment["PYTHONPATH"].split(os.pathsep)[0] == str(tmp_path)
    assert environment["ORT_DISABLE_TELEMETRY"] == "1"
    assert environment["RAGS_ADMISSION_POLICY"] == "compatibility"
    assert environment["RAGS_TOKENIZER_ARTIFACTS_JSON"] == registry


@pytest.mark.integration
@pytest.mark.parametrize("provider", ["sentence_transformers", "openai"])
def test_selected_embedding_contract_survives_actual_worker_settings(monkeypatch, tmp_path, provider):
    from rags.config import Settings
    from rags.research_worker import provider_settings

    for name in tuple(os.environ):
        if name.startswith(("RAGS_", "OPENAI_", "ANTHROPIC_")):
            monkeypatch.delenv(name)
    values = {
        "RAGS_ENV": "test", "RAGS_MODEL_PROVIDER": "openai", "OPENAI_API_KEY": "fixture-generation",
        "RAGS_GENERATION_TOKENIZER_JSON": '{"backend":"unicode"}',
        "RAGS_EMBEDDING_PROVIDER": provider, "RAGS_EMBEDDING_DIMENSIONS": "128",
        "RAGS_EMBEDDING_QUERY_PREFIX": "query: ", "RAGS_EMBEDDING_DOCUMENT_PREFIX": "document: ",
        "RAGS_EMBEDDING_NORMALIZE": "true", "RAGS_EMBEDDING_INPUT_POLICY": "exact",
        "RAGS_EMBEDDING_PROFILES_JSON": json.dumps({"unused": {
            "provider": "openai", "model": "unused", "api_key_env": "UNRELATED_PROVIDER_TOKEN",
        }}),
    }
    if provider == "sentence_transformers":
        values.update(RAGS_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2",
                      RAGS_EMBEDDING_REVISION="a" * 40, RAGS_EMBEDDING_AUTH_MODE="required")
    else:
        values.update(
            RAGS_EMBEDDING_MODEL="custom-exact-encoder", RAGS_EMBEDDING_AUTH_MODE="none",
            RAGS_EMBEDDING_BASE_URL="http://127.0.0.1:9999", RAGS_EMBEDDING_TOKEN_LIMIT="512",
            RAGS_EMBEDDING_SPECIAL_TOKENS="2",
            RAGS_EMBEDDING_TOKENIZER_JSON='{"backend":"huggingface","model":"selected"}',
            RAGS_TOKENIZER_ARTIFACTS_JSON=json.dumps({"selected": {
                "backend": "huggingface", "path": str(tmp_path / "not-loaded.json"), "sha256": "b" * 64,
            }}),
        )
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("UNRELATED_PROVIDER_TOKEN", "must-not-cross-boundary")
    before = Settings.from_env()
    environment = worker_environment(ResearchSettings("lightrag", {"service-key": "tenant"}))
    assert "RAGS_EMBEDDING_PROFILES_JSON" not in environment
    assert "UNRELATED_PROVIDER_TOKEN" not in environment
    monkeypatch.setattr("rags.tokenization.read_tokenizer_artifact",
                        lambda *_: pytest.fail("Worker configuration must not load tokenizer files"))
    with monkeypatch.context() as child:
        child.setattr(os, "environ", dict(environment))
        restored = provider_settings()
    assert restored.embedding_profile == before.embedding_profile
    assert restored.tokenizer_artifacts == before.tokenizer_artifacts
    assert restored.generation_tokenizer_config == before.generation_tokenizer_config


def test_actual_fixed_worker_empty_request_and_missing_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("RAGS_ENV", "test")
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1]))
    monkeypatch.setenv("RAGS_EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("RAGS_MODEL_PROVIDER", "extractive")
    runner = ProcessRunner(ResearchSettings("lightrag", {"key": "tenant-a"}, timeout_seconds=5, work_dir=str(tmp_path)))
    empty = asyncio.run(runner.run(ResearchRequest.model_validate(request_body(passages=[]))))
    assert empty["hits"] == [] and empty["diagnostics"]["engine_executed"] is False
    assert not list(tmp_path.iterdir())
    with pytest.raises(ResearchFailure) as failure:
        asyncio.run(runner.run(ResearchRequest.model_validate(request_body())))
    assert (failure.value.code, failure.value.status) == ("requires_provider", 503)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("behavior,expected", [("timeout", "upstream_timeout"), ("oversize", "invalid_worker_output"), ("invalid", "invalid_worker_output")])
def test_real_process_is_terminated_and_directory_removed(monkeypatch, tmp_path, behavior, expected):
    monkeypatch.setenv("RAGS_ENV", "test")
    real_spawn = asyncio.create_subprocess_exec
    children = []
    scripts = {"timeout": "import time; time.sleep(30)",
               "oversize": "import os; os.write(1, b'x' * 2000000)",
               "invalid": "print('source-secret-must-not-escape')"}

    async def spawn(*args, **kwargs):
        assert args == (sys.executable, "-m", "rags.research_worker")
        assert kwargs["start_new_session"] is True
        if behavior == "oversize":
            kwargs["limit"] = 1024  # Exercise real pipe backpressure during cleanup.
        child = await real_spawn(sys.executable, "-c", scripts[behavior], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    runner = ProcessRunner(ResearchSettings("lightrag", {"key": "a"}, timeout_seconds=0.2, work_dir=str(tmp_path)))
    started = time.monotonic()
    async def bounded_run():
        return await asyncio.wait_for(runner.run(ResearchRequest.model_validate(request_body())), timeout=3)

    with pytest.raises(ResearchFailure) as failure:
        asyncio.run(bounded_run())
    assert failure.value.code == expected
    assert time.monotonic() - started < 3
    assert children and children[0].returncode is not None
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(not Path("/proc").is_dir() or not hasattr(os, "fork"), reason="Linux process-group lifecycle check")
def test_descendants_are_cleaned_even_after_successful_leader_exit(monkeypatch, tmp_path):
    monkeypatch.setenv("RAGS_ENV", "test")
    work = tmp_path / "work"
    work.mkdir()
    pid_file = tmp_path / "descendant.pid"
    request = ResearchRequest.model_validate(request_body(passages=[]))
    payload = json.dumps(result_for(request))
    script = ("import os,time\nfrom pathlib import Path\npid=os.fork()\n"
              "if pid == 0:\n os.close(0)\n os.close(1)\n time.sleep(30)\n os._exit(0)\n"
              f"Path({str(pid_file)!r}).write_text(str(pid))\nprint({payload!r}, flush=True)\n")
    real_spawn = asyncio.create_subprocess_exec

    async def spawn(*args, **kwargs):
        return await real_spawn(sys.executable, "-c", script, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    result = asyncio.run(ProcessRunner(ResearchSettings("lightrag", {"key": "a"}, timeout_seconds=2,
                                                       work_dir=str(work))).run(request))
    assert result["hits"] == []
    pid = int(pid_file.read_text())
    state = Path(f"/proc/{pid}/stat")
    # SIGKILL delivery and orphan reaping are asynchronous to the parent's exit.
    deadline = time.monotonic() + 0.5
    while True:
        try:
            if state.read_text().split()[2] in {"Z", "X"}:
                break
        except (FileNotFoundError, ProcessLookupError):
            break
        assert time.monotonic() < deadline
        time.sleep(0.005)
    assert not list(work.iterdir())


def test_cancellation_cleans_actual_worker(monkeypatch, tmp_path):
    monkeypatch.setenv("RAGS_ENV", "test")
    real_spawn = asyncio.create_subprocess_exec
    children = []

    async def exercise():
        started = asyncio.Event()

        async def spawn(*args, **kwargs):
            child = await real_spawn(sys.executable, "-c", "import time; time.sleep(30)", **kwargs)
            children.append(child)
            started.set()
            return child

        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        runner = ProcessRunner(ResearchSettings("lightrag", {"key": "a"}, timeout_seconds=5, work_dir=str(tmp_path)))
        task = asyncio.create_task(runner.run(ResearchRequest.model_validate(request_body())))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
    assert children and children[0].returncode is not None
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("changes", [{"timeout_seconds": True}, {"timeout_seconds": float("nan")},
                                     {"timeout_seconds": 301}, {"concurrency": True}, {"concurrency": 5},
                                     {"engine": "caller.selected.module"}, {"raptor_source": "relative/source"}])
def test_configuration_limits(monkeypatch, changes):
    monkeypatch.setenv("RAGS_ENV", "test")
    with pytest.raises(ValueError):
        ResearchSettings(**({"engine": "lightrag", "keys": {"key": "tenant"}} | changes))
