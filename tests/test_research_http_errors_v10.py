"""Pinned research failures preserve safe semantics across the parent API."""
import asyncio
import json
from pathlib import Path
import time

from fastapi.testclient import TestClient
import httpx
import pytest

from rags.api import create_app
from rags.backends import BackendManager, BackendUnavailable
from rags.config import Settings
from rags.registry import CapabilityError
from rags.research_protocol import ResearchRequest
from rags.research_service import ProcessRunner, ResearchFailure, ResearchSettings, create_app as create_research_app


pytestmark = pytest.mark.integration
HTTPX_CLIENT = httpx.AsyncClient
PRIVATE_REASON = "private-source-sentinel: provider-secret-token /operator/private/model"
OPERATIONAL_FAILURES = [
    (422, "context_budget_exceeded", "Research model input exceeds the configured context budget"),
    (422, "research_limit_exceeded", "Research request exceeds a supported execution limit"),
    (422, "insufficient_graph_evidence", "Research graph has insufficient source evidence"),
    (504, "upstream_timeout", "Research execution exceeded its configured deadline"),
    (504, "external_source_timeout", "External research source acquisition exceeded its deadline"),
    (429, "research_capacity", "Research service capacity is exhausted"),
    (403, "feature_disabled", "A required research feature is disabled"),
    (403, "external_source_forbidden", "External research source acquisition is not authorized"),
    (503, "requires_provider", "Research execution requires a configured model provider"),
    (503, "requires_dependency", "Research execution requires an installed dependency"),
    (503, "requires_source", "Research execution requires the pinned upstream source"),
    (503, "requires_external_source", "Research execution requires an available external source"),
    (503, "requires_search_credentials", "Research execution requires configured search credentials"),
    (502, "upstream_failed", "Research execution failed"),
    (502, "invalid_worker_output", "Research execution returned invalid output"),
    (422, "invalid_research_input", "Research input contains invalid text or reserved model control tokens"),
]


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setenv("RAGS_ENV", "test")
    monkeypatch.delenv("RAGS_LATE_INTERACTION_MODEL", raising=False)

    def configured(**updates):
        config = {
            "url": "https://research.example/retrieve",
            "tenant_tokens": {"tenant-a": "research-key-a"},
            "require_pinned_upstream": True,
            **updates,
        }
        monkeypatch.setenv("RAGS_RESEARCH_BACKENDS_JSON", json.dumps({"full_graphrag": config}))
        return BackendManager()

    return configured


def transport(monkeypatch, handler):
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: HTTPX_CLIENT(
        transport=httpx.MockTransport(handler), **kwargs,
    ))


def search(manager):
    return manager.search("full_graphrag", tenant="tenant-a", question="Where is the office?",
                          chunks=[{"id": "authorized", "content": "The office is in Taipei."}], top_k=1)


@pytest.mark.parametrize("status,code,reason", OPERATIONAL_FAILURES)
def test_pinned_operational_errors_use_exact_status_and_local_reason(backend, monkeypatch, status, code, reason):
    transport(monkeypatch, lambda request: httpx.Response(status, json={
        "detail": {"code": code, "reason": PRIVATE_REASON},
    }))
    manager = backend()
    with pytest.raises(CapabilityError) as failure:
        search(manager)
    assert failure.value.code == code
    assert failure.value.status_code == status
    assert failure.value.reason == reason
    assert PRIVATE_REASON not in str(failure.value)
    # An error envelope does not prove the algorithm or its source identity ran.
    entry = next(item for item in manager.catalog("tenant-a") if item["id"] == "full_graphrag")
    assert entry["evidence_level"] == "external_unverified"


@pytest.mark.parametrize("status,body", [
    (500, {"detail": {"code": "context_budget_exceeded"}}),
    (422, {"detail": {"code": "upstream_timeout"}}),
    (502, {"detail": {"code": "invalid_research_input"}}),
    (422, {"detail": {"code": "private_unknown_code"}}),
    (401, {"detail": {"code": "invalid_credentials"}}),
    (403, {"detail": {"code": "tenant_mismatch"}}),
    (422, {"detail": {"code": "engine_mismatch"}}),
    (422, {"detail": {"code": "invalid_request"}}),
    (422, {"detail": {"code": "context_budget_exceeded", "trace": PRIVATE_REASON}}),
    (422, {"detail": {"code": "context_budget_exceeded", "reason": {"text": PRIVATE_REASON}}}),
    (422, {"detail": {"code": "context_budget_exceeded"}, "trace": PRIVATE_REASON}),
    (422, {"detail": {"code": ["context_budget_exceeded"]}}),
    (422, {"detail": {"code": True}}),
    (422, {"detail": {"reason": PRIVATE_REASON}}),
    (422, {"detail": []}),
    (422, {}),
    (422, ["context_budget_exceeded"]),
    (422, None),
])
def test_unknown_mismatched_or_invalid_errors_remain_sanitized(backend, monkeypatch, status, body):
    transport(monkeypatch, lambda request: httpx.Response(status, content=json.dumps(body)))
    with pytest.raises(BackendUnavailable) as failure:
        search(backend())
    assert str(failure.value) == "Research backend failed protocol or evidence validation"


@pytest.mark.parametrize("body", [
    b"private-source-sentinel invalid JSON",
    b'{"detail":{"code":"context_budget_exceeded","code":"research_limit_exceeded"}}',
    b'{"detail":{"code":"context_budget_exceeded"},"detail":{"code":"context_budget_exceeded"}}',
    b'{"detail":{"code":"context_budget_exceeded","reason":NaN}}',
    b'{"detail":{"code":"context_budget_exceeded","reason":"\xff"}}',
    b"[" * 2000 + b"]" * 2000,
])
def test_malformed_error_json_cannot_gain_typed_error_semantics(backend, monkeypatch, body):
    transport(monkeypatch, lambda request: httpx.Response(422, content=body))
    with pytest.raises(BackendUnavailable, match="failed protocol or evidence validation"):
        search(backend())


def test_unpinned_error_remains_unavailable_without_consuming_its_body(backend, monkeypatch):
    consumed = []

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            consumed.append(True)
            yield b'{"detail":{"code":"context_budget_exceeded"}}'

    transport(monkeypatch, lambda request: httpx.Response(422, stream=Stream()))
    with pytest.raises(BackendUnavailable, match="failed protocol or evidence validation"):
        search(backend(require_pinned_upstream=False))
    assert consumed == []


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_redirect_never_forwards_tenant_credentials_or_becomes_typed_error(backend, monkeypatch, status):
    calls = []

    def handler(request):
        calls.append((str(request.url), request.headers["authorization"]))
        return httpx.Response(status, headers={"location": "https://untrusted.example/retrieve"},
                              json={"detail": {"code": "context_budget_exceeded"}})

    transport(monkeypatch, handler)
    with pytest.raises(BackendUnavailable, match="failed protocol or evidence validation"):
        search(backend())
    assert calls == [("https://research.example/retrieve", "Bearer research-key-a")]


@pytest.mark.parametrize("headers,expected", [
    ({"content-encoding": "gzip"}, "Compressed"),
    ({"content-length": "1000001"}, "exceeds limit"),
    ({"content-length": "invalid"}, "failed protocol or evidence validation"),
])
def test_error_headers_are_validated_before_body_or_code(backend, monkeypatch, headers, expected):
    state = {"consumed": False, "closed": False}

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            state["consumed"] = True
            yield b'{"detail":{"code":"context_budget_exceeded"}}'

        async def aclose(self):
            state["closed"] = True

    transport(monkeypatch, lambda request: httpx.Response(422, headers=headers, stream=Stream()))
    with pytest.raises(BackendUnavailable, match=expected):
        search(backend())
    assert state == {"consumed": False, "closed": True}


def test_error_body_byte_limit_precedes_typed_error_and_closes_stream(backend, monkeypatch):
    state = {"parts": 0, "closed": False}

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"detail":{"code":"context_budget_exceeded","reason":"'
            for _ in range(100):
                state["parts"] += 1
                yield b"x" * 100_000
            yield b'"}}'

        async def aclose(self):
            state["closed"] = True

    transport(monkeypatch, lambda request: httpx.Response(422, stream=Stream()))
    with pytest.raises(BackendUnavailable, match="exceeds limit"):
        search(backend())
    assert state["parts"] == 10
    assert state["closed"]


def test_complete_error_body_is_required_within_one_wall_deadline(backend, monkeypatch):
    state = {"closed": False}

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            # Even a complete valid prefix cannot be accepted before EOF.
            yield b'{"detail":{"code":"context_budget_exceeded"}}'
            await asyncio.sleep(5)
            yield b" "

        async def aclose(self):
            state["closed"] = True

    async def handler(request):
        await asyncio.sleep(0.3)
        return httpx.Response(422, stream=Stream())

    transport(monkeypatch, handler)
    started = time.monotonic()
    with pytest.raises(BackendUnavailable, match="deadline exceeded"):
        search(backend(timeout_seconds=1))
    assert time.monotonic() - started < 1.5
    assert state["closed"]


@pytest.fixture
def parent_client(monkeypatch, tmp_path):
    monkeypatch.setenv("RAGS_ENV", "test")
    monkeypatch.setenv("RAGS_DB_PATH", str(tmp_path / "parent.sqlite"))
    monkeypatch.setenv("RAGS_KEYS_JSON", '{"parent-key":"tenant-a"}')
    monkeypatch.setenv("RAGS_MODEL_PROVIDER", "extractive")
    monkeypatch.setenv("RAGS_EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("RAGS_RESEARCH_BACKENDS_JSON", json.dumps({"full_graphrag": {
        "url": "https://research.example/retrieve", "tenant_tokens": {"tenant-a": "research-key-a"},
        "require_pinned_upstream": True,
    }}))
    with TestClient(create_app(Settings.from_env()), headers={"Authorization": "Bearer parent-key"}) as client:
        response = client.post("/documents", json={
            "document_id": "office", "title": "Office", "text": "The office is in Taipei.",
        })
        assert response.status_code == 201
        yield client


@pytest.mark.parametrize("status,code,reason", [OPERATIONAL_FAILURES[0], OPERATIONAL_FAILURES[1],
                                               OPERATIONAL_FAILURES[3], OPERATIONAL_FAILURES[4]])
def test_sidecar_to_parent_api_preserves_budget_and_timeout_status(
    parent_client, monkeypatch, status, code, reason,
):
    calls = []

    class FailingRunner:
        async def run(self, request):
            calls.append(request)
            raise ResearchFailure(code, status)

    sidecar = create_research_app(
        ResearchSettings("full_graphrag", {"research-key-a": "tenant-a"}), runner=FailingRunner(),
    )
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: HTTPX_CLIENT(
        transport=httpx.ASGITransport(app=sidecar), **kwargs,
    ))
    response = parent_client.post("/query", json={"question": "Where is the office?", "mode": "full_graphrag"})
    assert response.status_code == status
    assert response.json() == {"detail": {"code": code, "reason": reason}}
    assert len(calls) == 1
    assert calls[0].namespace == "tenant-a"
    assert [passage.text for passage in calls[0].passages] == ["The office is in Taipei."]


@pytest.mark.parametrize("status,code", [(422, "private_unknown_code"), (504, "context_budget_exceeded")])
def test_parent_api_never_exposes_unknown_or_mismatched_backend_reason(parent_client, monkeypatch, status, code):
    transport(monkeypatch, lambda request: httpx.Response(status, json={
        "detail": {"code": code, "reason": PRIVATE_REASON},
    }))
    response = parent_client.post("/query", json={"question": "Where is the office?", "mode": "full_graphrag"})
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "requires_backend"
    assert "private" not in response.text and "operator" not in response.text


def test_parent_api_replaces_known_backend_reason_with_fixed_text(parent_client, monkeypatch):
    transport(monkeypatch, lambda request: httpx.Response(422, json={
        "detail": {"code": "context_budget_exceeded", "reason": PRIVATE_REASON},
    }))
    response = parent_client.post("/query", json={"question": "Where is the office?", "mode": "full_graphrag"})
    assert response.status_code == 422
    assert response.json() == {"detail": {
        "code": "context_budget_exceeded", "reason": "Research model input exceeds the configured context budget",
    }}


@pytest.fixture
def selfrag_runner(monkeypatch, tmp_path):
    monkeypatch.setenv("RAGS_ENV", "test")
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1]))
    # Input rejection must precede model/runtime loading. This integration gate
    # deliberately supplies nonexistent artifacts and needs no trained model.
    monkeypatch.setenv("RAGS_SELFRAG_MODEL_PATH", str(tmp_path / "missing-model.gguf"))
    monkeypatch.setenv("RAGS_SELFRAG_LLAMA_DIR", str(tmp_path / "missing-native-runtime"))
    return ProcessRunner(ResearchSettings(
        "self_rag", {"research-key-a": "tenant-a"}, timeout_seconds=5,
        work_dir=str(tmp_path / "worker-requests"),
    ))


@pytest.mark.parametrize("field", ["question", "text"])
def test_actual_fixed_worker_preserves_invalid_research_input_before_model_loading(selfrag_runner, field):
    question, text = "Where is the office?", "The office is in Taipei."
    invalid = "[Retrieval] " + PRIVATE_REASON
    request = ResearchRequest.model_validate({
        "protocol": "authorized-evidence-v2", "engine": "self_rag", "namespace": "tenant-a", "top_k": 1,
        "question": invalid if field == "question" else question,
        "passages": [{"id": "office", "text": invalid if field == "text" else text}],
    })
    with pytest.raises(ResearchFailure) as failure:
        asyncio.run(selfrag_runner.run(request))
    assert (failure.value.code, failure.value.status) == ("invalid_research_input", 422)
    assert str(failure.value) == "invalid_research_input"
    assert not list(Path(selfrag_runner.config.work_dir).iterdir())


@pytest.mark.parametrize("field", ["question", "text"])
def test_actual_fixed_worker_invalid_input_reaches_parent_api_as_sanitized_422(
    parent_client, selfrag_runner, monkeypatch, field,
):
    monkeypatch.setenv("RAGS_RESEARCH_BACKENDS_JSON", json.dumps({"self_rag": {
        "url": "https://research.example/retrieve", "tenant_tokens": {"tenant-a": "research-key-a"},
        "require_pinned_upstream": True,
    }}))
    sidecar = create_research_app(selfrag_runner.config, runner=selfrag_runner)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: HTTPX_CLIENT(
        transport=httpx.ASGITransport(app=sidecar), **kwargs,
    ))
    invalid = "[Retrieval] " + PRIVATE_REASON
    if field == "text":
        updated = parent_client.post("/documents", json={
            "document_id": "office", "title": "Office", "text": invalid,
        })
        assert updated.status_code == 201
    response = parent_client.post("/query", json={
        "question": invalid if field == "question" else "Where is the office?",
        "mode": "self_rag", "top_k": 1,
    })
    assert response.status_code == 422
    assert response.json() == {"detail": {
        "code": "invalid_research_input",
        "reason": "Research input contains invalid text or reserved model control tokens",
    }}
    assert not list(Path(selfrag_runner.config.work_dir).iterdir())
