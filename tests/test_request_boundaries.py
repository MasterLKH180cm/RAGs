"""Request rejection stays serializable, private and free of index mutations."""
import io
import json
from dataclasses import replace
from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient

from rags import api
from rags.config import Settings
from rags.registry import CapabilityError

AUTH = {"Authorization": "Bearer boundary-key"}
PRIVATE_INPUT = "private-request-value"


def source(**changes):
    return {"document_id": "policy", "title": "Policy", "text": "Updated retention evidence.", **changes}


@pytest.fixture
def system(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGS_ENV", "test")
    settings = replace(
        Settings.from_env(), db_path=str(tmp_path / "boundaries.db"),
        keys={"boundary-key": "tenant", "other-key": "other"},
        embedding_provider="local", model_provider="extractive",
        tokenizer_config={"backend": "unicode"}, generation_tokenizer_config={"backend": "unicode"},
        chunking_config={"strategy": "fixed", "max_tokens": 64, "overlap_tokens": 0},
    )
    app = api.create_app(settings)
    with TestClient(app, raise_server_exceptions=False) as client:
        first = client.post("/documents", headers=AUTH, json=source(text="The retention period is thirty days."))
        assert first.status_code == 201
        foreign = client.post("/documents", headers={"Authorization": "Bearer other-key"},
                              json=source(text="Other tenant private evidence."))
        assert foreign.status_code == 201
        yield client, app.state.service, first.json(), settings


def snapshot(store):
    with store.connect() as connection:
        return {table: [tuple(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY rowid")]
                for table in ("documents", "document_revisions", "chunks", "chunks_fts", "graph_relations", "audit")}


@pytest.mark.unit
def test_validation_details_escape_only_invalid_unicode_and_omit_input_and_context():
    errors = [{"type": "custom_錯誤\ud800", "loc": ("body", "中文👩‍🚀\udfff", 2),
               "msg": "中文 café cafe\u0301 👩‍🚀\ud800", "input": float("inf"),
               "ctx": {"error": ValueError(PRIVATE_INPUT), "credential": "private-credential"}}]
    result = api.validation_details(errors)
    assert result == [{"type": "custom_錯誤\\ud800", "loc": ["body", "中文👩‍🚀\\udfff", 2],
                       "msg": "中文 café cafe\u0301 👩‍🚀\\ud800"}]
    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")
    assert "中文 café cafe\u0301 👩‍🚀".encode() in encoded
    assert b"private" not in encoded


@pytest.mark.integration
@pytest.mark.parametrize(("path", "body", "expected_location"), [
    ("/documents", ('{"document_id":"policy","title":"Policy","text":"Updated retention evidence",'
                    '"metadata":{"amount":1e309,"private":"private-request-value"}}'), ["body"]),
    ("/query", ('{"question":"amount","mode":"structured","row_filters":'
                '[{"field":"private-request-value","op":"gt","value":1e309}]}'), ["body", "row_filters", 0]),
    ("/documents", json.dumps(source(text="\ud800")), ["body", "text"]),
    ("/documents", json.dumps(source(title="\udfff", text=PRIVATE_INPUT)), ["body", "title"]),
    ("/documents", json.dumps(source(document_id="\ud800", text=PRIVATE_INPUT)), ["body", "document_id"]),
    ("/documents", json.dumps({**source(), "未知\ud800": PRIVATE_INPUT}), ["body"]),
    ("/documents", json.dumps(source(metadata={"nested": {"label": "\ud800", "private": PRIVATE_INPUT}})), ["body"]),
    ("/documents", json.dumps(source(metadata={"nested": {"bad\udfff": PRIVATE_INPUT}})), ["body"]),
    ("/query", json.dumps({"question": "retention", "filters": {"metadata": {"label": "\ud800", "private": PRIVATE_INPUT}}}),
     ["body", "filters"]),
    ("/documents", json.dumps(source(title=[PRIVATE_INPUT])), ["body", "title"]),
    ("/documents/policy/reindex", json.dumps({"expected_revision": "\ud800"}), ["body", "expected_revision"]),
    ("/documents", '{"title":"private-request-value","text":]}', None),
], ids=["metadata-exponent-overflow", "predicate-exponent-overflow", "source-surrogate", "title-surrogate",
        "id-surrogate", "error-path-surrogate", "nested-metadata-value", "nested-metadata-key",
        "filter-metadata-surrogate", "ordinary-field-error", "reindex-surrogate", "invalid-json"])
def test_invalid_requests_return_private_serializable_422_before_inference_or_writes(
    system, monkeypatch, path, body, expected_location,
):
    client, service, initial, _ = system
    before = snapshot(service.store)
    embedding = Mock(wraps=service.provider.embedding)
    generation = Mock(wraps=service.provider.complete)
    monkeypatch.setattr(service.provider, "embedding", embedding)
    monkeypatch.setattr(service.provider, "complete", generation)
    response = client.post(path, headers={**AUTH, "Content-Type": "application/json"}, content=body)
    assert response.status_code == 422, response.text
    details = response.json()["detail"]
    assert details and all(set(item) == {"type", "loc", "msg"} for item in details)
    assert all(isinstance(item["type"], str) and isinstance(item["msg"], str) for item in details)
    assert all(item["loc"][0] == "body" for item in details)
    if expected_location is not None:
        assert details[0]["loc"] == expected_location
    assert PRIVATE_INPUT not in response.text and "boundary-key" not in response.text
    json.dumps(details, ensure_ascii=False, allow_nan=False).encode("utf-8")
    assert embedding.call_count == generation.call_count == 0
    assert snapshot(service.store) == before
    assert client.get("/documents/policy", headers=AUTH).json()["revision"] == initial["revision"]
    assert len(client.get("/documents", headers=AUTH).json()) == 1


@pytest.mark.integration
def test_valid_unicode_ingestion_keeps_text_metadata_and_citation_offsets(system):
    client, _, _, _ = system
    text = "# 中文標題 🛰️\nCafé cafe\u0301，航程 30 天。\nEvidence: 星際導航 👩‍🚀 保留原文。"
    metadata = {"nested": {"label": "中文 café cafe\u0301 👩‍🚀"}}
    response = client.post("/documents", headers=AUTH, json={
        "document_id": "unicode", "title": "中文標題 🛰️", "source": "來源:中文👩‍🚀",
        "text": text, "metadata": metadata,
    })
    assert response.status_code == 201, response.text
    stored = client.get("/documents/unicode", headers=AUTH).json()
    assert stored["body"] == text and stored["metadata"] == metadata
    result = client.post("/query", headers=AUTH, json={
        "question": "Evidence", "mode": "sparse", "generate": False, "filters": {"source": "來源:中文👩‍🚀"},
    })
    assert result.status_code == 200 and result.json()["citations"]
    for citation in result.json()["citations"]:
        assert citation["document_id"] == "unicode"
        assert citation["text"] == text[citation["start_char"]:citation["end_char"]]
    assert client.get("/documents/unicode", headers={"Authorization": "Bearer other-key"}).status_code == 404


@pytest.mark.integration
def test_actual_upload_chunk_cap_matches_json_ingestion_and_preserves_old_evidence(system, monkeypatch):
    client, service, initial, _ = system
    before = snapshot(service.store)
    embedding = Mock(wraps=service.provider.embedding)
    monkeypatch.setattr(service.provider, "embedding", embedding)
    text = "界" * 65_537  # At 64 code points per chunk, the real plan needs 1025 records.
    upload = client.post("/documents/upload", headers=AUTH,
                         files={"file": ("notes.txt", text.encode("utf-8"), "text/plain")})
    ordinary = client.post("/documents", headers=AUTH, json=source(text=text))
    assert upload.status_code == ordinary.status_code == 400
    assert upload.json() == ordinary.json() == {"detail": "Chunk configuration produces more than 1024 index records"}
    assert embedding.call_count == 0
    assert snapshot(service.store) == before
    result = client.post("/query", headers=AUTH, json={"question": "retention", "mode": "sparse", "generate": False})
    assert result.status_code == 200 and result.json()["citations"]
    assert {citation["revision"] for citation in result.json()["citations"]} == {initial["revision"]}


@pytest.mark.integration
@pytest.mark.parametrize("path", ["/documents", "/documents/upload", "/images"])
def test_ingestion_provider_errors_keep_their_typed_status(system, monkeypatch, path):
    client, service, _, settings = system
    if path == "/images":
        from PIL import Image

        from rags import providers

        # Actual image parsing and a controlled valid caption reach the same ingestion handoff.
        image = io.BytesIO()
        Image.new("RGB", (2, 2)).save(image, format="PNG")
        settings = replace(settings, model_provider="openai", openai_key="fixture-key",
                           openai_base_url="https://caption-fixture.invalid/v1")
        app = api.create_app(settings)
        service = app.state.service
        monkeypatch.setattr(providers.provider_transport, "post", lambda url, **_: httpx.Response(
            200, request=httpx.Request("POST", url), json={"choices": [{"message": {"content": "Image description."}}]},
        ))
        request = {"files": {"file": ("image.png", image.getvalue(), "image/png")}}
    elif path == "/documents/upload":
        request = {"files": {"file": ("notes.txt", b"New source evidence.", "text/plain")}}
    else:
        request = {"json": source(text="New source evidence.")}

    def invalid_embedding(_):
        raise CapabilityError("invalid_embedding", "Provider returned an invalid embedding vector", 502)

    monkeypatch.setattr(service.provider, "embedding", invalid_embedding)
    before = snapshot(service.store)
    if path == "/images":
        with TestClient(app, raise_server_exceptions=False) as image_client:
            response = image_client.post(path, headers=AUTH, **request)
    else:
        response = client.post(path, headers=AUTH, **request)
    assert response.status_code == 502, response.text
    assert response.json()["detail"]["code"] == "invalid_embedding"
    assert snapshot(service.store) == before


@pytest.mark.integration
def test_unexpected_ingestion_failure_is_not_reclassified_as_client_validation(system, monkeypatch):
    client, service, _, _ = system

    def fail(*_, **__):
        raise RuntimeError("private unexpected server failure")

    monkeypatch.setattr(service, "ingest", fail)
    response = client.post("/documents/upload", headers=AUTH, files={"file": ("notes.txt", b"Source evidence.")})
    assert response.status_code == 500
    assert "private unexpected" not in response.text
