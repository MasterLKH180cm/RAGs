"""Multipart HTTP, real PDF extraction and database preservation on rejection."""
import builtins
import io
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from rags.api import create_app
from rags.config import Settings

pytestmark = pytest.mark.integration
AUTH = {"Authorization": "Bearer upload-key"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGS_ENV", "test")
    settings = replace(Settings.from_env(), db_path=str(tmp_path / "uploads.db"),
                       keys={"upload-key": "tenant", "other-key": "other"},
                       embedding_provider="local", model_provider="extractive")
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def pdf_bytes(text=None, *, pages=1):
    """Create a genuine small PDF using the same optional dependency as the API."""
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    for index in range(pages):
        page = writer.add_blank_page(width=300, height=200)
        if text is not None and index == 0:
            page[NameObject("/Resources")] = DictionaryObject({
                NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
            })
            escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            stream = DecodedStreamObject()
            stream.set_data(f"BT /F1 12 Tf 10 100 Td ({escaped}) Tj ET".encode("ascii"))
            page.replace_contents(stream)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


@pytest.mark.parametrize("extension", ["txt", "md", "csv", "json"])
def test_utf8_upload_preserves_source_and_basename_with_tenant_scope(client, extension):
    source = "# 中文政策\n會議紀錄：保留 30 天。\nEmoji: 🛰️"
    response = client.post("/documents/upload", headers=AUTH, files={
        "file": (f"../../policy.{extension}", source.encode("utf-8"), "application/octet-stream"),
    })
    assert response.status_code == 201, response.text
    document_id = response.json()["id"]
    document = client.get(f"/documents/{document_id}", headers=AUTH).json()
    assert document["title"] == f"policy.{extension}"
    assert document["source"] == f"upload:policy.{extension}"
    assert document["body"] == source
    query = client.post("/query", headers=AUTH, json={"question": "Emoji", "mode": "sparse", "generate": False})
    assert query.status_code == 200 and query.json()["citations"]
    citation = query.json()["citations"][0]
    assert citation["text"] == source[citation["start_char"]:citation["end_char"]]
    assert client.get(f"/documents/{document_id}", headers={"Authorization": "Bearer other-key"}).status_code == 404


def test_real_pdf_upload_extracts_and_indexes_text(client):
    content = pdf_bytes("The satellite retention period is 30 days.")
    response = client.post("/documents/upload", headers=AUTH, files={"file": ("manual.PDF", content, "application/pdf")})
    assert response.status_code == 201, response.text
    document = client.get(f"/documents/{response.json()['id']}", headers=AUTH).json()
    assert document["source"] == "upload:manual.PDF"
    assert "satellite retention period is 30 days" in document["body"]
    query = client.post("/query", headers=AUTH, json={"question": "satellite retention", "mode": "sparse", "generate": False})
    assert query.status_code == 200
    assert query.json()["citations"][0]["document_id"] == response.json()["id"]


@pytest.mark.parametrize(("filename", "content", "status", "detail"), [
    ("invalid.txt", b"\xff\xfe", 422, "UTF-8 file required"),
    ("empty.md", b" \n\t", 422, "No extractable text"),
    ("archive.zip", b"PK", 415, "Unsupported file type"),
    ("broken.pdf", b"this is not a PDF", 422, "Invalid PDF"),
    ("long.txt", b"x" * 200_001, 413, "Extracted text exceeds the document limit"),
    ("large.txt", b"x" * 2_000_001, 413, "file is too large"),
], ids=["invalid-utf8", "empty-text", "unsupported-type", "invalid-pdf", "extracted-text-limit", "file-size-limit"])
def test_rejected_uploads_preserve_an_existing_index(client, filename, content, status, detail):
    existing = client.post("/documents", headers=AUTH, json={
        "document_id": "existing", "title": "Existing", "text": "Existing policy evidence.",
    })
    assert existing.status_code == 201
    before = client.get("/documents", headers=AUTH).json()
    response = client.post("/documents/upload", headers=AUTH, files={"file": (filename, content)})
    assert response.status_code == status and response.json()["detail"] == detail
    assert client.get("/documents", headers=AUTH).json() == before
    result = client.post("/query", headers=AUTH, json={"question": "Existing", "mode": "sparse", "generate": False})
    assert {citation["revision"] for citation in result.json()["citations"]} == {existing.json()["revision"]}


@pytest.mark.parametrize(("pages", "text", "detail"), [
    (151, None, "too many pages"),
    (1, None, "No extractable text"),
    (1, "X" * 200_001, "Extracted text exceeds the document limit"),
], ids=["page-limit", "blank-pdf", "extracted-text-limit"])
def test_real_pdf_page_empty_text_and_extraction_limits(client, pages, text, detail):
    response = client.post("/documents/upload", headers=AUTH, files={
        "file": ("bounded.pdf", pdf_bytes(text, pages=pages), "application/pdf"),
    })
    assert response.status_code == (422 if text is None and pages == 1 else 413)
    assert response.json()["detail"] == detail
    assert client.get("/documents", headers=AUTH).json() == []


def test_upload_authentication_precedes_pdf_extraction_provider_and_storage_work(client, monkeypatch):
    import pypdf

    def forbidden(*_, **__):
        pytest.fail("Unauthorized uploads must not extract, embed or ingest")

    monkeypatch.setattr(pypdf, "PdfReader", forbidden)
    monkeypatch.setattr(client.app.state.service.provider, "embedding", forbidden)
    monkeypatch.setattr(client.app.state.service, "ingest", forbidden)
    response = client.post("/documents/upload", files={"file": ("private.pdf", b"private PDF body")})
    assert response.status_code == 401
    assert client.get("/documents", headers=AUTH).json() == []


def test_pdf_extra_unavailable_is_an_explicit_capability_response(client, monkeypatch):
    original_import = builtins.__import__

    def import_module(name, *args, **kwargs):
        if name == "pypdf":
            raise ImportError("fixture missing optional PDF dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_module)
    response = client.post("/documents/upload", headers=AUTH, files={"file": ("manual.pdf", b"pdf")})
    assert response.status_code == 501 and response.json()["detail"] == "Install the pdf extra"
    assert client.get("/documents", headers=AUTH).json() == []
