"""FastAPI service with tenant-scoped key authentication and bounded inputs."""
import hmac
import io
import os
import re
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from rags.config import Settings
from rags.providers import Providers
from rags.retrieval import MODES, RAGService, now
from rags.storage import Store


class DocumentIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=200_000)
    source: str = Field(default="manual", max_length=500)
    metadata: dict = Field(default_factory=dict)
    document_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_.:-]{1,128}$")


class QueryIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=2, max_length=2000)
    mode: Literal[
        "sparse", "dense", "hybrid", "fusion", "hyde", "contextual",
        "hierarchical", "graph", "corrective", "adaptive"
    ] = "hybrid"
    top_k: int = Field(default=5, ge=1, le=20)
    generate: bool = True


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or Settings.from_env()
    app = FastAPI(title="RAGs", version="0.1.0")
    store = Store(config.db_path)
    service = RAGService(store, Providers(config))
    app.state.service = service

    def auth(authorization: Annotated[str | None, Header()] = None) -> str:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Bearer token required")
        supplied = authorization[7:]
        for key, tenant in config.keys.items():
            if hmac.compare_digest(supplied, key):
                return tenant
        raise HTTPException(status_code=401, detail="Invalid API key")

    @app.get("/live")
    def live() -> dict:
        return {"status": "ok"}

    @app.get("/health")
    def health(tenant: Annotated[str, Depends(auth)]) -> dict:
        return {"status": "ok", "tenant": tenant,
                "embedding_model": service.provider.embedding_model}

    @app.get("/strategies")
    def strategies(tenant: Annotated[str, Depends(auth)]) -> dict:
        return {"strategies": MODES, "tenant": tenant}

    @app.post("/documents", status_code=201)
    def ingest(data: DocumentIn, tenant: Annotated[str, Depends(auth)]) -> dict:
        if len(str(data.metadata)) > 16_000:
            raise HTTPException(status_code=413, detail="metadata too large")
        try:
            return service.ingest(
                tenant, data.title, data.text, data.source,
                data.metadata, data.document_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/documents/upload", status_code=201)
    async def upload(
        tenant: Annotated[str, Depends(auth)],
        file: UploadFile = File(...),
    ) -> dict:
        # Uploads are explicitly capped; file formats are allowlisted, no URL fetching.
        raw = await file.read(2_000_001)
        if len(raw) > 2_000_000:
            raise HTTPException(status_code=413, detail="file is too large")
        filename = os.path.basename(file.filename or "upload")
        extension = filename.lower().rsplit(".", 1)[-1]
        if extension in {"txt", "md", "csv", "json"}:
            try:
                body = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise HTTPException(status_code=422, detail="UTF-8 file required") from exc
        elif extension == "pdf":
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise HTTPException(status_code=501, detail="Install the pdf extra") from exc
            try:
                reader = PdfReader(io.BytesIO(raw), strict=False)
                if len(reader.pages) > 150:
                    raise HTTPException(status_code=413, detail="too many pages")
                body = "\n".join(p.extract_text() or "" for p in reader.pages)
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(status_code=422, detail="Invalid PDF") from exc
        else:
            raise HTTPException(status_code=415, detail="Unsupported file type")
        if not body.strip():
            raise HTTPException(status_code=422, detail="No extractable text")
        return service.ingest(tenant, filename, body[:200_000], "upload:" + filename)

    @app.get("/documents")
    def documents(tenant: Annotated[str, Depends(auth)]) -> list[dict]:
        return store.list_documents(tenant)

    @app.get("/documents/{document_id}")
    def document(document_id: str, tenant: Annotated[str, Depends(auth)]) -> dict:
        row = store.get_document(tenant, document_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return row

    @app.delete("/documents/{document_id}")
    def delete(document_id: str, tenant: Annotated[str, Depends(auth)]) -> dict:
        if not store.delete(tenant, document_id, now()):
            raise HTTPException(status_code=404, detail="Document not found")
        return {"deleted": True, "id": document_id}

    @app.post("/query")
    def query(data: QueryIn, tenant: Annotated[str, Depends(auth)]) -> dict:
        try:
            return service.query(
                tenant, data.question, data.mode, data.top_k, data.generate
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = create_app()
