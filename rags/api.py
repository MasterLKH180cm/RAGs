"""FastAPI service with tenant-scoped key authentication and bounded inputs."""
import hmac
import io
import json
import math
import os
from typing import Annotated, Literal

import httpx
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rags.chunking import ChunkConfig
from rags.config import Settings
from rags.limits import BodyLimitMiddleware
from rags.multimodal import install_multimodal_routes
from rags.providers import Providers
from rags.registry import CapabilityError
from rags.retrieval import MODES, RAGService, now
from rags.storage import Store, timestamp
from rags.tokenization import TokenizerConfig


def validation_details(errors: list[dict]) -> list[dict]:
    """Keep useful validation paths/messages without echoing unsafe request values."""
    def safe_text(value: str) -> str:
        return value.encode("utf-8", "backslashreplace").decode("utf-8")

    return [{"type": safe_text(error["type"]),
             "loc": [safe_text(part) if isinstance(part, str) else part for part in error["loc"]],
             "msg": safe_text(error["msg"])} for error in errors]


class DocumentIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=200)
    text: str | None = Field(default=None, min_length=1, max_length=200_000)
    source: str = Field(default="manual", max_length=500)
    metadata: dict = Field(default_factory=dict)
    document_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_.:-]{1,128}$")
    chunking: dict | None = None
    tokenizer: dict | None = None
    index_features: list[Literal["hierarchy", "graph"]] = Field(default_factory=list, max_length=2)
    rows: list[dict] | None = Field(default=None, min_length=1, max_length=1000)
    valid_from: str | None = None
    valid_to: str | None = None
    expected_revision: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_document(self):
        if (self.text is None) == (self.rows is None):
            raise ValueError("Provide exactly one of text or structured rows")
        validate_processing(self.chunking, self.tokenizer)
        validate_validity(self.valid_from, self.valid_to)
        json.dumps(self.metadata, ensure_ascii=False, allow_nan=False).encode("utf-8")
        return self


def validate_processing(chunking: dict | None, tokenizer: dict | None) -> None:
    try:
        if chunking is not None:
            ChunkConfig(**chunking)
        if tokenizer is not None:
            TokenizerConfig(**tokenizer)
    except TypeError as exc:
        raise ValueError("Unknown processing configuration field") from exc


def validate_validity(start: str | None, end: str | None) -> None:
    if start:
        start = timestamp(start)
    if end:
        end = timestamp(end)
    if start and end and end <= start:
        raise ValueError("valid_to must be later than valid_from")


class QueryFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str | None = Field(default=None, max_length=500)
    metadata: dict = Field(default_factory=dict)
    valid_after: str | None = None
    valid_before: str | None = None

    @model_validator(mode="after")
    def validate_filter(self):
        if len(self.metadata) > 20 or any(not isinstance(k, str) or len(k) > 128 or type(v) not in (str, int, float, bool, type(None)) for k, v in self.metadata.items()):
            raise ValueError("Metadata filters require at most20 scalar equality fields")
        validate_validity(self.valid_after, self.valid_before)
        json.dumps(self.metadata, ensure_ascii=False, allow_nan=False).encode("utf-8")
        return self


class RowPredicate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str = Field(min_length=1, max_length=128)
    op: Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "contains"]
    value: str | float | int | bool | list | None = None

    @model_validator(mode="after")
    def validate_predicate(self):
        if isinstance(self.value, list) and (len(self.value) > 100 or any(type(v) not in (str, int, float, bool, type(None)) for v in self.value)):
            raise ValueError("Predicate list must contain at most100 scalar values")
        if isinstance(self.value, list) and any(type(v) is float and not math.isfinite(v) for v in self.value):
            raise ValueError("Predicate list numbers must be finite")
        if self.op == "in" and not isinstance(self.value, list):
            raise ValueError("The in predicate requires a list")
        if type(self.value) is float and not math.isfinite(self.value):
            raise ValueError("Predicate numbers must be finite")
        return self


class Aggregation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation: Literal["count", "sum", "mean", "min", "max"]
    field: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_aggregate(self):
        if self.operation != "count" and self.field is None:
            raise ValueError("Numeric aggregation requires a field")
        return self


class ReindexIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: str = Field(min_length=1, max_length=128)
    chunking: dict | None = None
    tokenizer: dict | None = None
    index_features: list[Literal["hierarchy", "graph"]] | None = Field(default=None, max_length=2)

    @model_validator(mode="after")
    def validate_reindex(self):
        validate_processing(self.chunking, self.tokenizer)
        return self


class QueryIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=2, max_length=2000)
    mode: str = "hybrid"
    top_k: int = Field(default=5, ge=1, le=20)
    generate: bool = True
    filters: QueryFilter | None = None
    sources: list[str] | None = Field(default=None, max_length=8)
    as_of: str | None = None
    max_iterations: int | None = Field(default=None, ge=1, le=3)
    row_filters: list[RowPredicate] = Field(default_factory=list, max_length=20)
    aggregate: Aggregation | None = None

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, value):
        if value not in MODES:
            raise ValueError("Unknown retrieval strategy")
        return value

    @field_validator("as_of")
    @classmethod
    def validate_as_of(cls, value):
        return timestamp(value) if value else value

    @model_validator(mode="after")
    def validate_options(self):
        if (self.row_filters or self.aggregate) and self.mode != "structured":
            raise ValueError("Row predicates and aggregation require structured mode")
        if self.sources is not None and self.mode != "federated":
            raise ValueError("Source selection requires federated mode")
        return self


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or Settings.from_env()
    app = FastAPI(title="RAGs", version="0.3.0")
    app.add_middleware(BodyLimitMiddleware, max_body_bytes=2_100_000)
    store = Store(config.db_path)
    service = RAGService(store, Providers(config))
    app.state.service = service

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse(status_code=422, content={"detail": validation_details(exc.errors())})

    @app.exception_handler(CapabilityError)
    async def capability_error(request, exc):
        return JSONResponse(status_code=exc.status_code, content={"detail": {"code": exc.code, "reason": exc.reason}})

    @app.exception_handler(httpx.HTTPError)
    async def provider_error(request, exc):
        return JSONResponse(status_code=502, content={"detail": {"code": "provider_failed", "reason": "Configured provider request failed"}})

    def auth(authorization: Annotated[str | None, Header()] = None) -> str:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Bearer token required")
        supplied = authorization[7:]
        for key, tenant in config.keys.items():
            if hmac.compare_digest(supplied.encode("utf-8"), key.encode("utf-8")):
                return tenant
        raise HTTPException(status_code=401, detail="Invalid API key")

    multimodal = install_multimodal_routes(app, auth, db_path=config.db_path)

    @app.get("/live")
    def live() -> dict:
        return {"status": "ok"}

    @app.get("/health")
    def health(tenant: Annotated[str, Depends(auth)]) -> dict:
        return {"status": "ok", "tenant": tenant,
                "embedding_model": service.provider.embedding_model}

    @app.get("/strategies")
    def strategies(tenant: Annotated[str, Depends(auth)]) -> dict:
        return {"strategies": MODES, "tenant": tenant, "capabilities": [*service.catalog(tenant), multimodal.capability()]}

    def ingest_source(tenant: str, **document) -> dict:
        try:
            return service.ingest(tenant, **document)
        except CapabilityError:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/documents", status_code=201)
    def ingest(data: DocumentIn, tenant: Annotated[str, Depends(auth)]) -> dict:
        if len(json.dumps(data.metadata)) > 16_000:
            raise HTTPException(status_code=413, detail="metadata too large")
        return ingest_source(
            tenant, title=data.title, text=data.text or "", source=data.source,
            metadata=data.metadata, document_id=data.document_id,
            chunking=data.chunking, tokenizer=data.tokenizer, index_features=data.index_features,
            rows=data.rows, valid_from=data.valid_from, valid_to=data.valid_to, expected_revision=data.expected_revision,
        )

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
        if len(body) > 200_000:
            raise HTTPException(status_code=413, detail="Extracted text exceeds the document limit")
        return ingest_source(tenant, title=filename, text=body, source="upload:" + filename)

    @app.post("/images", status_code=201)
    async def ingest_image(
        tenant: Annotated[str, Depends(auth)],
        file: UploadFile = File(...),
    ) -> dict:
        """Vision-assisted image caption ingestion (caption index, not image vectors)."""
        if config.model_provider == "extractive":
            raise HTTPException(status_code=501, detail="Configure a vision-capable model")
        raw = await file.read(2_000_001)
        if not 0 < len(raw) <= 2_000_000:
            raise HTTPException(status_code=413, detail="Image must be 1..2MB")
        from rags.vision import describe_image, validate_image

        try:
            media_type = validate_image(raw)
            description = describe_image(raw, media_type, config)
        except CapabilityError:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        filename = os.path.basename(file.filename or "image")
        return ingest_source(
            tenant, title=filename,
            text="# Image description (model-generated; verify against original)\n" + description,
            source="image:" + filename,
            metadata={"kind": "image", "media_type": media_type,
                      "caption_generated_by": config.model_provider},
        )

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

    @app.get("/documents/{document_id}/revisions")
    def revisions(document_id: str, tenant: Annotated[str, Depends(auth)]) -> list[dict]:
        if store.get_document(tenant, document_id) is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return store.revisions(tenant, document_id)

    @app.post("/documents/{document_id}/reindex")
    def reindex(document_id: str, data: ReindexIn, tenant: Annotated[str, Depends(auth)]) -> dict:
        try:
            return service.reindex(tenant, document_id, **data.model_dump())
        except CapabilityError:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/query")
    def query(data: QueryIn, tenant: Annotated[str, Depends(auth)]) -> dict:
        try:
            return service.query(tenant, **data.model_dump(exclude_none=True))
        except CapabilityError:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = create_app()
