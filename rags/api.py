"""FastAPI service with tenant-scoped key authentication and bounded inputs."""
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import asyncio
import io
import json
import math
import os
from threading import BoundedSemaphore
from typing import Annotated, Literal

import httpx
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rags.admin import install_admin
from rags.agents.config import AgentPlatformConfig
from rags.agents.api import mount_agent_api
from rags.auth import Authenticator, authorization_for_path, Principal
from rags.chunking import ChunkConfig
from rags.collection_api import mount_collection_api
from rags.collection_service import CollectionService
from rags.config import Settings
from rags.limits import BodyLimitMiddleware
from rags.multimodal import install_multimodal_routes
from rags.monitoring import Monitor, route_name
from rags.providers import Providers
from rags.registry import CapabilityError
from rags.retrieval import MODES, RAGService, now
from rags.features import strategy_features
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
    docs = config.features.enabled("api_docs")
    upload_workers = ThreadPoolExecutor(max_workers=8, thread_name_prefix="rags-upload")
    upload_slots = BoundedSemaphore(8)

    @asynccontextmanager
    async def lifespan(_application):
        try:
            yield
        finally:
            await asyncio.to_thread(upload_workers.shutdown, wait=True)
            monitor.shutdown()
    app = FastAPI(title="RAGs", version="0.9.0", docs_url="/docs" if docs else None,
                  redoc_url="/redoc" if docs else None, openapi_url="/openapi.json" if docs else None, lifespan=lifespan)
    app.add_middleware(BodyLimitMiddleware, max_body_bytes=2_100_000)
    store = Store(config.db_path)
    service = RAGService(store, Providers(config))
    app.state.service = service
    collections = CollectionService(store, service.provider)
    app.state.collections = collections
    authenticator = Authenticator(mode=config.auth_mode, keys=config.keys,
                                  key_roles=config.key_roles, oauth2=config.oauth2)

    monitor = Monitor(enabled=config.otel_enabled, endpoint=config.otel_endpoint,
                      service_name=config.otel_service_name)
    app.state.monitor = monitor

    @app.middleware("http")
    async def observe_http(request: Request, call_next):
        # No headers, bodies, untemplated paths, subjects, tenant IDs or prompts
        # enter OTLP. The tenant is used only for process-local admin metrics.
        import time
        start_ns = time.time_ns()
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
            principal = getattr(request.state, "principal", None)
            monitor.record(
                tenant=principal.tenant if principal is not None else None,
                route=route_name(request.scope), method=request.method, status=status,
                duration_ms=elapsed_ms, start_ns=start_ns, end_ns=time.time_ns(),
            )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse(status_code=422, content={"detail": validation_details(exc.errors())})

    @app.exception_handler(CapabilityError)
    async def capability_error(request, exc):
        return JSONResponse(status_code=exc.status_code, content={"detail": {"code": exc.code, "reason": exc.reason}})

    @app.exception_handler(httpx.HTTPError)
    async def provider_error(request, exc):
        return JSONResponse(status_code=502, content={"detail": {"code": "provider_failed", "reason": "Configured provider request failed"}})

    def auth(request: Request, authorization: Annotated[str | None, Header()] = None) -> str:
        principal = authenticator.authenticate(authorization)
        is_mcp = request.headers.get("X-RAGS-Interface") == "mcp"
        authorization_for_path(principal, path=request.url.path, method=request.method, is_mcp=is_mcp)
        if is_mcp:
            config.features.require("mcp")
        request.state.principal = principal
        path = request.url.path
        # Operator flags and identity privileges are independent restrictions.
        if path == "/query":
            config.features.require("query")
        elif path == "/images":
            config.features.require("uploads", "ingestion", "image_captioning", "model_generation",
                                    "embedding." + config.embedding_provider)
            service.preflight_ingestion()
        elif path == "/documents" or path.startswith("/documents/"):
            if request.method == "GET":
                config.features.require("documents")
            elif request.method == "DELETE":
                config.features.require("deletion")
            else:
                config.features.require("ingestion")
                if path == "/documents/upload":
                    config.features.require("uploads")
                    require_index_generation(principal, chunking=config.chunking_config,
                                             index_features=[])
                    service.preflight_ingestion()
                elif path.endswith("/reindex"):
                    config.features.require("reindex")
        return principal.tenant

    def require_index_generation(principal: Principal, *, chunking: dict,
                                 index_features: list[str]) -> None:
        # Model-provider configuration alone does not make ordinary indexing
        # model-generated. Evaluate the actual requested processing profile.
        if (config.model_provider != "extractive"
                and (chunking.get("strategy") == "contextual"
                     or "graph" in index_features or "hierarchy" in index_features)):
            principal.require("rags:generate")

    def effective_features(principal: Principal) -> dict[str, bool]:
        """Advertise only features the authenticated caller is allowed to use."""
        effective = config.features.as_dict()
        permissions = {
            "documents": "rags:read", "query": "rags:query",
            "ingestion": "rags:write", "deletion": "rags:delete",
            "reindex": "rags:reindex", "model_generation": "rags:generate",
            "original_images": "rags:original_image", "mcp": "rags:mcp",
        }
        for name, required in permissions.items():
            effective[name] = effective[name] and required in principal.permissions
        return effective

    def effective_catalog(tenant: str, principal: Principal) -> list[dict]:
        items = service.catalog(tenant)
        for item in items:
            requirements = strategy_features(item["id"])
            denied = set()
            if "rags:query" not in principal.permissions:
                denied.add("rags:query")
            if "model_generation" in requirements and "rags:generate" not in principal.permissions:
                denied.add("rags:generate")
            if denied:
                item.update(status="insufficient_scope", enabled=False, disabled_by=sorted(denied))
        return items

    multimodal = install_multimodal_routes(app, auth, db_path=config.db_path, features=config.features)

    @app.get("/live")
    def live() -> dict:
        return {"status": "ok"}

    @app.get("/health")
    def health(tenant: Annotated[str, Depends(auth)]) -> dict:
        return {"status": "ok", "tenant": tenant,
                "embedding_model": service.provider.embedding_model,
                "storage": {"backend": store.storage_backend, "vector_search": store.vector_search}}

    @app.get("/capabilities")
    def capabilities(request: Request, tenant: Annotated[str, Depends(auth)]) -> dict:
        principal = request.state.principal
        return {"features": effective_features(principal),
                "storage": {"backend": store.storage_backend, "vector_search": store.vector_search},
                "generation": {"provider": config.model_provider, "model": service.provider.model_identity},
                "permissions": sorted(principal.permissions),
                "capabilities": effective_catalog(tenant, principal),
                "collections": collections.capabilities(principal),
                "image_assets": multimodal.capability()}

    @app.get("/strategies")
    def strategies(request: Request, tenant: Annotated[str, Depends(auth)]) -> dict:
        principal = request.state.principal
        catalog = effective_catalog(tenant, principal)
        return {"strategies": [item["id"] for item in catalog if item["enabled"]], "tenant": tenant,
                "features": effective_features(principal),
                "permissions": sorted(principal.permissions),
                "capabilities": [*catalog, multimodal.capability()]}

    def ingest_source(tenant: str, **document) -> dict:
        try:
            return service.ingest(tenant, **document)
        except CapabilityError:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/documents", status_code=201)
    def ingest(data: DocumentIn, request: Request, tenant: Annotated[str, Depends(auth)]) -> dict:
        require_index_generation(
            request.state.principal, chunking=data.chunking or config.chunking_config,
            index_features=data.index_features,
        )
        if len(json.dumps(data.metadata)) > 16_000:
            raise HTTPException(status_code=413, detail="metadata too large")
        return ingest_source(
            tenant, title=data.title, text=data.text or "", source=data.source,
            metadata=data.metadata, document_id=data.document_id,
            chunking=data.chunking, tokenizer=data.tokenizer, index_features=data.index_features,
            rows=data.rows, valid_from=data.valid_from, valid_to=data.valid_to, expected_revision=data.expected_revision,
        )

    async def run_upload(callback):
        if not upload_slots.acquire(blocking=False):
            raise CapabilityError("ingestion_capacity", "Multipart ingestion capacity exhausted", 503)
        context = copy_context()

        def invoke():
            try:
                return context.run(callback)
            finally:
                # Cancellation of the awaiting HTTP task does not stop a
                # synchronous provider or database operation. Keep its slot
                # until the operation actually finishes.
                upload_slots.release()

        try:
            future = asyncio.get_running_loop().run_in_executor(upload_workers, invoke)
        except BaseException:
            upload_slots.release()
            raise
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            future.add_done_callback(lambda completed: completed.exception() if not completed.cancelled() else None)
            raise

    def process_upload(tenant: str, raw: bytes, filename: str) -> dict:
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
        return await run_upload(lambda: process_upload(tenant, raw, filename))

    def process_image(tenant: str, raw: bytes, filename: str) -> dict:
        from rags.vision import describe_image, validate_image

        try:
            media_type = validate_image(raw)
            description = describe_image(raw, media_type, config)
        except CapabilityError:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return ingest_source(
            tenant, title=filename,
            text="# Image description (model-generated; verify against original)\n" + description,
            source="image:" + filename,
            metadata={"kind": "image", "media_type": media_type,
                      "caption_generated_by": config.model_provider},
        )

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
        filename = os.path.basename(file.filename or "image")
        return await run_upload(lambda: process_image(tenant, raw, filename))

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
    def reindex(document_id: str, data: ReindexIn, request: Request,
                tenant: Annotated[str, Depends(auth)]) -> dict:
        existing = store.get_document(tenant, document_id)
        if existing is not None:
            profile = existing["index_profile"]
            require_index_generation(
                request.state.principal,
                chunking=data.chunking or profile.get("chunking") or config.chunking_config,
                index_features=(data.index_features if data.index_features is not None
                                else profile.get("features", [])),
            )
        try:
            return service.reindex(tenant, document_id, **data.model_dump())
        except CapabilityError:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/query")
    def query(data: QueryIn, request: Request, tenant: Annotated[str, Depends(auth)]) -> dict:
        if ((data.generate and config.model_provider != "extractive")
                or "model_generation" in strategy_features(data.mode)):
            request.state.principal.require("rags:generate")
        try:
            return service.query(tenant, **data.model_dump(exclude_none=True))
        except CapabilityError:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    install_admin(app, settings=config, authenticator=authenticator,
                  store=store, service=service, monitor=monitor)
    mount_collection_api(app, service=collections, auth=auth)
    mount_agent_api(app, store=store, rag=service, config=AgentPlatformConfig.from_env(),
                    auth=auth, features=config.features)
    return app


app = create_app()
