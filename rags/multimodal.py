"""Opt-in native CLIP image retrieval with separate, authenticated asset citations.

This stores original image bytes and aligned image vectors. It does not turn
images into invented document text, and does not provide audio/video retrieval.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import threading
from typing import Callable, Protocol
import uuid
import warnings

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from rags import postgres
from rags.features import FeatureFlags
from rags.registry import CapabilityError


MAX_IMAGE_BYTES = 2_000_000
MAX_IMAGE_PIXELS = 25_000_000
MAX_TENANT_ASSETS = 10_000
CLIP_PAIR = "clip-vit-b-32"
TEXT_MODEL = "Qdrant/clip-ViT-B-32-text"
IMAGE_MODEL = "Qdrant/clip-ViT-B-32-vision"
_ID = re.compile(r"^[a-zA-Z0-9_.:-]{1,128}$")
_MEDIA = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}


class AlignedEncoder(Protocol):
    id: str
    dimensions: int
    profile: dict

    def embed_text(self, query: str) -> list[float]: ...
    def embed_image(self, image) -> list[float]: ...


def normalized_vector(values, dimensions: int) -> list[float]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, (list, tuple)) or len(values) != dimensions:
        raise CapabilityError("invalid_asset_vector", "Aligned encoder returned an incompatible vector", 502)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise CapabilityError("invalid_asset_vector", "Aligned encoder returned a non-numeric vector", 502)
    try:
        vector = [float(value) for value in values]
        norm = math.hypot(*vector)
    except (ValueError, TypeError, OverflowError) as exc:
        raise CapabilityError("invalid_asset_vector", "Aligned encoder returned an invalid vector", 502) from exc
    if not math.isfinite(norm) or norm == 0 or not all(math.isfinite(value) for value in vector):
        raise CapabilityError("invalid_asset_vector", "Aligned encoder returned a zero or non-finite vector", 502)
    return [value / norm for value in vector]


def validated_image(raw: bytes):
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_IMAGE_BYTES:
        raise CapabilityError("invalid_image_size", "Image must contain 1..2000000 bytes", 413)
    try:
        from PIL import Image, ImageOps, UnidentifiedImageError
    except ImportError as exc:
        raise CapabilityError("requires_dependency", "Install the vision extra for image assets") from exc
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as probe:
                media_type = _MEDIA.get(probe.format)
                if media_type is None:
                    raise CapabilityError("unsupported_image", "Only JPEG, PNG and WebP images are supported", 415)
                if probe.width * probe.height > MAX_IMAGE_PIXELS:
                    raise CapabilityError("invalid_image_size", "Image exceeds 25000000 pixels", 413)
                if getattr(probe, "n_frames", 1) != 1:
                    raise CapabilityError("unsupported_image", "Animated images are not supported", 415)
                probe.verify()
            with Image.open(io.BytesIO(raw)) as source:
                # The returned pixels are independent of the file handle. Original
                # bytes, including orientation metadata, remain the stored source.
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.load()
                return image, media_type
    except CapabilityError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombWarning,
            Image.DecompressionBombError) as exc:
        raise CapabilityError("invalid_image", "Image is corrupt or unsafe to decode", 422) from exc


def _checkpoint_fingerprint(model) -> tuple[str, str]:
    # FastEmbed 0.9 exposes the resolved model directory on its implementation.
    # Fingerprint actual files, including tokenization/preprocessing configuration,
    # so cached checkpoint changes cannot silently reuse an old vector space.
    directory = Path(model.model._model_dir)
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if not files:
        raise ValueError("No checkpoint files were resolved")
    manifest = []
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        manifest.append((path.relative_to(directory).as_posix(), digest.hexdigest()))
    digest = hashlib.sha256(json.dumps(manifest, separators=(",", ":")).encode()).hexdigest()
    return directory.name, digest


class CLIPEncoder:
    """A fixed, verified-compatible encoder pair; requests cannot select models."""

    dimensions = 512

    def __init__(self, cache_dir: str | None = None):
        try:
            from fastembed import ImageEmbedding, TextEmbedding
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise CapabilityError("requires_dependency", "Install encoders and vision extras for CLIP asset retrieval") from exc
        try:
            self._text = TextEmbedding(TEXT_MODEL, cache_dir=cache_dir, threads=2, cuda=False)
            self._image = ImageEmbedding(IMAGE_MODEL, cache_dir=cache_dir, threads=2, cuda=False)
            # Clone before disabling truncation/padding, preserving the encoder's
            # shared preprocessing and avoiding concurrent tokenizer mutation.
            self._counter = Tokenizer.from_str(self._text.model.tokenizer.to_str())
            self._counter.no_truncation()
            self._counter.no_padding()
            text_revision, text_digest = _checkpoint_fingerprint(self._text)
            image_revision, image_digest = _checkpoint_fingerprint(self._image)
        except CapabilityError:
            raise
        except Exception as exc:
            raise CapabilityError("model_load_failed", "Unable to load the configured aligned CLIP checkpoints") from exc
        self.profile = {
            "implementation": "fastembed-aligned-clip-v1", "pair": CLIP_PAIR,
            "text_model": TEXT_MODEL, "image_model": IMAGE_MODEL, "dimensions": self.dimensions,
            "text_checkpoint": text_revision, "image_checkpoint": image_revision,
            "text_artifacts_sha256": text_digest, "image_artifacts_sha256": image_digest,
            "image_preprocessing": "validated single frame; EXIF transpose; RGB; CLIP preprocessing",
            "text_token_limit": 77, "normalization": "l2", "primary_language": "en",
        }
        fingerprint = hashlib.sha256(json.dumps(self.profile, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.id = "clip-vit-b-32:" + fingerprint

    def embed_text(self, query: str) -> list[float]:
        try:
            count = len(self._counter.encode(query, add_special_tokens=True).ids)
        except Exception as exc:
            raise CapabilityError("invalid_query", "CLIP query could not be tokenized", 422) from exc
        if count > 77:
            raise CapabilityError("query_too_long", "CLIP queries must fit 77 tokens including special tokens", 422)
        try:
            return next(self._text.embed([query])).tolist()
        except Exception as exc:
            raise CapabilityError("asset_encoding_failed", "CLIP text encoding failed", 502) from exc

    def embed_image(self, image) -> list[float]:
        try:
            return next(self._image.embed([image])).tolist()
        except Exception as exc:
            raise CapabilityError("asset_encoding_failed", "CLIP image encoding failed", 502) from exc


_ENCODERS: dict[str | None, CLIPEncoder] = {}
_ENCODER_LOCK = threading.Lock()


def clip_encoder(cache_dir: str | None = None) -> CLIPEncoder:
    with _ENCODER_LOCK:
        if cache_dir not in _ENCODERS:
            _ENCODERS[cache_dir] = CLIPEncoder(cache_dir)
        return _ENCODERS[cache_dir]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS image_assets (
 tenant TEXT NOT NULL, id TEXT NOT NULL, revision TEXT NOT NULL,
 title TEXT NOT NULL, description TEXT NOT NULL, filename TEXT NOT NULL,
 media_type TEXT NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL,
 checksum TEXT NOT NULL, original BLOB NOT NULL, vector TEXT NOT NULL,
 encoder_id TEXT NOT NULL, dimensions INTEGER NOT NULL, encoder_profile TEXT NOT NULL,
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
 PRIMARY KEY (tenant,id)
);
CREATE INDEX IF NOT EXISTS image_assets_encoder ON image_assets(tenant,encoder_id);
"""
_METADATA = "id,revision,title,description,filename,media_type,width,height,checksum,encoder_id,dimensions,encoder_profile,created_at,updated_at"
ASSET_FIELDS = ("tenant", "id", "revision", "title", "description", "filename", "media_type", "width", "height",
                "checksum", "original", "vector", "encoder_id", "dimensions", "encoder_profile", "created_at", "updated_at")


class AssetStore:
    def __init__(self, path: str):
        if path == ":memory:":
            raise ValueError("Image assets require a file-backed database")
        self.path = path
        self.is_postgres = postgres.is_postgres(path)
        self.storage_backend = "postgresql" if self.is_postgres else "sqlite"
        self.vector_search = "pgvector_exact" if self.is_postgres else "python_exact"
        if not self.is_postgres:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            if self.is_postgres:
                postgres.initialize(connection, _SCHEMA)
                postgres.migrate_vector_column(connection, "image_assets", "vector")
            else:
                connection.executescript(_SCHEMA)

    @contextmanager
    def connect(self):
        if self.is_postgres:
            with postgres.connect(self.path) as connection:
                yield connection
            return
        connection = sqlite3.connect(self.path, timeout=20)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout=20000")
            connection.execute("PRAGMA journal_mode=WAL")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def decode(row) -> dict | None:
        if row is None:
            return None
        result = dict(row)
        result["encoder_profile"] = json.loads(result["encoder_profile"])
        return result

    def get(self, tenant: str, asset_id: str, *, original: bool = False) -> dict | None:
        fields = _METADATA + (",original" if original else "")
        with self.connect() as connection:
            row = connection.execute(f"SELECT {fields} FROM image_assets WHERE tenant=? AND id=?", (tenant, asset_id)).fetchone()
        return self.decode(row)

    def list(self, tenant: str, limit: int = 100, offset: int = 0) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(f"SELECT {_METADATA} FROM image_assets WHERE tenant=? ORDER BY updated_at DESC,id LIMIT ? OFFSET ?",
                                      (tenant, limit, offset)).fetchall()
        return [self.decode(row) for row in rows]

    def vectors(self, tenant: str) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(f"SELECT {_METADATA},vector FROM image_assets WHERE tenant=?", (tenant,)).fetchall()
        return [self.decode(row) for row in rows]

    def save(self, tenant: str, value: dict, expected_revision: str | None):
        vector = (postgres.vector_literal(value["vector"], code="invalid_asset_vector") if self.is_postgres
                  else json.dumps(value["vector"], allow_nan=False))
        if self.is_postgres and len(value["vector"]) != value["dimensions"]:
            raise CapabilityError("invalid_asset_vector", "Image vector does not match its encoder dimensions", 502)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if self.is_postgres:
                postgres.lock_mutation(connection, "assets", tenant)
            current = connection.execute("SELECT revision,created_at FROM image_assets WHERE tenant=? AND id=?", (tenant, value["id"])).fetchone()
            if (current["revision"] if current else None) != expected_revision:
                raise CapabilityError("revision_conflict", "Image asset changed during encoding; retry with its current revision", 409)
            if current is None and connection.execute("SELECT count(*) FROM image_assets WHERE tenant=?", (tenant,)).fetchone()[0] >= MAX_TENANT_ASSETS:
                raise CapabilityError("asset_limit", "Tenant image asset limit reached", 409)
            created = current["created_at"] if current else value["updated_at"]
            placeholders = ["?::vector" if self.is_postgres and field == "vector" else "?" for field in ASSET_FIELDS]
            connection.execute(
                "INSERT INTO image_assets(" + ",".join(ASSET_FIELDS) + ") VALUES (" + ",".join(placeholders) + ") "
                "ON CONFLICT(tenant,id) DO UPDATE SET revision=excluded.revision,title=excluded.title,description=excluded.description,"
                "filename=excluded.filename,media_type=excluded.media_type,width=excluded.width,height=excluded.height,"
                "checksum=excluded.checksum,original=excluded.original,vector=excluded.vector,encoder_id=excluded.encoder_id,"
                "dimensions=excluded.dimensions,encoder_profile=excluded.encoder_profile,updated_at=excluded.updated_at",
                (tenant, value["id"], value["revision"], value["title"], value["description"], value["filename"], value["media_type"],
                 value["width"], value["height"], value["checksum"], value["original"], vector,
                 value["encoder_id"], value["dimensions"], json.dumps(value["encoder_profile"], sort_keys=True), created, value["updated_at"]),
            )

    def delete(self, tenant: str, asset_id: str) -> bool:
        with self.connect() as connection:
            if self.is_postgres:
                postgres.lock_mutation(connection, "assets", tenant)
            return connection.execute("DELETE FROM image_assets WHERE tenant=? AND id=?", (tenant, asset_id)).rowcount > 0

    def dense(self, tenant: str, vector: list[float], top_k: int, *, encoder_id: str,
              dimensions: int) -> tuple[list[dict], int]:
        """PostgreSQL ranking over one consistent tenant/encoder snapshot."""
        if not self.is_postgres:
            raise ValueError("Native SQL image search requires PostgreSQL")
        if type(top_k) is not int or not 1 <= top_k <= 20:
            raise ValueError("Image top_k must be 1..20")
        literal = postgres.vector_literal(vector, code="invalid_asset_vector")
        if len(vector) != dimensions:
            raise CapabilityError("asset_reindex_required", "Image query has incompatible encoder dimensions", 409)
        with self.connect() as connection:
            connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            candidates = connection.execute(
                "SELECT encoder_id,dimensions,vector_dims(vector) AS stored_dimensions FROM image_assets WHERE tenant=?",
                (tenant,),
            ).fetchall()
            if any(row["encoder_id"] != encoder_id or row["dimensions"] != dimensions
                   or row["stored_dimensions"] != dimensions for row in candidates):
                raise CapabilityError("asset_reindex_required", "Image assets use another encoder; reindex them before searching", 409)
            distance = postgres.cosine_distance("a.vector")
            rows = connection.execute(
                "WITH query AS (SELECT ?::vector AS embedding) "
                f"SELECT {_METADATA},{distance} AS distance FROM image_assets a CROSS JOIN query q "
                "WHERE a.tenant=? ORDER BY distance,a.id LIMIT ?", (literal, tenant, top_k),
            ).fetchall()
        result = []
        for row in rows:
            value = self.decode(row)
            value["score"] = max(-1.0, min(1.0, 1.0 - value.pop("distance")))
            result.append(value)
        return result, len(candidates)


def _asset_id(value: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise CapabilityError("invalid_asset_id", "Asset ID must contain 1..128 letters, digits, dots, underscores, colons or hyphens", 422)
    return value


class MultimodalService:
    def __init__(self, db_path: str, model_name: str | None = None, encoder: AlignedEncoder | None = None,
                 features: FeatureFlags | None = None):
        self.features = features if features is not None else FeatureFlags.from_env()
        self.store = AssetStore(db_path)
        self.model_name = (os.getenv("RAGS_MULTIMODAL_MODEL", "") if model_name is None else model_name).strip().lower()
        if self.model_name and self.model_name != CLIP_PAIR:
            raise ValueError("RAGS_MULTIMODAL_MODEL supports only clip-vit-b-32, the declared aligned pair")
        self.cache_dir = os.getenv("RAGS_MODEL_CACHE") or None
        self._encoder = encoder
        self._exercised = False

    def encoder(self) -> AlignedEncoder:
        self.features.require("multimodal", "embedding.fastembed")
        if self._encoder is None:
            if not self.model_name:
                raise CapabilityError("requires_backend", "Set RAGS_MULTIMODAL_MODEL=clip-vit-b-32 and install encoders,vision extras")
            self._encoder = clip_encoder(self.cache_dir)
        return self._encoder

    def capability(self) -> dict:
        dependencies = ("multimodal", "query", "strategy.multimodal_native", "embedding.fastembed")
        disabled = [name for name in dependencies if not self.features.enabled(name)]
        return {"id": "multimodal_assets", "family": "Multimodal RAG",
                "status": ("feature_disabled" if disabled else
                           "available" if self._encoder else "configured" if self.model_name else "requires_backend"),
                "enabled": not disabled, "disabled_by": disabled,
                "implementation": "Aligned CLIP text-to-image and image-to-image asset retrieval",
                "evidence_level": "model_exercised_in_process" if self._exercised else "not_exercised_in_process",
                "model": self.model_name or None, "model_loaded": self._encoder is not None,
                "search_endpoint": "/assets/search", "image_search_endpoint": "/assets/search/image",
                "original_images_stored": True, "audio_video": False,
                "original_images_enabled": self.features.enabled("original_images"),
                "max_image_bytes": MAX_IMAGE_BYTES, "max_image_pixels": MAX_IMAGE_PIXELS,
                "text_token_limit": 77, "primary_language": "en"}

    def ingest(self, tenant: str, raw: bytes, *, filename: str = "image", title: str | None = None,
               description: str = "", asset_id: str | None = None, expected_revision: str | None = None,
               force: bool = False) -> dict:
        self.features.require("ingestion", "multimodal", "embedding.fastembed")
        if force:
            self.features.require("reindex")
        identifier = _asset_id(asset_id or uuid.uuid4().hex)
        if not isinstance(filename, str):
            raise CapabilityError("invalid_asset_metadata", "Image filename must be a string", 422)
        filename = os.path.basename(filename.replace("\\", "/"))[:200] or "image"
        title = title if title is not None else filename
        if not isinstance(title, str) or not title.strip() or not 1 <= len(title) <= 200 or not isinstance(description, str) or len(description) > 2000:
            raise CapabilityError("invalid_asset_metadata", "Title must contain 1..200 characters and description at most 2000", 422)
        try:
            for value in (filename, title, description):
                value.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise CapabilityError("invalid_asset_metadata", "Image metadata contains invalid Unicode", 422) from exc
        image, media_type = validated_image(raw)
        try:
            encoder = self.encoder()
            old = self.store.get(tenant, identifier)
            if expected_revision is not None and (old is None or old["revision"] != expected_revision):
                raise CapabilityError("revision_conflict", "Image asset revision does not match", 409)
            checksum = hashlib.sha256(raw).hexdigest()
            if not force and old and all((old["checksum"] == checksum, old["encoder_id"] == encoder.id,
                                          old["title"] == title, old["description"] == description, old["filename"] == filename)):
                return {**old, "unchanged": True, "asset_uri": f"/assets/images/{identifier}"}
            vector = normalized_vector(encoder.embed_image(image), encoder.dimensions)
            self._exercised = True
            record = dict(id=identifier, revision=uuid.uuid4().hex, title=title, description=description, filename=filename,
                          media_type=media_type, width=image.width, height=image.height, checksum=checksum, original=raw,
                          vector=vector, encoder_id=encoder.id, dimensions=encoder.dimensions, encoder_profile=encoder.profile,
                          updated_at=datetime.now(timezone.utc).isoformat())
            self.store.save(tenant, record, old["revision"] if old else None)
            public = {key: value for key, value in record.items() if key not in {"vector", "original"}}
            public["created_at"] = old["created_at"] if old else record["updated_at"]
            return {**public, "unchanged": False, "asset_uri": f"/assets/images/{identifier}"}
        finally:
            image.close()

    def reindex(self, tenant: str, asset_id: str) -> dict:
        self.features.require("reindex", "ingestion", "multimodal", "embedding.fastembed")
        row = self.store.get(tenant, _asset_id(asset_id), original=True)
        if row is None:
            raise CapabilityError("asset_not_found", "Image asset not found", 404)
        return self.ingest(tenant, row["original"], filename=row["filename"], title=row["title"],
                           description=row["description"], asset_id=asset_id,
                           expected_revision=row["revision"], force=True)

    def get(self, tenant: str, asset_id: str, *, original: bool = False) -> dict | None:
        self.features.require("documents", "multimodal")
        if original:
            self.features.require("original_images")
        return self.store.get(tenant, _asset_id(asset_id), original=original)

    def list(self, tenant: str, limit: int = 100, offset: int = 0) -> list[dict]:
        self.features.require("documents", "multimodal")
        if type(limit) is not int or type(offset) is not int or not 1 <= limit <= 100 or not 0 <= offset <= MAX_TENANT_ASSETS:
            raise CapabilityError("invalid_asset_limit", "limit must be 1..100 and offset 0..10000", 422)
        return self.store.list(tenant, limit, offset)

    def delete(self, tenant: str, asset_id: str) -> bool:
        self.features.require("deletion", "multimodal")
        return self.store.delete(tenant, _asset_id(asset_id))

    def _search(self, tenant: str, vector, top_k: int, modality: str, encoder: AlignedEncoder) -> dict:
        self.features.require("query", "multimodal", "strategy.multimodal_native", "embedding.fastembed")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 20:
            raise CapabilityError("invalid_top_k", "top_k must be in 1..20", 422)
        query = normalized_vector(vector, encoder.dimensions)
        if self.store.is_postgres:
            rows, count = self.store.dense(tenant, query, top_k, encoder_id=encoder.id, dimensions=encoder.dimensions)
            ranked = [(row.pop("score"), row) for row in rows]
        else:
            rows = self.store.vectors(tenant)
            count = len(rows)
            if any(row["encoder_id"] != encoder.id or row["dimensions"] != encoder.dimensions for row in rows):
                raise CapabilityError("asset_reindex_required", "Image assets use another encoder; reindex them before searching", 409)
            ranked = []
            for row in rows:
                try:
                    stored = normalized_vector(json.loads(row.pop("vector")), encoder.dimensions)
                except (ValueError, CapabilityError) as exc:
                    raise CapabilityError("invalid_asset_index", "Stored image vector is invalid; reindex the asset", 409) from exc
                score = max(-1.0, min(1.0, sum(a * b for a, b in zip(query, stored, strict=True))))
                ranked.append((score, row))
            ranked.sort(key=lambda item: (-item[0], item[1]["id"]))
        citations = [{**row, "asset_id": row["id"], "asset_uri": f"/assets/images/{row['id']}",
                      "score": round(score, 8), "score_kind": "cosine_similarity",
                      "evidence_kind": "original_image", "description_origin": "user"}
                     for score, row in ranked[:top_k]]
        return {"query_modality": modality, "representation": "native_image_vector", "encoder_id": encoder.id,
                "encoder_profile": encoder.profile, "candidates": count, "citations": citations}

    def search_text(self, tenant: str, query: str, top_k: int = 5) -> dict:
        self.features.require("query", "multimodal", "strategy.multimodal_native", "embedding.fastembed")
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise CapabilityError("invalid_query", "Text query must contain 1..2000 characters", 422)
        encoder = self.encoder()
        vector = encoder.embed_text(query)
        self._exercised = True
        return self._search(tenant, vector, top_k, "text", encoder)

    def search_image(self, tenant: str, raw: bytes, top_k: int = 5) -> dict:
        self.features.require("query", "multimodal", "strategy.multimodal_native", "embedding.fastembed")
        image, _ = validated_image(raw)
        try:
            encoder = self.encoder()
            vector = encoder.embed_image(image)
            self._exercised = True
            return self._search(tenant, vector, top_k, "image", encoder)
        finally:
            image.close()


class AssetSearch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


def install_multimodal_routes(app: FastAPI, auth: Callable, *, db_path: str,
                              model_name: str | None = None, encoder: AlignedEncoder | None = None,
                              features: FeatureFlags | None = None) -> MultimodalService:
    service = MultimodalService(db_path, model_name=model_name, encoder=encoder, features=features)
    app.state.multimodal = service
    if CapabilityError not in app.exception_handlers:
        async def capability_error(request, exc):
            return JSONResponse(status_code=exc.status_code, content={"detail": {"code": exc.code, "reason": exc.reason}})
        app.add_exception_handler(CapabilityError, capability_error)

    @app.get("/assets/capabilities", tags=["Image assets"])
    def capabilities(tenant: str = Depends(auth)):
        return service.capability()

    @app.post("/assets/images", status_code=201, tags=["Image assets"])
    async def ingest_image(tenant: str = Depends(auth), file: UploadFile = File(...),
                           title: str | None = Form(default=None), description: str = Form(default=""),
                           asset_id: str | None = Form(default=None), expected_revision: str | None = Form(default=None)):
        service.features.require("uploads", "ingestion", "multimodal", "embedding.fastembed")
        raw = await file.read(MAX_IMAGE_BYTES + 1)
        return await run_in_threadpool(service.ingest, tenant, raw, filename=file.filename or "image",
                                       title=title, description=description, asset_id=asset_id, expected_revision=expected_revision)

    @app.get("/assets/images", tags=["Image assets"])
    def list_images(tenant: str = Depends(auth), limit: int = 100, offset: int = 0):
        return service.list(tenant, limit, offset)

    @app.get("/assets/images/{asset_id}", tags=["Image assets"])
    def get_image(asset_id: str, tenant: str = Depends(auth)):
        row = service.get(tenant, asset_id, original=True)
        if row is None:
            raise HTTPException(404, "Image asset not found")
        extension = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}[row["media_type"]]
        return Response(content=row["original"], media_type=row["media_type"],
                        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
                                 "Content-Disposition": f'attachment; filename="image-{asset_id}.{extension}"'})

    @app.delete("/assets/images/{asset_id}", tags=["Image assets"])
    def delete_image(asset_id: str, tenant: str = Depends(auth)):
        if not service.delete(tenant, asset_id):
            raise HTTPException(404, "Image asset not found")
        return {"deleted": True, "asset_id": asset_id}

    @app.post("/assets/images/{asset_id}/reindex", tags=["Image assets"])
    def reindex_image(asset_id: str, tenant: str = Depends(auth)):
        return service.reindex(tenant, asset_id)

    @app.post("/assets/search", tags=["Image assets"])
    def search_assets(data: AssetSearch, tenant: str = Depends(auth)):
        return service.search_text(tenant, data.query, data.top_k)

    @app.post("/assets/search/image", tags=["Image assets"])
    async def search_image(tenant: str = Depends(auth), file: UploadFile = File(...),
                           top_k: int = Form(default=5, ge=1, le=20)):
        service.features.require("query", "multimodal", "strategy.multimodal_native", "embedding.fastembed")
        raw = await file.read(MAX_IMAGE_BYTES + 1)
        return await run_in_threadpool(service.search_image, tenant, raw, top_k)

    return service
