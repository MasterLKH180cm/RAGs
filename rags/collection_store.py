"""Tenant-scoped collection records and copy-on-write immutable vector payloads.

Provider execution belongs to the application service. This module accepts only
prepared complete records, commits batches atomically, and keeps filtered reads
and PostgreSQL vector ranking in the same repeatable database snapshot.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from rags import postgres
from rags.collection_models import (
    MAX_DIMENSIONS,
    MAX_DOCUMENT_CHARS,
    canonical_dense,
    canonical_metadata,
    canonical_sparse,
    record_ids,
    valid_text,
    validate_collection_name,
)
from rags.native import cosine
from rags.registry import CapabilityError
from rags.storage import Store, timestamp

SCHEMA = """
CREATE TABLE IF NOT EXISTS rag_collections (
 tenant TEXT NOT NULL, id TEXT NOT NULL, name TEXT NOT NULL,
 metadata TEXT NOT NULL, configuration TEXT NOT NULL,
 generation BIGINT NOT NULL CHECK(generation>=0),
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
 fork_source_id TEXT, fork_source_generation BIGINT,
 PRIMARY KEY(tenant,id), UNIQUE(tenant,name)
);
CREATE TABLE IF NOT EXISTS rag_collection_versions (
 tenant TEXT NOT NULL, id TEXT NOT NULL, document TEXT,
 metadata TEXT NOT NULL, embedding TEXT NOT NULL,
 embedding_dimensions INTEGER NOT NULL CHECK(embedding_dimensions BETWEEN 1 AND 4096),
 sparse_embedding TEXT NOT NULL, created_at TEXT NOT NULL,
 PRIMARY KEY(tenant,id)
);
CREATE TABLE IF NOT EXISTS rag_collection_records (
 tenant TEXT NOT NULL, collection_id TEXT NOT NULL, id TEXT NOT NULL,
 version TEXT NOT NULL, created_at TEXT NOT NULL,
 PRIMARY KEY(tenant,collection_id,id),
 FOREIGN KEY(tenant,collection_id) REFERENCES rag_collections(tenant,id),
 FOREIGN KEY(tenant,version) REFERENCES rag_collection_versions(tenant,id)
);
CREATE INDEX IF NOT EXISTS rag_collection_version_refs
 ON rag_collection_records(tenant,version);
CREATE INDEX IF NOT EXISTS rag_collection_names ON rag_collections(tenant,name);
"""

TABLE_FIELDS = {
    "rag_collections": (
        "tenant", "id", "name", "metadata", "configuration", "generation",
        "created_at", "updated_at", "fork_source_id", "fork_source_generation",
    ),
    "rag_collection_versions": (
        "tenant", "id", "document", "metadata", "embedding", "embedding_dimensions",
        "sparse_embedding", "created_at",
    ),
    "rag_collection_records": ("tenant", "collection_id", "id", "version", "created_at"),
}
MAX_RECORDS = 10_000
MAX_SCAN_BYTES = 16 * 1024 * 1024
MAX_BATCH = 100
PAGE_SIZE = 32
MAX_GENERATION = 2 ** 63 - 1


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _error(code: str, reason: str, status: int = 422) -> CapabilityError:
    return CapabilityError(code, reason, status)


def _invalid() -> CapabilityError:
    return _error("invalid_collection_input", "Collection input does not satisfy its bounded contract")


def _generation(value: int | None, *, optional: bool = True) -> None:
    if value is None and optional:
        return
    if type(value) is not int or not 0 <= value <= MAX_GENERATION:
        raise _invalid()


def _next_generation(current: int) -> int:
    if current >= MAX_GENERATION:
        raise _error("collection_limit_exceeded", "Collection generation limit has been reached")
    return current + 1


def _configuration(value: Any) -> dict:
    if not isinstance(value, dict) or set(value) != {"embedding", "sparse_space_id"}:
        raise ValueError("Invalid collection configuration")
    embedding = value["embedding"]
    fields = {"mode", "model", "endpoint", "space_id", "dimensions", "normalization"}
    if not isinstance(embedding, dict) or set(embedding) not in (fields, fields | {"profile_id"}):
        raise ValueError("Invalid embedding configuration")
    if embedding["mode"] not in {"configured", "external"} or embedding["normalization"] != "l2-f32-v1":
        raise ValueError("Invalid embedding configuration")
    if "profile_id" in embedding:
        from rags.embedding_profiles import PROFILE_ID
        if (embedding["mode"] != "configured" or not isinstance(embedding["profile_id"], str)
                or not PROFILE_ID.fullmatch(embedding["profile_id"])):
            raise ValueError("Invalid embedding profile binding")
    valid_text(embedding["model"], 160)
    valid_text(embedding["endpoint"], 4096)
    valid_text(embedding["space_id"], 160)
    dimensions = embedding["dimensions"]
    if dimensions is not None and (type(dimensions) is not int or not 1 <= dimensions <= MAX_DIMENSIONS):
        raise ValueError("Invalid embedding dimensions")
    if embedding["mode"] == "external" and (dimensions is None or embedding["endpoint"] != "external"):
        raise ValueError("Invalid external embedding configuration")
    if value["sparse_space_id"] is not None:
        valid_text(value["sparse_space_id"], 160)
    return json.loads(_json(value))


def _prepared(records: list[dict]) -> list[dict]:
    try:
        if not isinstance(records, list) or not 1 <= len(records) <= MAX_BATCH:
            raise ValueError("Invalid record batch")
        record_ids([record["id"] for record in records])
        result = []
        for record in records:
            if set(record) != {"id", "document", "metadata", "embedding", "sparse_embedding"}:
                raise ValueError("Prepared records require every field")
            document = record["document"]
            if document is not None:
                valid_text(document, MAX_DOCUMENT_CHARS, allow_empty=True)
            result.append({
                "id": record["id"], "document": document,
                "metadata": canonical_metadata(record["metadata"]),
                "embedding": canonical_dense(record["embedding"]),
                "sparse_embedding": canonical_sparse(record["sparse_embedding"]),
            })
        if len(_json(result).encode("utf-8")) > MAX_SCAN_BYTES:
            raise _error("collection_limit_exceeded", "Prepared collection batch exceeds its byte limit")
        return result
    except (KeyError, TypeError, ValueError, UnicodeError) as exc:
        raise _invalid() from exc


def _decode_vector(raw: Any) -> list[float]:
    """Restore exact float32 values even from pgvector's shortest text format."""
    values = json.loads(raw) if isinstance(raw, str) else raw
    return json.loads(postgres.vector_literal(values, code="invalid_collection_input"))


class CollectionSnapshot:
    """A bounded materialized record scope with ranking on its live read transaction."""

    def __init__(self, owner: CollectionStore, connection, tenant: str,
                 collection: dict, records: list[dict], scanned_bytes: int):
        self.collection = collection
        self.records = records
        self._owner = owner
        self._connection = connection
        self._tenant = tenant
        self._ids = {record["id"] for record in records}
        self._versions = {record["id"]: record["version"] for record in records}
        self._embedding_cache = {record["id"]: record["embedding"] for record in records if "embedding" in record}
        self._scanned_bytes = scanned_bytes
        self._active = True

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError("Collection snapshot is closed")

    def embeddings(self, ids) -> dict[str, list[float]]:
        self._require_active()
        requested = set(ids) & self._ids
        missing = sorted(requested - self._embedding_cache.keys())
        for start in range(0, len(missing), PAGE_SIZE):
            selected = missing[start:start + PAGE_SIZE]
            placeholders = ",".join("?" for _ in selected)
            expression = "v.embedding::text" if self._owner.store.is_postgres else "v.embedding"
            rows = self._connection.execute(
                f"SELECT r.id,r.version,{expression} AS embedding FROM rag_collection_records r "
                "JOIN rag_collection_versions v ON v.tenant=r.tenant AND v.id=r.version "
                f"WHERE r.tenant=? AND r.collection_id=? AND r.id IN ({placeholders}) ORDER BY r.id",
                (self._tenant, self.collection["id"], *selected),
            ).fetchall()
            if {row["id"] for row in rows} != set(selected):
                raise _error("collection_integrity_error", "Selected record versions are unavailable", 500)
            for row in rows:
                if self._versions[row["id"]] != row["version"]:
                    raise _error("collection_integrity_error", "Selected record version changed within a snapshot", 500)
                vector = _decode_vector(row["embedding"])
                self._scanned_bytes += len(_json(vector).encode("utf-8"))
                self._owner._check_bytes(self._scanned_bytes)
                self._embedding_cache[row["id"]] = vector
        # Callers cannot mutate the snapshot's cached vectors.
        return {identifier: list(self._embedding_cache[identifier]) for identifier in sorted(requested)}

    def dense(self, vector, allowed_record_ids, limit: int) -> list[tuple[str, float]]:
        self._require_active()
        if type(limit) is not int or not 1 <= limit <= 100:
            raise _invalid()
        try:
            normalized = canonical_dense(vector)
        except (ValueError, TypeError) as exc:
            raise _invalid() from exc
        expected = self.collection["configuration"]["embedding"]["dimensions"]
        if expected is not None and len(normalized) != expected:
            raise _error("embedding_dimension_mismatch", "Vector dimensions do not match the collection", 409)
        allowed = sorted(set(allowed_record_ids) & self._ids)
        if not allowed:
            return []
        if self._owner.store.is_postgres:
            # Collection and record predicates select the authorized snapshot
            # before native distance. Version rows survive concurrent GC through
            # the same PostgreSQL MVCC snapshot used for metadata/filter reads.
            rows = self._connection.execute(
                "WITH query AS (SELECT ?::vector AS embedding) "
                "SELECT r.id,GREATEST(0.0,LEAST(2.0,"
                "l2_normalize(v.embedding) <=> l2_normalize(q.embedding))) AS distance "
                "FROM rag_collection_records r JOIN rag_collection_versions v "
                "ON v.tenant=r.tenant AND v.id=r.version CROSS JOIN query q "
                "WHERE r.tenant=? AND r.collection_id=? AND r.id=ANY(?) "
                'ORDER BY distance,r.id COLLATE "C" LIMIT ?',
                (postgres.vector_literal(normalized), self._tenant, self.collection["id"], allowed, limit),
            ).fetchall()
            return [(row["id"], float(row["distance"])) for row in rows]
        vectors = self.embeddings(allowed)
        scores = [(identifier, max(0.0, min(2.0, 1.0 - cosine(normalized, values))))
                  for identifier, values in vectors.items()]
        return sorted(scores, key=lambda item: (item[1], item[0]))[:limit]


class CollectionStore:
    def __init__(self, store: Store, max_records: int = MAX_RECORDS, max_scan_bytes: int = MAX_SCAN_BYTES,
                 *, result_validator: Callable[[dict], Any] | None = None):
        if type(max_records) is not int or not 1 <= max_records <= MAX_RECORDS:
            raise ValueError("Collection record limit must be 1..10000")
        if type(max_scan_bytes) is not int or not 1 <= max_scan_bytes <= MAX_SCAN_BYTES:
            raise ValueError("Collection scan byte limit must be 1..16777216")
        if result_validator is not None and not callable(result_validator):
            raise ValueError("Collection result validator must be callable")
        self.store = store
        self.max_records = max_records
        self.max_scan_bytes = max_scan_bytes
        self._result_validator = result_validator
        with store.connect() as connection:
            if store.is_postgres:
                # Identity/ordering must not depend on the deployment's locale,
                # including at a native LIMIT boundary before Python sees rows.
                schema = SCHEMA
                for field in ("tenant", "id", "name", "collection_id", "version"):
                    schema = schema.replace(f" {field} TEXT NOT NULL", f' {field} TEXT COLLATE "C" NOT NULL')
                postgres.initialize(connection, schema)
            else:
                connection.executescript(SCHEMA)

    def _check_bytes(self, size: int) -> None:
        if size > self.max_scan_bytes:
            raise _error("collection_limit_exceeded", "Collection snapshot exceeds its scan byte limit")

    def _validated_result(self, result: dict) -> dict:
        # A trusted application validator may reject an oversized exact response.
        # Mutation callers invoke it before leaving their transaction, so failure
        # rolls back both application rows and their audit event.
        if self._result_validator is not None:
            self._result_validator(result)
        return result

    def _lock(self, connection, family: str, tenant: str, identifier: str = "") -> None:
        if self.store.is_postgres:
            postgres.lock_mutation(connection, family, tenant, identifier)

    def _lock_versions(self, connection, tenant: str, versions) -> None:
        # Bounded, deterministic stripes avoid exhausting PostgreSQL's shared
        # advisory-lock memory when a 10,000-record collection is forked/deleted.
        # Every shared payload maps to the same stripe across processes; sorted
        # acquisition gives all callers the same lock ordering.
        stripes = {int.from_bytes(hashlib.sha256(version.encode("utf-8")).digest()[:2], "big") % 32
                   for version in versions}
        for stripe in sorted(stripes):
            self._lock(connection, "collection-version-stripe", tenant, str(stripe))

    @staticmethod
    def _info(row) -> dict:
        result = dict(row)
        result.pop("tenant", None)
        result["metadata"] = json.loads(result["metadata"])
        result["configuration"] = json.loads(result["configuration"])
        return result

    def _get(self, connection, tenant: str, collection_id: str) -> dict:
        row = connection.execute(
            "SELECT c.*,(SELECT COUNT(*) FROM rag_collection_records r "
            "WHERE r.tenant=c.tenant AND r.collection_id=c.id) AS record_count "
            "FROM rag_collections c WHERE c.tenant=? AND c.id=?", (tenant, collection_id),
        ).fetchone()
        if row is None:
            raise _error("collection_not_found", "Collection was not found", 404)
        return self._info(row)

    @staticmethod
    def _expect(collection: dict, expected_generation: int | None) -> None:
        if expected_generation is not None and collection["generation"] != expected_generation:
            raise _error("collection_generation_conflict", "Collection changed; refresh its generation and retry", 409)

    @staticmethod
    def _audit(connection, tenant: str, collection_id: str, action: str, detail: dict, at: str) -> None:
        connection.execute(
            "INSERT INTO audit(timestamp,tenant,action,document_id,detail) VALUES(?,?,?,?,?)",
            (at, tenant, "collection." + action, collection_id, _json(detail)),
        )

    def create(self, tenant: str, name: str, metadata: dict, configuration: dict,
               *, get_or_create: bool = False) -> dict:
        try:
            name = validate_collection_name(name)
            metadata = canonical_metadata(metadata)
            configuration = _configuration(configuration)
            if type(get_or_create) is not bool:
                raise ValueError("Invalid create mode")
        except (TypeError, ValueError) as exc:
            raise _invalid() from exc
        with self.store.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._lock(connection, "collection-catalog", tenant)
            existing = connection.execute("SELECT id FROM rag_collections WHERE tenant=? AND name=?", (tenant, name)).fetchone()
            if existing is not None:
                if get_or_create:
                    return self._validated_result(self._get(connection, tenant, existing["id"]))
                raise _error("collection_conflict", "Collection name already exists", 409)
            identifier, at = uuid.uuid4().hex, timestamp()
            connection.execute(
                "INSERT INTO rag_collections(tenant,id,name,metadata,configuration,generation,created_at,updated_at) "
                "VALUES(?,?,?,?,?,0,?,?)", (tenant, identifier, name, _json(metadata), _json(configuration), at, at),
            )
            self._audit(connection, tenant, identifier, "create", {"generation": 0}, at)
            return self._validated_result(self._get(connection, tenant, identifier))

    def get(self, tenant: str, collection_id: str) -> dict:
        with self.store.connect() as connection:
            return self._get(connection, tenant, collection_id)

    def find_by_name(self, tenant: str, name: str) -> dict | None:
        with self.store.connect() as connection:
            row = connection.execute(
                "SELECT c.*,(SELECT COUNT(*) FROM rag_collection_records r "
                "WHERE r.tenant=c.tenant AND r.collection_id=c.id) AS record_count "
                "FROM rag_collections c WHERE c.tenant=? AND c.name=?", (tenant, name),
            ).fetchone()
            return self._info(row) if row else None

    def collection_count(self, tenant: str) -> int:
        with self.store.connect() as connection:
            return connection.execute("SELECT COUNT(*) AS total FROM rag_collections WHERE tenant=?", (tenant,)).fetchone()["total"]

    def list(self, tenant: str, limit: int = 100, offset: int = 0) -> list[dict]:
        if type(limit) is not int or not 1 <= limit <= 100 or type(offset) is not int or not 0 <= offset <= 2 ** 31 - 1:
            raise _invalid()
        with self.store.connect() as connection:
            rows = connection.execute(
                "SELECT c.*,(SELECT COUNT(*) FROM rag_collection_records r "
                "WHERE r.tenant=c.tenant AND r.collection_id=c.id) AS record_count "
                "FROM rag_collections c WHERE c.tenant=? ORDER BY c.created_at,c.id LIMIT ? OFFSET ?",
                (tenant, limit, offset),
            ).fetchall()
            return [self._info(row) for row in rows]

    def modify(self, tenant: str, cid: str, *, name: str | None = None,
               metadata: dict | None = None, expected_generation: int | None = None) -> dict:
        _generation(expected_generation)
        try:
            if name is not None:
                name = validate_collection_name(name)
            if metadata is not None:
                metadata = canonical_metadata(metadata)
        except (TypeError, ValueError) as exc:
            raise _invalid() from exc
        with self.store.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._lock(connection, "collection-catalog", tenant)
            self._lock(connection, "collection", tenant, cid)
            current = self._get(connection, tenant, cid)
            self._expect(current, expected_generation)
            next_name = current["name"] if name is None else name
            next_metadata = current["metadata"] if metadata is None else metadata
            duplicate = connection.execute(
                "SELECT id FROM rag_collections WHERE tenant=? AND name=? AND id<>?", (tenant, next_name, cid),
            ).fetchone()
            if duplicate is not None:
                raise _error("collection_conflict", "Collection name already exists", 409)
            if next_name == current["name"] and _json(next_metadata) == _json(current["metadata"]):
                return self._validated_result(current)
            at, generation = timestamp(), _next_generation(current["generation"])
            connection.execute(
                "UPDATE rag_collections SET name=?,metadata=?,generation=?,updated_at=? WHERE tenant=? AND id=?",
                (next_name, _json(next_metadata), generation, at, tenant, cid),
            )
            self._audit(connection, tenant, cid, "modify", {"generation": generation}, at)
            return self._validated_result(self._get(connection, tenant, cid))

    @staticmethod
    def _gc(connection, tenant: str, versions) -> None:
        versions = sorted(set(versions))
        for start in range(0, len(versions), MAX_BATCH):
            selected = versions[start:start + MAX_BATCH]
            placeholders = ",".join("?" for _ in selected)
            connection.execute(
                f"DELETE FROM rag_collection_versions WHERE tenant=? AND id IN ({placeholders}) AND NOT EXISTS "
                "(SELECT 1 FROM rag_collection_records r WHERE r.tenant=rag_collection_versions.tenant "
                "AND r.version=rag_collection_versions.id)",
                (tenant, *selected),
            )

    def delete_collection(self, tenant: str, cid: str, *, expected_generation: int | None = None) -> dict:
        _generation(expected_generation)
        with self.store.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._lock(connection, "collection-catalog", tenant)
            self._lock(connection, "collection", tenant, cid)
            current = self._get(connection, tenant, cid)
            self._expect(current, expected_generation)
            versions = [row["version"] for row in connection.execute(
                "SELECT version FROM rag_collection_records WHERE tenant=? AND collection_id=?", (tenant, cid),
            )]
            self._lock_versions(connection, tenant, versions)
            connection.execute("DELETE FROM rag_collection_records WHERE tenant=? AND collection_id=?", (tenant, cid))
            self._gc(connection, tenant, versions)
            connection.execute("DELETE FROM rag_collections WHERE tenant=? AND id=?", (tenant, cid))
            self._audit(connection, tenant, cid, "delete", {"generation": current["generation"], "deleted_records": len(versions)}, timestamp())
            return {"collection_id": cid, "generation": current["generation"], "deleted": True, "deleted_records": len(versions)}

    def fork(self, tenant: str, cid: str, name: str, *, metadata: dict | None = None,
             expected_generation: int | None = None) -> dict:
        _generation(expected_generation)
        try:
            name = validate_collection_name(name)
            if metadata is not None:
                metadata = canonical_metadata(metadata)
        except (TypeError, ValueError) as exc:
            raise _invalid() from exc
        with self.store.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._lock(connection, "collection-catalog", tenant)
            self._lock(connection, "collection", tenant, cid)
            source = self._get(connection, tenant, cid)
            self._expect(source, expected_generation)
            if connection.execute("SELECT 1 FROM rag_collections WHERE tenant=? AND name=?", (tenant, name)).fetchone():
                raise _error("collection_conflict", "Collection name already exists", 409)
            versions = [row["version"] for row in connection.execute(
                "SELECT version FROM rag_collection_records WHERE tenant=? AND collection_id=? ORDER BY id", (tenant, cid),
            )]
            if len(versions) > self.max_records:
                raise _error("collection_limit_exceeded", "Collection exceeds its record limit")
            self._lock_versions(connection, tenant, versions)
            identifier, at = uuid.uuid4().hex, timestamp()
            connection.execute(
                "INSERT INTO rag_collections(tenant,id,name,metadata,configuration,generation,created_at,updated_at,"
                "fork_source_id,fork_source_generation) VALUES(?,?,?,?,?,0,?,?,?,?)",
                (tenant, identifier, name, _json(source["metadata"] if metadata is None else metadata),
                 _json(source["configuration"]), at, at, cid, source["generation"]),
            )
            connection.execute(
                "INSERT INTO rag_collection_records(tenant,collection_id,id,version,created_at) "
                "SELECT tenant,?,id,version,created_at FROM rag_collection_records WHERE tenant=? AND collection_id=?",
                (identifier, tenant, cid),
            )
            self._audit(connection, tenant, identifier, "fork", {
                "generation": 0, "source_id": cid, "source_generation": source["generation"], "records": len(versions),
            }, at)
            return self._validated_result(self._get(connection, tenant, identifier))

    @contextmanager
    def read_snapshot(self, tenant: str, cid: str, *, include_embeddings: bool = False,
                      record_ids=None) -> Iterator[CollectionSnapshot]:
        if type(include_embeddings) is not bool:
            raise _invalid()
        selected_ids = None
        if record_ids is not None:
            try:
                if not isinstance(record_ids, (list, tuple, set)) or len(record_ids) > self.max_records:
                    raise ValueError("Invalid record selection")
                selected_ids = {valid_text(item, 128) for item in record_ids}
            except (TypeError, ValueError) as exc:
                raise _invalid() from exc
        with self.store.connect() as connection:
            if self.store.is_postgres:
                connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            else:
                connection.execute("BEGIN")
            current = self._get(connection, tenant, cid)
            if selected_ids is None and current["record_count"] > self.max_records:
                raise _error("collection_limit_exceeded", "Collection exceeds its record limit")
            records, scanned, last = [], 0, None
            # Keyset pages bound each driver's fetched result, instead of fetching
            # the entire collection and noticing a materialization limit too late.
            while selected_ids is None or selected_ids:
                fields = "r.id,r.version,r.created_at,v.document,v.metadata,v.sparse_embedding"
                if include_embeddings:
                    expression = "v.embedding::text" if self.store.is_postgres else "v.embedding"
                    fields += f",{expression} AS embedding"
                predicate, parameters = "", [tenant, cid]
                if selected_ids is not None:
                    # SQLite's bind count stays bounded even for an internal
                    # caller selecting up to the entire configured collection.
                    selected = sorted(selected_ids)[:PAGE_SIZE]
                    predicate += " AND r.id IN (" + ",".join("?" for _ in selected) + ")"
                    parameters.extend(selected)
                elif last is not None:
                    key = 'r.id COLLATE "C"' if self.store.is_postgres else "r.id"
                    predicate += f" AND {key}>?"
                    parameters.append(last)
                order = 'r.id COLLATE "C"' if self.store.is_postgres else "r.id"
                rows = connection.execute(
                    f"SELECT {fields} FROM rag_collection_records r JOIN rag_collection_versions v "
                    "ON v.tenant=r.tenant AND v.id=r.version WHERE r.tenant=? AND r.collection_id=?" +
                    predicate + f" ORDER BY {order} LIMIT ?", (*parameters, PAGE_SIZE),
                ).fetchall()
                for row in rows:
                    record = dict(row)
                    record["metadata"] = json.loads(record["metadata"])
                    record["sparse_embedding"] = json.loads(record["sparse_embedding"])
                    if include_embeddings:
                        record["embedding"] = _decode_vector(record["embedding"])
                    scanned += len(_json(record).encode("utf-8"))
                    self._check_bytes(scanned)
                    records.append(record)
                if selected_ids is not None:
                    selected_ids.difference_update(selected)
                elif len(rows) < PAGE_SIZE:
                    break
                else:
                    last = rows[-1]["id"]
            snapshot = CollectionSnapshot(self, connection, tenant, current, records, scanned)
            try:
                yield snapshot
            finally:
                snapshot._active = False

    def write(self, tenant: str, cid: str, mode: str, records: list[dict], *, expected_generation: int) -> dict:
        _generation(expected_generation, optional=False)
        if mode not in {"add", "update", "upsert"}:
            raise _invalid()
        prepared = _prepared(records)
        with self.store.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._lock(connection, "collection", tenant, cid)
            current = self._get(connection, tenant, cid)
            self._expect(current, expected_generation)
            dimensions = {len(record["embedding"]) for record in prepared}
            expected = current["configuration"]["embedding"]["dimensions"]
            if len(dimensions) != 1 or (expected is not None and dimensions != {expected}):
                raise _error("embedding_dimension_mismatch", "Vector dimensions do not match the collection", 409)
            sparse_space = current["configuration"]["sparse_space_id"]
            if any(record["sparse_embedding"] is not None and record["sparse_embedding"]["space_id"] != sparse_space
                   for record in prepared):
                raise _error("embedding_space_mismatch", "Sparse vector space does not match the collection", 409)
            identifiers = [record["id"] for record in prepared]
            placeholders = ",".join("?" for _ in identifiers)
            expression = "v.embedding::text" if self.store.is_postgres else "v.embedding"
            rows = connection.execute(
                f"SELECT r.id,r.version,r.created_at,v.document,v.metadata,v.sparse_embedding,{expression} AS embedding "
                "FROM rag_collection_records r JOIN rag_collection_versions v ON v.tenant=r.tenant AND v.id=r.version "
                f"WHERE r.tenant=? AND r.collection_id=? AND r.id IN ({placeholders})", (tenant, cid, *identifiers),
            ).fetchall()
            existing = {row["id"]: dict(row) for row in rows}
            if mode == "add" and existing:
                raise _error("collection_conflict", "Add requires every record ID to be new", 409)
            if mode == "update" and len(existing) != len(prepared):
                raise _error("collection_conflict", "Update requires every record ID to exist", 409)
            added = len(prepared) - len(existing)
            if current["record_count"] + added > self.max_records:
                raise _error("collection_limit_exceeded", "Collection exceeds its record limit")
            changed = []
            for record in prepared:
                previous = existing.get(record["id"])
                same = previous is not None and (
                    previous["document"] == record["document"]
                    and _json(json.loads(previous["metadata"])) == _json(record["metadata"])
                    and _decode_vector(previous["embedding"]) == record["embedding"]
                    and _json(json.loads(previous["sparse_embedding"])) == _json(record["sparse_embedding"])
                )
                if not same:
                    changed.append(record)
            if not changed:
                return {"collection_id": cid, "generation": current["generation"], "changed": 0, "added": 0, "updated": 0}
            generation = _next_generation(current["generation"])
            retired = {existing[record["id"]]["version"] for record in changed if record["id"] in existing}
            self._lock_versions(connection, tenant, retired)
            at = timestamp()
            for record in changed:
                version = uuid.uuid4().hex
                vector = postgres.vector_literal(record["embedding"])
                vector_parameter = "?::vector" if self.store.is_postgres else "?"
                connection.execute(
                    "INSERT INTO rag_collection_versions(tenant,id,document,metadata,embedding,embedding_dimensions,"
                    "sparse_embedding,created_at) VALUES(?,?,?,?," + vector_parameter + ",?,?,?)",
                    (tenant, version, record["document"], _json(record["metadata"]), vector, len(record["embedding"]),
                     _json(record["sparse_embedding"]), at),
                )
                created = existing[record["id"]]["created_at"] if record["id"] in existing else at
                connection.execute(
                    "INSERT INTO rag_collection_records(tenant,collection_id,id,version,created_at) VALUES(?,?,?,?,?) "
                    "ON CONFLICT(tenant,collection_id,id) DO UPDATE SET version=excluded.version",
                    (tenant, cid, record["id"], version, created),
                )
            self._gc(connection, tenant, retired)
            configuration = current["configuration"]
            configuration["embedding"]["dimensions"] = next(iter(dimensions))
            connection.execute(
                "UPDATE rag_collections SET configuration=?,generation=?,updated_at=? WHERE tenant=? AND id=?",
                (_json(configuration), generation, at, tenant, cid),
            )
            self._audit(connection, tenant, cid, mode, {
                "generation": generation, "changed": len(changed), "added": added, "updated": len(changed) - added,
            }, at)
            return {"collection_id": cid, "generation": generation, "changed": len(changed), "added": added,
                    "updated": len(changed) - added}

    def delete_records(self, tenant: str, cid: str, ids: list[str], *, expected_generation: int) -> dict:
        _generation(expected_generation, optional=False)
        try:
            if ids:
                ids = record_ids(ids)
            elif not isinstance(ids, list):
                raise ValueError("Invalid record selection")
        except (ValueError, TypeError) as exc:
            raise _invalid() from exc
        with self.store.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._lock(connection, "collection", tenant, cid)
            current = self._get(connection, tenant, cid)
            self._expect(current, expected_generation)
            if not ids:
                return {"collection_id": cid, "generation": current["generation"], "deleted": 0}
            placeholders = ",".join("?" for _ in ids)
            rows = connection.execute(
                f"SELECT id,version FROM rag_collection_records WHERE tenant=? AND collection_id=? AND id IN ({placeholders})",
                (tenant, cid, *ids),
            ).fetchall()
            if not rows:
                return {"collection_id": cid, "generation": current["generation"], "deleted": 0}
            generation = _next_generation(current["generation"])
            retired = {row["version"] for row in rows}
            self._lock_versions(connection, tenant, retired)
            connection.execute(
                f"DELETE FROM rag_collection_records WHERE tenant=? AND collection_id=? AND id IN ({placeholders})",
                (tenant, cid, *ids),
            )
            self._gc(connection, tenant, retired)
            at = timestamp()
            connection.execute(
                "UPDATE rag_collections SET generation=?,updated_at=? WHERE tenant=? AND id=?", (generation, at, tenant, cid),
            )
            self._audit(connection, tenant, cid, "delete_records", {"generation": generation, "deleted": len(rows)}, at)
            return {"collection_id": cid, "generation": generation, "deleted": len(rows)}

    def validate_import(self, connection) -> None:
        """Validate imported relationships/configuration without invoking models."""
        collections = {}
        for row in connection.execute("SELECT * FROM rag_collections"):
            item = dict(row)
            validate_collection_name(item["name"])
            canonical_metadata(json.loads(item["metadata"]))
            configuration = _configuration(json.loads(item["configuration"]))
            _generation(item["generation"], optional=False)
            timestamp(item["created_at"])
            timestamp(item["updated_at"])
            if (item["fork_source_id"] is None) != (item["fork_source_generation"] is None):
                raise ValueError("Invalid imported fork lineage")
            _generation(item["fork_source_generation"])
            collections[(item["tenant"], item["id"])] = configuration
        expression = "embedding::text" if self.store.is_postgres else "embedding"
        versions = {}
        for row in connection.execute(
            f"SELECT tenant,id,document,metadata,{expression} AS embedding,embedding_dimensions,sparse_embedding,created_at "
            "FROM rag_collection_versions",
        ):
            item = dict(row)
            if item["document"] is not None:
                valid_text(item["document"], MAX_DOCUMENT_CHARS, allow_empty=True)
            canonical_metadata(json.loads(item["metadata"]))
            vector = _decode_vector(item["embedding"])
            canonical = canonical_dense(vector)
            if canonical != vector or len(vector) != item["embedding_dimensions"]:
                raise ValueError("Invalid imported normalized vector")
            sparse = canonical_sparse(json.loads(item["sparse_embedding"]))
            timestamp(item["created_at"])
            versions[(item["tenant"], item["id"])] = (len(vector), sparse)
        referenced, counts = set(), {}
        for row in connection.execute("SELECT * FROM rag_collection_records"):
            item = dict(row)
            valid_text(item["id"], 128)
            timestamp(item["created_at"])
            key = (item["tenant"], item["collection_id"])
            version = (item["tenant"], item["version"])
            if key not in collections or version not in versions:
                raise ValueError("Invalid imported record reference")
            configuration = collections[key]
            dimension, sparse = versions[version]
            if configuration["embedding"]["dimensions"] != dimension:
                raise ValueError("Imported collection dimensions do not match records")
            if sparse is not None and configuration["sparse_space_id"] != sparse["space_id"]:
                raise ValueError("Imported collection sparse space does not match records")
            counts[key] = counts.get(key, 0) + 1
            if counts[key] > self.max_records:
                raise ValueError("Imported collection exceeds its record limit")
            referenced.add(version)
        if set(versions) != referenced:
            raise ValueError("Imported collection contains unreferenced payload versions")
