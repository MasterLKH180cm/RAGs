"""Tenant-scoped, versioned source records and transactional derived indexes."""
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from rags.native import cosine, equal_value
from rags import database, postgres
from rags.registry import CapabilityError

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
 tenant TEXT NOT NULL, id TEXT NOT NULL, title TEXT NOT NULL, source TEXT NOT NULL,
 body TEXT NOT NULL, metadata TEXT NOT NULL, checksum TEXT NOT NULL, updated_at TEXT NOT NULL,
 PRIMARY KEY(tenant,id)
);
CREATE TABLE IF NOT EXISTS chunks (
 id TEXT PRIMARY KEY, tenant TEXT NOT NULL, document_id TEXT NOT NULL,
 ordinal INTEGER NOT NULL, section TEXT NOT NULL, content TEXT NOT NULL,
 embedding TEXT NOT NULL, embedding_model TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_tenant_document ON chunks(tenant,document_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(chunk_id UNINDEXED,content,tokenize='unicode61');
CREATE TABLE IF NOT EXISTS edges (
 tenant TEXT NOT NULL, document_id TEXT NOT NULL, source TEXT NOT NULL, target TEXT NOT NULL,
 PRIMARY KEY(tenant,document_id,source,target)
);
CREATE TABLE IF NOT EXISTS audit (
 id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, tenant TEXT NOT NULL,
 action TEXT NOT NULL, document_id TEXT, detail TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS document_revisions (
 tenant TEXT NOT NULL, id TEXT NOT NULL, revision TEXT NOT NULL,
 title TEXT NOT NULL, source TEXT NOT NULL, body TEXT NOT NULL, metadata TEXT NOT NULL,
 checksum TEXT NOT NULL, recorded_at TEXT NOT NULL, valid_from TEXT NOT NULL, valid_to TEXT,
 profile_id TEXT NOT NULL, index_profile TEXT NOT NULL,
 PRIMARY KEY(tenant,id,revision)
);
CREATE INDEX IF NOT EXISTS revision_validity ON document_revisions(tenant,id,valid_from,valid_to);
CREATE TABLE IF NOT EXISTS graph_relations (
 tenant TEXT NOT NULL, document_id TEXT NOT NULL, revision TEXT NOT NULL,
 source TEXT NOT NULL, predicate TEXT NOT NULL, target TEXT NOT NULL, evidence_id TEXT NOT NULL,
 PRIMARY KEY(tenant,document_id,revision,source,predicate,target,evidence_id)
);
CREATE INDEX IF NOT EXISTS graph_relations_evidence ON graph_relations(tenant,evidence_id);
"""
DOC_COLUMNS = {"revision": "TEXT NOT NULL DEFAULT 'legacy'", "profile_id": "TEXT NOT NULL DEFAULT 'legacy'",
               "index_profile": "TEXT NOT NULL DEFAULT '{\"legacy\":true}'", "valid_from": "TEXT", "valid_to": "TEXT"}
CHUNK_COLUMNS = {
    "revision": "TEXT NOT NULL DEFAULT 'legacy'", "profile_id": "TEXT NOT NULL DEFAULT 'legacy'",
    "role": "TEXT NOT NULL DEFAULT 'leaf'", "parent_id": "TEXT", "start_char": "INTEGER",
    "end_char": "INTEGER", "token_count": "INTEGER", "tokenizer_id": "TEXT", "token_unit": "TEXT",
    "context": "TEXT NOT NULL DEFAULT ''", "section_path": "TEXT NOT NULL DEFAULT '[]'",
    "evidence_ids": "TEXT NOT NULL DEFAULT '[]'", "level": "INTEGER NOT NULL DEFAULT 0",
    "processing_metadata": "TEXT NOT NULL DEFAULT '{}'",
}
CHUNK_FIELDS = ("id", "tenant", "document_id", "ordinal", "section", "content", "embedding", "embedding_model", *CHUNK_COLUMNS)
REVISION_FIELDS = ("tenant", "id", "revision", "title", "source", "body", "metadata", "checksum",
                   "recorded_at", "valid_from", "valid_to", "profile_id", "index_profile")
REVISION_SCOPE_FIELDS = tuple(field for field in REVISION_FIELDS if field not in {"body", "checksum"})


def _sqlite_allowed_ids(connection, identifiers: set[str]) -> None:
    """Bind large authorized sets without JSON extensions or parameter limits.

    Every caller owns a fresh connection, so this private TEMP table is isolated
    from other requests and disappears when the shared lifecycle closes it.
    """
    connection.execute("CREATE TEMP TABLE authorized_chunks(id TEXT PRIMARY KEY) WITHOUT ROWID")
    connection.executemany("INSERT INTO authorized_chunks(id) VALUES(?)", ((key,) for key in identifiers))


def timestamp(value: str | None = None) -> str:
    """Canonical UTC ISO timestamps make SQLite comparisons deterministic."""
    if value is None:
        parsed = datetime.now(UTC)
    else:
        try:
            parsed = datetime.fromisoformat(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Timestamps must be ISO 8601 with an explicit timezone") from exc
        if parsed.tzinfo is None:
            raise ValueError("Timestamps require an explicit timezone")
    return parsed.astimezone(UTC).isoformat(timespec="microseconds")


class Store:
    def __init__(self, path: str):
        self.path = path
        self.is_postgres = postgres.is_postgres(path)
        self.storage_backend = "postgresql" if self.is_postgres else "sqlite"
        self.vector_search = "pgvector_exact" if self.is_postgres else "python_exact"
        if path == ":memory:":
            raise ValueError("Use a temporary file: independent SQLite connections cannot share :memory:")
        if not self.is_postgres:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            if self.is_postgres:
                postgres.initialize(conn, SCHEMA)
            else:
                conn.executescript(SCHEMA)
            conn.execute("BEGIN IMMEDIATE")
            for table, additions in (("documents", DOC_COLUMNS), ("chunks", CHUNK_COLUMNS)):
                present = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
                for name, definition in additions.items():
                    if name not in present:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")
            conn.execute("CREATE INDEX IF NOT EXISTS chunks_revision ON chunks(tenant,document_id,revision,role)")
            if self.is_postgres:
                revisions = {row["name"] for row in conn.execute("PRAGMA table_info(document_revisions)")}
                if "revision_order" not in revisions:
                    conn.execute("ALTER TABLE document_revisions ADD COLUMN revision_order BIGSERIAL")
                postgres.migrate_vector_column(conn, "chunks", "embedding")
                conn.execute("CREATE INDEX IF NOT EXISTS chunks_vector_profile ON chunks(tenant,embedding_model,profile_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS chunks_fts_search ON chunks_fts USING GIN(search)")
            # Migration never calls models or invents offsets/token counts.
            conn.execute("UPDATE documents SET valid_from=updated_at WHERE valid_from IS NULL")
            conn.execute(
                "INSERT OR IGNORE INTO document_revisions(" + ",".join(REVISION_FIELDS) + ") "
                "SELECT tenant,id,revision,title,source,body,metadata,checksum,updated_at,valid_from,"
                "valid_to,profile_id,index_profile FROM documents"
            )

    def connect(self):
        return database.connect(self.path, foreign_keys=True)

    @staticmethod
    def decode_document(row) -> dict:
        value = dict(row)
        for key in ("metadata", "index_profile"):
            if key in value:
                value[key] = json.loads(value[key])
        return value

    def get_document(self, tenant: str, doc_id: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE tenant=? AND id=?", (tenant, doc_id)).fetchone()
        return self.decode_document(row) if row else None

    def list_documents(self, tenant: str) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT id,title,source,metadata,updated_at,revision,profile_id,index_profile,valid_from,valid_to "
                "FROM documents WHERE tenant=? ORDER BY updated_at DESC", (tenant,)
            ).fetchall()
        return [self.decode_document(row) for row in rows]

    def revisions(self, tenant: str, doc_id: str) -> list[dict]:
        order_column = "revision_order" if self.is_postgres else "rowid"
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT id,revision,title,source,metadata,recorded_at,valid_from,valid_to,profile_id,index_profile "
                f"FROM document_revisions WHERE tenant=? AND id=? ORDER BY recorded_at DESC,{order_column} DESC", (tenant, doc_id)
            ).fetchall()
        return [self.decode_document(row) for row in rows]

    def has_index(self, tenant: str, doc_id: str, model: str, profile_id: str | None = None) -> bool:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT c.embedding_model,c.profile_id FROM chunks c JOIN documents d "
                "ON d.tenant=c.tenant AND d.id=c.document_id AND d.revision=c.revision "
                "WHERE c.tenant=? AND c.document_id=? AND c.role='leaf'", (tenant, doc_id)
            ).fetchall()
        return bool(rows) and all(r["embedding_model"] == model and (profile_id is None or r["profile_id"] == profile_id) for r in rows)

    def commit_revision(self, tenant: str, document: dict, chunks: list[dict], relations: list[dict], expected_revision: str | None) -> None:
        """Provider work finishes before this short compare-and-swap transaction."""
        vectors = {}
        if self.is_postgres:
            for item in chunks:
                vectors[item["id"]] = postgres.vector_literal(item["embedding"])
                if (len(item["embedding"]) != document["index_profile"].get("embedding_dimensions")
                        or item["embedding_model"] != document["index_profile"].get("embedding_model")):
                    raise CapabilityError("invalid_embedding", "Vector does not match its document index profile", 502)
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if self.is_postgres:
                postgres.lock_mutation(conn, "document", tenant, document["id"])
            current = conn.execute("SELECT revision,valid_from,valid_to FROM documents WHERE tenant=? AND id=?", (tenant, document["id"])).fetchone()
            if (current["revision"] if current else None) != expected_revision:
                raise CapabilityError("revision_conflict", "Document changed during indexing; retry against its current revision", 409)
            if current:
                # Reindexing can produce several processing revisions for one
                # source interval. Every superseded copy must close together.
                conn.execute(
                    "UPDATE document_revisions SET valid_to=? WHERE tenant=? AND id=? AND valid_from<? "
                    "AND (valid_to IS NULL OR valid_to>?)",
                    (document["valid_from"], tenant, document["id"], document["valid_from"], document["valid_from"]),
                )
            values = (
                tenant, document["id"], document["revision"], document["title"], document["source"], document["body"],
                json.dumps(document["metadata"]), document["checksum"], document["recorded_at"],
                document["valid_from"], document.get("valid_to"), document["profile_id"], json.dumps(document["index_profile"]),
            )
            conn.execute("INSERT INTO document_revisions(" + ",".join(REVISION_FIELDS) + ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", values)
            conn.execute(
                "INSERT INTO documents(tenant,id,revision,title,source,body,metadata,checksum,updated_at,valid_from,valid_to,profile_id,index_profile) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(tenant,id) DO UPDATE SET "
                "revision=excluded.revision,title=excluded.title,source=excluded.source,body=excluded.body,"
                "metadata=excluded.metadata,checksum=excluded.checksum,updated_at=excluded.updated_at,"
                "valid_from=excluded.valid_from,valid_to=excluded.valid_to,profile_id=excluded.profile_id,index_profile=excluded.index_profile",
                values,
            )
            for item in chunks:
                fields = CHUNK_FIELDS
                encoded = dict(item, tenant=tenant, document_id=document["id"], revision=document["revision"], profile_id=document["profile_id"])
                for key in ("embedding", "section_path", "evidence_ids", "processing_metadata"):
                    encoded[key] = json.dumps(encoded.get(key, {} if key == "processing_metadata" else []))
                if self.is_postgres:
                    encoded["embedding"] = vectors[item["id"]]
                defaults = {"role": "leaf", "context": "", "level": 0}
                placeholders = ["?::vector" if self.is_postgres and field == "embedding" else "?" for field in fields]
                conn.execute(
                    "INSERT INTO chunks(" + ",".join(fields) + ") VALUES(" + ",".join(placeholders) + ")",
                    [encoded.get(key, defaults.get(key)) for key in fields],
                )
                conn.execute("INSERT INTO chunks_fts(chunk_id,content) VALUES(?,?)", (
                    item["id"], f"{document['title']}\n{item['section']}\n{item.get('context', '')}\n{item['content']}"
                ))
            for edge in relations:
                conn.execute("INSERT OR IGNORE INTO graph_relations VALUES(?,?,?,?,?,?,?)", (
                    tenant, document["id"], document["revision"], edge["source"], edge["predicate"], edge["target"], edge["evidence_id"]
                ))
            conn.execute("INSERT INTO audit(timestamp,tenant,action,document_id,detail) VALUES(?,?,?,?,?)", (
                document["recorded_at"], tenant, "upsert", document["id"],
                json.dumps({"revision": document["revision"], "profile_id": document["profile_id"], "chunks": len(chunks)}),
            ))

    @staticmethod
    def remove_chunks(conn, tenant: str, doc_id: str) -> None:
        conn.execute("DELETE FROM chunks_fts WHERE chunk_id IN (SELECT id FROM chunks WHERE tenant=? AND document_id=?)", (tenant, doc_id))
        conn.execute("DELETE FROM chunks WHERE tenant=? AND document_id=?", (tenant, doc_id))
        conn.execute("DELETE FROM edges WHERE tenant=? AND document_id=?", (tenant, doc_id))
        conn.execute("DELETE FROM graph_relations WHERE tenant=? AND document_id=?", (tenant, doc_id))

    def delete(self, tenant: str, doc_id: str, now: str,
               expected_revision: str | None = None) -> bool:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if self.is_postgres:
                postgres.lock_mutation(conn, "document", tenant, doc_id)
            row = conn.execute("SELECT id,revision FROM documents WHERE tenant=? AND id=?",
                               (tenant, doc_id)).fetchone()
            if not row:
                return False
            if expected_revision is not None and row["revision"] != expected_revision:
                raise CapabilityError("revision_conflict",
                                      "Document changed since it was selected; refresh and retry", 409)
            self.remove_chunks(conn, tenant, doc_id)
            conn.execute("DELETE FROM document_revisions WHERE tenant=? AND id=?", (tenant, doc_id))
            conn.execute("DELETE FROM documents WHERE tenant=? AND id=?", (tenant, doc_id))
            conn.execute("INSERT INTO audit(timestamp,tenant,action,document_id,detail) VALUES(?,?,?,?,?)", (now, tenant, "delete", doc_id, "{}"))
        return True

    def scope(self, tenant: str, *, as_of: str | None = None, filters: dict | None = None,
              include_embeddings: bool = True) -> list[dict]:
        """Choose one valid revision per document before retrieval or expansion.

        Read revisions and chunks from one transaction snapshot. Both backends
        push authorized revision pairs into the chunk query; unrelated historical
        content and vectors never become Python candidate rows.
        """
        at = timestamp(as_of)
        filters = filters or {}
        order_column = "revision_order" if self.is_postgres else "rowid"
        revision_fields = ",".join(REVISION_SCOPE_FIELDS)
        with self.connect() as conn:
            if self.is_postgres:
                conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            else:
                # SELECT alone does not begin a Python sqlite3 transaction.
                conn.execute("BEGIN")
            rows = conn.execute(
                f"SELECT {revision_fields} FROM (SELECT *,ROW_NUMBER() OVER ("
                "PARTITION BY id ORDER BY recorded_at DESC,selection_order DESC) AS source_order "
                f"FROM (SELECT {revision_fields},"
                f"{order_column} AS selection_order,ROW_NUMBER() OVER ("
                f"PARTITION BY id,valid_from ORDER BY recorded_at DESC,{order_column} DESC) AS processing_order "
                "FROM document_revisions WHERE tenant=? AND valid_from<=?) AS revisions "
                "WHERE processing_order=1 AND (valid_to IS NULL OR valid_to>?)) AS sources "
                "WHERE source_order=1 "
                "ORDER BY recorded_at DESC,selection_order DESC",
                (tenant, at, at),
            ).fetchall()
            selected = {}
            for row in rows:
                doc = self.decode_document(row)
                if filters.get("source") is not None and doc["source"] != filters["source"]:
                    continue
                if any(k not in doc["metadata"] or not equal_value(doc["metadata"][k], v)
                       for k, v in filters.get("metadata", {}).items()):
                    continue
                if filters.get("valid_after") and doc["valid_from"] < timestamp(filters["valid_after"]):
                    continue
                if filters.get("valid_before") and doc["valid_from"] >= timestamp(filters["valid_before"]):
                    continue
                selected[(doc["id"], doc["revision"])] = doc
            if not selected:
                return []
            fields = "c.*" if include_embeddings else ",".join(
                "c." + field for field in CHUNK_FIELDS if field != "embedding"
            )
            if self.is_postgres:
                revision_keys = json.dumps(
                    [{"document_id": doc_id, "revision": revision}
                     for doc_id, revision in selected], separators=(",", ":")
                )
                chunk_rows = conn.execute(
                    f"SELECT {fields} FROM chunks c "
                    "JOIN jsonb_to_recordset(?::jsonb) AS scope(document_id text, revision text) "
                    "ON scope.document_id=c.document_id AND scope.revision=c.revision "
                    "WHERE c.tenant=? ORDER BY c.id",
                    (revision_keys, tenant),
                ).fetchall()
            else:
                conn.execute(
                    "CREATE TEMP TABLE authorized_revisions("
                    "document_id TEXT NOT NULL,revision TEXT NOT NULL,"
                    "PRIMARY KEY(document_id,revision)) WITHOUT ROWID"
                )
                conn.executemany("INSERT INTO authorized_revisions VALUES(?,?)", selected)
                chunk_rows = conn.execute(
                    f"SELECT {fields} FROM chunks c WHERE c.tenant=? AND (c.document_id,c.revision) "
                    "IN (SELECT document_id,revision FROM authorized_revisions) "
                    # Preserve the existing chunks_revision index traversal,
                    # including ingestion order for nodes with the same role.
                    "ORDER BY c.document_id,c.revision,c.role,c.rowid", (tenant,)
                ).fetchall()
        result = []
        for row in chunk_rows:
            chunk = dict(row)
            doc = selected.get((chunk["document_id"], chunk["revision"]))
            if doc is None:
                continue
            for key in ("section_path", "evidence_ids", "processing_metadata"):
                chunk[key] = json.loads(chunk[key])
            chunk.update(title=doc["title"], source=doc["source"], metadata=doc["metadata"],
                         updated_at=doc["recorded_at"], valid_from=doc["valid_from"], valid_to=doc["valid_to"],
                         index_profile=doc["index_profile"])
            result.append(chunk)
        return result

    def all_chunks(self, tenant: str, embedding_model: str) -> list[dict]:
        return [c for c in self.scope(tenant) if c["embedding_model"] == embedding_model and c["role"] == "leaf"]

    def fetch_chunks(self, tenant: str, ids: list[str]) -> list[dict]:
        allowed = set(ids)
        return [c for c in self.scope(tenant) if c["id"] in allowed]

    def dense(self, tenant: str, vector: list[float], limit: int, allowed_ids: set[str], *,
              embedding_model: str, embedding_endpoint: str, embedding_space_id: str = "") -> list[str]:
        """Rank precisely the server-authorized revision/filter snapshot."""
        if not allowed_ids:
            return []
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Dense candidate limit must be 1..100")
        literal = postgres.vector_literal(vector) if self.is_postgres else None
        with self.connect() as connection:
            if self.is_postgres:
                connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            dimensions = "vector_dims(c.embedding) AS dimensions," if self.is_postgres else "c.embedding,"
            if not self.is_postgres:
                _sqlite_allowed_ids(connection, allowed_ids)
            predicate = (" AND c.id=ANY(?)" if self.is_postgres else
                         " AND c.id IN (SELECT id FROM authorized_chunks)")
            parameters = (tenant, sorted(allowed_ids)) if self.is_postgres else (tenant,)
            rows = connection.execute(
                "SELECT c.id,c.embedding_model,c.profile_id," + dimensions +
                "r.profile_id AS revision_profile_id,r.index_profile FROM chunks c JOIN document_revisions r "
                "ON r.tenant=c.tenant AND r.id=c.document_id AND r.revision=c.revision "
                "WHERE c.tenant=?" + predicate, parameters,
            ).fetchall()
            candidates = rows
            try:
                for row in candidates:
                    profile = json.loads(row["index_profile"])
                    actual_dimensions = row["dimensions"] if self.is_postgres else len(json.loads(row["embedding"]))
                    if (row["embedding_model"] != embedding_model or profile.get("embedding_model") != embedding_model
                            or profile.get("embedding_endpoint") != embedding_endpoint
                            or profile.get("embedding_space_id", "") != embedding_space_id
                            or type(profile.get("embedding_dimensions")) is not int
                            or profile["embedding_dimensions"] != len(vector) or actual_dimensions != len(vector)
                            or row["profile_id"] != row["revision_profile_id"]):
                        raise ValueError("Incompatible embedding profile")
            except (ValueError, TypeError, AttributeError) as exc:
                raise CapabilityError("requires_reindex", "Visible vectors have incompatible index profiles or dimensions; reindex them", 409) from exc
            if self.is_postgres:
                distance = postgres.cosine_distance("c.embedding")
                result = connection.execute(
                    "WITH query AS (SELECT ?::vector AS embedding) "
                    f"SELECT c.id,{distance} AS distance FROM chunks c CROSS JOIN query q "
                    "WHERE c.tenant=? AND c.id=ANY(?) ORDER BY distance,c.id LIMIT ?",
                    (literal, tenant, [row["id"] for row in candidates], limit),
                ).fetchall()
                return [row["id"] for row in result]
        try:
            scores = [(cosine(vector, json.loads(row["embedding"])), row["id"]) for row in candidates]
        except (ValueError, TypeError) as exc:
            raise CapabilityError("requires_reindex", "Visible vectors have incompatible dimensions; reindex them", 409) from exc
        return [identifier for _, identifier in sorted(scores, key=lambda item: (-item[0], item[1]))[:limit]]

    def sparse(self, tenant: str, match_query: str, limit: int, allowed_ids: set[str] | None = None) -> list[str]:
        if not match_query:
            return []
        if allowed_ids is None:
            allowed_ids = {c["id"] for c in self.scope(tenant) if c["role"] == "leaf"}
        if not allowed_ids:
            return []
        if self.is_postgres:
            # The caller constructs an OR of quoted literal tokens. Each token
            # stays a bound value in PostgreSQL's plain-query parser.
            tokens = re.findall(r'"([^"]+)"', match_query) or re.findall(r"\w+", match_query)
            tokens = [token for token in tokens if token.upper() != "OR"][:12]
            if not tokens:
                return []
            query = " || ".join("plainto_tsquery('simple',?)" for _ in tokens)
            with self.connect() as connection:
                rows = connection.execute(
                    f"WITH query AS (SELECT {query} AS terms) SELECT c.id FROM chunks_fts f "
                    "JOIN chunks c ON c.id=f.chunk_id CROSS JOIN query q "
                    "WHERE c.tenant=? AND c.id=ANY(?) AND f.search @@ q.terms "
                    "ORDER BY ts_rank_cd(f.search,q.terms) DESC,c.id LIMIT ?",
                    (*tokens, tenant, sorted(allowed_ids), limit),
                ).fetchall()
            return [row["id"] for row in rows]
        with self.connect() as conn:
            _sqlite_allowed_ids(conn, allowed_ids)
            rows = conn.execute(
                "SELECT c.id FROM chunks_fts f JOIN chunks c ON c.id=f.chunk_id "
                "WHERE c.tenant=? AND c.id IN (SELECT id FROM authorized_chunks) AND chunks_fts MATCH ? "
                "ORDER BY bm25(chunks_fts),c.id LIMIT ?", (tenant, match_query, limit)
            ).fetchall()
        return [r["id"] for r in rows]

    def relations(self, tenant: str, allowed_ids: set[str]) -> list[dict]:
        if not allowed_ids:
            return []
        with self.connect() as conn:
            if self.is_postgres:
                predicate, parameters = "evidence_id=ANY(?)", (tenant, sorted(allowed_ids))
            else:
                _sqlite_allowed_ids(conn, allowed_ids)
                predicate, parameters = "evidence_id IN (SELECT id FROM authorized_chunks)", (tenant,)
            rows = conn.execute(
                "SELECT * FROM graph_relations WHERE tenant=? AND " + predicate +
                " ORDER BY document_id,revision,source,predicate,target,evidence_id", parameters,
            ).fetchall()
        return [dict(r) for r in rows]

    def graph_documents(self, tenant: str, terms: list[str], limit: int = 20) -> list[str]:
        allowed = {c["id"] for c in self.scope(tenant)}
        edges = self.relations(tenant, allowed)
        matched = [e["document_id"] for e in edges if any(t in (e["source"] + " " + e["target"]).casefold() for t in terms)]
        return list(dict.fromkeys(matched))[:limit]
