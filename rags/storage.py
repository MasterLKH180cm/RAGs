"""Tenant-scoped, versioned source records and transactional derived indexes."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Iterator

from rags.registry import CapabilityError
from rags.native import equal_value

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


def timestamp(value: str | None = None) -> str:
    """Canonical UTC ISO timestamps make SQLite comparisons deterministic."""
    if value is None:
        parsed = datetime.now(timezone.utc)
    else:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (TypeError, ValueError) as exc:
            raise ValueError("Timestamps must be ISO 8601 with an explicit timezone") from exc
        if parsed.tzinfo is None:
            raise ValueError("Timestamps require an explicit timezone")
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds")


class Store:
    def __init__(self, path: str):
        self.path = path
        if path == ":memory:":
            raise ValueError("Use a temporary file: independent SQLite connections cannot share :memory:")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            conn.execute("BEGIN IMMEDIATE")
            for table, additions in (("documents", DOC_COLUMNS), ("chunks", CHUNK_COLUMNS)):
                present = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
                for name, definition in additions.items():
                    if name not in present:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")
            conn.execute("CREATE INDEX IF NOT EXISTS chunks_revision ON chunks(tenant,document_id,revision,role)")
            # Migration never calls models or invents offsets/token counts.
            conn.execute("UPDATE documents SET valid_from=updated_at WHERE valid_from IS NULL")
            conn.execute(
                "INSERT OR IGNORE INTO document_revisions "
                "SELECT tenant,id,revision,title,source,body,metadata,checksum,updated_at,valid_from,"
                "valid_to,profile_id,index_profile FROM documents"
            )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=20)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=20000")
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

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
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT id,revision,title,source,metadata,recorded_at,valid_from,valid_to,profile_id,index_profile "
                "FROM document_revisions WHERE tenant=? AND id=? ORDER BY recorded_at DESC", (tenant, doc_id)
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
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute("SELECT revision,valid_from,valid_to FROM documents WHERE tenant=? AND id=?", (tenant, document["id"])).fetchone()
            if (current["revision"] if current else None) != expected_revision:
                raise CapabilityError("revision_conflict", "Document changed during indexing; retry against its current revision", 409)
            if current and current["valid_from"] < document["valid_from"]:
                if current["valid_to"] is None or current["valid_to"] > document["valid_from"]:
                    conn.execute(
                        "UPDATE document_revisions SET valid_to=? WHERE tenant=? AND id=? AND revision=?",
                        (document["valid_from"], tenant, document["id"], current["revision"]),
                    )
            values = (
                tenant, document["id"], document["revision"], document["title"], document["source"], document["body"],
                json.dumps(document["metadata"]), document["checksum"], document["recorded_at"],
                document["valid_from"], document.get("valid_to"), document["profile_id"], json.dumps(document["index_profile"]),
            )
            conn.execute("INSERT INTO document_revisions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", values)
            conn.execute(
                "INSERT INTO documents(tenant,id,revision,title,source,body,metadata,checksum,updated_at,valid_from,valid_to,profile_id,index_profile) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(tenant,id) DO UPDATE SET "
                "revision=excluded.revision,title=excluded.title,source=excluded.source,body=excluded.body,"
                "metadata=excluded.metadata,checksum=excluded.checksum,updated_at=excluded.updated_at,"
                "valid_from=excluded.valid_from,valid_to=excluded.valid_to,profile_id=excluded.profile_id,index_profile=excluded.index_profile",
                values,
            )
            for item in chunks:
                fields = ["id", "tenant", "document_id", "ordinal", "section", "content", "embedding", "embedding_model", *CHUNK_COLUMNS]
                encoded = dict(item, tenant=tenant, document_id=document["id"], revision=document["revision"], profile_id=document["profile_id"])
                for key in ("embedding", "section_path", "evidence_ids", "processing_metadata"):
                    encoded[key] = json.dumps(encoded.get(key, {} if key == "processing_metadata" else []))
                defaults = {"role": "leaf", "context": "", "level": 0}
                conn.execute(
                    "INSERT INTO chunks(" + ",".join(fields) + ") VALUES(" + ",".join("?" for _ in fields) + ")",
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

    def delete(self, tenant: str, doc_id: str, now: str) -> bool:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT id FROM documents WHERE tenant=? AND id=?", (tenant, doc_id)).fetchone()
            if not row:
                return False
            self.remove_chunks(conn, tenant, doc_id)
            conn.execute("DELETE FROM document_revisions WHERE tenant=? AND id=?", (tenant, doc_id))
            conn.execute("DELETE FROM documents WHERE tenant=? AND id=?", (tenant, doc_id))
            conn.execute("INSERT INTO audit(timestamp,tenant,action,document_id,detail) VALUES(?,?,?,?,?)", (now, tenant, "delete", doc_id, "{}"))
        return True

    def scope(self, tenant: str, *, as_of: str | None = None, filters: dict | None = None) -> list[dict]:
        """Choose one valid revision per document before retrieval or expansion."""
        at = timestamp(as_of)
        filters = filters or {}
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM document_revisions WHERE tenant=? AND valid_from<=? "
                "AND (valid_to IS NULL OR valid_to>?) ORDER BY recorded_at DESC,rowid DESC", (tenant, at, at)
            ).fetchall()
        docs = {}
        for row in rows:
            doc = self.decode_document(row)
            if doc["id"] not in docs:
                docs[doc["id"]] = doc
        selected = {}
        for doc_id, doc in docs.items():
            if filters.get("source") is not None and doc["source"] != filters["source"]:
                continue
            if any(k not in doc["metadata"] or not equal_value(doc["metadata"][k], v) for k, v in filters.get("metadata", {}).items()):
                continue
            if filters.get("valid_after") and doc["valid_from"] < timestamp(filters["valid_after"]):
                continue
            if filters.get("valid_before") and doc["valid_from"] >= timestamp(filters["valid_before"]):
                continue
            selected[(doc_id, doc["revision"])] = doc
        with self.connect() as conn:
            chunk_rows = conn.execute("SELECT * FROM chunks WHERE tenant=?", (tenant,)).fetchall()
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

    def sparse(self, tenant: str, match_query: str, limit: int, allowed_ids: set[str] | None = None) -> list[str]:
        if not match_query:
            return []
        if allowed_ids is None:
            allowed_ids = {c["id"] for c in self.scope(tenant) if c["role"] == "leaf"}
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT c.id FROM chunks_fts f JOIN chunks c ON c.id=f.chunk_id "
                "WHERE c.tenant=? AND chunks_fts MATCH ? ORDER BY bm25(chunks_fts),c.id", (tenant, match_query)
            ).fetchall()
        return [r["id"] for r in rows if r["id"] in allowed_ids][:limit]

    def relations(self, tenant: str, allowed_ids: set[str]) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM graph_relations WHERE tenant=?", (tenant,)).fetchall()
        return [dict(r) for r in rows if r["evidence_id"] in allowed_ids]

    def graph_documents(self, tenant: str, terms: list[str], limit: int = 20) -> list[str]:
        allowed = {c["id"] for c in self.scope(tenant)}
        edges = self.relations(tenant, allowed)
        matched = [e["document_id"] for e in edges if any(t in (e["source"] + " " + e["target"]).casefold() for t in terms)]
        return list(dict.fromkeys(matched))[:limit]
