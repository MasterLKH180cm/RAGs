"""SQLite transactional document store, FTS5 index and lightweight graph edges."""
from contextlib import contextmanager
import json
import sqlite3
from pathlib import Path
from typing import Iterator


SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  tenant TEXT NOT NULL, id TEXT NOT NULL, title TEXT NOT NULL,
  source TEXT NOT NULL, body TEXT NOT NULL, metadata TEXT NOT NULL,
  checksum TEXT NOT NULL, updated_at TEXT NOT NULL,
  PRIMARY KEY (tenant, id)
);
CREATE TABLE IF NOT EXISTS chunks (
  id TEXT PRIMARY KEY, tenant TEXT NOT NULL, document_id TEXT NOT NULL,
  ordinal INTEGER NOT NULL, section TEXT NOT NULL,
  content TEXT NOT NULL, embedding TEXT NOT NULL, embedding_model TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_tenant_document ON chunks(tenant, document_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(chunk_id UNINDEXED, content, tokenize='unicode61');
CREATE TABLE IF NOT EXISTS edges (
  tenant TEXT NOT NULL, document_id TEXT NOT NULL,
  source TEXT NOT NULL, target TEXT NOT NULL,
  PRIMARY KEY (tenant, document_id, source, target)
);
CREATE TABLE IF NOT EXISTS audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
  tenant TEXT NOT NULL, action TEXT NOT NULL, document_id TEXT,
  detail TEXT NOT NULL
);
"""


class Store:
    def __init__(self, path: str):
        self.path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=20)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=20000")
        if self.path != ":memory:":
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
    def remove_chunks(conn: sqlite3.Connection, tenant: str, document_id: str) -> None:
        conn.execute(
            "DELETE FROM chunks_fts WHERE chunk_id IN "
            "(SELECT id FROM chunks WHERE tenant=? AND document_id=?)",
            (tenant, document_id),
        )
        conn.execute(
            "DELETE FROM chunks WHERE tenant=? AND document_id=?", (tenant, document_id)
        )
        conn.execute(
            "DELETE FROM edges WHERE tenant=? AND document_id=?", (tenant, document_id)
        )

    def delete(self, tenant: str, doc_id: str, now: str) -> bool:
        with self.connect() as conn:
            self.remove_chunks(conn, tenant, doc_id)
            row = conn.execute(
                "DELETE FROM documents WHERE tenant=? AND id=? RETURNING id", (tenant, doc_id)
            ).fetchone()
            if row:
                conn.execute(
                    "INSERT INTO audit(timestamp,tenant,action,document_id,detail)"
                    " VALUES(?,?,?,?,?)", (now, tenant, "delete", doc_id, "{}")
                )
            return row is not None

    def list_documents(self, tenant: str) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT id,title,source,metadata,updated_at FROM documents "
                "WHERE tenant=? ORDER BY updated_at DESC", (tenant,)
            ).fetchall()
            return [
                {**dict(row), "metadata": json.loads(row["metadata"])}
                for row in rows
            ]

    def get_document(self, tenant: str, doc_id: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE tenant=? AND id=?", (tenant, doc_id)
            ).fetchone()
            return {**dict(row), "metadata": json.loads(row["metadata"])} if row else None

    def all_chunks(self, tenant: str, embedding_model: str) -> list[dict]:
        with self.connect() as conn:
            return [
                dict(row) for row in conn.execute(
                    "SELECT c.*,d.title,d.source,d.updated_at FROM chunks c "
                    "JOIN documents d ON d.tenant=c.tenant AND d.id=c.document_id "
                    "WHERE c.tenant=? AND c.embedding_model=?",
                    (tenant, embedding_model),
                )
            ]

    def fetch_chunks(self, tenant: str, ids: list[str]) -> list[dict]:
        if not ids:
            return []
        marks = ",".join("?" for _ in ids)
        with self.connect() as conn:
            return [
                dict(row) for row in conn.execute(
                    "SELECT c.*,d.title,d.source,d.updated_at FROM chunks c "
                    "JOIN documents d ON d.tenant=c.tenant AND d.id=c.document_id "
                    f"WHERE c.tenant=? AND c.id IN ({marks})",
                    [tenant, *ids],
                )
            ]

    def sparse(self, tenant: str, match_query: str, limit: int) -> list[str]:
        if not match_query:
            return []
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT c.id FROM chunks_fts f "
                "JOIN chunks c ON c.id=f.chunk_id "
                "WHERE c.tenant=? AND chunks_fts MATCH ? "
                "ORDER BY bm25(chunks_fts) LIMIT ?",
                (tenant, match_query, limit),
            ).fetchall()
            return [row["id"] for row in rows]

    def graph_documents(self, tenant: str, terms: list[str], limit: int = 20) -> list[str]:
        if not terms:
            return []
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT document_id, source, target FROM edges WHERE tenant=?", (tenant,)
            ).fetchall()
        ranked = {}
        for row in rows:
            score = sum(
                1 for word in terms
                if word.casefold() in (row["source"] + " " + row["target"]).casefold()
            )
            if score:
                ranked[row["document_id"]] = max(ranked.get(row["document_id"], 0), score)
        return sorted(ranked, key=lambda d: (-ranked[d], d))[:limit]
