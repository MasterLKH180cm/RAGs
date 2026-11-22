"""Explicit, offline SQLite-to-PostgreSQL import without provider execution."""
import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3

from rags import postgres
from rags.multimodal import ASSET_FIELDS, AssetStore
from rags.registry import CapabilityError
from rags.storage import CHUNK_FIELDS, DOC_COLUMNS, REVISION_FIELDS, Store


TABLE_FIELDS = {
    "document_revisions": REVISION_FIELDS,
    "documents": ("tenant", "id", "title", "source", "body", "metadata", "checksum", "updated_at", *DOC_COLUMNS),
    "chunks": CHUNK_FIELDS,
    "chunks_fts": ("chunk_id", "content"),
    "edges": ("tenant", "document_id", "source", "target"),
    "graph_relations": ("tenant", "document_id", "revision", "source", "predicate", "target", "evidence_id"),
    "audit": ("id", "timestamp", "tenant", "action", "document_id", "detail"),
    "image_assets": ASSET_FIELDS,
}


def _source_tables(source) -> set[str]:
    tables = {row["name"] for row in source.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if set(TABLE_FIELDS) - {"image_assets"} - tables:
        raise ValueError("Source must contain the complete migrated RAG SQLite schema")
    for table, fields in TABLE_FIELDS.items():
        if table not in tables:
            continue
        actual = {row["name"] for row in source.execute(f"PRAGMA table_info({table})")}
        if actual != set(fields):
            raise ValueError("Unsupported source columns in " + table + "; upgrade or review the SQLite schema before importing")
    return tables


def _require_empty(connection) -> None:
    for table in TABLE_FIELDS:
        present = connection.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_schema=current_schema() AND table_name=?", (table,),
        ).fetchone()
        if present and connection.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone():
            raise ValueError("PostgreSQL import target must be empty")


def import_sqlite(source_path: str | Path, target_dsn: str) -> dict:
    """Copy one read-only SQLite snapshot into an empty PostgreSQL application schema.

The operator must quiesce both applications. Table locks additionally prevent
concurrent target writes while the empty-target check and data copy execute.
Schema bootstrap may persist on failure; imported application rows never do.
    """
    if not postgres.is_postgres(target_dsn):
        raise ValueError("Import requires a PostgreSQL RAGS_DATABASE_URL")
    path = Path(source_path).resolve()
    if not path.is_file():
        raise ValueError("SQLite source must be an existing regular file")
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as source:
        source.row_factory = sqlite3.Row
        source.execute("PRAGMA query_only=ON")
        source.execute("BEGIN")
        tables = _source_tables(source)
        # Refuse existing data before running even an additive target migration.
        with postgres.connect(target_dsn) as connection:
            _require_empty(connection)
        store = Store(target_dsn)
        AssetStore(target_dsn)
        counts = {}
        with store.connect() as connection:
            connection.execute("LOCK TABLE " + ",".join(sorted(TABLE_FIELDS)) + " IN ACCESS EXCLUSIVE MODE")
            _require_empty(connection)
            for table, fields in TABLE_FIELDS.items():
                counts[table] = 0
                if table not in tables:
                    continue
                selected = ",".join(fields)
                target_fields = fields
                if table == "document_revisions":
                    # Preserve the exact SQLite tie-breaker, not UUID ordering.
                    selected += ",rowid AS revision_order"
                    target_fields = (*fields, "revision_order")
                vector_column = {"chunks": "embedding", "image_assets": "vector"}.get(table)
                placeholders = ["?::vector" if name == vector_column else "?" for name in target_fields]
                statement = (f"INSERT INTO {table}(" + ",".join(target_fields) + ") VALUES(" +
                             ",".join(placeholders) + ")")
                for row in source.execute(f"SELECT {selected} FROM {table} ORDER BY rowid"):
                    values = dict(row)
                    if vector_column:
                        try:
                            vector = json.loads(values[vector_column])
                        except (TypeError, ValueError) as exc:
                            raise ValueError("Source contains an invalid vector encoding") from exc
                        values[vector_column] = postgres.vector_literal(vector)
                    connection.execute(statement, [values[name] for name in target_fields])
                    counts[table] += 1
            for table, column in (("audit", "id"), ("document_revisions", "revision_order")):
                connection.execute(
                    f"SELECT setval(pg_get_serial_sequence(?,?),COALESCE((SELECT MAX({column}) FROM {table}),1),"
                    f"EXISTS(SELECT 1 FROM {table}))", (table, column),
                )
        return {"status": "imported", "storage_backend": "postgresql", "vector_search": "pgvector_exact",
                "source_read_only": True, "tables": counts}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", required=True, help="Existing, quiesced SQLite source file (opened read-only)")
    arguments = parser.parse_args(argv)
    destination = os.getenv("RAGS_DATABASE_URL", "")
    if not destination:
        parser.error("RAGS_DATABASE_URL is required; the target must be empty")
    try:
        from psycopg import Error as PostgresError
    except ImportError:
        parser.exit(1, "Install the postgres extra for RAGS_DATABASE_URL\n")
    try:
        result = import_sqlite(arguments.sqlite, destination)
    except (ValueError, CapabilityError) as exc:
        parser.exit(1, "Import rejected: " + str(exc) + "\n")
    except sqlite3.Error:
        parser.exit(1, "Import failed while reading the SQLite source\n")
    except PostgresError:
        parser.exit(1, "Import failed while accessing PostgreSQL; verify bootstrap, permissions and connectivity\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
