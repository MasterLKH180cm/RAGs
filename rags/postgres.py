"""Small PostgreSQL adapter and native pgvector storage primitives.

Values always use bound parameters. Only the application's static SQLite SQL
vocabulary is translated; vector ranking is explicit PostgreSQL SQL.
"""
from contextlib import contextmanager
import json
import math
import re
import struct

from rags.registry import CapabilityError


MAX_VECTOR_DIMENSIONS = 16_000
SCHEMA_LOCK = 731824204


def is_postgres(path: str) -> bool:
    return path.startswith(("postgresql://", "postgres://"))


def vector_literal(values, *, code: str = "invalid_embedding") -> str:
    """Reject invalid dimensions/numbers before a transaction can mutate sources.

Round through IEEE float32, the pgvector element representation. Bound the
literal to values PostgreSQL can parse, including underflow to signed zero.
    """
    reason = "Vectors require 1..16000 finite float32 values"
    if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= MAX_VECTOR_DIMENSIONS:
        raise CapabilityError(code, reason, 502)
    rounded = []
    try:
        for value in values:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("Invalid vector element")
            converted = struct.unpack("!f", struct.pack("!f", value))[0]
            if not math.isfinite(converted):
                raise ValueError("Float32 overflow")
            rounded.append(converted)
    except (ValueError, TypeError, OverflowError, struct.error) as exc:
        raise CapabilityError(code, reason, 502) from exc
    return json.dumps(rounded, allow_nan=False, separators=(",", ":"))


def bind_parameters(statement: str) -> str:
    """Translate placeholders outside single-quoted SQL literals only."""
    result, quoted, index = [], False, 0
    while index < len(statement):
        character = statement[index]
        if character == "'":
            if quoted and index + 1 < len(statement) and statement[index + 1] == "'":
                result.append("''")
                index += 2
                continue
            quoted = not quoted
        result.append("%s" if character == "?" and not quoted else character)
        index += 1
    return "".join(result)


def translate_schema(schema: str) -> str:
    """Adapt the two fixed application schemas, never request-supplied DDL."""
    schema = schema.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "BIGSERIAL PRIMARY KEY")
    schema = schema.replace("original BLOB NOT NULL", "original BYTEA NOT NULL")
    schema = schema.replace("embedding TEXT NOT NULL", "embedding vector NOT NULL")
    schema = schema.replace("vector TEXT NOT NULL", "vector vector NOT NULL")
    return schema.replace(
        "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(chunk_id UNINDEXED,content,tokenize='unicode61')",
        "CREATE TABLE IF NOT EXISTS chunks_fts (chunk_id TEXT PRIMARY KEY, content TEXT NOT NULL, "
        "search tsvector GENERATED ALWAYS AS (to_tsvector('simple',content)) STORED)",
    )


class Row(dict):
    def __getitem__(self, key):
        return tuple(self.values())[key] if isinstance(key, int) else super().__getitem__(key)


class Cursor:
    def __init__(self, raw):
        self.raw = raw

    @property
    def rowcount(self):
        return self.raw.rowcount

    def fetchone(self):
        row = self.raw.fetchone()
        return None if row is None else Row(row)

    def fetchall(self):
        return [Row(row) for row in self.raw.fetchall()]

    def __iter__(self):
        for row in self.raw:
            yield Row(row)


class Connection:
    def __init__(self, raw):
        self.raw = raw
        raw.execute("SET LOCAL statement_timeout = '20s'")
        raw.execute("SET LOCAL lock_timeout = '20s'")

    def execute(self, statement: str, parameters=()):
        sql = statement.strip().rstrip(";")
        if sql == "BEGIN IMMEDIATE":
            # PostgreSQL already owns a transaction. Conflicting mutations take
            # an explicit document/tenant advisory lock, never a global read lock.
            return Cursor(self.raw.execute("SELECT 1"))
        info = re.fullmatch(r"PRAGMA table_info\(([a-z_]+)\)", sql)
        if info:
            return Cursor(self.raw.execute(
                "SELECT ordinal_position-1 AS cid,column_name AS name,"
                "CASE WHEN data_type='USER-DEFINED' THEN udt_name ELSE data_type END AS type "
                "FROM information_schema.columns WHERE table_schema=current_schema() AND table_name=%s "
                "ORDER BY ordinal_position", (info.group(1),),
            ))
        if sql.startswith("PRAGMA") or re.search(r"\browid\b", sql, flags=re.IGNORECASE):
            raise ValueError("PostgreSQL requires explicit storage columns and ordering")
        if sql.startswith("INSERT OR IGNORE INTO "):
            sql = sql.replace("INSERT OR IGNORE INTO ", "INSERT INTO ", 1) + " ON CONFLICT DO NOTHING"
        return Cursor(self.raw.execute(bind_parameters(sql), parameters))

    def executescript(self, script: str):
        # Only application schema declarations use this helper. Trigger/function
        # bodies must be executed as one statement via execute/raw.execute.
        for statement in translate_schema(script).split(";"):
            if statement.strip():
                self.execute(statement)


@contextmanager
def connect(dsn: str):
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise ValueError("Install the postgres extra for RAGS_DATABASE_URL") from exc
    raw = psycopg.connect(dsn, row_factory=dict_row, connect_timeout=5)
    try:
        yield Connection(raw)
        raw.commit()
    except BaseException:
        raw.rollback()
        raise
    finally:
        raw.close()


def initialize(connection: Connection, schema: str) -> None:
    connection.execute("SELECT pg_advisory_xact_lock(?)", (SCHEMA_LOCK,))
    extension = connection.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()
    if extension is None:
        raise ValueError("PostgreSQL requires the vector extension; run the documented administrator bootstrap")
    connection.executescript(schema)


def lock_mutation(connection: Connection, family: str, tenant: str, identifier: str = "") -> None:
    key = json.dumps(["rags-v04", family, tenant, identifier], ensure_ascii=True, separators=(",", ":"))
    connection.execute("SELECT pg_advisory_xact_lock(hashtextextended(?,0))", (key,))


def migrate_vector_column(connection: Connection, table: str, column: str) -> None:
    if (table, column) not in {("chunks", "embedding"), ("image_assets", "vector")}:
        raise ValueError("Unknown application vector column")
    columns = {row["name"]: row["type"] for row in connection.execute(f"PRAGMA table_info({table})")}
    if columns.get(column) == "text":
        connection.execute(f"ALTER TABLE {table} ALTER COLUMN {column} TYPE vector USING {column}::vector")
    elif columns.get(column) != "vector":
        raise ValueError("Application vector column has an unsupported storage type")


def cosine_distance(column: str, query: str = "q.embedding") -> str:
    """Exact native distance with the existing zero-vector similarity convention.

Normalize with pgvector before its float32 cosine accumulation so finite but
large/small float32 values cannot overflow intermediate dot products.
    """
    if column not in {"c.embedding", "a.vector"} or query != "q.embedding":
        raise ValueError("Unknown vector expression")
    return (f"CASE WHEN vector_norm({column})=0 OR vector_norm({query})=0 THEN 1.0 "
            f"ELSE l2_normalize({column}) <=> l2_normalize({query}) END")
