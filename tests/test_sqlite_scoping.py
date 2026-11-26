"""Actual SQLite candidate materialization, snapshots and ranking equivalence."""
from datetime import UTC, datetime, timedelta
import hashlib
import json
import sqlite3

import pytest

from rags.native import cosine
from rags.storage import Store

pytestmark = pytest.mark.integration


def add_source(store, identifier, revision, *, tenant="tenant", expected=None,
               metadata=None, source="fixture", vector=(1.0, 0.0),
               text="Alpha needle -> Beta", valid_from="2024-01-01T00:00:00.000000+00:00"):
    """Stored source fixtures exercise the real revision/FTS/graph transaction."""
    profile = {"embedding_model": "fixture-model", "embedding_endpoint": "local", "embedding_dimensions": 2}
    profile_id = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()
    chunk_id = f"{tenant}:{identifier}:{revision}"
    document = {"id": identifier, "revision": str(revision), "title": identifier, "source": source,
                "body": text, "metadata": metadata or {}, "checksum": hashlib.sha256(text.encode()).hexdigest(),
                "recorded_at": (datetime(2024, 2, 1, tzinfo=UTC) + timedelta(seconds=int(revision))).isoformat(timespec="microseconds"),
                "valid_from": valid_from, "valid_to": None, "profile_id": profile_id, "index_profile": profile}
    chunk = {"id": chunk_id, "ordinal": 0, "section": "", "content": text,
             "embedding": list(vector), "embedding_model": "fixture-model", "role": "leaf",
             "start_char": 0, "end_char": len(text), "section_path": [], "evidence_ids": [chunk_id],
             "processing_metadata": {"fixture": True}}
    relations = [{"source": "Alpha", "predicate": "links", "target": "Beta", "evidence_id": chunk_id}]
    store.commit_revision(tenant, document, [chunk], relations, expected)
    return chunk_id


def observe_materialization(monkeypatch, *, on_revisions=None, parameter_limit=None):
    original = sqlite3.connect
    reads = []

    class MeasuredCursor:
        def __init__(self, cursor, statement):
            self.cursor = cursor
            self.statement = statement

        def fetchall(self):
            rows = self.cursor.fetchall()
            reads.append((self.statement, [dict(row) for row in rows]))
            if on_revisions is not None and "FROM document_revisions WHERE" in self.statement:
                on_revisions()
            return rows

    class MeasuredConnection(sqlite3.Connection):
        def execute(self, statement, *args, **kwargs):
            cursor = super().execute(statement, *args, **kwargs)
            if statement.startswith("SELECT") and any(part in statement for part in (
                "FROM document_revisions WHERE", "FROM chunks c", "FROM chunks_fts", "FROM graph_relations",
            )):
                return MeasuredCursor(cursor, statement)
            return cursor

    def connect(*args, **kwargs):
        connection = original(*args, factory=MeasuredConnection, **kwargs)
        if parameter_limit is not None:
            connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, parameter_limit)
        return connection

    monkeypatch.setattr(sqlite3, "connect", connect)
    return reads


@pytest.mark.parametrize("history", [0, 40, 160])
def test_scope_materializes_only_live_authorized_chunks_as_history_grows(tmp_path, monkeypatch, history):
    store = Store(str(tmp_path / "scoped.db"))
    selected = set()
    for identifier in ("first", "second", "third"):
        previous = None
        for revision in range(history + 1):
            chunk_id = add_source(store, identifier, revision, expected=previous, metadata={"visible": True})
            previous = str(revision)
        selected.add(chunk_id)
    add_source(store, "hidden", 0, metadata={"visible": False}, source="private")
    add_source(store, "foreign", 0, tenant="other", metadata={"visible": True})
    reads = observe_materialization(monkeypatch)
    scope = store.scope("tenant", filters={"metadata": {"visible": True}}, include_embeddings=False)
    assert {chunk["id"] for chunk in scope} == selected
    revision_reads = [(sql, rows) for sql, rows in reads if "FROM document_revisions WHERE" in sql]
    assert len(revision_reads) == 1
    assert len(revision_reads[0][1]) == 4  # One chosen row per tenant document, independent of history.
    assert all("body" not in row and "checksum" not in row for row in revision_reads[0][1])
    chunk_reads = [(sql, rows) for sql, rows in reads if "FROM chunks c" in sql]
    assert len(chunk_reads) == 1 and len(chunk_reads[0][1]) == 3
    assert {row["id"] for row in chunk_reads[0][1]} == selected
    assert all("embedding" not in row for row in chunk_reads[0][1])
    with store.connect() as connection:
        assert connection.execute("SELECT count(*) FROM chunks WHERE tenant=?", ("tenant",)).fetchone()[0] == 3 * (history + 1) + 1
        assert connection.execute("SELECT count(*) FROM sqlite_temp_master").fetchone()[0] == 0


def test_scoped_dense_sparse_and_graph_match_prior_formulas_and_tie_order(tmp_path, monkeypatch):
    store = Store(str(tmp_path / "rank.db"))
    add_source(store, "changed", 0, text="needle " * 20)
    current = add_source(store, "changed", 1, expected="0", vector=(0.0, 1.0), text="needle sparse")
    same = add_source(store, "tie", 0, vector=(0.0, 1.0), text="needle sparse")
    near = add_source(store, "near", 0, vector=(0.9, 0.1), text="needle near")
    add_source(store, "excluded", 0, vector=(1.0, 0.0), text="needle " * 30)
    foreign = add_source(store, "foreign", 0, tenant="other")
    allowed = {current, same, near, foreign}
    query_vector = [1.0, 0.0]
    with store.connect() as connection:
        vectors = connection.execute("SELECT id,embedding FROM chunks WHERE tenant=?", ("tenant",)).fetchall()
        expected_dense = [identifier for _, identifier in sorted(
            ((cosine(query_vector, json.loads(row["embedding"])), row["id"]) for row in vectors if row["id"] in allowed),
            key=lambda pair: (-pair[0], pair[1]),
        )]
        lexical = connection.execute(
            "SELECT c.id FROM chunks_fts f JOIN chunks c ON c.id=f.chunk_id "
            "WHERE c.tenant=? AND chunks_fts MATCH ? ORDER BY bm25(chunks_fts),c.id", ("tenant", '"needle"'),
        ).fetchall()
        expected_sparse = [row["id"] for row in lexical if row["id"] in allowed][:2]
        expected_graph = [dict(row) for row in connection.execute(
            "SELECT * FROM graph_relations WHERE tenant=?", ("tenant",)
        ) if row["evidence_id"] in allowed]
    reads = observe_materialization(monkeypatch)
    assert store.dense("tenant", query_vector, 3, allowed,
                       embedding_model="fixture-model", embedding_endpoint="local") == expected_dense
    assert store.sparse("tenant", '"needle"', 2, allowed) == expected_sparse
    assert store.relations("tenant", allowed) == expected_graph
    assert len(reads) == 3
    assert [len(rows) for _, rows in reads] == [3, 2, 3]
    assert all(row.get("id", row.get("evidence_id")) in allowed for _, rows in reads for row in rows)


def test_revision_choice_and_historical_sources_remain_exact(tmp_path):
    store = Store(str(tmp_path / "history.db"))
    add_source(store, "policy", 0, source="archived", metadata={"active": False})
    reprocessed = add_source(store, "policy", 1, expected="0", source="archived", metadata={"active": False})
    current = add_source(store, "policy", 2, expected="1", source="current", metadata={"active": True},
                         valid_from="2024-06-01T00:00:00.000000+00:00")
    assert [row["id"] for row in store.scope("tenant", as_of="2024-03-01T00:00:00Z")] == [reprocessed]
    assert [row["id"] for row in store.scope("tenant", as_of="2024-07-01T00:00:00Z")] == [current]
    assert store.scope("tenant", filters={"source": "archived"}) == []
    assert store.scope("tenant", filters={"metadata": {"active": 1}}) == []
    assert [row["id"] for row in store.scope("tenant", filters={"metadata": {"active": True}})] == [current]


def test_scope_keeps_one_snapshot_when_source_is_deleted_between_reads(tmp_path, monkeypatch):
    store = Store(str(tmp_path / "snapshot.db"))
    original_id = add_source(store, "policy", 0)
    triggered = []

    def delete_after_revision_read():
        if not triggered:
            triggered.append(True)
            assert store.delete("tenant", "policy", "2024-09-01T00:00:00.000000+00:00")

    reads = observe_materialization(monkeypatch, on_revisions=delete_after_revision_read)
    during = store.scope("tenant")
    assert triggered == [True]
    assert [row["id"] for row in during] == [original_id]
    assert len(reads) == 2
    assert store.scope("tenant") == []


def test_large_scopes_do_not_require_json1_or_unbounded_sql_parameters(tmp_path, monkeypatch):
    store = Store(str(tmp_path / "large-scope.db"))
    identifiers = {add_source(store, f"source-{index:03d}", 0) for index in range(90)}
    reads = observe_materialization(monkeypatch, parameter_limit=32)
    assert {row["id"] for row in store.scope("tenant")} == identifiers
    allowed = identifiers | {f"missing-{index}" for index in range(1200)}
    assert set(store.dense("tenant", [1.0, 0.0], 100, allowed,
                           embedding_model="fixture-model", embedding_endpoint="local")) == identifiers
    assert set(store.sparse("tenant", '"needle"', 100, allowed)) == identifiers
    assert {row["evidence_id"] for row in store.relations("tenant", allowed)} == identifiers
    assert all("json_each" not in sql for sql, _ in reads)
