"""Actual SQLite rollback, source-revision races and connection lifecycle."""
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

import pytest

from rags.config import Settings
from rags.providers import Providers
from rags.registry import CapabilityError
from rags.retrieval import RAGService, now
from rags.storage import Store

pytestmark = pytest.mark.integration


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGS_ENV", "test")
    settings = replace(Settings.from_env(), db_path=str(tmp_path / "atomicity.db"),
                       embedding_provider="local", model_provider="extractive")
    return RAGService(Store(settings.db_path), Providers(settings))


def snapshot(store):
    with store.connect() as connection:
        return {table: [tuple(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY rowid")]
                for table in ("documents", "document_revisions", "chunks", "chunks_fts",
                              "edges", "graph_relations", "audit")}


def test_connection_is_closed_when_pragma_initialization_fails(service, monkeypatch):
    connections = []
    original_connect = sqlite3.connect

    class FailingSetupConnection(sqlite3.Connection):
        closed = False

        def execute(self, sql, *args, **kwargs):
            if sql == "PRAGMA journal_mode=WAL":
                raise sqlite3.OperationalError("fixture journal initialization failure")
            return super().execute(sql, *args, **kwargs)

        def close(self):
            self.closed = True
            return super().close()

    def connect(*args, **kwargs):
        connection = original_connect(*args, factory=FailingSetupConnection, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", connect)
    with (pytest.raises(sqlite3.OperationalError, match="fixture journal initialization failure"),
          service.store.connect()):
        pytest.fail("A failed connection setup must not yield a connection")
    assert len(connections) == 1 and connections[0].closed
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connections[0].execute("SELECT 1")


def test_sql_failure_rolls_back_source_revision_fts_graph_and_hierarchy(service):
    chunking = {"strategy": "fixed", "max_tokens": 64, "overlap_tokens": 8}
    first = service.ingest("tenant", "Network", "Alpha -> Beta\n" * 20, "fixture", document_id="policy",
                           chunking=chunking, index_features=["hierarchy"], valid_from="2024-01-01T00:00:00Z")
    assert first["derived_nodes"] > 0
    before = snapshot(service.store)
    assert before["graph_relations"] and before["chunks_fts"]
    with service.store.connect() as connection:
        connection.executescript("""
            CREATE TRIGGER fail_new_graph BEFORE INSERT ON graph_relations
            BEGIN SELECT RAISE(ABORT, 'fixture graph write failure'); END;
        """)
    with pytest.raises(sqlite3.IntegrityError, match="fixture graph write failure"):
        service.ingest("tenant", "Changed network", "Alpha -> Gamma\n" * 20, "fixture-v2", document_id="policy",
                       chunking=chunking, index_features=["hierarchy"], valid_from="2024-06-01T00:00:00Z",
                       expected_revision=first["revision"])
    assert snapshot(service.store) == before
    assert service.store.get_document("tenant", "policy")["revision"] == first["revision"]
    result = service.query("tenant", "Beta", "sparse", generate=False)
    assert result["citations"] and {item["revision"] for item in result["citations"]} == {first["revision"]}
    assert service.query("tenant", "Gamma", "sparse", generate=False)["citations"] == []


def test_simultaneous_reindex_writers_have_one_cas_winner_without_partial_records(service, monkeypatch):
    initial = service.ingest("tenant", "Network", "Alpha -> Beta", "fixture", document_id="policy")
    other = service.ingest("other", "Private", "Private -> Secret", "private", document_id="policy")
    barrier = Barrier(2, timeout=5)
    workers = [RAGService(Store(service.settings.db_path), Providers(service.settings)) for _ in range(2)]
    for worker in workers:
        original = worker.provider.embedding

        def embedding(text, original=original):
            barrier.wait()
            return original(text)

        monkeypatch.setattr(worker.provider, "embedding", embedding)

    def update(index):
        try:
            result = workers[index].ingest("tenant", f"Writer {index}", f"Alpha -> Writer{index}", "fixture",
                                           document_id="policy", expected_revision=initial["revision"])
            return index, result
        except CapabilityError as error:
            return index, error

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(update, index) for index in range(2)]
        results = [future.result(timeout=15) for future in futures]
    winners = [(index, result) for index, result in results if isinstance(result, dict)]
    losers = [(index, result) for index, result in results if isinstance(result, CapabilityError)]
    assert len(winners) == len(losers) == 1
    winner_index, winner = winners[0]
    loser_index, error = losers[0]
    assert error.code == "revision_conflict" and error.status_code == 409
    current = service.store.get_document("tenant", "policy")
    assert current["revision"] == winner["revision"] and current["body"] == f"Alpha -> Writer{winner_index}"
    expected_revisions = {initial["revision"], winner["revision"]}
    assert {row["revision"] for row in service.store.revisions("tenant", "policy")} == expected_revisions
    with service.store.connect() as connection:
        assert {row["revision"] for row in connection.execute(
            "SELECT revision FROM chunks WHERE tenant='tenant'")} == expected_revisions
        assert {row["revision"] for row in connection.execute(
            "SELECT revision FROM graph_relations WHERE tenant='tenant'")} == expected_revisions
        assert connection.execute("SELECT count(*) FROM audit WHERE tenant='tenant' AND action='upsert'").fetchone()[0] == 2
        assert connection.execute("SELECT count(*) FROM chunks_fts WHERE content LIKE ?", (f"%Writer{loser_index}%",)).fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM chunks_fts WHERE chunk_id NOT IN (SELECT id FROM chunks)").fetchone()[0] == 0
    assert service.store.get_document("other", "policy")["revision"] == other["revision"]


def test_delete_during_provider_work_prevents_source_resurrection(service, monkeypatch):
    first = service.ingest("tenant", "Network", "Alpha -> Beta", "fixture", document_id="policy")
    original = service.provider.embedding

    def embedding(text):
        assert service.store.delete("tenant", "policy", now())
        return original(text)

    monkeypatch.setattr(service.provider, "embedding", embedding)
    with pytest.raises(CapabilityError) as error:
        service.ingest("tenant", "Network", "Alpha -> Gamma", "fixture", document_id="policy",
                       expected_revision=first["revision"])
    assert error.value.code == "revision_conflict" and error.value.status_code == 409
    assert service.store.get_document("tenant", "policy") is None
    assert service.store.revisions("tenant", "policy") == []
    after = snapshot(service.store)
    assert all(after[table] == [] for table in ("documents", "document_revisions", "chunks", "chunks_fts", "graph_relations"))
    assert [row[3] for row in after["audit"]] == ["upsert", "delete"]


def test_delete_purges_all_derived_history_for_only_the_authorized_tenant(service):
    chunking = {"strategy": "parent_child", "max_tokens": 48, "overlap_tokens": 8, "parent_max_tokens": 120}
    for tenant in ("tenant", "other"):
        service.ingest(tenant, "Network", "Alpha -> Beta\n" * 20, "fixture", document_id="shared-id",
                       chunking=chunking, index_features=["hierarchy"])
        service.ingest(tenant, "Network", "Alpha -> Gamma\n" * 20, "fixture-v2", document_id="shared-id",
                       chunking=chunking, index_features=["hierarchy"])
    other_document = service.store.get_document("other", "shared-id")
    other_chunks = service.store.scope("other")
    assert {chunk["role"] for chunk in other_chunks} >= {"leaf", "parent", "summary"}
    assert service.store.delete("tenant", "shared-id", now())
    assert not service.store.delete("tenant", "shared-id", now())
    assert service.store.get_document("other", "shared-id") == other_document
    assert service.store.scope("other") == other_chunks
    with service.store.connect() as connection:
        for table in ("documents", "document_revisions", "chunks", "edges", "graph_relations"):
            assert connection.execute(f"SELECT count(*) FROM {table} WHERE tenant='tenant'").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM chunks_fts").fetchone()[0] > 0
        assert connection.execute("SELECT count(*) FROM chunks_fts WHERE chunk_id NOT IN (SELECT id FROM chunks)").fetchone()[0] == 0


def test_scope_applies_source_metadata_and_validity_filters_after_revision_selection(service):
    service.ingest("tenant", "Archived", "Alpha -> Beta", "archive", document_id="policy",
                   metadata={"active": False}, valid_from="2024-01-01T00:00:00Z")
    current = service.ingest("tenant", "Current", "Alpha -> Gamma", "current", document_id="policy",
                             metadata={"active": True, "amount": 1}, valid_from="2024-06-01T00:00:00Z")
    service.ingest("tenant", "Later", "Delta -> Epsilon", "current", document_id="later",
                   metadata={"active": True, "amount": 2}, valid_from="2024-09-01T00:00:00Z")

    def scope(**filters):
        return service.store.scope("tenant", as_of="2024-10-01T00:00:00Z", filters=filters)

    assert scope(source="archive") == []  # A source filter cannot revive a superseded revision.
    assert scope(metadata={"active": 1}) == []
    matched = scope(source="current", metadata={"amount": 1.0},
                    valid_after="2024-06-01T00:00:00Z", valid_before="2024-09-01T00:00:00Z")
    assert matched and {chunk["revision"] for chunk in matched} == {current["revision"]}
    assert {chunk["document_id"] for chunk in scope(valid_after="2024-07-01T00:00:00Z")} == {"later"}
    assert scope(valid_before="2024-06-01T00:00:00Z") == []


def test_storage_lookup_helpers_preserve_current_revision_tenant_and_source_identity(service):
    old = service.ingest("tenant", "Old", "Alpha -> Beta", "fixture", document_id="policy")
    old_ids = {chunk["id"] for chunk in service.store.scope("tenant")}
    current = service.ingest("tenant", "Current", "Alpha -> Gamma", "fixture", document_id="policy")
    service.ingest("other", "Private", "Alpha -> Private", "private", document_id="policy")
    current_ids = {chunk["id"] for chunk in service.store.scope("tenant")}
    private_ids = {chunk["id"] for chunk in service.store.scope("other")}
    selected = service.store.fetch_chunks("tenant", list(old_ids | current_ids | private_ids | {"unknown"}))
    assert {chunk["id"] for chunk in selected} == current_ids
    assert {chunk["revision"] for chunk in selected} == {current["revision"]}
    assert old["revision"] != current["revision"]
    assert service.store.all_chunks("tenant", "unavailable-model") == []
    assert {chunk["id"] for chunk in service.store.all_chunks("tenant", "offline-hash-v1")} == current_ids
    assert service.store.sparse("tenant", "", 10) == []
    assert service.store.sparse("tenant", "Beta", 10) == []
    assert set(service.store.sparse("tenant", "Gamma", 10)) == current_ids
    assert service.store.sparse("tenant", "Alpha", 10, private_ids) == []
    assert service.store.relations("tenant", private_ids) == []
    assert service.store.graph_documents("tenant", ["alpha"], limit=1) == ["policy"]
    assert service.store.graph_documents("tenant", ["private"]) == []
