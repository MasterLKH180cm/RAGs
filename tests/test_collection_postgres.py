"""Actual PostgreSQL/pgvector collection isolation, MVCC, GC and import gates."""
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

import pytest

from rags.collection_models import canonical_dense
from rags.collection_store import TABLE_FIELDS, CollectionStore
from rags.migrate import import_sqlite
from rags.registry import CapabilityError
from rags.storage import Store

pytestmark = [pytest.mark.integration, pytest.mark.external, pytest.mark.postgres]


@pytest.fixture
def database():
    endpoint = os.getenv("RAGS_TEST_POSTGRES_URL")
    if not endpoint:
        pytest.fail("RAGS_TEST_POSTGRES_URL is required for selected PostgreSQL tests")
    import psycopg
    from psycopg import sql
    parsed = urlsplit(endpoint)
    assert parsed.scheme in {"postgres", "postgresql"}
    schema = "rags_collections_" + uuid.uuid4().hex
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["options"] = (query.get("options", "") + f" -c search_path={schema},public").strip()
    isolated = urlunsplit(parsed._replace(query=urlencode(query, quote_via=quote)))
    with psycopg.connect(endpoint, autocommit=True, connect_timeout=5) as administrator:
        assert administrator.execute("SELECT 1 FROM pg_extension WHERE extname='vector'").fetchone()
        administrator.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            yield isolated
        finally:
            administrator.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def config(dimensions=None, sparse=None):
    return {"embedding": {"mode": "configured", "model": "fixture-v1", "endpoint": "local",
                          "space_id": "fixture-space", "dimensions": dimensions, "normalization": "l2-f32-v1"},
            "sparse_space_id": sparse}


def record(identifier="one", document="Synthetic source", embedding=(1, 0, 0), metadata=None, sparse=None):
    return {"id": identifier, "document": document, "metadata": metadata or {}, "embedding": list(embedding),
            "sparse_embedding": sparse}


def setup_records(database, records=None, *, tenant="tenant", name="main", sparse=None):
    collections = CollectionStore(Store(database))
    collection = collections.create(tenant, name, {}, config(sparse=sparse))
    collections.write(tenant, collection["id"], "add", records or [record()], expected_generation=0)
    return collections, collections.get(tenant, collection["id"])


def snapshot(collections):
    with collections.store.connect() as connection:
        return {table: sorted(json.dumps(dict(row), sort_keys=True) for row in connection.execute(f"SELECT * FROM {table}"))
                for table in (*TABLE_FIELDS, "audit")}


def data(collections, cid, tenant="tenant"):
    with collections.read_snapshot(tenant, cid, include_embeddings=True) as scoped:
        return scoped.records


def test_native_pgvector_ranks_only_authorized_records_and_unicode_ties(database, monkeypatch):
    identifiers = ["a", "z", "Z", "中", "Å", "_", "10"]
    records = [record(identifier, embedding=(1, 2, 3)) for identifier in reversed(identifiers)]
    collections, collection = setup_records(database, records)
    other = collections.create("other", "main", {}, config())
    collections.write("other", other["id"], "add", [record("foreign", embedding=(1, 2, 3))], expected_generation=0)
    monkeypatch.setattr("rags.collection_store.cosine", lambda *_: pytest.fail("PostgreSQL must perform native vector ranking"))
    with collections.read_snapshot("tenant", collection["id"]) as scoped:
        ranked = scoped.dense([1, 2, 3], set(identifiers) | {"foreign"}, 3)
        assert [identifier for identifier, _ in ranked] == sorted(identifiers)[:3]
        assert all(distance == pytest.approx(0.0, abs=2e-7) for _, distance in ranked)
        assert scoped.embeddings(["Å", "foreign"]) == {"Å": canonical_dense([1, 2, 3])}
    with collections.store.connect() as connection:
        assert {row["kind"] for row in connection.execute("SELECT pg_typeof(embedding)::text AS kind FROM rag_collection_versions")} == {"vector"}
        assert {row["kind"] for row in connection.execute("SELECT pg_typeof(sparse_embedding)::text AS kind FROM rag_collection_versions")} == {"text"}
        assert connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0


def test_native_dense_query_and_projection_retain_mvcc_snapshot_during_gc(database):
    collections, collection = setup_records(database)
    writer = CollectionStore(Store(database))
    with collections.read_snapshot("tenant", collection["id"]) as scoped:
        original_version = scoped.records[0]["version"]
        assert scoped.collection["generation"] == 1
        writer.write("tenant", collection["id"], "update", [record(document="new", embedding=(0, 1, 0))], expected_generation=1)
        with writer.store.connect() as connection:
            assert connection.execute("SELECT 1 FROM rag_collection_versions WHERE tenant=? AND id=?",
                                      ("tenant", original_version)).fetchone() is None
        assert scoped.records[0]["document"] == "Synthetic source"
        assert scoped.dense([1, 0, 0], {"one"}, 1) == [("one", 0.0)]
        assert scoped.embeddings(["one"])["one"] == [1.0, 0.0, 0.0]
    assert data(collections, collection["id"])[0]["document"] == "new"


def test_postgres_payload_failure_rolls_back_entire_batch_and_dimension_binding(database):
    import psycopg
    collections = CollectionStore(Store(database))
    collection = collections.create("tenant", "main", {}, config())
    before = snapshot(collections)
    with collections.store.connect() as connection:
        connection.execute("CREATE FUNCTION fail_collection_payload() RETURNS trigger LANGUAGE plpgsql AS $$ "
                           "BEGIN IF NEW.document='explode' THEN RAISE EXCEPTION 'fixture write failure' USING ERRCODE='23514'; "
                           "END IF; RETURN NEW; END; $$")
        connection.execute("CREATE TRIGGER collection_failure BEFORE INSERT ON rag_collection_versions "
                           "FOR EACH ROW EXECUTE FUNCTION fail_collection_payload()")
    with pytest.raises(psycopg.errors.CheckViolation):
        collections.write("tenant", collection["id"], "add", [record("a", document="first"), record("b", document="explode")], expected_generation=0)
    assert snapshot(collections) == before
    assert collections.get("tenant", collection["id"])["configuration"]["embedding"]["dimensions"] is None


@pytest.mark.parametrize("operation", ["create", "modify", "fork"])
def test_postgres_mutation_response_validator_failure_rolls_back_audit_and_rows(database, operation):
    collections, origin = setup_records(database)
    before = snapshot(collections)
    def reject(result):
        assert result["name"] == "candidate"
        raise CapabilityError("collection_response_limit", "Fixture response limit", 422)
    guarded = CollectionStore(collections.store, result_validator=reject)
    actions = {
        "create": lambda: guarded.create("tenant", "candidate", {}, config()),
        "modify": lambda: guarded.modify("tenant", origin["id"], name="candidate", expected_generation=1),
        "fork": lambda: guarded.fork("tenant", origin["id"], "candidate", expected_generation=1),
    }
    with pytest.raises(CapabilityError) as failure:
        actions[operation]()
    assert failure.value.code == "collection_response_limit"
    assert snapshot(collections) == before


def test_postgres_competing_writers_commit_one_generation(database):
    collections, collection = setup_records(database)
    other = CollectionStore(Store(database))
    barrier = Barrier(2)
    def attempt(item):
        owner, name = item
        barrier.wait(timeout=10)
        try:
            return owner.write("tenant", collection["id"], "upsert", [record(document=name), record(name)], expected_generation=1)
        except CapabilityError as exc:
            return exc
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, [(collections, "a"), (other, "b")]))
    assert len([result for result in results if isinstance(result, dict)]) == 1
    assert [result.code for result in results if isinstance(result, CapabilityError)] == ["collection_generation_conflict"]
    current = collections.get("tenant", collection["id"])
    assert current["generation"] == 2 and current["record_count"] == 2
    with collections.store.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM rag_collection_versions").fetchone()[0] == 2


def test_postgres_fork_copies_only_mappings_and_retains_last_reference(database):
    collections, original = setup_records(database, [record("a"), record("b", embedding=(0, 1, 0))])
    before = data(collections, original["id"])
    branch = collections.fork("tenant", original["id"], "branch", expected_generation=1)
    with collections.store.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM rag_collection_versions").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM rag_collection_records").fetchone()[0] == 4
    assert data(collections, branch["id"]) == before
    collections.write("tenant", original["id"], "update", [record("a", document="new")], expected_generation=1)
    assert data(collections, branch["id"]) == before
    collections.delete_collection("tenant", original["id"], expected_generation=2)
    assert data(collections, branch["id"]) == before
    collections.delete_collection("tenant", branch["id"], expected_generation=0)
    with collections.store.connect() as connection:
        assert all(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0 for table in TABLE_FIELDS)


def test_postgres_concurrent_last_reference_gc_leaves_no_orphan(database):
    # More versions than lock stripes guarantees collisions. Reverse batch order
    # on the second branch to exercise a common ordering for the shared locks.
    identifiers = [f"record-{index:03d}" for index in range(100)]
    collections, source = setup_records(database, [record(identifier) for identifier in identifiers])
    branch = collections.fork("tenant", source["id"], "branch")
    barrier = Barrier(2)
    def remove(item):
        collection, ids = item
        barrier.wait(timeout=10)
        return collections.delete_records("tenant", collection["id"], ids,
                                          expected_generation=collection["generation"])
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(remove, [(source, identifiers), (branch, list(reversed(identifiers)))]))
    assert [result["deleted"] for result in results] == [100, 100]
    with collections.store.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM rag_collection_versions").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM rag_collection_records").fetchone()[0] == 0


def test_postgres_fork_and_source_delete_race_preserves_references(database):
    collections, source = setup_records(database)
    barrier = Barrier(2)
    def fork():
        barrier.wait(timeout=10)
        try:
            return collections.fork("tenant", source["id"], "branch", expected_generation=1)
        except CapabilityError as exc:
            return exc
    def remove():
        barrier.wait(timeout=10)
        return collections.delete_collection("tenant", source["id"], expected_generation=1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        created, deleted = pool.submit(fork), pool.submit(remove)
        result = created.result()
        assert deleted.result()["deleted"]
    with collections.store.connect() as connection:
        versions = connection.execute("SELECT COUNT(*) FROM rag_collection_versions").fetchone()[0]
    if isinstance(result, dict):
        assert versions == 1 and data(collections, result["id"])[0]["id"] == "one"
    else:
        assert result.code == "collection_not_found" and versions == 0


def test_version_reference_locks_are_bounded_for_maximum_collection_size(database):
    collections = CollectionStore(Store(database))
    with collections.store.connect() as connection:
        collections._lock_versions(connection, "tenant", (f"version-{i}" for i in range(10_000)))
        count = connection.execute("SELECT COUNT(*) FROM pg_locks WHERE pid=pg_backend_pid() AND locktype='advisory'").fetchone()[0]
        assert 1 <= count <= 32


def test_postgres_scan_limits_projection_and_sparse_float32_persistence(database):
    sparse = {"indices": [1, 20], "values": [0.1, 10], "space_id": "vocab"}
    _collections, collection = setup_records(database, [record(document="x" * 1000, sparse=sparse)], sparse="vocab")
    limited = CollectionStore(Store(database), max_scan_bytes=256)
    with pytest.raises(CapabilityError) as failure, limited.read_snapshot("tenant", collection["id"]):
        pytest.fail("Oversized data must not yield a partial collection")
    assert failure.value.code == "collection_limit_exceeded"
    reopened = CollectionStore(Store(database))
    assert data(reopened, collection["id"])[0]["sparse_embedding"]["values"][0] == pytest.approx(0.1)
    with reopened.read_snapshot("tenant", collection["id"], record_ids=[]) as scoped:
        assert scoped.records == []


def test_sqlite_import_preserves_exact_vectors_shared_versions_and_fork_mutability(database, tmp_path):
    source = CollectionStore(Store(str(tmp_path / "source.db")))
    collection = source.create("tenant", "main", {"tags": ["a", "b"]}, config(sparse="vocab"))
    sparse = {"indices": [1, 5], "values": [0.1, 3], "space_id": "vocab"}
    source.write("tenant", collection["id"], "add", [record("a", embedding=(0.1, 0.2, 0.3), sparse=sparse), record("b")], expected_generation=0)
    branch = source.fork("tenant", collection["id"], "branch", expected_generation=1)
    source.write("tenant", collection["id"], "update", [record("a", document="updated")], expected_generation=1)
    foreign = source.create("other", "main", {}, config())
    source.write("other", foreign["id"], "add", [record("a", document="private")], expected_generation=0)
    source_before = snapshot(source)
    expected_source, expected_branch = data(source, collection["id"]), data(source, branch["id"])
    result = import_sqlite(source.store.path, database)
    assert result["source_read_only"] and result["status"] == "imported"
    assert result["tables"]["rag_collection_versions"] == 4
    target = CollectionStore(Store(database))
    assert target.get("tenant", collection["id"]) == source.get("tenant", collection["id"])
    assert target.get("tenant", branch["id"]) == source.get("tenant", branch["id"])
    assert data(target, collection["id"]) == expected_source
    assert data(target, branch["id"]) == expected_branch
    assert snapshot(source) == source_before
    target.delete_collection("tenant", collection["id"])
    assert data(target, branch["id"]) == expected_branch
    assert data(target, foreign["id"], tenant="other")[0]["document"] == "private"


def test_sqlite_import_rejects_inconsistent_collection_configuration_atomically(database, tmp_path):
    source = CollectionStore(Store(str(tmp_path / "corrupt.db")))
    collection = source.create("tenant", "main", {}, config())
    source.write("tenant", collection["id"], "add", [record()], expected_generation=0)
    with source.store.connect() as connection:
        connection.execute("UPDATE rag_collections SET configuration=?", (json.dumps(config(dimensions=2)),))
    with pytest.raises(ValueError, match="dimensions"):
        import_sqlite(source.store.path, database)
    target = CollectionStore(Store(database))
    assert all(not rows for rows in snapshot(target).values())


def test_import_refuses_nonempty_collection_target_and_preserves_both_sides(database, tmp_path):
    source = CollectionStore(Store(str(tmp_path / "source.db")))
    source.create("tenant", "source", {}, config())
    target, _ = setup_records(database)
    before_source, before_target = snapshot(source), snapshot(target)
    with pytest.raises(ValueError, match="must be empty"):
        import_sqlite(source.store.path, database)
    assert snapshot(source) == before_source and snapshot(target) == before_target
