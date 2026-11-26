"""Real SQLite collection atomicity, MVCC, fork ownership and bounded reads."""
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from rags.collection_models import canonical_dense
from rags.collection_store import TABLE_FIELDS, CollectionStore
from rags.migrate import _source_tables
from rags.registry import CapabilityError
from rags.storage import Store

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("operation", ["create", "modify", "fork", "get_or_create", "noop_modify"])
def test_exact_mutation_result_validation_rolls_back_rows_and_audit(collections, operation):
    origin = create(collections)
    insert(collections, origin)
    before = database_snapshot(collections)
    seen = []
    def reject(result):
        seen.append(result)
        raise CapabilityError("collection_response_limit", "Fixture response limit", 422)
    guarded = CollectionStore(collections.store, result_validator=reject)
    operations = {
        "create": lambda: guarded.create("tenant", "new", {}, configuration()),
        "modify": lambda: guarded.modify("tenant", origin["id"], name="renamed", expected_generation=1),
        "fork": lambda: guarded.fork("tenant", origin["id"], "branch", expected_generation=1),
        "get_or_create": lambda: guarded.create("tenant", origin["name"], {}, configuration(), get_or_create=True),
        "noop_modify": lambda: guarded.modify("tenant", origin["id"], name=origin["name"], expected_generation=1),
    }
    with pytest.raises(CapabilityError) as failure:
        operations[operation]()
    assert failure.value.code == "collection_response_limit"
    assert len(seen) == 1
    expected = {"create": ("new", 0, 0), "modify": ("renamed", 2, 1), "fork": ("branch", 0, 1),
                "get_or_create": (origin["name"], 1, 1), "noop_modify": (origin["name"], 1, 1)}[operation]
    assert (seen[0]["name"], seen[0]["generation"], seen[0]["record_count"]) == expected
    assert database_snapshot(collections) == before


def test_collection_result_validator_requires_callable(collections):
    with pytest.raises(ValueError, match="callable"):
        CollectionStore(collections.store, result_validator=True)


@pytest.mark.parametrize("operation", ["create", "modify", "fork"])
def test_unicode_endpoint_control_response_limit_rejects_before_commit(tmp_path, operation):
    from rags.auth import PERMISSIONS, Principal
    from rags.collection_models import CollectionConfig
    from rags.collection_service import CollectionService
    from rags.config import Settings
    from rags.providers import Providers

    settings = Settings(db_path=str(tmp_path / "unicode-endpoint.db"), keys={"key": "tenant"},
                        embedding_provider="openai", model_provider="extractive", openai_key="",
                        openai_base_url="https://example.invalid/" + "\U0001f4a1" * 4070,
                        embedding_model="text-embedding-3-small", chat_model="", anthropic_key="",
                        anthropic_model="", collection_config=CollectionConfig(max_response_bytes=32768))
    service = CollectionService(Store(settings.db_path), Providers(settings))
    principal = Principal("tenant", "fixture", PERMISSIONS, "api_key")
    origin = service.dispatch(principal, "create", data={"name": "origin"})
    before = database_snapshot(service.database)
    metadata = {"a": "x" * 4096, "b": "x" * 4096, "c": "x" * 4096, "d": "x" * 4050}
    payload = {"metadata": metadata}
    if operation in {"create", "fork"}:
        payload["name"] = "destination"
    with pytest.raises(CapabilityError) as failure:
        service.dispatch(principal, operation, None if operation == "create" else origin["id"], payload)
    assert failure.value.code == "collection_response_limit"
    assert database_snapshot(service.database) == before


def configuration(*, dimensions=None, external=False, sparse=None):
    return {"embedding": {
        "mode": "external" if external else "configured", "model": "fixture-v1",
        "endpoint": "external" if external else "local", "space_id": "fixture-space",
        "dimensions": dimensions, "normalization": "l2-f32-v1",
    }, "sparse_space_id": sparse}


def record(identifier="one", *, document="Synthetic record", metadata=None,
           embedding=(1, 0, 0), sparse=None):
    return {"id": identifier, "document": document, "metadata": metadata or {},
            "embedding": list(embedding), "sparse_embedding": sparse}


@pytest.fixture
def collections(tmp_path):
    return CollectionStore(Store(str(tmp_path / "collections.db")))


def create(collections, name="collection", tenant="tenant", **options):
    return collections.create(tenant, name, {}, configuration(**options))


def insert(collections, collection, records=None, tenant="tenant"):
    return collections.write(tenant, collection["id"], "add", records or [record()],
                             expected_generation=collection["generation"])


def data(collections, cid, tenant="tenant"):
    with collections.read_snapshot(tenant, cid, include_embeddings=True) as snapshot:
        return snapshot.records


def database_snapshot(collections):
    with collections.store.connect() as connection:
        return {table: sorted(json.dumps(dict(row), sort_keys=True) for row in connection.execute(f"SELECT * FROM {table}"))
                for table in (*TABLE_FIELDS, "audit")}


def counts(collections):
    with collections.store.connect() as connection:
        return {table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in TABLE_FIELDS}


def test_collection_crud_names_pagination_tenant_isolation_and_get_or_create(collections):
    first = create(collections, "same")
    other = create(collections, "same", tenant="other")
    second = create(collections, "second")
    assert first["id"] != other["id"]
    assert collections.collection_count("tenant") == 2
    assert collections.collection_count("unknown") == 0
    assert [c["id"] for c in collections.list("tenant", limit=1, offset=1)] == [second["id"]]
    assert collections.list("unknown") == []
    assert collections.find_by_name("tenant", "same") == first
    assert collections.find_by_name("tenant", "missing") is None
    with pytest.raises(CapabilityError, match="already exists") as failure:
        create(collections, "same")
    assert failure.value.status_code == 409
    reused = collections.create("tenant", "same", {"changed": True}, configuration(dimensions=7), get_or_create=True)
    assert reused == first
    changed = collections.modify("tenant", first["id"], name="renamed", metadata={"tags": ["a", "b"]}, expected_generation=0)
    assert changed["id"] == first["id"] and changed["generation"] == 1
    assert changed["metadata"] == {"tags": ["a", "b"]}
    assert collections.find_by_name("tenant", "same") is None
    assert collections.modify("tenant", first["id"], name="renamed", expected_generation=1) == changed
    before = database_snapshot(collections)
    with pytest.raises(CapabilityError):
        collections.modify("tenant", first["id"], name="second")
    assert database_snapshot(collections) == before
    with pytest.raises(CapabilityError) as forbidden:
        collections.get("other", first["id"])
    assert forbidden.value.code == "collection_not_found" and forbidden.value.status_code == 404


@pytest.mark.parametrize("name", ["", "bad/name", ".starts", "127.0.0.1", "a" * 129, 1])
def test_invalid_names_never_create_collection_or_audit(collections, name):
    before = database_snapshot(collections)
    with pytest.raises(CapabilityError) as failure:
        create(collections, name)
    assert failure.value.code == "invalid_collection_input"
    assert database_snapshot(collections) == before


@pytest.mark.parametrize("change", [
    {"mode": "bad"}, {"model": ""}, {"endpoint": ""}, {"space_id": ""},
    {"normalization": "none"}, {"dimensions": True}, {"dimensions": 4097},
    {"mode": "external", "dimensions": None}, {"unknown": True},
])
def test_invalid_immutable_configuration_rejected_before_write(collections, change):
    config = configuration()
    config["embedding"].update(change)
    with pytest.raises(CapabilityError):
        collections.create("tenant", "bad", {}, config)
    assert collections.collection_count("tenant") == 0


@pytest.mark.parametrize("kwargs", [{"max_records": 0}, {"max_records": 10_001}, {"max_records": True},
                                    {"max_scan_bytes": 0}, {"max_scan_bytes": 16 * 1024 * 1024 + 1}])
def test_store_bounds_cannot_expand_contract(collections, kwargs):
    with pytest.raises(ValueError):
        CollectionStore(collections.store, **kwargs)


@pytest.mark.parametrize("limit,offset", [(0, 0), (101, 0), (True, 0), (1, -1), (1, True), (1, 2 ** 31)])
def test_collection_list_is_bounded(collections, limit, offset):
    with pytest.raises(CapabilityError):
        collections.list("tenant", limit=limit, offset=offset)


def test_record_modes_are_atomic_bind_dimensions_and_keep_stable_creation(collections):
    collection = create(collections)
    cid = collection["id"]
    assert insert(collections, collection, [record("a"), record("b")]) == {
        "collection_id": cid, "generation": 1, "changed": 2, "added": 2, "updated": 0,
    }
    initial = {item["id"]: item for item in data(collections, cid)}
    assert collections.get("tenant", cid)["configuration"]["embedding"]["dimensions"] == 3
    for mode, records in [("add", [record("c"), record("a")]), ("update", [record("a", document="changed"), record("missing")])]:
        before = database_snapshot(collections)
        with pytest.raises(CapabilityError) as failure:
            collections.write("tenant", cid, mode, records, expected_generation=1)
        assert failure.value.code == "collection_conflict"
        assert database_snapshot(collections) == before
    result = collections.write("tenant", cid, "upsert", [record("a"), record("b", document="changed"), record("c")], expected_generation=1)
    assert result == {"collection_id": cid, "generation": 2, "changed": 2, "added": 1, "updated": 1}
    current = {item["id"]: item for item in data(collections, cid)}
    assert current["a"] == initial["a"]
    assert current["b"]["version"] != initial["b"]["version"]
    assert current["b"]["created_at"] == initial["b"]["created_at"]
    assert counts(collections)["rag_collection_versions"] == 3
    before = database_snapshot(collections)
    assert collections.write("tenant", cid, "update", [record("b", document="changed")], expected_generation=2)["changed"] == 0
    assert database_snapshot(collections) == before


@pytest.mark.parametrize("records", [
    [], [record("a"), record("a")], [record(embedding=(0, 0, 0))],
    [record(embedding=(float("inf"), 0, 0))], [record(metadata={"nested": {"x": 1}})],
    [dict(record(), unexpected=True)], [record(document="x" * 20_001)],
    [{"document": "missing fields"}], [record("x\x00")],
])
def test_bad_prepared_batch_cannot_bind_dimensions_or_partially_mutate(collections, records):
    collection = create(collections)
    before = database_snapshot(collections)
    with pytest.raises(CapabilityError) as failure:
        collections.write("tenant", collection["id"], "add", records, expected_generation=0)
    assert failure.value.code == "invalid_collection_input"
    assert database_snapshot(collections) == before


def test_dimension_sparse_identity_and_record_count_failures_are_atomic(collections):
    limited = CollectionStore(collections.store, max_records=2)
    collection = create(limited, sparse="vocab-v1")
    sparse = {"indices": [1, 4], "values": [1, 2], "space_id": "vocab-v1"}
    insert(limited, collection, [record("a", sparse=sparse), record("b")])
    cid = collection["id"]
    before = database_snapshot(limited)
    failures = [
        (record("a", embedding=(1, 0)), "embedding_dimension_mismatch"),
        (record("a", sparse={**sparse, "space_id": "wrong"}), "embedding_space_mismatch"),
        (record("c"), "collection_limit_exceeded"),
    ]
    for prepared, code in failures:
        with pytest.raises(CapabilityError) as failure:
            limited.write("tenant", cid, "upsert", [prepared], expected_generation=1)
        assert failure.value.code == code
        assert database_snapshot(limited) == before
    with pytest.raises(CapabilityError):
        limited.write("tenant", cid, "upsert", [record("a"), record("b", embedding=(1, 0))], expected_generation=1)
    assert database_snapshot(limited) == before
    assert data(limited, cid)[0]["sparse_embedding"] == sparse


@pytest.mark.parametrize("generation", [-1, True, 1.5, 2 ** 63])
def test_invalid_generations_fail_without_mutation(collections, generation):
    collection = create(collections)
    before = database_snapshot(collections)
    operations = [
        lambda: collections.write("tenant", collection["id"], "add", [record()], expected_generation=generation),
        lambda: collections.modify("tenant", collection["id"], name="new", expected_generation=generation),
        lambda: collections.fork("tenant", collection["id"], "fork", expected_generation=generation),
        lambda: collections.delete_collection("tenant", collection["id"], expected_generation=generation),
        lambda: collections.delete_records("tenant", collection["id"], ["one"], expected_generation=generation),
    ]
    for operation in operations:
        with pytest.raises(CapabilityError):
            operation()
        assert database_snapshot(collections) == before


def test_final_int64_generation_remains_readable_forkable_and_deletable(collections):
    collection = create(collections)
    insert(collections, collection)
    cid, maximum = collection["id"], 2 ** 63 - 1
    with collections.store.connect() as connection:
        connection.execute("UPDATE rag_collections SET generation=? WHERE tenant=? AND id=?",
                           (maximum - 1, "tenant", cid))
    changed = collections.modify("tenant", cid, metadata={"final": True}, expected_generation=maximum - 1)
    assert changed["generation"] == maximum
    assert collections.modify("tenant", cid, metadata={"final": True}, expected_generation=maximum) == changed
    assert collections.write("tenant", cid, "update", [record()], expected_generation=maximum)["changed"] == 0
    assert collections.delete_records("tenant", cid, ["absent"], expected_generation=maximum)["deleted"] == 0
    before = database_snapshot(collections)
    for action in (
        lambda: collections.modify("tenant", cid, name="new", expected_generation=maximum),
        lambda: collections.write("tenant", cid, "update", [record(document="new")], expected_generation=maximum),
        lambda: collections.delete_records("tenant", cid, ["one"], expected_generation=maximum),
    ):
        with pytest.raises(CapabilityError) as failure:
            action()
        assert failure.value.code == "collection_limit_exceeded"
        assert database_snapshot(collections) == before
    fork = collections.fork("tenant", cid, "branch", expected_generation=maximum)
    assert fork["generation"] == 0 and fork["fork_source_generation"] == maximum
    assert collections.delete_collection("tenant", cid, expected_generation=maximum)["deleted"]
    assert data(collections, fork["id"])[0]["document"] == "Synthetic record"


def test_fork_shares_payloads_and_preserves_both_branches_until_last_reference(collections):
    origin = create(collections, sparse="vocab")
    insert(collections, origin, [record("a", metadata={"tags": ["one"]}), record("b")])
    old = data(collections, origin["id"])
    fork = collections.fork("tenant", origin["id"], "branch", metadata={"branch": True}, expected_generation=1)
    assert fork["fork_source_id"] == origin["id"] and fork["fork_source_generation"] == 1
    assert fork["generation"] == 0 and fork["record_count"] == 2
    assert fork["configuration"] == collections.get("tenant", origin["id"])["configuration"]
    assert counts(collections)["rag_collection_versions"] == 2
    assert counts(collections)["rag_collection_records"] == 4
    assert data(collections, fork["id"]) == old
    collections.write("tenant", origin["id"], "update", [record("a", document="parent new")], expected_generation=1)
    assert data(collections, fork["id"]) == old
    assert counts(collections)["rag_collection_versions"] == 3
    collections.delete_records("tenant", fork["id"], ["a", "missing"], expected_generation=0)
    assert counts(collections)["rag_collection_versions"] == 2
    assert data(collections, origin["id"])[0]["document"] == "parent new"
    deleted = collections.delete_collection("tenant", origin["id"], expected_generation=2)
    assert deleted["deleted_records"] == 2
    assert [item["id"] for item in data(collections, fork["id"])] == ["b"]
    assert counts(collections)["rag_collection_versions"] == 1
    collections.delete_collection("tenant", fork["id"], expected_generation=1)
    assert all(count == 0 for count in counts(collections).values())


def test_same_record_ids_and_tenants_never_share_unrelated_payloads(collections):
    a = create(collections, "a")
    b = create(collections, "b")
    private = create(collections, "a", tenant="private")
    insert(collections, a, [record("same", document="A")])
    insert(collections, b, [record("same", document="B")])
    insert(collections, private, [record("same", document="Private")], tenant="private")
    assert data(collections, a["id"])[0]["document"] == "A"
    assert data(collections, b["id"])[0]["document"] == "B"
    before = database_snapshot(collections)
    for operation in [
        lambda: collections.write("tenant", private["id"], "update", [record()], expected_generation=1),
        lambda: collections.delete_records("tenant", private["id"], ["same"], expected_generation=1),
        lambda: collections.delete_collection("tenant", private["id"]),
        lambda: collections.fork("tenant", private["id"], "stolen"),
    ]:
        with pytest.raises(CapabilityError) as failure:
            operation()
        assert failure.value.code == "collection_not_found"
    assert database_snapshot(collections) == before
    assert collections.store.scope("tenant") == []
    assert collections.store.list_documents("tenant") == []


def test_record_deletion_noops_and_generation_conflicts(collections):
    collection = create(collections)
    cid = collection["id"]
    insert(collections, collection)
    before = database_snapshot(collections)
    for ids in ([], ["missing"]):
        assert collections.delete_records("tenant", cid, ids, expected_generation=1) == {
            "collection_id": cid, "generation": 1, "deleted": 0,
        }
    assert database_snapshot(collections) == before
    for operation in [
        lambda: collections.write("tenant", cid, "upsert", [record("other")], expected_generation=0),
        lambda: collections.modify("tenant", cid, name="renamed", expected_generation=0),
        lambda: collections.fork("tenant", cid, "branch", expected_generation=0),
        lambda: collections.delete_records("tenant", cid, ["one"], expected_generation=0),
        lambda: collections.delete_collection("tenant", cid, expected_generation=0),
    ]:
        with pytest.raises(CapabilityError) as failure:
            operation()
        assert failure.value.code == "collection_generation_conflict"
        assert database_snapshot(collections) == before


def test_sql_failure_after_first_record_rolls_back_payload_mapping_generation_and_audit(collections):
    collection = create(collections)
    insert(collections, collection, [record("a"), record("b")])
    before = database_snapshot(collections)
    with collections.store.connect() as connection:
        connection.execute("CREATE TRIGGER fixture_fail BEFORE INSERT ON rag_collection_versions "
                           "WHEN NEW.document='explode' BEGIN SELECT RAISE(ABORT,'fixture failure'); END")
    with pytest.raises(sqlite3.IntegrityError):
        collections.write("tenant", collection["id"], "update", [record("a", document="first"), record("b", document="explode")],
                          expected_generation=1)
    assert database_snapshot(collections) == before


@pytest.mark.parametrize("action", ["write", "fork", "delete"])
def test_final_audit_failure_rolls_back_whole_collection_transaction(collections, action):
    collection = create(collections)
    insert(collections, collection)
    before = database_snapshot(collections)
    with collections.store.connect() as connection:
        connection.execute("CREATE TRIGGER fixture_audit_fail BEFORE INSERT ON audit "
                           "BEGIN SELECT RAISE(ABORT,'audit fixture failure'); END")
    with pytest.raises(sqlite3.IntegrityError):
        if action == "write":
            collections.write("tenant", collection["id"], "update", [record(document="changed")], expected_generation=1)
        elif action == "fork":
            collections.fork("tenant", collection["id"], "branch")
        else:
            collections.delete_collection("tenant", collection["id"])
    assert database_snapshot(collections) == before


def test_sqlite_read_filter_ranking_keeps_one_snapshot_through_concurrent_update_and_gc(collections):
    collection = create(collections)
    insert(collections, collection)
    writer = CollectionStore(Store(collections.store.path))
    with collections.read_snapshot("tenant", collection["id"]) as snapshot:
        old_version = snapshot.records[0]["version"]
        writer.write("tenant", collection["id"], "update", [record(document="new", embedding=(0, 1, 0))], expected_generation=1)
        assert snapshot.records[0]["document"] == "Synthetic record"
        assert snapshot.dense([1, 0, 0], {"one"}, 1) == [("one", 0.0)]
        assert snapshot.embeddings(["one"]) == {"one": [1.0, 0.0, 0.0]}
        with writer.store.connect() as connection:
            assert connection.execute("SELECT 1 FROM rag_collection_versions WHERE id=?", (old_version,)).fetchone() is None
    assert data(collections, collection["id"])[0]["document"] == "new"
    with pytest.raises(RuntimeError, match="closed"):
        snapshot.dense([1, 0, 0], {"one"}, 1)


def test_competing_batch_writers_have_one_cas_winner(collections):
    collection = create(collections)
    insert(collections, collection)
    barrier = Barrier(2)
    def attempt(index):
        barrier.wait(timeout=5)
        try:
            return collections.write("tenant", collection["id"], "upsert", [record("one", document=str(index)), record(str(index))],
                                     expected_generation=1)
        except CapabilityError as exc:
            return exc
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, range(2)))
    assert len([item for item in results if isinstance(item, dict)]) == 1
    assert [item.code for item in results if isinstance(item, CapabilityError)] == ["collection_generation_conflict"]
    assert collections.get("tenant", collection["id"])["generation"] == 2
    assert collections.get("tenant", collection["id"])["record_count"] == 2
    assert counts(collections)["rag_collection_versions"] == 2


def test_concurrent_last_reference_deletion_has_no_orphan_payload(collections):
    source = create(collections)
    insert(collections, source)
    branch = collections.fork("tenant", source["id"], "branch")
    barrier = Barrier(2)
    def remove(item):
        barrier.wait(timeout=5)
        return collections.delete_records("tenant", item["id"], ["one"], expected_generation=item["generation"])
    current = collections.get("tenant", source["id"])
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(remove, (current, branch)))
    assert [result["deleted"] for result in results] == [1, 1]
    assert counts(collections)["rag_collection_versions"] == 0


def test_fork_racing_source_delete_is_atomic(collections):
    source = create(collections)
    insert(collections, source)
    barrier = Barrier(2)
    def fork():
        barrier.wait(timeout=5)
        try:
            return collections.fork("tenant", source["id"], "branch", expected_generation=1)
        except CapabilityError as exc:
            return exc
    def remove():
        barrier.wait(timeout=5)
        return collections.delete_collection("tenant", source["id"], expected_generation=1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.submit(fork), pool.submit(remove)
        result, deleted = first.result(), second.result()
    assert deleted["deleted"]
    if isinstance(result, dict):
        assert len(data(collections, result["id"])) == 1
        assert counts(collections)["rag_collection_versions"] == 1
    else:
        assert result.code == "collection_not_found"
        assert all(value == 0 for value in counts(collections).values())


def test_snapshot_paging_order_id_filtering_projection_and_restart(collections):
    collection = create(collections)
    identifiers = [f"record-{index:03}" for index in range(80)] + ["Å", "中", "Z", "a"]
    insert(collections, collection, [record(identifier, embedding=(1, 2, 3)) for identifier in reversed(identifiers)])
    reopened = CollectionStore(Store(collections.store.path))
    with reopened.read_snapshot("tenant", collection["id"]) as snapshot:
        assert [item["id"] for item in snapshot.records] == sorted(identifiers)
        assert all("embedding" not in item for item in snapshot.records)
        assert [identifier for identifier, _ in snapshot.dense([1, 2, 3], set(identifiers) | {"foreign"}, 10)] == sorted(identifiers)[:10]
        projected = snapshot.embeddings(["中", "foreign"])
        assert projected == {"中": canonical_dense([1, 2, 3])}
        projected["中"][0] = 99
        assert snapshot.embeddings(["中"])["中"] == canonical_dense([1, 2, 3])
    with reopened.read_snapshot("tenant", collection["id"], record_ids=identifiers[::2]) as snapshot:
        assert [item["id"] for item in snapshot.records] == sorted(identifiers[::2])
    with reopened.read_snapshot("tenant", collection["id"], record_ids=[]) as snapshot:
        assert snapshot.records == [] and snapshot.dense([1, 2, 3], set(identifiers), 1) == []


def test_snapshot_materialization_and_vector_projection_limits_fail_explicitly(collections):
    collection = create(collections)
    insert(collections, collection, [record("big", document="x" * 1000)])
    vector_collection = create(collections, "vectors")
    insert(collections, vector_collection, [record("vector", document=None, embedding=[1] * 4096)])
    limited = CollectionStore(collections.store, max_scan_bytes=512)
    with pytest.raises(CapabilityError) as failure, limited.read_snapshot("tenant", collection["id"]):
        pytest.fail("An oversized corpus must not yield a partial snapshot")
    assert failure.value.code == "collection_limit_exceeded"
    with limited.read_snapshot("tenant", vector_collection["id"]) as snapshot:
        assert snapshot.records[0]["document"] is None
        with pytest.raises(CapabilityError) as failure:
            snapshot.embeddings(["vector"])
        assert failure.value.code == "collection_limit_exceeded"
    with pytest.raises(CapabilityError), limited.read_snapshot("tenant", vector_collection["id"], include_embeddings=True):
        pytest.fail("Oversized vector projection must fail before yielding")
    with limited.read_snapshot("tenant", collection["id"], record_ids=[]) as snapshot:
        assert snapshot.records == []


def test_lowered_record_limit_can_read_selected_ids_but_cannot_silently_scan_or_fork(collections):
    collection = create(collections)
    insert(collections, collection, [record("a"), record("b")])
    limited = CollectionStore(collections.store, max_records=1)
    with pytest.raises(CapabilityError), limited.read_snapshot("tenant", collection["id"]):
        pytest.fail("Cannot silently scan part of an oversized collection")
    with limited.read_snapshot("tenant", collection["id"], record_ids=["a"]) as snapshot:
        assert [item["id"] for item in snapshot.records] == ["a"]
    before = database_snapshot(collections)
    with pytest.raises(CapabilityError):
        limited.fork("tenant", collection["id"], "branch")
    assert database_snapshot(collections) == before


@pytest.mark.parametrize("selection", ["one", 1, {"bad": "value"}, [""], [1], ["x\x00"]])
def test_snapshot_record_selection_rejects_unsafe_inputs(collections, selection):
    collection = create(collections)
    with pytest.raises(CapabilityError), collections.read_snapshot("tenant", collection["id"], record_ids=selection):
        pytest.fail("Invalid selection was accepted")


@pytest.mark.parametrize("ids", [["one", "one"], [1], "one"])
def test_delete_id_validation_is_atomic(collections, ids):
    collection = create(collections)
    insert(collections, collection)
    before = database_snapshot(collections)
    with pytest.raises(CapabilityError):
        collections.delete_records("tenant", collection["id"], ids, expected_generation=1)
    assert database_snapshot(collections) == before


def test_null_documents_sparse_clear_and_normalized_float32_round_trip(collections):
    collection = create(collections, dimensions=3, external=True, sparse="vocab")
    sparse = {"indices": [1, 10], "values": [0.1, 2], "space_id": "vocab"}
    insert(collections, collection, [record(document=None, embedding=(3e38, 3e38, 1), sparse=sparse)])
    stored = data(collections, collection["id"])[0]
    assert stored["document"] is None
    assert stored["embedding"] == canonical_dense([3e38, 3e38, 1])
    assert stored["sparse_embedding"]["values"][0] != 0.1  # Exact float32, not caller float64.
    collections.write("tenant", collection["id"], "update", [record(document="", embedding=stored["embedding"])], expected_generation=1)
    updated = data(collections, collection["id"])[0]
    assert updated["document"] == "" and updated["sparse_embedding"] is None


def test_dense_shape_bounds_and_invalid_vectors_do_not_mutate_snapshot(collections):
    collection = create(collections)
    insert(collections, collection)
    with collections.read_snapshot("tenant", collection["id"]) as snapshot:
        with pytest.raises(CapabilityError) as mismatch:
            snapshot.dense([1, 0], {"one"}, 1)
        assert mismatch.value.code == "embedding_dimension_mismatch"
        for vector, limit in [([0, 0, 0], 1), ([1, 0, 0], 0), ([1, 0, 0], 101), ([1, 0, 0], True)]:
            with pytest.raises(CapabilityError):
                snapshot.dense(vector, {"one"}, limit)
        assert snapshot.dense([1, 0, 0], {"unknown"}, 1) == []
        assert snapshot.dense([1, 0, 0], {"one"}, 1) == [("one", 0.0)]


def test_new_collection_tables_are_an_optional_complete_migration_group(tmp_path):
    legacy = Store(str(tmp_path / "legacy.db"))
    with legacy.connect() as connection:
        assert not set(TABLE_FIELDS) & _source_tables(connection)
    collections = CollectionStore(legacy)
    with legacy.connect() as connection:
        assert set(TABLE_FIELDS) <= _source_tables(connection)
        connection.execute("DROP TABLE rag_collection_records")
    with legacy.connect() as connection, pytest.raises(ValueError, match="complete collection schema"):
        _source_tables(connection)
    assert collections.store.path == legacy.path


def test_import_validation_rejects_corrupt_dimension_bindings_and_orphans(collections):
    collection = create(collections)
    insert(collections, collection)
    with collections.store.connect() as connection:
        collections.validate_import(connection)
    config = configuration(dimensions=2)
    with collections.store.connect() as connection:
        connection.execute("UPDATE rag_collections SET configuration=? WHERE tenant=? AND id=?", (json.dumps(config), "tenant", collection["id"]))
    with collections.store.connect() as connection:
        with pytest.raises(ValueError, match="dimensions"):
            collections.validate_import(connection)
        connection.execute("DELETE FROM rag_collection_records")
        with pytest.raises(ValueError, match="unreferenced"):
            collections.validate_import(connection)

