"""Real pgvector ranking, source transactions, assets and read-only migration."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
import hashlib
import io
import json
import os
import sqlite3
from threading import Barrier
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit
import uuid

import pytest

from rags.config import Settings
from rags.migrate import TABLE_FIELDS, import_sqlite
from rags.multimodal import AssetStore, MultimodalService
from rags.providers import Providers
from rags.registry import CapabilityError
from rags.retrieval import RAGService
from rags.storage import Store


pytestmark = [pytest.mark.integration, pytest.mark.external, pytest.mark.postgres]


@pytest.fixture
def database(monkeypatch):
    endpoint = os.getenv("RAGS_TEST_POSTGRES_URL")
    if not endpoint:
        pytest.fail("RAGS_TEST_POSTGRES_URL is required for selected PostgreSQL tests")
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import conninfo_to_dict

    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"postgres", "postgresql"}:
        pytest.fail("RAGS_TEST_POSTGRES_URL must be a PostgreSQL URL")
    schema = "rags_storage_" + uuid.uuid4().hex
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["options"] = (query.get("options", "") + f" -c search_path={schema},public").strip()
    isolated = urlunsplit(parsed._replace(query=urlencode(query, quote_via=quote)))
    assert conninfo_to_dict(isolated)["options"] == query["options"]
    monkeypatch.setenv("RAGS_ENV", "test")
    monkeypatch.setenv("RAGS_FEATURES_JSON", "{}")
    monkeypatch.setenv("RAGS_ADMISSION_POLICY", "compatibility")
    for key in ("RAGS_FEATURES_FILE", "RAGS_ADMISSION_PROFILE", "RAGS_DATABASE_URL"):
        monkeypatch.delenv(key, raising=False)
    with psycopg.connect(endpoint, autocommit=True, connect_timeout=5) as admin:
        extension = admin.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()
        assert extension is not None, "The external gate must bootstrap the vector extension"
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            with psycopg.connect(isolated, connect_timeout=5) as connection:
                assert connection.execute("SELECT current_schema()").fetchone()[0] == schema
            yield isolated
        finally:
            admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def add_document(store, tenant="tenant", identifier="policy", vector=(1.0, 0.0, 0.0), *,
                 text="Source needle", source="fixture", metadata=None, expected=None,
                 valid_from="2024-01-01T00:00:00.000000+00:00", recorded_at="2024-02-01T00:00:00.000000+00:00",
                 model="offline-hash-v1", endpoint="local", graph=False):
    profile = {"embedding_model": model, "embedding_endpoint": endpoint, "embedding_dimensions": len(vector)}
    profile_id = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()
    revision = uuid.uuid4().hex
    chunk_id = tenant + ":" + identifier + ":" + revision
    document = {"id": identifier, "revision": revision, "title": identifier, "source": source, "body": text,
                "metadata": metadata or {}, "checksum": hashlib.sha256(text.encode()).hexdigest(),
                "recorded_at": recorded_at, "valid_from": valid_from, "valid_to": None,
                "profile_id": profile_id, "index_profile": profile}
    chunk = {"id": chunk_id, "ordinal": 0, "section": "", "content": text, "embedding": list(vector),
             "embedding_model": model, "role": "leaf", "start_char": 0, "end_char": len(text),
             "token_count": len(text), "tokenizer_id": "unicode-v1", "token_unit": "unicode_units",
             "section_path": [], "evidence_ids": [chunk_id], "processing_metadata": {"fixture": True}}
    relations = ([{"source": "Alpha", "predicate": "links", "target": "Beta", "evidence_id": chunk_id}]
                 if graph else [])
    store.commit_revision(tenant, document, [chunk], relations, expected)
    return document, chunk


def dense(store, ids, vector=(1.0, 0.0, 0.0), limit=1, tenant="tenant", **profile):
    return store.dense(tenant, list(vector), limit, set(ids), embedding_model=profile.get("model", "offline-hash-v1"),
                       embedding_endpoint=profile.get("endpoint", "local"))


def snapshot(store):
    with store.connect() as connection:
        names = [row["table_name"] for row in connection.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema=current_schema()",
        )]
        return {name: sorted(json.dumps(dict(row), sort_keys=True, default=lambda value: bytes(value).hex())
                             for row in connection.execute(f"SELECT * FROM {name}"))
                for name in names}


def test_native_vector_ranking_applies_tenant_revision_and_metadata_before_top_k(database, monkeypatch):
    store = Store(database)
    old, _ = add_document(store, identifier="changed", metadata={"group": "visible"})
    _, current = add_document(store, identifier="changed", vector=(0, 1, 0), metadata={"group": "visible"},
                              expected=old["revision"], valid_from="2024-06-01T00:00:00.000000+00:00",
                              recorded_at="2024-06-02T00:00:00.000000+00:00")
    _, near = add_document(store, identifier="near", vector=(0.9, 0.1, 0), metadata={"group": "visible"})
    add_document(store, identifier="filtered-perfect", metadata={"group": "hidden"})
    _, foreign = add_document(store, tenant="foreign", identifier="foreign-perfect", vector=(1, 0, 0, 0))
    allowed = {row["id"] for row in store.scope("tenant", filters={"metadata": {"group": "visible"}})}
    assert allowed == {current["id"], near["id"]}
    without_vectors = store.scope("tenant", filters={"metadata": {"group": "visible"}}, include_embeddings=False)
    assert {row["id"] for row in without_vectors} == allowed
    assert all("embedding" not in row and row["index_profile"]["embedding_dimensions"] == 3 for row in without_vectors)
    monkeypatch.setattr("rags.storage.cosine", lambda *_: pytest.fail("PostgreSQL must execute native vector ranking"))
    assert dense(store, allowed | {foreign["id"]}) == [near["id"]]
    with store.connect() as connection:
        assert {row["kind"] for row in connection.execute("SELECT pg_typeof(embedding)::text AS kind FROM chunks")} == {"vector"}
        score = connection.execute("SELECT 1-(embedding <=> '[1,0,0]'::vector) AS score FROM chunks WHERE id=?",
                                   (near["id"],)).fetchone()["score"]
        assert score == pytest.approx(0.9938837, abs=1e-6)
    settings = replace(Settings.from_env(), db_path=database, embedding_provider="local", model_provider="extractive")
    provider = Providers(settings)
    monkeypatch.setattr(provider, "query_embedding", lambda _: [1, 0, 0])
    service = RAGService(store, provider)
    result = service.query("tenant", "Which source?", "dense", top_k=1, generate=False,
                           filters={"metadata": {"group": "visible"}})
    assert [row["chunk_id"] for row in result["citations"]] == [near["id"]]
    assert store.storage_backend == "postgresql" and store.vector_search == "pgvector_exact"


def test_postgres_full_text_filtering_and_gin_are_real(database):
    store = Store(database)
    _, allowed = add_document(store, identifier="allowed", text="needle policy")
    add_document(store, identifier="filtered", text="needle " * 20)
    _, foreign = add_document(store, tenant="foreign", text="needle " * 30)
    assert store.sparse("tenant", '"needle"', 1, {allowed["id"], foreign["id"]}) == [allowed["id"]]
    assert store.sparse("tenant", '"absent"', 1) == []
    assert store.sparse("tenant", "OR", 1) == []
    assert store.sparse("tenant", '"needle"', 1, set()) == []
    with store.connect() as connection:
        definition = connection.execute(
            "SELECT indexdef FROM pg_indexes WHERE schemaname=current_schema() AND indexname='chunks_fts_search'",
        ).fetchone()["indexdef"]
        assert "USING gin (search)" in definition


@pytest.mark.parametrize("corruption", ["dimensions", "model", "endpoint", "profile"])
def test_every_authorized_incompatible_vector_fails_instead_of_being_filtered(database, corruption):
    store = Store(database)
    document, bad = add_document(store, identifier="bad", vector=(0, 1, 0))
    _, good = add_document(store, identifier="good")
    with store.connect() as connection:
        if corruption == "dimensions":
            connection.execute("UPDATE chunks SET embedding='[0,1]'::vector WHERE id=?", (bad["id"],))
        elif corruption == "model":
            connection.execute("UPDATE chunks SET embedding_model='other-model' WHERE id=?", (bad["id"],))
        elif corruption == "endpoint":
            profile = {**document["index_profile"], "embedding_endpoint": "https://other.invalid"}
            connection.execute("UPDATE document_revisions SET index_profile=? WHERE revision=?",
                               (json.dumps(profile), document["revision"]))
        else:
            connection.execute("UPDATE chunks SET profile_id='corrupted' WHERE id=?", (bad["id"],))
    with pytest.raises(CapabilityError) as error:
        dense(store, {good["id"], bad["id"]})
    assert error.value.code == "requires_reindex" and error.value.status_code == 409
    assert dense(store, {good["id"]}) == [good["id"]]


def test_zero_and_extreme_representable_vectors_have_finite_deterministic_cosine(database):
    store = Store(database)
    _, zero = add_document(store, identifier="zero", vector=(0, 0))
    _, positive = add_document(store, identifier="positive", vector=(3.4028234663852886e38, 0))
    _, negative = add_document(store, identifier="negative", vector=(-3.4028234663852886e38, 0))
    identifiers = {row["id"] for row in (zero, positive, negative)}
    assert dense(store, identifiers, vector=(0, 0), limit=3) == sorted(identifiers)
    assert dense(store, identifiers, vector=(3.4028234663852886e38, 0), limit=3) == [positive["id"], zero["id"], negative["id"]]


@pytest.mark.parametrize("invalid", [(float("nan"), 0), (True, 0), (1e100, 0), (0,) * 16_001])
def test_unrepresentable_update_cannot_mutate_existing_source(database, invalid):
    store = Store(database)
    original, _ = add_document(store)
    before = snapshot(store)
    with pytest.raises(CapabilityError) as error:
        add_document(store, vector=invalid, expected=original["revision"])
    assert error.value.code == "invalid_embedding"
    assert snapshot(store) == before


def test_history_timestamp_ties_idempotent_initialization_and_delete(database):
    store = Store(database)
    original, _ = add_document(store, source="old", metadata={"active": True})
    newer, current = add_document(store, source="new", text="Updated source", expected=original["revision"],
                                 metadata={"active": True})
    add_document(store, tenant="foreign")
    assert {row["id"] for row in store.scope("tenant")} == {current["id"]}
    assert store.scope("tenant", filters={"source": "old"}) == []
    assert store.scope("tenant", filters={"metadata": {"active": 1}}) == []
    assert store.scope("tenant", as_of="2024-03-01T00:00:00Z")[0]["revision"] == newer["revision"]
    assert [row["revision"] for row in store.revisions("tenant", "policy")] == [newer["revision"], original["revision"]]
    before = snapshot(store)
    reopened = Store(database)
    assert snapshot(reopened) == before
    with reopened.connect() as connection:
        orders = connection.execute("SELECT revision_order FROM document_revisions WHERE tenant=? ORDER BY revision_order",
                                    ("tenant",)).fetchall()
        assert len(orders) == 2 and orders[0]["revision_order"] < orders[1]["revision_order"]
    assert reopened.delete("tenant", "policy", "2024-12-01T00:00:00+00:00")
    assert not reopened.delete("tenant", "policy", "2024-12-01T00:00:00+00:00")
    assert reopened.revisions("tenant", "policy") == [] and reopened.scope("tenant") == []
    assert reopened.sparse("tenant", "needle", 5) == []
    assert reopened.get_document("foreign", "policy") is not None


@pytest.mark.parametrize("existing", [False, True])
def test_concurrent_create_or_update_has_one_cas_winner(database, existing):
    store = Store(database)
    expected = add_document(store)[0]["revision"] if existing else None
    before = len(store.revisions("tenant", "policy"))
    workers = [Store(database), Store(database)]
    barrier = Barrier(2, timeout=10)

    def update(index):
        barrier.wait()
        try:
            return add_document(workers[index], text=f"Writer {index}", expected=expected)[0]
        except CapabilityError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(update, (0, 1)))
    winner, = [result for result in results if isinstance(result, dict)]
    loser, = [result for result in results if isinstance(result, CapabilityError)]
    assert loser.code == "revision_conflict"
    assert store.get_document("tenant", "policy")["revision"] == winner["revision"]
    assert len(store.revisions("tenant", "policy")) == before + 1
    with store.connect() as connection:
        assert connection.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()["count"] == before + 1
        assert connection.execute("SELECT COUNT(*) AS count FROM chunks_fts").fetchone()["count"] == before + 1


def test_postgres_constraint_failure_rolls_back_source_vectors_fts_graph_and_audit(database):
    import psycopg

    store = Store(database)
    original, _ = add_document(store, graph=True)
    before = snapshot(store)
    with store.connect() as connection:
        connection.execute("CREATE FUNCTION reject_graph() RETURNS trigger LANGUAGE plpgsql AS $$ "
                           "BEGIN RAISE EXCEPTION 'fixture graph write failure' USING ERRCODE='23514'; END; $$")
        connection.execute("CREATE TRIGGER fail_graph BEFORE INSERT ON graph_relations "
                           "FOR EACH ROW EXECUTE FUNCTION reject_graph()")
    with pytest.raises(psycopg.errors.CheckViolation):
        add_document(store, text="Rejected replacement", expected=original["revision"], graph=True,
                     valid_from="2024-06-01T00:00:00.000000+00:00")
    assert snapshot(store) == before


class ImageEncoder:
    id = "fixture:rgb-v1"
    dimensions = 3
    profile = {"implementation": "protocol_fixture", "dimensions": 3}

    def embed_image(self, image):
        return list(image.getpixel((0, 0)))

    def embed_text(self, query):
        return {"red": [1, 0, 0], "blue": [0, 0, 1]}[query]


def image_bytes(color):
    from PIL import Image

    image = Image.new("RGB", (2, 2), color)
    try:
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        return stream.getvalue()
    finally:
        image.close()


def test_pgvector_image_search_bytes_tenant_isolation_reindex_and_delete(database):
    encoder = ImageEncoder()
    service = MultimodalService(database, encoder=encoder)
    original = service.ingest("tenant", image_bytes("red"), asset_id="same")
    service.ingest("foreign", image_bytes("blue"), asset_id="same")
    found = service.search_text("tenant", "red")
    assert found["candidates"] == 1 and found["citations"][0]["score"] == 1
    assert service.search_image("foreign", image_bytes("blue"))["citations"][0]["score"] == 1
    assert service.get("tenant", "same", original=True)["original"] == image_bytes("red")
    with service.store.connect() as connection:
        assert {row["kind"] for row in connection.execute("SELECT pg_typeof(vector)::text AS kind FROM image_assets")} == {"vector"}
    encoder.id = "fixture:rgb-v2"
    with pytest.raises(CapabilityError) as error:
        service.search_text("tenant", "red")
    assert error.value.code == "asset_reindex_required"
    refreshed = service.reindex("tenant", "same")
    assert refreshed["revision"] != original["revision"]
    assert service.search_text("tenant", "red")["citations"][0]["score"] == 1
    assert service.delete("tenant", "same") and not service.delete("tenant", "same")
    assert service.get("foreign", "same", original=True)["original"] == image_bytes("blue")


def test_image_cas_and_invalid_vector_leave_original_bytes_unchanged(database, monkeypatch):
    first = MultimodalService(database, encoder=ImageEncoder())
    concurrent = MultimodalService(database, encoder=ImageEncoder())
    first.ingest("tenant", image_bytes("red"), asset_id="same")
    encode = first._encoder.embed_image

    def racing_encoder(image):
        concurrent.ingest("tenant", image_bytes("blue"), asset_id="same")
        return encode(image)

    monkeypatch.setattr(first._encoder, "embed_image", racing_encoder)
    with pytest.raises(CapabilityError) as error:
        first.ingest("tenant", image_bytes("green"), asset_id="same")
    assert error.value.code == "revision_conflict"
    before = first.get("tenant", "same", original=True)
    monkeypatch.setattr(first._encoder, "embed_image", lambda _: [0, 0, 0])
    with pytest.raises(CapabilityError):
        first.ingest("tenant", image_bytes("green"), asset_id="same")
    assert first.get("tenant", "same", original=True) == before


def test_existing_text_vector_columns_migrate_atomically_without_losing_records(database):
    store = Store(database)
    document, chunk = add_document(store)
    assets = MultimodalService(database, encoder=ImageEncoder())
    asset = assets.ingest("tenant", image_bytes("red"), asset_id="original")
    with store.connect() as connection:
        connection.execute("ALTER TABLE chunks ALTER COLUMN embedding TYPE text USING embedding::text")
        connection.execute("ALTER TABLE image_assets ALTER COLUMN vector TYPE text USING vector::text")
    reopened = Store(database)
    reopened_assets = MultimodalService(database, encoder=ImageEncoder())
    assert reopened.get_document("tenant", "policy")["revision"] == document["revision"]
    assert dense(reopened, {chunk["id"]}) == [chunk["id"]]
    assert reopened_assets.get("tenant", "original")["revision"] == asset["revision"]
    assert reopened_assets.search_text("tenant", "red")["citations"][0]["score"] == 1


def test_sqlite_import_preserves_history_provenance_images_and_source_read_only(database, tmp_path, monkeypatch):
    source_path = tmp_path / "source.db"
    source = Store(str(source_path))
    old, _ = add_document(source, graph=True, metadata={"標籤": "保留 🧠"})
    current, chunk = add_document(source, expected=old["revision"], graph=True,
                                  metadata={"標籤": "保留 🧠"}, text="Current source")
    add_document(source, tenant="foreign", identifier="private")
    assets = MultimodalService(str(source_path), encoder=ImageEncoder())
    asset = assets.ingest("tenant", image_bytes("red"), asset_id="original")
    before = source_path.read_bytes()
    monkeypatch.setattr(Providers, "embedding", lambda *_: pytest.fail("Import must reuse stored vectors"))
    monkeypatch.setattr(Providers, "complete", lambda *_: pytest.fail("Import must not call generation"))
    report = import_sqlite(source_path, database)
    assert report["status"] == "imported" and report["source_read_only"]
    assert source_path.read_bytes() == before
    target = Store(database)
    assert target.get_document("tenant", "policy") == source.get_document("tenant", "policy")
    assert target.scope("tenant")[0]["revision"] == current["revision"]
    assert len(target.revisions("tenant", "policy")) == 2
    assert dense(target, {chunk["id"]}) == [chunk["id"]]
    target_assets = AssetStore(database)
    imported_image = target_assets.get("tenant", "original", original=True)
    assert imported_image["revision"] == asset["revision"] and imported_image["original"] == image_bytes("red")
    with closing(sqlite3.connect(source_path.as_uri() + "?mode=ro", uri=True)) as original:
        assert report["tables"] == {table: original.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                                    for table in TABLE_FIELDS}
    before_target = snapshot(target)
    with pytest.raises(ValueError, match="target must be empty"):
        import_sqlite(source_path, database)
    assert snapshot(target) == before_target
    add_document(target, identifier="after-import")
    with target.connect() as connection:
        assert connection.execute("SELECT MAX(id) AS maximum FROM audit").fetchone()["maximum"] > report["tables"]["audit"]
        assert connection.execute("SELECT MAX(revision_order) AS maximum FROM document_revisions").fetchone()["maximum"] > report["tables"]["document_revisions"]


def test_invalid_late_import_rolls_back_all_target_rows_and_keeps_source_intact(database, tmp_path):
    source_path = tmp_path / "invalid-source.db"
    source = Store(str(source_path))
    add_document(source, identifier="valid-first")
    _, invalid = add_document(source, identifier="invalid-last")
    with source.connect() as connection:
        connection.execute("UPDATE chunks SET embedding='[NaN]' WHERE id=?", (invalid["id"],))
    before = source_path.read_bytes()
    with pytest.raises(CapabilityError) as error:
        import_sqlite(source_path, database)
    assert error.value.code == "invalid_embedding"
    assert source_path.read_bytes() == before
    target = Store(database)
    assert all(not rows for rows in snapshot(target).values())


def test_import_rejects_unsupported_read_only_source_before_target_schema_changes(database, tmp_path):
    source_path = tmp_path / "unsupported.db"
    with closing(sqlite3.connect(source_path)) as connection:
        connection.execute("CREATE TABLE documents (private TEXT)")
        connection.commit()
    before = source_path.read_bytes()
    with pytest.raises(ValueError, match="complete migrated"):
        import_sqlite(source_path, database)
    assert source_path.read_bytes() == before
    from rags.postgres import connect
    with connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) AS count FROM information_schema.tables WHERE table_schema=current_schema()").fetchone()["count"] == 0


def test_scope_pushes_revision_filters_to_postgres_without_loading_unauthorized_chunks(database, monkeypatch):
    """Only allowed document revisions are selected in SQL within one read snapshot."""
    from rags import postgres

    store = Store(database)
    _, visible = add_document(store, identifier="allowed", source="manual",
                              metadata={"active": True, "category": "ok"})
    add_document(store, identifier="hidden", source="manual",
                 metadata={"active": False}, vector=(0.1, 0.8, 0.1))
    add_document(store, tenant="other", identifier="foreign", source="manual")
    calls = []
    execute = postgres.Connection.execute

    def inspect(self, statement, parameters=()):
        calls.append((statement, parameters))
        return execute(self, statement, parameters)

    monkeypatch.setattr(postgres.Connection, "execute", inspect)
    subset = store.scope("tenant", filters={"metadata": {"active": True}, "source": "manual"},
                         include_embeddings=False)
    assert [chunk["id"] for chunk in subset] == [visible["id"]]
    assert "embedding" not in subset[0]
    assert len([sql for sql, _ in calls if "FROM document_revisions WHERE" in sql]) == 1
    scoped = [(sql, params) for sql, params in calls if "FROM chunks c" in sql]
    assert len(scoped) == 1
    query, args = scoped[0]
    assert "jsonb_to_recordset(?::jsonb)" in query
    assert "c.embedding" not in query and "c.*" not in query
    assert args[-1] == "tenant"
    keys = json.loads(args[0])
    assert keys == [{"document_id": "allowed", "revision": subset[0]["revision"]}]
    assert any("REPEATABLE READ, READ ONLY" in sql for sql, _ in calls)
    before = len(scoped)
    assert store.scope("tenant", filters={"metadata": {"active": "True"}}) == []
    after = len([sql for sql, _ in calls if "FROM chunks c" in sql])
    assert before == after


def test_benchmark_uses_only_temp_tables_and_reproduces_expected_counts(database):
    from rags.benchmark import benchmark

    result = benchmark(database, documents=20, chunks_per_document=3, iterations=3)
    assert result["workload"] == "synthetic_temp_tables"
    assert result["database"] == "postgresql+pgvector"
    assert result["vector_dimensions"] == 32
    assert result["candidate_fraction"] == 0.1
    assert result["measurements"]["scoped_revision_join"]["rows"] == 6
    assert result["measurements"]["unfiltered_tenant_scan"]["rows"] == 6
    assert result["measurements"]["exact_scoped_cosine"]["rows"] == 6
    assert all(sample["samples"] == 3 for sample in result["measurements"].values())
    with Store(database).connect() as connection:
        assert connection.execute(
            "SELECT count(*) AS n FROM information_schema.tables "
            "WHERE table_schema=current_schema() AND table_name='rags_bench_chunks'"
        ).fetchone()["n"] == 0
