"""Offline public persistent client exercises the authenticated application path."""
from dataclasses import replace
import os
import sqlite3

import pytest

from rags.client import ClientError, PersistentClient
from rags.config import Settings
from rags.features import FeatureFlags


pytestmark = pytest.mark.integration
KEY = "local-collection-credential"


def settings(path, **changes):
    value = Settings(
        db_path=str(path), keys={"tenant-a-key": "tenant-a", "tenant-b-key": "tenant-b",
                                 "reader-key": "tenant-a"},
        key_roles={"tenant-a-key": ("admin",), "tenant-b-key": ("admin",), "reader-key": ("reader",)},
        embedding_provider="local", model_provider="extractive", openai_key="",
        openai_base_url="https://api.openai.com/v1", embedding_model="offline-hash-v1",
        chat_model="extractive", anthropic_key="", anthropic_model="",
    )
    return replace(value, **changes)


def test_default_local_client_ignores_ambient_credentials_and_model_configuration(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGS_KEYS_JSON", "invalid authentication environment")
    monkeypatch.setenv("RAGS_AUTH_MODE", "oauth2_introspection")
    monkeypatch.setenv("RAGS_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("RAGS_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-secret-must-not-be-used")
    monkeypatch.setenv("RAGS_FEATURES_JSON", '{"collections":false}')
    monkeypatch.setattr(Settings, "from_env", lambda: pytest.fail("Default local client must not read env"))
    with PersistentClient(tmp_path / "offline.db", KEY) as client:
        collection = client.create_collection("notes")
        collection.add(ids=["retention"], documents=["Alpha retention policy requires seven days."])
        assert collection.search(query_texts=["retention policy"], mode="dense").rows()[0]["id"] == "retention"
        status = collection.indexing_status()
        assert status["embedding"]["endpoint"] == "local"
        assert status["embedding"]["mode"] == "configured"
        assert status["state"] == "ready" and status["pending_operations"] == 0
    with pytest.raises(ClientError, match="client_closed"):
        collection.count()


def test_persistent_record_lifecycle_batch_search_and_restart(tmp_path):
    path = tmp_path / "lifecycle.db"
    with PersistentClient(path, KEY) as client:
        collection = client.create_collection("notes", metadata={"owner": "platform"})
        identifier = collection.id
        created = collection.add(
            ids=["retention", "recovery", "billing"],
            documents=["Alpha retention lasts seven days.", "Recovery uses daily snapshots.", "Billing uses monthly invoices."],
            metadatas=[{"team": "platform", "tags": ["policy", "backup"]},
                       {"team": "platform", "tags": ["backup"]}, {"team": "finance"}],
        )
        assert (created["changed"], created["added"], created["generation"]) == (3, 3, 1)
        assert collection.count() == 3
        result = collection.search(searches=[
            {"mode": "lexical", "query_text": "retention", "where": {"tags": {"$contains": "policy"}}},
            {"mode": "hybrid", "query_text": "daily recovery", "where": {"team": "platform"}},
        ])
        assert result["generation"] == 1
        assert result["evidence_status"] == "ranked_unassessed"
        assert result.rows(0)[0]["id"] == "retention"
        assert result.rows(1)[0]["id"] == "recovery"
        before = collection.get(ids=["retention"], include=["document", "metadata", "embedding"])["items"][0]
        updated = collection.update(ids=["retention"], metadatas=[{"team": "security"}], expected_generation=1)
        assert (updated["updated"], updated["generation"]) == (1, 2)
        after = collection.get(ids=["retention"], include=["document", "metadata", "embedding"])["items"][0]
        assert after["document"] == before["document"] and after["embedding"] == before["embedding"]
        assert after["version"] != before["version"]
        unchanged = collection.update(ids=["retention"], metadatas=[{"team": "security"}], expected_generation=2)
        assert unchanged["changed"] == 0 and unchanged["generation"] == 2
        collection.modify(name="renamed", expected_generation=2)
        assert collection.id == identifier and collection.name == "renamed"
    with PersistentClient(path, KEY) as reopened:
        collection = reopened.get_collection("renamed")
        assert collection.id == identifier and collection.count() == 3
        assert reopened.list_collections_page()["total"] == 1
        assert reopened.list_collections()[0].name == "renamed"
        assert collection.search(query_text="invoices", mode="lexical").rows()[0]["id"] == "billing"
        cleared = collection.update(ids=["retention"], documents=[None])
        row = collection.get(ids=["retention"], include=["document", "metadata", "embedding"])["items"][0]
        assert row["document"] is None and row["embedding"] == before["embedding"]
        deleted = collection.delete(where={"team": "finance"}, expected_generation=cleared["generation"])
        assert deleted["deleted"] == 1 and collection.count() == 2
        reopened.delete_collection(collection_id=identifier, expected_generation=deleted["generation"])
        assert reopened.list_collections() == []


def test_fork_shares_initial_versions_and_isolates_all_later_mutations(tmp_path):
    with PersistentClient(tmp_path / "fork.db", KEY) as client:
        source = client.create_collection("source")
        source.add(ids=["same-id", "retained"], documents=["Original source policy.", "Preserved evidence."])
        branch = source.fork("experiment", expected_generation=1)
        info = branch.to_dict()
        assert info["fork_source_id"] == source.id and info["fork_source_generation"] == 1
        assert info["generation"] == 0 and branch.count() == 2
        source_rows = {row["id"]: row for row in source.get()["items"]}
        branch_rows = {row["id"]: row for row in branch.get()["items"]}
        assert source_rows == branch_rows
        branch.update(ids=["same-id"], documents=["Changed branch policy."])
        source.delete(ids=["retained"])
        assert source.get(ids=["same-id"])["items"][0]["document"] == "Original source policy."
        assert branch.get(ids=["retained"])["items"][0]["document"] == "Preserved evidence."
        client.delete_collection(collection_id=source.id)
        assert branch.count() == 2
        assert branch.search(query_text="changed policy", mode="lexical").rows()[0]["id"] == "same-id"


def test_tenant_credentials_scope_names_records_and_forks(tmp_path):
    path = tmp_path / "tenants.db"
    config = settings(path)
    with PersistentClient(path, "tenant-a-key", settings=config) as a, PersistentClient(
        path, "tenant-b-key", settings=config,
    ) as b:
        ca, cb = a.create_collection("same-name"), b.create_collection("same-name")
        assert ca.id != cb.id
        ca.add(ids=["same-id"], documents=["Tenant alpha private retention."])
        cb.add(ids=["same-id"], documents=["Tenant beta private billing."])
        assert [item.id for item in a.list_collections()] == [ca.id]
        assert [item.id for item in b.list_collections()] == [cb.id]
        assert cb.search(query_text="alpha retention", mode="lexical").rows() == []
        for client, identifier in ((a, cb.id), (b, ca.id)):
            with pytest.raises(ClientError) as denied:
                client.get_collection(collection_id=identifier)
            assert (denied.value.code, denied.value.status_code) == ("collection_not_found", 404)
        fork = ca.fork("a-branch")
        assert b.list_collections_page()["total"] == 1 and fork.count() == 1


def test_authentication_precedes_database_initialization_and_each_dispatch(tmp_path):
    path = tmp_path / "unopened.db"
    config = settings(path)
    with pytest.raises(ClientError, match="invalid_token"):
        PersistentClient(path, "wrong-credential", settings=config)
    assert not path.exists()
    with PersistentClient(path, "tenant-a-key", settings=config) as client:
        collection = client.create_collection("notes")
        del config.keys["tenant-a-key"]
        with pytest.raises(ClientError) as denied:
            collection.add(ids=["a"], documents=["Never stored"])
        assert (denied.value.code, denied.value.status_code) == ("invalid_token", 401)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM rag_collection_records").fetchone()[0] == 0


def test_reader_can_retrieve_but_cannot_mutate_or_fork(tmp_path):
    path = tmp_path / "scopes.db"
    config = settings(path)
    with PersistentClient(path, "tenant-a-key", settings=config) as admin:
        collection = admin.create_collection("notes")
        collection.add(ids=["a"], documents=["Retention is seven days."])
        with PersistentClient(path, "reader-key", settings=config) as reader:
            read = reader.get_collection("notes")
            assert read.count() == 1 and read.search(query_text="retention", mode="lexical").rows()
            calls = [lambda: read.add(ids=["b"], documents=["unauthorized"]),
                     lambda: read.update(ids=["a"], documents=["unauthorized"]),
                     lambda: read.delete(ids=["a"]), lambda: read.modify(name="new"),
                     lambda: read.fork("branch"), lambda: reader.create_collection("forbidden"),
                     lambda: reader.delete_collection(collection_id=collection.id)]
            for call in calls:
                with pytest.raises(ClientError) as denied:
                    call()
                assert (denied.value.code, denied.value.status_code) == ("insufficient_scope", 403)
            assert read.get()["items"][0]["document"] == "Retention is seven days."


def test_local_feature_flags_and_generation_conflicts_are_typed(tmp_path):
    path = tmp_path / "features.db"
    config = settings(path)
    with PersistentClient(path, "tenant-a-key", settings=config) as client:
        collection = client.create_collection("notes")
        collection.add(ids=["a"], documents=["Original retention policy."])
        with pytest.raises(ClientError) as conflict:
            collection.update(ids=["a"], documents=["Stale replacement"], expected_generation=0)
        assert (conflict.value.code, conflict.value.status_code) == ("collection_generation_conflict", 409)
        assert collection.get()["items"][0]["document"] == "Original retention policy."
    disabled = replace(config, features=FeatureFlags((("collections", False),)))
    with PersistentClient(path, "tenant-a-key", settings=disabled) as client:
        with pytest.raises(ClientError, match="feature_disabled"):
            client.list_collections()
    disabled = replace(config, features=FeatureFlags((("collections.regex", False), ("collections.fork", False))))
    with PersistentClient(path, "tenant-a-key", settings=disabled) as client:
        collection = client.get_collection("notes")
        for call in (lambda: collection.fork("branch"),
                     lambda: collection.get(where_document={"$regex": "retention.*"})):
            with pytest.raises(ClientError, match="feature_disabled"):
                call()
        assert collection.get()["total"] == 1


def test_external_vectors_sparse_filters_projection_and_grouping(tmp_path):
    with PersistentClient(tmp_path / "vectors.db", KEY) as client:
        collection = client.create_collection(
            "precomputed", embedding={"mode": "external", "model": "example-encoder-v1",
                                      "space_id": "example-dense-v1", "dimensions": 2},
            sparse_space_id="example-sparse-v1",
        )
        collection.add(
            ids=["a", "b", "c"], documents=["Alpha policy", "Alpha backup", "Beta billing"],
            embeddings=[[1, 0], [0.9, 0.1], [0, 1]], embedding_space_id="example-dense-v1",
            metadatas=[{"team": "a", "year": 2026}, {"team": "a", "year": 2025}, {"team": "b", "year": 2026}],
            sparse_embeddings=[{"indices": [1], "values": [1], "space_id": "example-sparse-v1"},
                               {"indices": [1], "values": [0.5], "space_id": "example-sparse-v1"}, None],
        )
        result = collection.search(query_embeddings=[[1, 0]], mode="dense", embedding_space_id="example-dense-v1",
                                   group_by={"field": "team", "max_per_group": 1}, select=["metadata", "score"])
        assert [row["id"] for row in result.rows()] == ["a", "c"]
        assert all(set(row) == {"id", "version", "metadata", "score"} for row in result.rows())
        sparse = collection.search(mode="sparse", query_sparse={
            "indices": [1], "values": [1], "space_id": "example-sparse-v1"},
            where={"year": {"$gte": 2026}}, where_document={"$regex": "^Alpha"})
        assert [row["id"] for row in sparse.rows()] == ["a"]
        with pytest.raises(ClientError, match="embedding_space_mismatch"):
            collection.search(query_text="alpha", mode="dense")
        with pytest.raises(ClientError, match="embedding_space_mismatch"):
            collection.search(query_embeddings=[[1, 0]], mode="dense", embedding_space_id="wrong-space")


def test_collection_ask_preserves_record_and_immutable_version_citations(tmp_path):
    with PersistentClient(tmp_path / "answer.db", KEY) as client:
        collection = client.create_collection("policy")
        collection.add(ids=["retention"], documents=["The retention period is exactly seven days."])
        version = collection.get(ids=["retention"])["items"][0]["version"]
        answer = collection.ask("What is the retention period?", mode="lexical", generate=False)
        assert answer["collection_id"] == collection.id and answer["generation"] == 1
        assert answer["citations"]
        citation = answer["citations"][0]
        assert citation["collection_id"] == collection.id
        assert citation["record_id"] == "retention" and citation["revision"] == version
        before_id = citation["chunk_id"]
        collection.update(ids=["retention"], documents=["The retention period is exactly thirty days."])
        after = collection.ask("What is the retention period?", mode="lexical", generate=False)
        assert after["citations"][0]["chunk_id"] != before_id


def test_explicit_from_env_uses_operator_credentials_and_database(tmp_path, monkeypatch):
    for name in tuple(os.environ):
        if name.startswith(("RAGS_", "OPENAI_", "ANTHROPIC_")):
            monkeypatch.delenv(name)
    path = tmp_path / "configured.db"
    monkeypatch.setenv("RAGS_ENV", "test")
    monkeypatch.setenv("RAGS_DB_PATH", str(path))
    monkeypatch.setenv("RAGS_KEYS_JSON", '{"configured-key":"operator-tenant"}')
    with PersistentClient.from_env("configured-key") as client:
        client.create_collection("configured")
    assert path.exists()
    with PersistentClient.from_env("configured-key") as client:
        assert client.get_collection("configured").name == "configured"
    with pytest.raises(ClientError, match="invalid_token"):
        PersistentClient.from_env(KEY)

