"""Real SQLite/API/SDK selection boundaries; offline vectors are demo fixtures."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from fastapi.testclient import TestClient
import pytest

from rags.api import create_app
from rags.auth import PERMISSIONS, Principal
from rags.client import ClientError, PersistentClient, ReembeddingError
from rags.collection_service import CollectionService
from rags.config import Settings
from rags.features import FeatureFlags
from rags.providers import Providers, local_embedding
from rags.registry import CapabilityError
from rags.storage import Store

pytestmark = pytest.mark.integration
ADMIN = Principal("one", "fixture", PERMISSIONS, "api_key")


@pytest.fixture
def settings(tmp_path):
    return Settings(db_path=str(tmp_path / "profiles.db"), keys={"profile-key": "one", "other-key": "two"},
                    embedding_provider="local", model_provider="extractive", openai_key="never-public-secret",
                    openai_base_url="https://unused.invalid/v1", embedding_model="offline-hash-v1",
                    chat_model="", anthropic_key="", anthropic_model="", embedding_profiles={
                        "english": {"provider": "local", "model": "offline-hash-v1", "document_prefix": "english: "},
                        "chinese": {"provider": "local", "model": "offline-hash-v1", "document_prefix": "中文："},
                    })


def local_client(settings):
    return PersistentClient(settings.db_path, api_key="profile-key", settings=settings)


def test_independent_approved_collection_profiles_and_concurrent_providers(settings):
    service = CollectionService(Store(settings.db_path), Providers(settings))
    original = service.provider
    def exercise(alias):
        created = service.dispatch(ADMIN, "create", data={"name": alias, "embedding": {
            "mode": "profile", "profile_id": alias}})
        cid = created["id"]
        written = service.dispatch(ADMIN, "add", cid, {"ids": ["one"], "documents": ["same source"]})
        assert written["generation"] == 1
        result = service.dispatch(ADMIN, "get_records", cid, {"include": ["document", "embedding"]})
        return created, result["items"][0]
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(exercise, ("english", "chinese")))
    for (created, record), alias in zip(outcomes, ("english", "chinese")):
        assert created["configuration"]["embedding"]["profile_id"] == alias
        assert record["document"] == "same source"
        expected = local_embedding(settings.embedding_profiles[alias].document_prefix + "same source")
        assert record["embedding"] == pytest.approx(expected, abs=1e-7)
    assert outcomes[0][0]["configuration"]["embedding"]["space_id"] != outcomes[1][0]["configuration"]["embedding"]["space_id"]
    assert service.provider is original and service.settings.embedding_document_prefix == ""


def test_profile_resolution_follows_authorization_and_tenant_scope(settings, monkeypatch):
    service = CollectionService(Store(settings.db_path), Providers(settings))
    called = []
    monkeypatch.setattr(service.provider, "for_embedding_profile", lambda alias: called.append(alias))
    denied = replace(ADMIN, permissions=frozenset())
    with pytest.raises(Exception) as error:
        service.dispatch(denied, "create", data={"name": "private", "embedding": {"mode": "profile", "profile_id": "english"}})
    assert getattr(error.value, "status_code", None) == 403 and not called
    with pytest.raises(CapabilityError) as error:
        service.dispatch(replace(ADMIN, tenant="two"), "search", "missing", {"searches": [{"query_text": "secret"}]})
    assert error.value.status_code == 404 and not called


def test_unknown_disabled_profiles_fail_before_collection_schema_or_model_load(settings):
    disabled = replace(settings, features=FeatureFlags((("embedding.local", False),)))
    for current, alias, code in [(settings, "missing", "unknown_embedding_profile"), (disabled, "english", "feature_disabled")]:
        service = CollectionService(Store(current.db_path), Providers(current))
        with pytest.raises(CapabilityError) as error:
            service.dispatch(ADMIN, "create", data={"name": "notes", "embedding": {"mode": "profile", "profile_id": alias}})
        assert error.value.code == code and "database" not in service.__dict__


def test_alias_rebinding_does_not_reinterpret_existing_vectors_and_fork_retains_binding(settings):
    client = local_client(settings)
    source = client.create_collection("original", embedding_profile="english")
    source.add(ids=["one"], documents=["source evidence"])
    fork = source.fork("copy")
    assert fork.to_dict()["configuration"] == source.refresh().to_dict()["configuration"]
    changed = replace(settings, embedding_profiles={"english": {
        "provider": "local", "model": "offline-hash-v1", "document_prefix": "different: "}})
    with local_client(changed) as other:
        bound = other.get_collection("original")
        assert bound.indexing_status()["text_query_profile_matches"] is False
        with pytest.raises(ClientError, match="embedding_space_mismatch"):
            bound.search(query_text="source", mode="dense")
        assert bound.count() == 1


def test_reembedding_copies_original_text_metadata_and_preserves_source(settings):
    with local_client(settings) as client:
        source = client.create_collection("source", embedding_profile="english", metadata={"purpose": "fixture"})
        source.add(ids=["one", "two", "three"], documents=["Exact\r\noriginal é", "繁體中文證據", "third"],
                   metadatas=[{"rank": 1}, {"rank": 2}, {}])
        initial = source.refresh().to_dict()
        original = source.get(include=["document", "metadata", "embedding"])
        result = source.reembed("new-space", embedding_profile="chinese", batch_size=2, expected_generation=1)
        assert result["completed"] is True and result["copied_records"] == 3 and result["target_generation"] == 2
        target = client.get_collection(collection_id=result["target_collection_id"])
        copied = target.get(include=["document", "metadata", "embedding"])
        assert [(v["id"], v["document"], v["metadata"]) for v in copied["items"]] == [
            (v["id"], v["document"], v["metadata"]) for v in original["items"]]
        assert copied["items"][0]["embedding"] != original["items"][0]["embedding"]
        assert source.refresh().to_dict() == initial
        assert source.get(include=["document", "metadata", "embedding"]) == original
        assert target.metadata == {"purpose": "fixture"}


def test_source_change_during_reembedding_retains_inspectable_partial_target(settings, monkeypatch):
    with local_client(settings) as client:
        source = client.create_collection("source", embedding_profile="english")
        source.add(ids=["one", "two"], documents=["first", "second"])
        original = Providers.embedding
        changed = []
        def encode(provider, text):
            if provider.settings.embedding_document_prefix == "中文：" and not changed:
                changed.append(True)
                source.modify(metadata={"changed": True}, expected_generation=1)
            return original(provider, text)
        monkeypatch.setattr(Providers, "embedding", encode)
        with pytest.raises(ReembeddingError) as error:
            source.reembed("partial", embedding_profile="chinese", batch_size=1)
        progress = error.value.progress
        assert progress["cause"] == "reembedding_source_changed" and progress["copied_records"] == 1
        assert progress["completed"] is False
        assert client.get_collection(collection_id=progress["target_collection_id"]).count() == 1
        assert source.count() == 2 and source.refresh().metadata == {"changed": True}


def test_target_concurrent_writer_is_not_deleted_or_overwritten(settings, monkeypatch):
    with local_client(settings) as client:
        source = client.create_collection("source", embedding_profile="english")
        source.add(ids=["one"], documents=["first"])
        original = Providers.embedding
        def encode(provider, text):
            if provider.settings.embedding_document_prefix == "中文：":
                client.get_collection("partial").modify(metadata={"other_writer": True}, expected_generation=0)
            return original(provider, text)
        monkeypatch.setattr(Providers, "embedding", encode)
        with pytest.raises(ReembeddingError) as error:
            source.reembed("partial", embedding_profile="chinese")
        assert error.value.progress["cause"] == "collection_generation_conflict"
        target = client.get_collection("partial")
        assert target.metadata == {"other_writer": True} and target.count() == 0


def test_reembedding_missing_text_fails_with_progress_and_stale_source_creates_no_target(settings):
    with local_client(settings) as client:
        source = client.create_collection("vectors", embedding={"mode": "external", "model": "fixture",
                                                                 "space_id": "fixture", "dimensions": 2})
        source.add(ids=["one"], embeddings=[[1, 0]], embedding_space_id="fixture")
        with pytest.raises(ClientError, match="collection_generation_conflict"):
            source.reembed("not-created", embedding_profile="english", expected_generation=0)
        assert len(client.list_collections()) == 1
        with pytest.raises(ReembeddingError) as error:
            source.reembed("no-text", embedding_profile="english")
        assert error.value.progress["copied_records"] == 0
        assert client.get_collection("no-text").count() == 0


def test_discovery_is_authenticated_and_does_not_load_files_models_or_expose_secrets(settings, tmp_path, monkeypatch):
    configured = replace(settings, tokenizer_artifacts={"native": {
        "backend": "huggingface", "path": str(tmp_path / "not-downloaded.json"), "sha256": "a" * 64}},
        embedding_profiles={"remote": {"provider": "openai", "model": "text-embedding-3-small",
                                       "base_url": "https://private-server.invalid/v1", "api_key_env": "PRIVATE_EMBED_KEY"}})
    monkeypatch.setattr(Providers, "embedding", lambda *_: pytest.fail("discovery ran inference"))
    app = create_app(configured)
    with TestClient(app) as http:
        for path, key in [("/v1/models", "embeddings"), ("/v1/tokenizers", "tokenizers")]:
            assert http.get(path).status_code == 401
            response = http.get(path, headers={"Authorization": "Bearer profile-key"})
            assert response.status_code == 200 and response.json()[key]
            assert all(value not in response.text for value in ("never-public-secret", "PRIVATE_EMBED_KEY", "private-server.invalid", str(tmp_path)))
    with local_client(configured) as client:
        assert client.list_models()["approved_profiles"][0]["id"] == "remote"
        assert client.list_tokenizers()["registered_artifacts"][0]["id"] == "native"
        assert all("path" not in entry for entry in client.list_tokenizers()["registered_artifacts"])


def test_http_collection_profile_alias_contract_and_tenant_isolation(settings):
    with TestClient(create_app(settings)) as http:
        auth = {"Authorization": "Bearer profile-key"}
        created = http.post("/collections", headers=auth, json={"name": "http-profile", "embedding": {
            "mode": "profile", "profile_id": "english"}})
        assert created.status_code == 201
        cid = created.json()["id"]
        assert http.post(f"/collections/{cid}/records/add", headers=auth,
                         json={"ids": ["one"], "documents": ["exact source"]}).status_code == 200
        denied = http.post(f"/collections/{cid}/search", headers={"Authorization": "Bearer other-key"},
                           json={"searches": [{"query_text": "exact source"}]})
        assert denied.status_code == 404
        assert http.post("/collections", headers=auth, json={"name": "bad", "embedding": {
            "mode": "profile", "profile_id": "english", "base_url": "https://unapproved.invalid"}}).status_code == 422
