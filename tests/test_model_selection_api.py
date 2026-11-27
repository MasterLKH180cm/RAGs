"""Selection metadata and authorized local-artifact indexing over the real API."""
from dataclasses import replace
import json
import os

from fastapi.testclient import TestClient
import pytest

from rags.api import create_app
from rags.config import Settings
from rags.features import FeatureFlags

pytestmark = pytest.mark.integration
AUTH = {"Authorization": "Bearer catalog-test"}


@pytest.fixture
def settings(tmp_path, monkeypatch):
    for name in tuple(os.environ):
        if name.startswith(("RAGS_", "OPENAI_", "ANTHROPIC_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("RAGS_ENV", "test")
    return replace(Settings.from_env(), db_path=str(tmp_path / "models.db"), keys={"catalog-test": "tenant"})


def artifact(path, extra=False):
    from tokenizers import Tokenizer, models, pre_tokenizers
    words = ["[UNK]", "Policy", "sessions", "expire", "after", "minutes", "30"]
    if extra:
        words.append("changed")
    tokenizer = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.save(str(path))
    return {"backend": "huggingface", "path": str(path)}


def test_authenticated_catalog_is_static_and_respects_feature_policy(settings, monkeypatch):
    import rags.providers
    monkeypatch.setattr(rags.providers, "_semantic_encoder", lambda *_: pytest.fail("Catalog must not load a model"))
    flags = FeatureFlags((("embedding.fastembed", False), ("tokenizer.sentencepiece", False)))
    with TestClient(create_app(replace(settings, features=flags))) as client:
        assert client.get("/capabilities").status_code == 401
        response = client.get("/capabilities", headers=AUTH)
        assert response.status_code == 200
        catalog = response.json()["model_selection"]
        assert len(catalog["embeddings"]) == 22
        assert len(catalog["tokenizers"]) == 8
        assert all(not item["enabled"] for item in catalog["embeddings"] if item["provider"] == "fastembed")
        assert not next(item for item in catalog["tokenizers"] if item["id"] == "sentencepiece")["enabled"]
        assert "catalog-test" not in response.text
        assert str(settings.db_path) not in response.text


@pytest.mark.parametrize("backend", ["huggingface", "sentencepiece"])
def test_request_cannot_read_unapproved_local_artifact(settings, monkeypatch, backend):
    import rags.tokenizer_backends
    monkeypatch.setattr(rags.tokenizer_backends, "read_regular_bytes",
                        lambda *_args, **_kwargs: pytest.fail("Unapproved file was opened"))
    app = create_app(settings)
    with TestClient(app) as client:
        response = client.post("/documents", headers=AUTH, json={
            "document_id": "denied", "title": "Policy", "text": "sessions expire after 30 minutes",
            "tokenizer": {"backend": backend, "path": "/private/unauthorized-model"},
        })
        assert response.status_code == 403, response.text
        assert response.json()["detail"]["code"] == "tokenizer_not_allowed"
        assert "unauthorized-model" not in response.text
        assert app.state.service.store.get_document("tenant", "denied") is None


def test_artifact_index_profile_reindex_and_explicit_change(settings, tmp_path):
    path = tmp_path / "tokenizer.json"
    config = artifact(path)
    app = create_app(replace(settings, tokenizer_config=config))
    with TestClient(app) as client:
        first = client.post("/documents", headers=AUTH, json={
            "document_id": "policy", "title": "Policy", "text": "sessions expire after 30 minutes",
        })
        assert first.status_code == 201, first.text
        revision = first.json()["revision"]
        stored = app.state.service.store.get_document("tenant", "policy")
        profile = stored["index_profile"]["tokenizer"]
        assert profile["artifact_sha256"]
        assert "path" not in profile and str(path) not in json.dumps(profile)
        original_hash = profile["artifact_sha256"]
        same = client.post("/documents/policy/reindex", headers=AUTH, json={"expected_revision": revision})
        assert same.status_code == 200, same.text
        revision = same.json()["revision"]
        artifact(path, extra=True)
        mismatch = client.post("/documents/policy/reindex", headers=AUTH, json={"expected_revision": revision})
        assert mismatch.status_code == 409, mismatch.text
        assert mismatch.json()["detail"]["code"] == "tokenizer_mismatch"
        assert app.state.service.store.get_document("tenant", "policy")["revision"] == revision
        changed = client.post("/documents/policy/reindex", headers=AUTH,
                              json={"expected_revision": revision, "tokenizer": config})
        assert changed.status_code == 200, changed.text
        assert app.state.service.store.get_document("tenant", "policy")["index_profile"]["tokenizer"]["artifact_sha256"] != original_hash
        answer = client.post("/query", headers=AUTH, json={"question": "sessions expire", "mode": "sparse", "generate": False})
        assert answer.status_code == 200, answer.text
        assert answer.json()["citations"][0]["document_id"] == "policy"


def test_operator_artifact_cannot_be_substituted_by_same_bytes_at_other_path(settings, tmp_path):
    original = tmp_path / "configured.json"
    config = artifact(original)
    other = tmp_path / "not-allowlisted.json"
    other.write_bytes(original.read_bytes())
    with TestClient(create_app(replace(settings, tokenizer_config=config))) as client:
        response = client.post("/documents", headers=AUTH, json={"title": "Policy", "text": "sessions expire",
            "tokenizer": {"backend": "huggingface", "path": str(other)}})
        assert response.status_code == 403


@pytest.mark.parametrize("feature", ["tokenizer.huggingface", "embedding.local", "chunking.structure", "reindex", "ingestion"])
def test_reindex_checks_feature_policy_before_opening_artifact(settings, tmp_path, monkeypatch, feature):
    import rags.tokenizer_backends
    config = artifact(tmp_path / "operator.json")
    configured = replace(settings, tokenizer_config=config)
    with TestClient(create_app(configured)) as client:
        response = client.post("/documents", headers=AUTH, json={
            "document_id": "policy", "title": "Policy", "text": "sessions expire after 30 minutes"})
        assert response.status_code == 201
        revision = response.json()["revision"]
    disabled = replace(configured, features=FeatureFlags(((feature, False),)))
    monkeypatch.setattr(rags.tokenizer_backends, "read_regular_bytes",
                        lambda *_args, **_kwargs: pytest.fail("Disabled feature read a model artifact"))
    with TestClient(create_app(disabled)) as client:
        response = client.post("/documents/policy/reindex", headers=AUTH, json={"expected_revision": revision})
        assert response.status_code == 403, response.text
        assert response.json()["detail"]["code"] == "feature_disabled"
