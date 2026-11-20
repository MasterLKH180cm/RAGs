from fastapi.testclient import TestClient

from rags.api import create_app
from rags.config import Settings


def test_source_metadata_and_embedding_changes_trigger_reindex(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGS_ENV", "test")
    monkeypatch.setenv("RAGS_DB_PATH", str(tmp_path / "reindex.db"))
    monkeypatch.setenv("RAGS_KEYS_JSON", '{"testing":"tenant"}')
    settings = Settings.from_env()
    with TestClient(create_app(settings)) as client:
        headers = {"Authorization": "Bearer testing"}
        request = {
            "document_id": "policy-v1", "title": "Policy",
            "text": "# Session\nSession timeout is 30 minutes.",
            "source": "spec:v1", "metadata": {"version": 1},
        }
        first = client.post("/documents", json=request, headers=headers)
        assert first.status_code == 201
        assert client.post("/documents", json=request, headers=headers).json()["unchanged"]
        changed = {**request, "source": "spec:v2", "metadata": {"version": 2}}
        assert not client.post("/documents", json=changed, headers=headers).json()["unchanged"]
        row = client.get("/documents/policy-v1", headers=headers).json()
        assert row["metadata"]["version"] == 2
        with client.app.state.service.store.connect() as conn:
            conn.execute("UPDATE chunks SET embedding_model='previous-model'")
        rebuilt = client.post("/documents", json=changed, headers=headers)
        assert rebuilt.status_code == 201
        assert not rebuilt.json()["unchanged"]
