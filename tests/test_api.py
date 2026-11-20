import pytest
from fastapi.testclient import TestClient

from rags.api import create_app
from rags.config import Settings


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGS_ENV", "test")
    monkeypatch.setenv("RAGS_DB_PATH", str(tmp_path / "rag.db"))
    monkeypatch.setenv("RAGS_KEYS_JSON", '{"a-key":"tenant-a","b-key":"tenant-b"}')
    monkeypatch.setenv("RAGS_MODEL_PROVIDER", "extractive")
    monkeypatch.setenv("RAGS_EMBEDDING_PROVIDER", "local")
    with TestClient(create_app(Settings.from_env())) as c:
        yield c


A = {"Authorization": "Bearer a-key"}
B = {"Authorization": "Bearer b-key"}


def add(client, **kw):
    return client.post("/documents", headers=A, json=kw)


def test_ingest_all_strategies_and_evidence(client):
    response = add(client, title="Authentication design", document_id="auth-v1",
                   source="spec://auth", text=(
                       "# Authentication\n"
                       "Login service -> Token service\n"
                       "Refresh tokens rotate after login. JWT expiration is 30 minutes.\n"
                       "# Session Management\n"
                       "User session expires after 30 minutes of inactivity."
                   ))
    assert response.status_code == 201, response.text
    assert response.json()["chunks"] >= 2
    assert add(client, title="Authentication design", document_id="auth-v1",
               source="spec://auth", text=(
                   "# Authentication\nLogin service -> Token service\n"
                   "Refresh tokens rotate after login. JWT expiration is 30 minutes.\n"
                   "# Session Management\nUser session expires after 30 minutes of inactivity."
               )).json()["unchanged"]
    for mode in (
        "sparse", "dense", "hybrid", "fusion", "hyde",
        "contextual", "hierarchical", "graph", "corrective", "adaptive"
    ):
        r = client.post("/query", headers=A, json={
            "question": "When does session expire?", "mode": mode
        })
        assert r.status_code == 200, (mode, r.text)
        data = r.json()
        assert data["answer"]
        assert data["citations"]
        assert all(c["source"] == "spec://auth" for c in data["citations"])
    assert client.post("/query", headers=A, json={
        "question": "session expires", "generate": False
    }).json()["answer"] is None


def test_isolation_replace_and_delete(client):
    r = add(client, title="Private", document_id="private", text="SECRET ABC password policy")
    assert r.status_code == 201
    assert client.get("/documents", headers=B).json() == []
    assert client.get("/documents/private", headers=B).status_code == 404
    assert client.post("/query", headers=B, json={"question": "SECRET ABC"}).json()["citations"] == []
    assert add(client, title="Updated", document_id="private",
               text="NEW version of the authentication specification").status_code == 201
    old = client.post("/query", headers=A, json={
        "question": "SECRET ABC", "mode": "sparse"
    }).json()
    assert old["citations"] == []
    assert client.delete("/documents/private", headers=A).status_code == 200
    assert client.get("/documents/private", headers=A).status_code == 404


def test_fail_closed_and_validation(client):
    assert client.get("/documents").status_code == 401
    assert client.get("/documents", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert add(client, title="Empty", text="").status_code == 422
    assert client.post("/query", headers=A, json={
        "question": "ok", "mode": "not-real"
    }).status_code == 422
    assert client.post("/query", headers=A, json={
        "question": "hello", "top_k": 2000
    }).status_code == 422


def test_config_production_requires_keys(monkeypatch):
    monkeypatch.setenv("RAGS_ENV", "production")
    monkeypatch.delenv("RAGS_KEYS_JSON", raising=False)
    with pytest.raises(ValueError, match="RAGS_KEYS_JSON"):
        Settings.from_env()
