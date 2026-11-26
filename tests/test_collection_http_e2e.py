"""Public SDK and installed CLI cross a real loopback HTTP process boundary."""
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx
import pytest

from rags.client import ClientError, HttpClient

pytestmark = pytest.mark.e2e
ROOT = Path(__file__).parents[1]
ADMIN_A = "collection-e2e-admin-tenant-a"
ADMIN_B = "collection-e2e-admin-tenant-b"
READER = "collection-e2e-reader-tenant-a"


class LiveServer:
    def __init__(self, directory):
        self.directory, self.process, self.log, self.started = directory, None, None, 0
        self.env = {key: value for key, value in os.environ.items()
                    if not key.startswith(("RAGS_", "OPENAI_", "ANTHROPIC_"))}
        self.env.update(
            PYTHONPATH=str(ROOT), RAGS_ENV="test", RAGS_DB_PATH=str(directory / "records.db"),
            RAGS_KEYS_JSON=json.dumps({ADMIN_A: "tenant-a", ADMIN_B: "tenant-b", READER: "tenant-a"}),
            RAGS_KEY_ROLES_JSON=json.dumps({ADMIN_A: ["admin"], ADMIN_B: ["admin"], READER: ["reader"]}),
            RAGS_EMBEDDING_PROVIDER="local", RAGS_MODEL_PROVIDER="extractive",
        )

    def start(self):
        self.started += 1
        self.log_path = self.directory / f"server-{self.started}.log"
        self.log = self.log_path.open("wb")
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.bind(("127.0.0.1", 0))
            listener.listen(128)
            self.url = f"http://127.0.0.1:{listener.getsockname()[1]}"
            code = (
                "import socket,sys,uvicorn; "
                "listener=socket.socket(fileno=int(sys.argv[1])); "
                "server=uvicorn.Server(uvicorn.Config('rags.api:app',host='127.0.0.1',"
                "log_level='error',access_log=False)); server.run(sockets=[listener])"
            )
            self.process = subprocess.Popen(
                [sys.executable, "-c", code, str(listener.fileno())], cwd=self.directory,
                env=self.env, stdout=self.log, stderr=subprocess.STDOUT, pass_fds=(listener.fileno(),),
            )
        finally:
            listener.close()
        deadline = time.monotonic() + 15
        with httpx.Client(timeout=0.5, trust_env=False) as client:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    break
                try:
                    if client.get(self.url + "/live").status_code == 200:
                        return self
                except httpx.HTTPError:
                    pass
                time.sleep(0.02)
        self.stop()
        pytest.fail("Collection HTTP service did not become ready: " + self.log_path.read_text())

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.log is not None:
            self.log.close()

    def restart(self):
        self.stop()
        return self.start()

    def cli(self, *arguments, key=ADMIN_A, success=True):
        env = {**self.env, "RAGS_URL": self.url, "RAGS_API_KEY": key}
        result = subprocess.run([sys.executable, "-m", "rags", *map(str, arguments)],
                                cwd=self.directory, env=env, capture_output=True, text=True, timeout=15)
        assert key not in result.stdout and key not in result.stderr
        if success:
            assert result.returncode == 0, result.stderr
            assert result.stderr == ""
            return json.loads(result.stdout)
        assert result.returncode == 1, result.stdout
        assert result.stdout == ""
        return json.loads(result.stderr)


@pytest.fixture
def live_server(tmp_path):
    server = LiveServer(tmp_path)
    try:
        yield server.start()
    finally:
        server.stop()


def test_http_public_client_persists_and_forks_across_server_restart(live_server):
    server = live_server
    with HttpClient(server.url, ADMIN_A) as client:
        collection = client.create_collection("http-notes", metadata={"team": "platform"})
        identifier = collection.id
        collection.add(ids=["retention", "recovery"],
                       documents=["Retention is seven days.", "Recovery uses daily snapshots."],
                       metadatas=[{"kind": "policy"}, {"kind": "runbook"}])
        assert collection.indexing_status()["state"] == "ready"
        assert collection.count() == 2
        assert collection.search(query_texts=["retention"], mode="lexical").rows()[0]["id"] == "retention"
        assert collection.search(query_texts=["daily snapshots"], mode="hybrid").rows()[0]["id"] == "recovery"
        before = collection.ask("What is the retention?", mode="lexical", generate=False)["citations"][0]
        assert before["record_id"] == "retention" and before["collection_id"] == identifier
        branch = collection.fork("http-branch", expected_generation=1)
        branch_id = branch.id
        collection.update(ids=["retention"], documents=["Retention is thirty days."], expected_generation=1)
        assert branch.get(ids=["retention"])["items"][0]["document"] == "Retention is seven days."
        with pytest.raises(ClientError) as stale:
            client.delete_collection(collection_id=identifier, expected_generation=1)
        assert (stale.value.code, stale.value.status_code) == ("collection_generation_conflict", 409)
        assert collection.count() == 2
    server.restart()
    with HttpClient(server.url, ADMIN_A) as client:
        collection = client.get_collection("http-notes")
        branch = client.get_collection("http-branch")
        assert (collection.id, branch.id) == (identifier, branch_id)
        assert client.list_collections_page()["total"] == 2
        assert collection.get(ids=["retention"])["items"][0]["document"] == "Retention is thirty days."
        assert branch.get(ids=["retention"])["items"][0]["version"] == before["revision"]
        client.delete_collection(collection_id=identifier, expected_generation=2)
        assert branch.count() == 2
        deleted = branch.delete(ids=["retention"], expected_generation=0)
        assert deleted["deleted"] == 1
        client.delete_collection("http-branch", expected_generation=deleted["generation"])
        assert client.list_collections() == []


def test_http_client_and_cli_respect_credentials_scopes_and_tenant_names(live_server):
    server = live_server
    with HttpClient(server.url, ADMIN_A) as a, HttpClient(server.url, ADMIN_B) as b:
        ca, cb = a.create_collection("same-name"), b.create_collection("same-name")
        ca.add(ids=["same-id"], documents=["Private tenant alpha policy."])
        cb.add(ids=["same-id"], documents=["Private tenant beta billing."])
        assert a.get_collection("same-name").id == ca.id and b.get_collection("same-name").id == cb.id
        with pytest.raises(ClientError, match="collection_not_found"):
            b.get_collection(collection_id=ca.id)
        assert server.cli("records", "get", "same-name", "--ids", "same-id", key=ADMIN_B)["items"][0]["document"] == "Private tenant beta billing."
        assert server.cli("collection", "count", "same-name", key=READER)["count"] == 1
        denied = server.cli("collection", "fork", "same-name", "--name", "stolen", key=READER, success=False)
        assert denied == {"error": "insufficient_scope", "status_code": 403}
        denied = server.cli("collection", "list", key="unknown-test-credential", success=False)
        assert denied == {"error": "invalid_token", "status_code": 401}
        with HttpClient(server.url, READER) as reader:
            read = reader.get_collection("same-name")
            assert read.search(query_text="alpha policy", mode="lexical").rows()
            with pytest.raises(ClientError, match="insufficient_scope"):
                read.delete(ids=["same-id"])
        # Legacy collection-unaware document queries cannot accidentally expose records.
        with httpx.Client(base_url=server.url, trust_env=False, timeout=5,
                          headers={"Authorization": "Bearer " + ADMIN_A}) as raw:
            response = raw.get("/documents")
            assert response.status_code == 200 and response.json() == []


def test_cli_subprocess_complete_collection_lifecycle(live_server):
    server = live_server
    directory = server.directory
    records = directory / "records.json"
    records.write_text(json.dumps({"ids": ["policy", "recovery"],
        "documents": ["Retention lasts seven days.", "Recovery uses daily snapshots."],
        "metadatas": [{"team": "platform"}, {"team": "platform"}]}))
    query = directory / "query.txt"
    query.write_text("retention policy")
    where = directory / "where.json"
    where.write_text('{"team":"platform"}')
    created = server.cli("collection", "create", "cli-notes")
    assert created["name"] == "cli-notes"
    inserted = server.cli("records", "add", "cli-notes", "--file", records)
    assert inserted["added"] == 2 and inserted["generation"] == 1
    assert server.cli("collection", "status", "cli-notes")["pending_operations"] == 0
    assert server.cli("collection", "count", "cli-notes")["count"] == 2
    result = server.cli("search", "cli-notes", "--query-file", query, "--where-file", where,
                        "--mode", "lexical", "--select", "document", "score")
    assert result["results"][0]["items"][0]["id"] == "policy"
    branch = server.cli("collection", "fork", "cli-notes", "--name", "cli-branch", "--expected-generation", "1")
    assert branch["fork_source_id"] == created["id"] and branch["generation"] == 0
    records.write_text(json.dumps({"ids": ["policy"], "documents": ["Retention lasts thirty days."],
                                  "expected_generation": 1}))
    updated = server.cli("records", "update", "cli-notes", "--file", records)
    assert updated["updated"] == 1 and updated["generation"] == 2
    stale = server.cli("records", "update", "cli-notes", "--file", records, success=False)
    assert stale == {"error": "collection_generation_conflict", "status_code": 409}
    assert server.cli("records", "get", "cli-branch", "--ids", "policy")["items"][0]["document"] == "Retention lasts seven days."
    modified = server.cli("collection", "modify", "cli-notes", "--name", "cli-renamed", "--expected-generation", "2")
    assert modified["id"] == created["id"] and modified["generation"] == 3
    assert {row["name"] for row in server.cli("collection", "list")["items"]} == {"cli-renamed", "cli-branch"}
    server.restart()
    assert server.cli("collection", "get", "cli-renamed")["id"] == created["id"]
    server.cli("collection", "delete", "cli-renamed", "--expected-generation", "3")
    deleted = server.cli("records", "delete", "cli-branch", "--ids", "policy", "--expected-generation", "0")
    assert deleted["deleted"] == 1
    server.cli("collection", "delete", "cli-branch", "--expected-generation", "1")
    assert server.cli("collection", "list")["items"] == []


@pytest.mark.postgres
@pytest.mark.external
def test_postgres_http_sdk_and_cli_preserve_native_vectors_forks_and_tenant_scope(tmp_path):
    """The public protocol exercises real pgvector through a separate API process."""
    from rags.collection_store import CollectionStore
    from rags.storage import Store

    database = os.getenv("RAGS_E2E_POSTGRES_URL") or os.getenv("RAGS_TEST_POSTGRES_URL")
    if not database:
        pytest.fail("RAGS_E2E_POSTGRES_URL or RAGS_TEST_POSTGRES_URL is required")
    namespace = "collection-http-" + uuid.uuid4().hex
    tenant_a, tenant_b = namespace + "-a", namespace + "-b"
    server = LiveServer(tmp_path)
    server.env.update(RAGS_DATABASE_URL=database,
                      RAGS_KEYS_JSON=json.dumps({ADMIN_A: tenant_a, ADMIN_B: tenant_b, READER: tenant_a}))
    sparse = {"indices": [1, 8], "values": [0.1, 3.0], "space_id": "http-fixture-vocab"}
    try:
        server.start()
        with HttpClient(server.url, ADMIN_A) as a, HttpClient(server.url, ADMIN_B) as b:
            original = a.create_collection("pg-http-notes", sparse_space_id=sparse["space_id"])
            foreign = b.create_collection("pg-http-notes")
            original.add(ids=["policy", "private"],
                         documents=["Retention is seven days.", "Private unrelated account details."],
                         metadatas=[{"team": "platform"}, {"team": "private"}],
                         sparse_embeddings=[sparse, None])
            foreign.add(ids=["policy"], documents=["Foreign tenant secret retention is ninety days."])
            status = original.indexing_status()
            assert status["index"] == "pgvector_exact" and status["vector_count"] == 2
            before = original.get(ids=["policy"], include=["document", "metadata", "embedding", "sparse_embedding"])["items"][0]
            assert before["sparse_embedding"]["values"][0] == pytest.approx(0.1)
            binding = original.to_dict()["configuration"]["embedding"]
            searches = original.search(searches=[
                {"mode": "dense", "query_embedding": before["embedding"], "embedding_space_id": binding["space_id"],
                 "where": {"team": "platform"}},
                {"mode": "hybrid", "query_text": "retention", "where_document": {"$contains": "seven"}},
                {"mode": "sparse", "query_sparse": sparse, "where": {"team": "platform"}},
            ])
            assert all([row["id"] for row in searches.rows(index)] == ["policy"] for index in range(3))
            assert searches.rows(0)[0]["distance"] == pytest.approx(0.0, abs=2e-7)
            with pytest.raises(ClientError) as forbidden:
                b.get_collection(collection_id=original.id)
            assert (forbidden.value.code, forbidden.value.status_code) == ("collection_not_found", 404)
            with pytest.raises(ClientError, match="collection_conflict"):
                original.add(ids=["new", "policy"], documents=["Must not be committed", "Duplicate record"])
            assert original.count() == 2 and original.get(ids=["new"])["items"] == []
            branch = original.fork("pg-http-branch", expected_generation=1)
            original.update(ids=["policy"], documents=["Retention is thirty days."], expected_generation=1)
            with pytest.raises(ClientError, match="collection_generation_conflict"):
                original.delete(ids=["private"], expected_generation=1)
            assert branch.get(ids=["policy"], include=["document", "metadata", "embedding", "sparse_embedding"])["items"][0] == before
            citation = branch.ask("What is the retention?", mode="lexical", generate=False)["citations"][0]
            assert citation["revision"] == before["version"] and citation["collection_id"] == branch.id
            original_id, branch_id = original.id, branch.id
        server.restart()
        assert server.cli("collection", "status", "pg-http-notes")["index"] == "pgvector_exact"
        assert server.cli("records", "get", "pg-http-branch", "--ids", "policy")["items"][0]["version"] == before["version"]
        denied = server.cli("collection", "fork", "pg-http-notes", "--name", "denied", key=READER, success=False)
        assert denied == {"error": "insufficient_scope", "status_code": 403}
        with HttpClient(server.url, ADMIN_A) as a, HttpClient(server.url, ADMIN_B) as b:
            a.delete_collection(collection_id=original_id, expected_generation=2)
            branch = a.get_collection(collection_id=branch_id)
            assert branch.get(ids=["policy"], include=["embedding"])["items"][0]["embedding"] == before["embedding"]
            assert b.get_collection("pg-http-notes").count() == 1
            a.delete_collection(collection_id=branch_id, expected_generation=0)
            b.delete_collection("pg-http-notes", expected_generation=1)
            assert a.list_collections() == [] and b.list_collections() == []
    finally:
        server.stop()
        # Cleanup is scoped to this run even if an assertion fails midway.
        collections = CollectionStore(Store(database))
        for tenant in (tenant_a, tenant_b):
            for item in collections.list(tenant):
                collections.delete_collection(tenant, item["id"])
            with collections.store.connect() as connection:
                connection.execute("DELETE FROM audit WHERE tenant=?", (tenant,))

