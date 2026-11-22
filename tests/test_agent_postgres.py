"""Real Postgres+pgvector tenant-scoped agent state and memory transaction tests."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import os
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit
import uuid

import pytest

from rags.agents.memory import ConversationStore, ObservationStore
from rags.agents.state import StateStore
from rags.registry import CapabilityError
from rags.storage import Store

pytestmark = [pytest.mark.integration, pytest.mark.external, pytest.mark.postgres]


@pytest.fixture
def pg_database(monkeypatch):
    from psycopg import connect, sql
    from psycopg.conninfo import conninfo_to_dict

    endpoint = os.getenv("RAGS_TEST_POSTGRES_URL")
    if not endpoint:
        pytest.fail("RAGS_TEST_POSTGRES_URL required for PostgreSQL agent state test")
    parsed = urlsplit(endpoint)
    schema = "rags_agent_" + uuid.uuid4().hex
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["options"] = (query.get("options", "") + f" -c search_path={schema},public").strip()
    isolated = urlunsplit(parsed._replace(query=urlencode(query, quote_via=quote)))
    assert conninfo_to_dict(isolated)["options"] == query["options"]
    monkeypatch.setenv("RAGS_ENV", "test")
    with connect(endpoint, autocommit=True, connect_timeout=5) as admin:
        assert admin.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            yield isolated
        finally:
            admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def test_native_postgres_agent_checkpoint_cas_and_memory(pg_database):
    store = Store(pg_database)
    state = StateStore(store)
    cp = state.create("tenant-a", kind="agent", fingerprint="a" * 64,
                      state={"step": 0, "pending": None})
    assert cp.revision == 0
    assert state.get("tenant-b", cp.run_id) is None
    assert state.get("tenant-a", cp.run_id).state["step"] == 0
    updated = state.update("tenant-a", cp.run_id, expected_revision=0, fingerprint="a" * 64,
                           status="waiting_for_approval", state={"step": 1, "pending": "approval"})
    assert updated.revision == 1
    with pytest.raises(CapabilityError) as stale:
        state.update("tenant-a", cp.run_id, expected_revision=0, fingerprint="a"*64,
                     status="running", state={"step": 2})
    assert stale.value.status_code == 409
    assert state.get("tenant-a", cp.run_id).status == "waiting_for_approval"
    history = ConversationStore(state)
    assert history.append("tenant-a", "shared", "user", "Private A") == 1
    assert history.append("tenant-b", "shared", "user", "Private B") == 1
    assert history.last("tenant-a", "shared") == [{"role": "user", "content": "Private A"}]
    memory = ObservationStore(state, ttl_days=10)
    memory.observe("tenant-a", "shared", fact="Observation A", source_id="approved")
    memory.observe("tenant-b", "shared", fact="Observation B", source_id="approved")
    assert memory.recent("tenant-a", "shared")[0]["fact"] == "Observation A"
    assert memory.recent("tenant-b", "shared")[0]["fact"] == "Observation B"


def test_agent_tables_created_in_tenant_database_schema(pg_database):
    state = StateStore(Store(pg_database))
    with state.store.connect() as db:
        rows = db.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema=current_schema() AND table_name LIKE ?",
            ("agent_%",),
        ).fetchall()
    assert {row["table_name"] for row in rows} >= {
        "agent_runs", "agent_messages", "agent_observations"}


def test_parallel_postgres_checkpoint_writers_only_one_wins(pg_database):
    """CAS plus the advisory transaction lock prevents double approval/execution."""
    store = Store(pg_database)
    state = StateStore(store)
    saved = state.create("tenant-a", kind="agent", fingerprint="a" * 64,
                         status="waiting_for_approval", state={"cursor": 1})
    synchronizer = Barrier(2)

    def mutate(value):
        synchronizer.wait(timeout=10)
        try:
            updated = state.update(
                "tenant-a", saved.run_id, expected_revision=0,
                fingerprint="a" * 64, status="running",
                state={"cursor": 2, "approved_by": value},
            )
            return ("success", updated.revision, value)
        except CapabilityError as exc:
            return (exc.code, None, value)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(mutate, name) for name in ("approver-1", "approver-2")]
        results = [future.result(timeout=20) for future in futures]

    assert {result[0] for result in results} == {"success", "revision_conflict"}
    checkpoint = state.get("tenant-a", saved.run_id)
    assert checkpoint.revision == 1 and checkpoint.status == "running"
    winner = next(value for status, _, value in results if status == "success")
    assert checkpoint.state["approved_by"] == winner
    assert state.get("tenant-b", saved.run_id) is None


def test_postgres_legacy_message_migration_preserves_existing_history(pg_database):
    db = Store(pg_database)
    with db.connect() as conn:
        conn.execute(
            "CREATE TABLE agent_messages (tenant TEXT NOT NULL, session_id TEXT NOT NULL, "
            "sequence INTEGER NOT NULL, role TEXT NOT NULL, body TEXT NOT NULL, "
            "created_at TEXT NOT NULL, PRIMARY KEY(tenant,session_id,sequence))"
        )
        conn.execute("INSERT INTO agent_messages "
                     "(tenant,session_id,sequence,role,body,created_at) "
                     "VALUES(?,?,?,?,?,?)",
                     ("tenant", "s", 1, "assistant", "Previously verified", "2024-01-01"))
    state = StateStore(db)
    memory = ConversationStore(state)
    assert memory.last("tenant", "s", with_evidence=True) == [
        {"role": "assistant", "content": "Previously verified", "evidence_ids": []}
    ]
    memory.append("tenant", "s", "assistant", "Follow-up [source_v1]",
                  evidence_ids=["source_v1"])
    assert memory.last("tenant", "s", with_evidence=True)[-1][
        "evidence_ids"] == ["source_v1"]
