"""Tenant-scoped durable agent/workflow checkpoint store.

No executable graph, Python callables or gateway secrets are serialized.
SQLite and PostgreSQL share transactionally safe compare-and-swap revisions.
"""
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import re
import secrets

from rags import postgres
from rags.registry import CapabilityError
from rags.storage import Store


IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
MAX_STATE_BYTES = 256 * 1024
SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_runs (
 tenant TEXT NOT NULL,run_id TEXT NOT NULL,kind TEXT NOT NULL,
 revision INTEGER NOT NULL,status TEXT NOT NULL,fingerprint TEXT NOT NULL,
 state TEXT NOT NULL,updated_at TEXT NOT NULL,
 PRIMARY KEY(tenant,run_id)
);
CREATE INDEX IF NOT EXISTS agent_runs_status ON agent_runs(tenant,status,updated_at);
CREATE TABLE IF NOT EXISTS agent_messages (
 tenant TEXT NOT NULL,session_id TEXT NOT NULL,sequence INTEGER NOT NULL,
 role TEXT NOT NULL,body TEXT NOT NULL,created_at TEXT NOT NULL,
 PRIMARY KEY(tenant,session_id,sequence)
);
CREATE INDEX IF NOT EXISTS agent_messages_recent ON agent_messages(tenant,session_id,sequence);
CREATE TABLE IF NOT EXISTS agent_observations (
 tenant TEXT NOT NULL,session_id TEXT NOT NULL,digest TEXT NOT NULL,
 fact TEXT NOT NULL,source_id TEXT NOT NULL,confidence TEXT NOT NULL,
 expires_at TEXT NOT NULL,updated_at TEXT NOT NULL,
 PRIMARY KEY(tenant,session_id,digest)
);
CREATE INDEX IF NOT EXISTS agent_observations_ttl ON agent_observations(tenant,session_id,expires_at);
"""


def _id(value: str, field: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError(f"Invalid {field} identifier")
    return value


def _state_json(value: dict) -> str:
    if not isinstance(value, dict):
        raise ValueError("Checkpoint state must be a JSON object")
    try:
        data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Checkpoint state must contain JSON-safe values") from exc
    if len(data.encode("utf-8")) > MAX_STATE_BYTES:
        raise ValueError("Checkpoint state exceeds storage budget")
    return data


def _when() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


@dataclass(frozen=True)
class Checkpoint:
    tenant: str
    run_id: str
    kind: str
    revision: int
    status: str
    fingerprint: str
    state: dict
    updated_at: str

    def public(self) -> dict:
        return {"run_id": self.run_id, "kind": self.kind, "revision": self.revision,
                "status": self.status, "updated_at": self.updated_at,
                "state": self.state}


class StateStore:
    def __init__(self, store: Store):
        self.store = store
        with store.connect() as db:
            if store.is_postgres:
                postgres.lock_mutation(db, "agent_schema", "global")
            db.executescript(SCHEMA)

    @staticmethod
    def _read(row) -> Checkpoint:
        data = dict(row)
        data["state"] = json.loads(data["state"])
        return Checkpoint(**data)

    def create(self, tenant: str, *, kind: str, fingerprint: str, state: dict,
               status: str = "running") -> Checkpoint:
        _id(tenant, "tenant")
        if kind not in {"agent", "workflow"}:
            raise ValueError("Unknown run type")
        if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            raise ValueError("Invalid workflow/agent fingerprint")
        encoded = _state_json(state)
        identifier = secrets.token_hex(16)
        updated = _when()
        with self.store.connect() as db:
            db.execute("INSERT INTO agent_runs "
                       "(tenant,run_id,kind,revision,status,fingerprint,state,updated_at) "
                       "VALUES(?,?,?,?,?,?,?,?)",
                       (tenant, identifier, kind, 0, status, fingerprint, encoded, updated))
        return Checkpoint(tenant, identifier, kind, 0, status, fingerprint, state, updated)

    def get(self, tenant: str, run_id: str) -> Checkpoint | None:
        _id(tenant, "tenant")
        _id(run_id, "run")
        with self.store.connect() as db:
            row = db.execute("SELECT * FROM agent_runs WHERE tenant=? AND run_id=?",
                             (tenant, run_id)).fetchone()
        return self._read(row) if row is not None else None

    def update(self, tenant: str, run_id: str, *, expected_revision: int,
               fingerprint: str, status: str, state: dict) -> Checkpoint:
        _id(tenant, "tenant")
        _id(run_id, "run")
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("Invalid checkpoint revision")
        if status not in {"running", "waiting_for_approval", "needs_resume",
                          "completed", "failed", "stopped"}:
            raise ValueError("Invalid state transition target")
        encoded = _state_json(state)
        now = _when()
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if self.store.is_postgres:
                postgres.lock_mutation(db, "agent_run", tenant, run_id)
            row = db.execute("SELECT kind,revision,fingerprint,status FROM agent_runs "
                             "WHERE tenant=? AND run_id=?", (tenant, run_id)).fetchone()
            if row is None:
                raise CapabilityError("not_found", "Run not found", 404)
            if row["fingerprint"] != fingerprint:
                raise CapabilityError("definition_mismatch", "Run definition has changed", 409)
            if row["revision"] != expected_revision:
                raise CapabilityError("revision_conflict", "Run checkpoint is stale", 409)
            if row["status"] in {"completed", "failed", "stopped"}:
                raise CapabilityError("run_terminal", "Terminal run cannot be resumed", 409)
            cursor = db.execute(
                "UPDATE agent_runs SET revision=?,status=?,state=?,updated_at=? "
                "WHERE tenant=? AND run_id=? AND revision=?",
                (expected_revision + 1, status, encoded, now, tenant, run_id, expected_revision),
            )
            if cursor.rowcount != 1:
                raise CapabilityError("revision_conflict", "Concurrent checkpoint update", 409)
        return Checkpoint(tenant, run_id, row["kind"], expected_revision + 1, status,
                          fingerprint, state, now)
