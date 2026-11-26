"""Real shared SQLite lifecycle and explicit store policy regressions."""
from contextlib import contextmanager
import sqlite3

import pytest

from rags import database, postgres
from rags.multimodal import AssetStore
from rags.storage import Store

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("store_type,foreign_keys", [(Store, 1), (AssetStore, 0)])
def test_store_connections_preserve_policy_commit_rollback_and_close(tmp_path, store_type, foreign_keys):
    store = store_type(str(tmp_path / "lifecycle.db"))
    with store.connect() as connection:
        assert connection.row_factory is sqlite3.Row
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == foreign_keys
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 20_000
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        connection.execute("CREATE TABLE lifecycle(value TEXT)")
        connection.execute("INSERT INTO lifecycle VALUES(?)", ("committed",))
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")

    class Interrupted(BaseException):
        pass

    with pytest.raises(Interrupted), store.connect() as interrupted:
        interrupted.execute("INSERT INTO lifecycle VALUES(?)", ("rolled-back",))
        raise Interrupted()
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        interrupted.execute("SELECT 1")
    with store.connect() as reopened:
        assert [row[0] for row in reopened.execute("SELECT value FROM lifecycle")] == ["committed"]


@pytest.mark.parametrize("store_type", [Store, AssetStore])
def test_shared_setup_failure_closes_each_store_connection(tmp_path, monkeypatch, store_type):
    store = store_type(str(tmp_path / "failure.db"))
    original = sqlite3.connect
    connections = []

    class BrokenSetup(sqlite3.Connection):
        def execute(self, statement, *args, **kwargs):
            if statement == "PRAGMA journal_mode=WAL":
                raise sqlite3.OperationalError("fixture setup failure")
            return super().execute(statement, *args, **kwargs)

    def connect(*args, **kwargs):
        connection = original(*args, factory=BrokenSetup, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", connect)
    with pytest.raises(sqlite3.OperationalError, match="fixture setup failure"), store.connect():
        pytest.fail("Failed setup cannot yield a connection")
    assert len(connections) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connections[0].execute("SELECT 1")


@pytest.mark.parametrize("failure", [False, True])
def test_shared_postgres_facade_preserves_native_transaction_owner(monkeypatch, failure):
    native = object()
    events = []

    @contextmanager
    def connect(path):
        assert path == "postgresql://fixture"
        events.append("open")
        try:
            yield native
            events.append("commit")
        except RuntimeError:
            events.append("rollback")
            raise
        finally:
            events.append("close")

    monkeypatch.setattr(postgres, "connect", connect)
    monkeypatch.setattr(sqlite3, "connect", lambda *_a, **_k: pytest.fail("Native PostgreSQL must own its connection"))
    if failure:
        with pytest.raises(RuntimeError, match="fixture"), database.connect("postgresql://fixture", foreign_keys=True) as connection:
            assert connection is native
            raise RuntimeError("fixture")
        assert events == ["open", "rollback", "close"]
    else:
        with database.connect("postgresql://fixture", foreign_keys=False) as connection:
            assert connection is native
        assert events == ["open", "commit", "close"]
