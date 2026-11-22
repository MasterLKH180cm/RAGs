"""Native-vector input and the bounded PostgreSQL adapter contracts."""
import json
import math
import sys
from types import ModuleType

import pytest

from rags import postgres
from rags.registry import CapabilityError
from rags.storage import SCHEMA


pytestmark = pytest.mark.unit


@pytest.mark.parametrize("values", [None, [], [0.0] * 16_001, [True], ["1"], [None],
                                   [float("nan")], [float("inf")], [-float("inf")], [1e100], [10**400]])
def test_native_vector_rejects_nonfinite_or_unrepresentable_values(values):
    with pytest.raises(CapabilityError) as error:
        postgres.vector_literal(values)
    assert error.value.code == "invalid_embedding" and error.value.status_code == 502
    assert "1..16000 finite float32" in str(error.value)


def test_native_vector_literal_rounds_float32_and_preserves_zero_semantics():
    values = json.loads(postgres.vector_literal([0, -0.0, 0.1, 1e-100, 3.4028234663852886e38]))
    assert values[:2] == [0.0, -0.0] and math.copysign(1.0, values[1]) == -1
    assert values[2] == 0.10000000149011612 and values[3] == 0.0
    assert values[4] == 3.4028234663852886e38
    assert len(json.loads(postgres.vector_literal([1.0] * 16_000))) == 16_000
    with pytest.raises(CapabilityError) as error:
        postgres.vector_literal([True], code="invalid_asset_vector")
    assert error.value.code == "invalid_asset_vector"


def test_bound_values_do_not_rewrite_question_marks_inside_sql_literals():
    statement = "SELECT '?' AS literal, 'it''s ?' AS quoted FROM documents WHERE tenant=? AND id=?"
    assert postgres.bind_parameters(statement) == (
        "SELECT '?' AS literal, 'it''s ?' AS quoted FROM documents WHERE tenant=%s AND id=%s"
    )
    translated = postgres.translate_schema(SCHEMA)
    assert "embedding vector NOT NULL" in translated
    assert "tsvector GENERATED ALWAYS" in translated
    assert "VIRTUAL TABLE" not in translated and "AUTOINCREMENT" not in translated


class RawCursor:
    rowcount = 1

    def __init__(self, rows=()):
        self.rows = list(rows)

    def fetchone(self):
        return self.rows.pop(0) if self.rows else None

    def fetchall(self):
        result, self.rows = self.rows, []
        return result

    def __iter__(self):
        return iter(self.rows)


class RawConnection:
    def __init__(self, setup_failure=False):
        self.calls = []
        self.events = []
        self.setup_failure = setup_failure

    def execute(self, statement, parameters=()):
        self.calls.append((statement, parameters))
        if self.setup_failure:
            raise OSError("fixture setup failure")
        return RawCursor()

    def commit(self):
        self.events.append("commit")

    def rollback(self):
        self.events.append("rollback")

    def close(self):
        self.events.append("close")


def test_adapter_preserves_named_and_positional_rows_without_serializing_reads():
    raw = RawConnection()
    connection = postgres.Connection(raw)
    connection.execute("BEGIN IMMEDIATE")
    connection.execute("SELECT value FROM records WHERE id=?", ("private?value",))
    connection.execute("INSERT OR IGNORE INTO edges VALUES(?)", ("value",))
    connection.execute("PRAGMA table_info(chunks)")
    assert not any("advisory" in statement for statement, _ in raw.calls)
    assert ("SELECT value FROM records WHERE id=%s", ("private?value",)) in raw.calls
    assert ("INSERT INTO edges VALUES(%s) ON CONFLICT DO NOTHING", ("value",)) in raw.calls
    assert raw.calls[-1][1] == ("chunks",)
    cursor = postgres.Cursor(RawCursor([{"first": 1, "second": 2}]))
    row = cursor.fetchone()
    assert row[0] == row["first"] == 1 and row[1] == row["second"] == 2
    assert cursor.fetchone() is None and cursor.rowcount == 1
    assert postgres.Cursor(RawCursor([{"value": 3}])).fetchall() == [{"value": 3}]
    assert list(postgres.Cursor(RawCursor([{"value": 4}]))) == [{"value": 4}]
    for unsupported in ("PRAGMA journal_mode=WAL", "SELECT rowid FROM records"):
        with pytest.raises(ValueError, match="explicit storage"):
            connection.execute(unsupported)


@pytest.mark.parametrize("failure", [None, "setup", "body"])
def test_postgres_transaction_closes_after_success_body_or_setup_failure(monkeypatch, failure):
    raw = RawConnection(setup_failure=failure == "setup")
    module, rows = ModuleType("psycopg"), ModuleType("psycopg.rows")
    module.connect = lambda *args, **kwargs: raw
    rows.dict_row = object()
    monkeypatch.setitem(sys.modules, "psycopg", module)
    monkeypatch.setitem(sys.modules, "psycopg.rows", rows)
    if failure:
        with pytest.raises(OSError):
            with postgres.connect("postgresql://disposable-fixture"):
                raise OSError("fixture body failure")
        assert raw.events == ["rollback", "close"]
    else:
        with postgres.connect("postgresql://disposable-fixture"):
            pass
        assert raw.events == ["commit", "close"]


def test_mutation_lock_namespaces_do_not_alias_tenant_and_document_boundaries():
    raw = RawConnection()
    connection = postgres.Connection(raw)
    postgres.lock_mutation(connection, "document", "a:b", "c")
    postgres.lock_mutation(connection, "document", "a", "b:c")
    assert raw.calls[-2][0] == raw.calls[-1][0]
    assert raw.calls[-2][1] != raw.calls[-1][1]
    assert "a:b" not in raw.calls[-2][0]


def test_missing_vector_extension_does_not_attempt_privileged_bootstrap():
    raw = RawConnection()
    with pytest.raises(ValueError, match="administrator bootstrap"):
        postgres.initialize(postgres.Connection(raw), SCHEMA)
    assert not any("CREATE" in statement for statement, _ in raw.calls)
