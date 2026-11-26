"""Shared connection lifecycle; schemas and foreign-key policy belong to stores."""
from collections.abc import Iterator
from contextlib import contextmanager
import sqlite3

from rags import postgres


@contextmanager
def connect(path: str, *, foreign_keys: bool) -> Iterator[sqlite3.Connection | postgres.Connection]:
    """Open one transaction owner and always close it, including setup failures.

    PostgreSQL owns its native connection policy. SQLite callers declare their
    existing foreign-key policy explicitly; this helper never initializes tables.
    """
    if postgres.is_postgres(path):
        with postgres.connect(path) as connection:
            yield connection
        return
    connection = sqlite3.connect(path, timeout=20)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=" + ("ON" if foreign_keys else "OFF"))
        connection.execute("PRAGMA busy_timeout=20000")
        connection.execute("PRAGMA journal_mode=WAL")
        yield connection
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()
