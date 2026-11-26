"""Retained-payload budgets, LRU reuse and unchanged late-interaction evidence."""
from concurrent.futures import ThreadPoolExecutor
import hashlib

import pytest

from rags.backends import BackendManager, _TokenVectorCache, maxsim

pytestmark = pytest.mark.unit


def matrix(rows=1):
    return tuple((1.0, 0.0) for _ in range(rows))


def test_cache_evicts_only_lru_entries_until_variable_payload_fits():
    small = matrix()
    large = matrix(5)
    small_bytes = _TokenVectorCache.payload_bytes("a", small)
    large_bytes = _TokenVectorCache.payload_bytes("d", large)
    cache = _TokenVectorCache(large_bytes + small_bytes)
    for key in "abc":
        cache.put(key, small)
    assert cache.get("a") == small  # b, then c, are least recently used.
    cache.put("d", large)
    assert cache.get("a") == small
    assert cache.get("b") is None and cache.get("c") is None
    assert cache.get("d") == large
    assert cache.info() == {"entries": 2, "payload_bytes": large_bytes + small_bytes,
                            "max_entries": 1000, "max_bytes": large_bytes + small_bytes}


def test_entry_bound_recency_replacement_and_oversized_skip():
    cache = _TokenVectorCache(10_000, max_entries=2)
    cache.put("a", matrix())
    cache.put("b", matrix())
    cache.get("a")
    cache.put("c", matrix())
    assert cache.get("b") is None
    assert cache.get("a") == matrix()
    cache.put("a", matrix(3))
    expected = cache.payload_bytes("a", matrix(3)) + cache.payload_bytes("c", matrix())
    assert cache.info()["payload_bytes"] == expected
    cache.put("large", matrix(1000))
    assert cache.get("large") is None and cache.info()["payload_bytes"] == expected
    cache.put("a", matrix(1000))
    assert cache.get("a") is None
    assert cache.info()["payload_bytes"] == cache.payload_bytes("c", matrix())


def test_retained_matrix_is_immutable_and_zero_budget_does_not_cache():
    original = [[1.0, 0.0]]
    cache = _TokenVectorCache(1000)
    cache.put("a", original)
    before = cache.info()
    original[0].append(7.0)
    original.append([3.0])
    assert cache.get("a") == ((1.0, 0.0),)
    assert cache.info() == before
    disabled = _TokenVectorCache(0)
    disabled.put("a", matrix())
    assert disabled.get("a") is None
    assert disabled.info()["entries"] == disabled.info()["payload_bytes"] == 0


@pytest.mark.parametrize("max_bytes,max_entries", [(-1, 1), (True, 1), (1.5, 1), (10, 0), (10, True)])
def test_invalid_cache_limits_fail_explicitly(max_bytes, max_entries):
    with pytest.raises(ValueError, match="cache limits"):
        _TokenVectorCache(max_bytes, max_entries)


def test_concurrent_replacements_evictions_and_reads_keep_exact_accounting():
    cache = _TokenVectorCache(1800, max_entries=4)

    def write(index):
        key = str(index % 7)
        cache.put(key, matrix(1 + index % 5))
        cache.get(str((index + 1) % 7))
        info = cache.info()
        assert 0 <= info["payload_bytes"] <= info["max_bytes"]
        assert 0 <= info["entries"] <= info["max_entries"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(400)))
    retained = [(key, cache.get(key)) for key in map(str, range(7))]
    retained = [(key, value) for key, value in retained if value is not None]
    assert cache.info()["entries"] == len(retained)
    assert cache.info()["payload_bytes"] == sum(cache.payload_bytes(key, value) for key, value in retained)


@pytest.mark.parametrize("budget", ["0", "1", "100000"])
def test_search_preserves_maxsim_and_tenant_evidence_with_or_without_retention(monkeypatch, budget):
    monkeypatch.setenv("RAGS_BACKEND_CACHE_BYTES", budget)
    monkeypatch.setenv("RAGS_LATE_INTERACTION_MODEL", "fixture-token-encoder")
    manager = BackendManager()
    calls = []
    matrices = {"aligned": [[2.0, 0.0]], "orthogonal": [[0.0, 3.0]]}

    def encode(text, *, is_query=False):
        calls.append((text, is_query))
        return [[1.0, 0.0]] if is_query else matrices[text]

    monkeypatch.setattr(manager, "encode_tokens", encode)
    results = []
    for tenant in ("first", "second"):
        chunks = [{"id": tenant + ":" + text, "content": text} for text in matrices]
        result = manager.search("colbert", tenant=tenant, question="query", chunks=chunks, top_k=2)
        assert result["hits"] == [
            {"chunk_id": tenant + ":aligned", "score": maxsim([[1.0, 0.0]], matrices["aligned"])},
            {"chunk_id": tenant + ":orthogonal", "score": maxsim([[1.0, 0.0]], matrices["orthogonal"])},
        ]
        results.append(result)
    assert results[0]["trace"] == results[1]["trace"]
    assert len([call for call in calls if not call[1]]) == (2 if budget == "100000" else 4)
    assert manager._vectors.info()["payload_bytes"] <= int(budget)
    assert manager._vectors.get(hashlib.sha256(b"aligned").hexdigest()) == (
        ((2.0, 0.0),) if budget == "100000" else None
    )


def test_invalid_encoder_matrix_is_never_retained(monkeypatch):
    monkeypatch.setenv("RAGS_LATE_INTERACTION_MODEL", "fixture-token-encoder")
    manager = BackendManager()
    monkeypatch.setattr(manager, "encode_tokens", lambda text, **_k: [[1.0, 0.0]] if text == "query" else [[float("nan"), 0.0]])
    with pytest.raises(ValueError, match="finite"):
        manager.search("colbert", tenant="tenant", question="query", chunks=[{"id": "source", "content": "invalid"}], top_k=1)
    assert manager._vectors.info()["entries"] == 0


@pytest.mark.parametrize("value", ["-1", "NaN", "1.5", ""])
def test_invalid_cache_environment_has_a_sanitized_configuration_error(monkeypatch, value):
    monkeypatch.setenv("RAGS_BACKEND_CACHE_BYTES", value)
    with pytest.raises(ValueError, match="RAGS_BACKEND_CACHE_BYTES must be a nonnegative integer"):
        BackendManager()
