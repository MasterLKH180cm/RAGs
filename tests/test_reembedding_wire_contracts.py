"""Malformed HTTP responses must not disable re-embedding CAS or lose progress."""
import json

import httpx
import pytest

from rags.client import ClientError, Collection, HttpClient, ReembeddingError

pytestmark = pytest.mark.unit
SOURCE = {"id": "source", "name": "source", "metadata": {}, "generation": 1,
          "record_count": 1, "configuration": {"sparse_space_id": None}}
TARGET = {"id": "target", "name": "copy", "metadata": {}, "generation": 0,
          "record_count": 0, "configuration": {"sparse_space_id": None}}


def wire_client(monkeypatch, *, source=None, target=None, row=None):
    writes = []
    requests = []

    def respond(request):
        path = request.url.path
        requests.append((request.method, path))
        if path == "/collections/source":
            return httpx.Response(200, json=SOURCE if source is None else source)
        if path == "/collections":
            return httpx.Response(201, json=TARGET if target is None else target)
        if path == "/collections/source/records/get":
            record = {"id": "one", "document": "Exact original source", "metadata": {}}
            return httpx.Response(200, json={"generation": 1, "items": [record if row is None else row]})
        if path == "/collections/target/records/add":
            writes.append(json.loads(request.content))
            return httpx.Response(200, json={"generation": 1})
        if path == "/collections/target":
            return httpx.Response(200, json={**TARGET, "generation": 1, "record_count": 1})
        pytest.fail("Unexpected workflow request")

    transport = httpx.MockTransport(respond)
    original = httpx.AsyncClient

    def factory(**kwargs):
        return original(transport=transport, **kwargs)

    monkeypatch.setattr("rags.client.httpx.AsyncClient", factory)
    return HttpClient("https://fixture.invalid", "fixture-key"), requests, writes


@pytest.mark.parametrize("generation", [None, True, -1, 1, "0", 2 ** 63])
def test_invalid_created_target_generation_never_allows_an_unconditional_record_write(monkeypatch, generation):
    target = {**TARGET, "generation": generation}
    if generation is None:
        target.pop("generation")
    client, _, writes = wire_client(monkeypatch, target=target)
    with client, pytest.raises(ReembeddingError) as error:
        Collection(client, SOURCE).reembed("copy", embedding_profile="approved")
    assert writes == [], "A malformed response must not turn expected_generation into an unconditional write"
    assert error.value.progress["target_collection_id"] == "target"
    assert error.value.progress["completed"] is False
    assert error.value.progress["copied_records"] == 0
    assert error.value.progress["cause"] == "invalid_response"


@pytest.mark.parametrize("row", [
    {"document": "Exact text", "metadata": {}},
    {"id": None, "document": "Exact text", "metadata": {}},
    {"id": "", "document": "Exact text", "metadata": {}},
])
def test_malformed_record_identity_is_sanitized_and_retains_target_progress(monkeypatch, row):
    client, _, writes = wire_client(monkeypatch, row=row)
    with client, pytest.raises(ReembeddingError) as error:
        Collection(client, SOURCE).reembed("copy", embedding_profile="approved")
    assert writes == []
    assert error.value.progress["target_collection_id"] == "target"
    assert error.value.progress["copied_records"] == 0
    assert error.value.progress["completed"] is False


@pytest.mark.parametrize("source", [
    {**SOURCE, "configuration": []}, {**SOURCE, "configuration": None},
    {**SOURCE, "generation": -1}, {**SOURCE, "generation": 2 ** 63},
])
def test_invalid_source_snapshot_fails_before_creating_a_target(monkeypatch, source):
    client, requests, writes = wire_client(monkeypatch, source=source)
    with client, pytest.raises(ClientError) as error:
        Collection(client, SOURCE).reembed("copy", embedding_profile="approved")
    assert error.value.code == "invalid_response"
    assert ("POST", "/collections") not in requests
    assert writes == []


def test_well_formed_wire_copy_keeps_explicit_generation_checks(monkeypatch):
    client, _, writes = wire_client(monkeypatch)
    with client:
        result = Collection(client, SOURCE).reembed("copy", embedding_profile="approved")
    assert result["completed"] is True and result["copied_records"] == 1
    assert writes == [{"ids": ["one"], "documents": ["Exact original source"],
                       "metadatas": [{}], "expected_generation": 0}]
