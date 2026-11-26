"""Installed command grammar, bounded input files and authenticated SDK wiring."""
import json
import os

import httpx
import pytest

from rags import cli
from rags.collection_cli import _read_file


pytestmark = pytest.mark.unit
KEY = "collection-cli-scoped-test-key"


@pytest.fixture
def wire(monkeypatch):
    monkeypatch.setenv("RAGS_API_KEY", KEY)
    monkeypatch.setenv("RAGS_URL", "https://rag.example")
    calls = []

    def handler(request):
        assert request.headers["authorization"] == "Bearer " + KEY
        payload = json.loads(request.content) if request.content else dict(request.url.params)
        calls.append((request.method, request.url.path, payload))
        info = {"id": "col_cli", "name": payload.get("name", "notes"), "metadata": {}, "generation": 2}
        if request.url.path == "/collections" and request.method == "GET":
            return httpx.Response(200, json={"items": [info], "total": 1, "limit": 100, "offset": 0})
        if request.url.path == "/collections" or request.url.path.endswith("/fork") or request.method == "PATCH":
            return httpx.Response(200, json=info)
        if request.url.path.endswith("/search"):
            return httpx.Response(200, json={"results": [{"items": [{"id": "one", "version": "v1"}]}]})
        if request.url.path.endswith("/count"):
            return httpx.Response(200, json={"count": 3})
        if request.url.path.endswith("/indexing-status"):
            return httpx.Response(200, json={"status": "ready", "consistency": "transactional"})
        return httpx.Response(200, json={"collection_id": "col_cli", "generation": 3, "changed": 1})

    original = httpx.AsyncClient
    transport = httpx.MockTransport(handler)

    def factory(**options):
        assert options["trust_env"] is False and options["follow_redirects"] is False
        return original(transport=transport, **options)

    monkeypatch.setattr("rags.client.httpx.AsyncClient", factory)
    return calls


@pytest.mark.parametrize("args", [
    ["collection", "list"], ["collection", "get", "notes"], ["collection", "count", "notes"],
    ["collection", "status", "notes"], ["collection", "delete", "notes", "--expected-generation", "2"],
    ["collection", "modify", "notes", "--name", "renamed"],
    ["collection", "fork", "notes", "--name", "branch"],
])
def test_collection_commands_use_scoped_sdk_and_json_output(wire, args, capsys):
    assert cli.main(args) == 0
    output = capsys.readouterr()
    assert isinstance(json.loads(output.out), dict) and not output.err
    assert KEY not in output.out
    assert wire


def test_create_uses_metadata_embedding_files_and_get_or_create(wire, tmp_path, capsys):
    metadata = tmp_path / "metadata.json"
    metadata.write_text('{"owner":"platform"}')
    embedding = tmp_path / "embedding.json"
    embedding.write_text('{"mode":"external","model":"fixture","space_id":"fixture-v1","dimensions":2}')
    assert cli.main(["collection", "create", "notes", "--metadata-file", str(metadata),
                     "--embedding-file", str(embedding), "--sparse-space-id", "lexical-v1",
                     "--get-or-create"]) == 0
    assert wire[-1][2] == {"name": "notes", "metadata": {"owner": "platform"},
                           "embedding": {"mode": "external", "model": "fixture", "space_id": "fixture-v1", "dimensions": 2},
                           "sparse_space_id": "lexical-v1", "get_or_create": True}
    assert json.loads(capsys.readouterr().out)["id"] == "col_cli"


@pytest.mark.parametrize("action", ["add", "upsert", "update"])
def test_record_write_file_arrays_reach_fixed_operation(wire, tmp_path, action, capsys):
    file = tmp_path / "records.json"
    file.write_text('{"ids":["one"],"documents":["A source"],"metadatas":[{"team":"platform"}]}')
    assert cli.main(["records", action, "notes", "--file", str(file)]) == 0
    assert wire[-1] == ("POST", "/collections/col_cli/records/" + action, json.loads(file.read_text()))
    assert json.loads(capsys.readouterr().out)["changed"] == 1


def test_get_and_delete_ids_keep_generation_and_pagination(wire, capsys):
    assert cli.main(["records", "get", "notes", "--ids", "one", "two", "--limit", "2", "--offset", "1"]) == 0
    assert wire[-1][2] == {"ids": ["one", "two"], "limit": 2, "offset": 1}
    assert cli.main(["records", "delete", "notes", "--ids", "one", "--expected-generation", "3"]) == 0
    assert wire[-1][2] == {"ids": ["one"], "expected_generation": 3}
    assert KEY not in capsys.readouterr().out


def test_get_file_and_modify_metadata(wire, tmp_path, capsys):
    file = tmp_path / "select.json"
    file.write_text('{"where":{"team":"platform"},"include":["metadata"],"limit":3}')
    assert cli.main(["records", "get", "notes", "--file", str(file)]) == 0
    assert wire[-1][2]["include"] == ["metadata"]
    metadata = tmp_path / "meta.json"
    metadata.write_text('{"owner":"infra"}')
    assert cli.main(["collection", "modify", "notes", "--metadata-file", str(metadata)]) == 0
    assert wire[-1][2] == {"metadata": {"owner": "infra"}}
    capsys.readouterr()


def test_search_reads_query_and_filters_from_files(wire, tmp_path, capsys):
    query, where, document = [tmp_path / name for name in ("question.txt", "where.json", "document.json")]
    query.write_text("database recovery\n")
    where.write_text('{"team":{"$in":["platform"]}}')
    document.write_text('{"$contains":"backup"}')
    assert cli.main(["search", "notes", "--query-file", str(query), "--mode", "lexical", "--limit", "3",
                     "--offset", "1", "--where-file", str(where), "--where-document-file", str(document),
                     "--select", "document", "score"]) == 0
    request = wire[-1][2]["searches"][0]
    assert request == {"query_text": "database recovery", "mode": "lexical", "limit": 3, "offset": 1,
                       "where": {"team": {"$in": ["platform"]}}, "where_document": {"$contains": "backup"},
                       "select": ["document", "score"]}
    assert json.loads(capsys.readouterr().out)["results"][0]["items"][0]["id"] == "one"


def test_advanced_search_json_is_supported(wire, tmp_path, capsys):
    file = tmp_path / "search.json"
    file.write_text('{"searches":[{"query_text":"backup","mode":"hybrid","dense_weight":0,"sparse_weight":1}]}')
    assert cli.main(["search", "notes", "--file", str(file)]) == 0
    assert wire[-1][2] == json.loads(file.read_text())
    capsys.readouterr()


@pytest.mark.parametrize("body", ["[]", "not-json", '{"ids":[],"ids":["secret"]}', '{"ids":[NaN]}',
                                  '{"ids":["source"],"documents":["fact"],"unused":1e999}'])
def test_bad_json_files_fail_before_any_request(wire, tmp_path, capsys, body):
    file = tmp_path / "bad.json"
    file.write_text(body)
    assert cli.main(["records", "upsert", "notes", "--file", str(file)]) == 1
    assert not wire
    output = capsys.readouterr()
    assert not output.out and json.loads(output.err)["error"] == "invalid_input_file"
    assert "secret" not in output.err


def test_missing_key_and_unsafe_endpoint_fail_without_http(wire, monkeypatch, capsys):
    monkeypatch.delenv("RAGS_API_KEY")
    assert cli.main(["collection", "list"]) == 1
    assert json.loads(capsys.readouterr().err)["error"] == "invalid_token"
    monkeypatch.setenv("RAGS_API_KEY", KEY)
    assert cli.main(["collection", "list", "--url", "https://name:private-password@example.com"]) == 1
    assert "private-password" not in capsys.readouterr().err
    assert not wire


def test_input_files_reject_symlinks_directories_oversize_and_fifos(wire, tmp_path, capsys):
    source = tmp_path / "source.json"
    source.write_text("{}")
    link = tmp_path / "link.json"
    link.symlink_to(source)
    big = tmp_path / "large.json"
    with big.open("wb") as stream:
        stream.truncate(2 * 1024 * 1024 + 1)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    for file in (link, tmp_path, big, fifo, tmp_path / "missing"):
        assert cli.main(["records", "upsert", "notes", "--file", str(file)]) == 1
        assert json.loads(capsys.readouterr().err)["error"] == "invalid_input_file"
    assert not wire


def test_file_read_is_bounded_even_if_stat_size_is_stale(tmp_path, monkeypatch):
    file = tmp_path / "changing.txt"
    file.write_bytes(b"x" * 10)
    original = os.fstat

    def stale(fd):
        result = original(fd)
        values = list(result)
        values[6] = 0
        return os.stat_result(values)

    monkeypatch.setattr(os, "fstat", stale)
    with pytest.raises(ValueError, match="request_too_large"):
        _read_file(file, limit=3)


@pytest.mark.parametrize("text", [b"", b"x" * 2001, b"x" * 8001, b"\xff"])
def test_query_file_is_valid_bounded_utf8(wire, tmp_path, text, capsys):
    file = tmp_path / "query.txt"
    file.write_bytes(text)
    assert cli.main(["search", "notes", "--query-file", str(file)]) == 1
    assert not wire
    assert json.loads(capsys.readouterr().err)["error"] == "invalid_input_file"


def test_ambiguous_json_and_flags_are_rejected(wire, tmp_path, capsys):
    file = tmp_path / "search.json"
    file.write_text('{"searches":[{"mode":"lexical","query_text":"one"}]}')
    assert cli.main(["search", "notes", "--file", str(file), "--limit", "3"]) == 1
    assert not wire and json.loads(capsys.readouterr().err)["error"] == "ambiguous_input"
    file.write_text('{"limit":3}')
    assert cli.main(["records", "get", "notes", "--file", str(file), "--limit", "4"]) == 1
    assert not wire and json.loads(capsys.readouterr().err)["error"] == "ambiguous_input"


def test_parser_requires_explicit_record_delete_selection():
    with pytest.raises(SystemExit) as error:
        cli.build_parser().parse_args(["records", "delete", "notes"])
    assert error.value.code == 2
