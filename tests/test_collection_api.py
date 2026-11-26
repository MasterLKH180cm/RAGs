"""Authenticated collection HTTP contracts over the real SQLite application."""
from dataclasses import replace
import json
import os

from fastapi.testclient import TestClient
import pytest

from rags.admission import EvidenceAdmission
from rags.api import create_app
from rags.collection_models import CollectionConfig, MAX_REQUEST_BYTES
from rags.config import Settings
from rags.features import FeatureFlags
from rags.registry import CapabilityError


pytestmark = pytest.mark.integration

ROUTES = [
    ("POST", "/collections"), ("GET", "/collections"),
    ("GET", "/collections/missing"), ("PATCH", "/collections/missing"),
    ("DELETE", "/collections/missing"), ("GET", "/collections/missing/count"),
    ("GET", "/collections/missing/indexing-status"), ("POST", "/collections/missing/fork"),
    *(("POST", "/collections/missing/records/" + name)
      for name in ("add", "update", "upsert", "get", "delete")),
    ("POST", "/collections/missing/search"), ("POST", "/collections/missing/ask"),
]


@pytest.fixture
def settings(tmp_path, monkeypatch):
    for name in tuple(os.environ):
        if name.startswith(("RAGS_", "OPENAI_", "ANTHROPIC_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("RAGS_ENV", "test")
    return replace(
        Settings.from_env(), db_path=str(tmp_path / "collection-api.db"),
        keys={"admin-a": "tenant-a", "admin-b": "tenant-b", "reader-a": "tenant-a", "editor-a": "tenant-a"},
        key_roles={"admin-a": ("admin",), "admin-b": ("admin",),
                   "reader-a": ("reader",), "editor-a": ("editor",)},
        embedding_provider="local", model_provider="extractive",
    )


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as session:
        yield session


def headers(token="admin-a", **extra):
    return {"Authorization": "Bearer " + token, **extra}


def rejected(response, status, code):
    assert response.status_code == status, response.text
    assert response.json()["detail"]["code"] == code


def create(client, name="notes", token="admin-a", **fields):
    response = client.post("/collections", headers=headers(token), json={"name": name, **fields})
    assert response.status_code == 201, response.text
    return response.json()


def add(client, collection_id, *, token="admin-a", **fields):
    response = client.post("/collections/" + collection_id + "/records/add", headers=headers(token), json=fields)
    assert response.status_code == 200, response.text
    return response.json()


def no_database_or_provider(app, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Denied or malformed collection request reached storage/provider")
    monkeypatch.setattr(app.state.collections.store, "connect", forbidden)
    for name in ("embedding", "query_embedding", "complete", "json_task"):
        monkeypatch.setattr(app.state.collections.provider, name, forbidden)


@pytest.mark.parametrize("token", [None, "invalid-credential"])
def test_every_collection_route_authenticates_before_malformed_json_or_database(settings, monkeypatch, token):
    app = create_app(settings)
    no_database_or_provider(app, monkeypatch)
    with TestClient(app) as client:
        for method, path in ROUTES:
            response = client.request(method, path, headers={"Content-Type": "application/json",
                                                           **(headers(token) if token else {})}, content=b"{")
            rejected(response, 401, "invalid_token" if token else "authentication_required")
            assert response.headers["www-authenticate"].startswith("Bearer")
    assert "database" not in app.state.collections.__dict__


@pytest.mark.parametrize("method,path", ROUTES)
def test_disabled_collections_precedes_json_query_parsing_storage_and_provider(settings, monkeypatch, method, path):
    app = create_app(replace(settings, features=FeatureFlags((("collections", False),))))
    no_database_or_provider(app, monkeypatch)
    with TestClient(app) as client:
        response = client.request(method, path + "?unrecognized=private-sentinel", content=b"{",
                                  headers=headers(**{"Content-Type": "application/json"}))
    rejected(response, 403, "feature_disabled")
    assert "private-sentinel" not in response.text
    assert "database" not in app.state.collections.__dict__


@pytest.mark.parametrize("feature,method,path", [
    ("ingestion", "POST", "/collections"),
    ("ingestion", "PATCH", "/collections/missing"),
    ("ingestion", "POST", "/collections/missing/records/add"),
    ("documents", "GET", "/collections"),
    ("documents", "POST", "/collections/missing/records/get"),
    ("documents", "GET", "/collections/missing/indexing-status"),
    ("deletion", "DELETE", "/collections/missing"),
    ("deletion", "POST", "/collections/missing/records/delete"),
    ("query", "POST", "/collections/missing/search"),
    ("query", "POST", "/collections/missing/ask"),
    ("collections.fork", "POST", "/collections/missing/fork"),
])
def test_operation_features_precede_malformed_body_and_database(settings, monkeypatch, feature, method, path):
    app = create_app(replace(settings, features=FeatureFlags(((feature, False),))))
    no_database_or_provider(app, monkeypatch)
    with TestClient(app) as client:
        response = client.request(method, path, content=b"{", headers=headers(**{"Content-Type": "application/json"}))
    rejected(response, 403, "feature_disabled")
    assert response.json()["detail"]["reason"] == "Feature is disabled: " + feature


@pytest.mark.parametrize("token,method,path", [
    ("reader-a", "POST", "/collections"), ("reader-a", "PATCH", "/collections/missing"),
    ("reader-a", "POST", "/collections/missing/fork"),
    *(("reader-a", "POST", "/collections/missing/records/" + operation)
      for operation in ("add", "update", "upsert", "delete")),
    ("reader-a", "DELETE", "/collections/missing"), ("editor-a", "DELETE", "/collections/missing"),
    ("editor-a", "POST", "/collections/missing/records/delete"),
])
def test_operation_scopes_precede_body_validation_and_database(settings, monkeypatch, token, method, path):
    app = create_app(settings)
    no_database_or_provider(app, monkeypatch)
    with TestClient(app) as client:
        response = client.request(method, path, content=b"{", headers=headers(token, **{"Content-Type": "application/json"}))
    rejected(response, 403, "insufficient_scope")
    assert "insufficient_scope" in response.headers["www-authenticate"]


@pytest.mark.parametrize("token,disabled,code", [
    ("reader-a", False, "insufficient_scope"), ("admin-a", True, "feature_disabled"),
])
def test_mcp_origin_adds_scope_and_feature_checks_before_collection_body(settings, monkeypatch, token, disabled, code):
    configuration = replace(settings, features=FeatureFlags((("mcp", False),)) if disabled else FeatureFlags())
    app = create_app(configuration)
    no_database_or_provider(app, monkeypatch)
    with TestClient(app) as client:
        response = client.post("/collections/missing/search", content=b"{",
                               headers=headers(token, **{"Content-Type": "application/json", "X-RAGS-Interface": "mcp"}))
    rejected(response, 403, code)


@pytest.mark.parametrize("feature,searches", [
    ("collections.regex", [{"mode": "lexical", "query_text": "retention", "where_document": {"$regex": "policy"}}]),
    ("collections.grouping", [{"mode": "lexical", "query_text": "retention", "group_by": {"field": "group"}}]),
    ("collections.sparse_vectors", [{"mode": "sparse", "query_sparse": {"space_id": "vocab-v1", "indices": [1], "values": [1]}}]),
    ("collections.batch", [{"mode": "lexical", "query_text": "retention"}] * 2),
    ("strategy.hybrid", [{"query_text": "retention", "mode": "hybrid"}]),
    ("strategy.dense", [{"query_text": "retention", "mode": "dense"}]),
    ("strategy.sparse", [{"query_text": "retention", "mode": "lexical"}]),
    ("embedding.local", [{"query_text": "retention", "mode": "dense"}]),
])
def test_search_subfeatures_precede_record_snapshot_and_provider(client, settings, monkeypatch, feature, searches):
    cid = create(client, sparse_space_id="vocab-v1")["id"]
    add(client, cid, ids=["policy"], documents=["Retention policy applies."])
    app = create_app(replace(settings, features=FeatureFlags(((feature, False),))))

    def forbidden(*args, **kwargs):
        pytest.fail("Disabled search capability reached records or provider")

    monkeypatch.setattr(app.state.collections.database, "read_snapshot", forbidden)
    monkeypatch.setattr(app.state.collections.provider, "query_embedding", forbidden)
    with TestClient(app) as restricted:
        response = restricted.post("/collections/" + cid + "/search", headers=headers(), json={"searches": searches})
    rejected(response, 403, "feature_disabled")


def test_zero_weight_dense_leg_needs_no_dense_feature_or_encoder(client, settings, monkeypatch):
    cid = create(client)["id"]
    add(client, cid, ids=["policy"], documents=["Retention policy applies."])
    app = create_app(replace(settings, features=FeatureFlags((("strategy.dense", False), ("embedding.local", False)))))
    monkeypatch.setattr(app.state.collections.provider, "query_embedding", lambda *a: pytest.fail("Inactive dense leg called provider"))
    with TestClient(app) as restricted:
        response = restricted.post("/collections/" + cid + "/search", headers=headers(),
                                   json={"searches": [{"query_text": "retention", "mode": "hybrid", "dense_weight": 0}]})
    assert response.status_code == 200, response.text
    assert response.json()["results"][0]["items"][0]["id"] == "policy"


def test_late_invalid_regex_branch_rejects_whole_batch_before_encoder_or_snapshot(client, monkeypatch):
    cid = create(client)["id"]
    add(client, cid, ids=["policy"], documents=["Retention policy applies."])

    def forbidden(*args, **kwargs):
        pytest.fail("Partially validated search batch reached provider or record snapshot")

    monkeypatch.setattr(client.app.state.collections.database, "read_snapshot", forbidden)
    monkeypatch.setattr(client.app.state.collections.provider, "query_embedding", forbidden)
    response = client.post("/collections/" + cid + "/search", headers=headers(), json={"searches": [
        {"query_text": "retention", "mode": "dense"},
        {"query_text": "retention", "mode": "lexical", "where_document": {"$or": [
            {"$contains": "policy"}, {"$regex": "["},
        ]}},
    ]})
    rejected(response, 422, "invalid_collection_filter")
    assert "results" not in response.json()


@pytest.mark.parametrize("method,path", ROUTES)
def test_unknown_query_parameters_cannot_smuggle_tenant_or_options(settings, monkeypatch, method, path):
    app = create_app(settings)
    no_database_or_provider(app, monkeypatch)
    with TestClient(app) as client:
        response = client.request(method, path, params={"tenant": "private-tenant-sentinel"},
                                  headers=headers(), json={})
    rejected(response, 422, "invalid_collection_input")
    assert "private-tenant-sentinel" not in response.text


@pytest.mark.parametrize("query", [
    "limit=1&limit=2", "name=a&name=b", "offset=0&offset=1", "limit=-1", "limit=+1",
    "limit=1.0", "limit=", "limit=%EF%BC%91", "limit=101", "offset=1000001",
    "limit=99999999999999999999",
])
def test_duplicate_nondecimal_and_out_of_range_query_values_fail_before_database(settings, monkeypatch, query):
    app = create_app(settings)
    no_database_or_provider(app, monkeypatch)
    with TestClient(app) as client:
        response = client.get("/collections?" + query, headers=headers())
    rejected(response, 422, "invalid_collection_input")


@pytest.mark.parametrize("body,content_type", [
    (b'{"name":"first","name":"second"}', "application/json"),
    (b'{"name":"notes","metadata":{"x":1,"x":2}}', "application/json"),
    (b'{"name":"notes","metadata":{"x":NaN}}', "application/json"),
    (b'{"name":"notes","metadata":{"x":1e999}}', "application/json"),
    (b'{"name":"notes","metadata":{"x":Infinity}}', "application/json"),
    (b'{"name":"notes","metadata":{"x":-Infinity}}', "application/json"),
    (b'{"name":"notes"} trailing', "application/json"), (b"[1,2]", "application/json"),
    (b"null", "application/json"), (b"{}", "text/plain"), (b"{}", ""),
    (b'{"name":"\xff"}', "application/json"),
    (b'{"name":"notes","credential":"private-value-sentinel"}', "application/json"),
    (b'{"name":"notes","get_or_create":"true"}', "application/json"),
    (b"[" * 1100 + b"]" * 1100, "application/json"),
])
def test_strict_json_and_schema_errors_are_fixed_safe_and_provider_free(settings, monkeypatch, body, content_type):
    app = create_app(settings)
    no_database_or_provider(app, monkeypatch)
    with TestClient(app) as client:
        response = client.post("/collections", content=body,
                               headers=headers(**{"Content-Type": content_type}))
    rejected(response, 422, "invalid_collection_input")
    assert response.json()["detail"]["reason"] == "Invalid collection request"
    assert "private-value-sentinel" not in response.text


def test_collection_body_limit_is_checked_before_parser_database_and_provider(settings, monkeypatch):
    app = create_app(settings)
    no_database_or_provider(app, monkeypatch)
    with TestClient(app) as client:
        response = client.post("/collections", content=b" " * (MAX_REQUEST_BYTES + 1),
                               headers=headers(**{"Content-Type": "application/json"}))
        rejected(response, 413, "collection_request_limit")
        # The outer intake cap also rejects larger or streamed HTTP bodies.
        response = client.post("/collections", content=iter([b" " * 1_100_000, b" " * 1_100_000]),
                               headers=headers(**{"Content-Type": "application/json"}))
        assert response.status_code == 413


def test_tenant_credentials_scope_every_collection_operation_and_ignore_spoofed_headers(client):
    first = create(client, "same-name")
    other = create(client, "same-name", token="admin-b")
    cid = first["id"]
    add(client, cid, ids=["same-record"], documents=["Private alpha retention policy."])
    add(client, other["id"], token="admin-b", ids=["same-record"], documents=["Private beta retention policy."])
    cases = [
        ("GET", "", None), ("GET", "/count", None), ("GET", "/indexing-status", None),
        ("PATCH", "", {"name": "forbidden-change"}), ("DELETE", "", None),
        ("POST", "/fork", {"name": "forbidden-fork"}),
        ("POST", "/records/get", {}), ("POST", "/records/delete", {"ids": ["same-record"]}),
        *(("POST", "/records/" + operation, {"ids": ["same-record"], "documents": ["forbidden text"]})
          for operation in ("add", "update", "upsert")),
        ("POST", "/search", {"searches": [{"query_text": "retention", "mode": "lexical"}]}),
        ("POST", "/ask", {"question": "retention policy", "mode": "lexical", "generate": False}),
    ]
    for method, suffix, body in cases:
        response = client.request(method, "/collections/" + cid + suffix,
                                  headers=headers("admin-b", **{"X-RAGS-Tenant": "tenant-a"}),
                                  **({"json": body} if body is not None else {}))
        rejected(response, 404, "collection_not_found")
        assert "alpha" not in response.text
    listed = client.get("/collections", headers=headers("admin-b", **{"X-RAGS-Tenant": "tenant-a"})).json()
    assert [item["id"] for item in listed["items"]] == [other["id"]]
    assert client.get("/collections/" + cid + "/count", headers=headers()).json()["count"] == 1
    result = client.post("/collections/" + other["id"] + "/search", headers=headers("admin-b"),
                         json={"searches": [{"query_text": "retention", "mode": "lexical"}]}).json()
    assert result["results"][0]["items"][0]["document"] == "Private beta retention policy."
    assert "alpha" not in json.dumps(result)


def test_collection_delete_generation_can_use_query_or_body_but_not_both(client):
    collection = create(client)
    cid = collection["id"]
    add(client, cid, ids=["record"], documents=["Retention policy text."])
    url = "/collections/" + cid
    for query, body in [("expected_generation=1", {"expected_generation": 1}),
                        ("expected_generation=0", {"expected_generation": 1})]:
        response = client.request("DELETE", url + "?" + query, headers=headers(), json=body)
        rejected(response, 422, "invalid_collection_input")
    rejected(client.delete(url + "?expected_generation=0", headers=headers()), 409, "collection_generation_conflict")
    rejected(client.delete(url + "?expected_generation=9223372036854775808", headers=headers()), 422, "invalid_collection_input")
    assert client.get(url + "/count", headers=headers()).json()["count"] == 1
    deleted = client.request("DELETE", url, headers=headers(), json={"expected_generation": 1})
    assert deleted.status_code == 200, deleted.text
    rejected(client.get(url, headers=headers()), 404, "collection_not_found")
    next_collection = create(client, "next")
    deleted = client.delete("/collections/" + next_collection["id"] + "?expected_generation=0", headers=headers())
    assert deleted.status_code == 200, deleted.text


def test_record_delete_requires_selector_and_generation_is_unchanged_on_rejection(client):
    cid = create(client)["id"]
    add(client, cid, ids=["a", "b"], documents=["First retention policy.", "Second retention policy."])
    path = "/collections/" + cid + "/records/delete"
    rejected(client.post(path, headers=headers(), json={}), 422, "invalid_collection_input")
    rejected(client.post(path, headers=headers(), json={"ids": ["a"], "expected_generation": 0}),
             409, "collection_generation_conflict")
    unchanged = client.get("/collections/" + cid + "/count", headers=headers()).json()
    assert unchanged["generation"] == 1 and unchanged["count"] == 2
    result = client.post(path, headers=headers(), json={"where_document": {"$contains": "First"},
                                                      "expected_generation": 1})
    assert result.status_code == 200 and result.json()["deleted"] == 1


@pytest.mark.parametrize("disabled", [None, "collections", "ingestion", "documents", "query", "collections.fork"])
def test_capabilities_report_effective_collection_scopes_features_and_limits(settings, monkeypatch, disabled):
    configuration = replace(settings, features=FeatureFlags(((disabled, False),)) if disabled else FeatureFlags(),
                            openai_key="private-provider-key-sentinel")
    app = create_app(configuration)
    monkeypatch.setattr("rags.collection_service.CollectionStore", lambda *a, **kw: pytest.fail("Capabilities initialized collection storage"))
    with TestClient(app) as client:
        response = client.get("/capabilities", headers=headers("reader-a"))
        assert response.status_code == 200, response.text
        info = response.json()["collections"]
        assert "create" not in info["methods"] and "delete_collection" not in info["methods"] and "fork" not in info["methods"]
        if disabled == "collections":
            assert info["enabled"] is False and info["methods"] == []
            assert not any(info["features"].values())
        else:
            assert info["enabled"] is True
            assert ("search" in info["methods"]) == (disabled != "query")
            assert ("get_records" in info["methods"]) == (disabled != "documents")
        admin = client.get("/capabilities", headers=headers()).json()["collections"]
        assert ("fork" in admin["methods"]) == (disabled not in {"collections", "ingestion", "documents", "collections.fork"})
        assert info["dense_index"] == "float32_cosine_exact" and info["score_order"] == "descending"
        assert info["limits"]["max_queries"] == 16 and info["limits"]["max_records"] == 10_000
        assert "private-provider-key-sentinel" not in response.text
    assert "database" not in app.state.collections.__dict__


def test_raw_search_is_unassessed_and_ask_keeps_exact_versioned_source_evidence(client):
    cid = create(client)["id"]
    text = "Retention policy keeps audit records for thirty days."
    add(client, cid, ids=["policy/原文"], documents=[text], metadatas=[{"category": "policy"}])
    path = "/collections/" + cid
    search = client.post(path + "/search", headers=headers("reader-a"),
                         json={"searches": [{"query_text": "retention policy", "mode": "lexical"}]}).json()
    assert search["evidence_status"] == "ranked_unassessed"
    assert not {"supported", "admission", "citations"} & search.keys()
    row = search["results"][0]["items"][0]
    response = client.post(path + "/ask", headers=headers("reader-a"),
                           json={"question": "What is the retention policy?", "mode": "lexical", "generate": False})
    assert response.status_code == 200, response.text
    ask = response.json()
    assert ask["answer"] is None and ask["supported"] is None
    assert ask["admission"]["decision"] == "unassessed" and ask["outcome"] == "success"
    citation = ask["citations"][0]
    assert citation["collection_id"] == cid and citation["record_id"] == row["id"]
    assert citation["revision"] == row["version"] and citation["text"] == text
    assert citation["start_char"] == 0 and citation["end_char"] == len(text)
    assert citation["origin_chunk_id"] == citation["chunk_id"]
    generated = client.post(path + "/ask", headers=headers("reader-a"),
                            json={"question": "retention policy", "mode": "lexical"}).json()
    assert text in generated["answer"] and citation["chunk_id"] in generated["budget"]["included_ids"]
    empty = client.post(path + "/ask", headers=headers("reader-a"),
                        json={"question": "unfindableword", "mode": "lexical"}).json()
    assert empty["outcome"] == "insufficient_evidence" and empty["citations"] == [] and empty["supported"] is False


def test_ask_calls_existing_evidence_admission_and_refusal_cannot_leave_citations(client, monkeypatch):
    cid = create(client)["id"]
    add(client, cid, ids=["policy"], documents=["Retention policy is thirty days."])
    calls = []

    def refuse(self, question, chunks, **options):
        calls.append((question, chunks))
        return EvidenceAdmission.refuse("model_rejected", "model_review", len(chunks))

    monkeypatch.setattr(EvidenceAdmission, "assess", refuse)
    path = "/collections/" + cid
    assert client.post(path + "/search", headers=headers(),
                       json={"searches": [{"query_text": "retention", "mode": "lexical"}]}).status_code == 200
    assert calls == []
    response = client.post(path + "/ask", headers=headers(), json={"question": "retention policy", "mode": "lexical"})
    assert response.status_code == 200, response.text
    assert len(calls) == 1 and calls[0][1][0]["record_id"] == "policy"
    body = response.json()
    assert body["outcome"] == "insufficient_evidence" and body["citations"] == []
    assert body["refusal_reason"] == "model_rejected"


def test_ask_generation_scope_is_checked_before_database_or_provider(settings, monkeypatch):
    app = create_app(replace(settings, model_provider="openai", openai_key="configured-test-key"))
    no_database_or_provider(app, monkeypatch)
    with TestClient(app) as client:
        response = client.post("/collections/missing/ask", headers=headers("reader-a"),
                               json={"question": "retention policy", "mode": "lexical", "generate": True})
    rejected(response, 403, "insufficient_scope")


def test_configured_response_cap_covers_get_and_whole_batch_and_projection_recovers(settings):
    app = create_app(replace(settings, collection_config=CollectionConfig(max_response_bytes=32768)))
    with TestClient(app) as client:
        cid = create(client, embedding={"mode": "external", "model": "fixture", "space_id": "fixture-v1", "dimensions": 2})["id"]
        add(client, cid, ids=["a", "b"], documents=["match " + "x" * 19_994] * 2,
            embeddings=[[1, 0], [0, 1]], embedding_space_id="fixture-v1")
        path = "/collections/" + cid
        rejected(client.post(path + "/records/get", headers=headers(), json={}), 422, "collection_response_limit")
        reduced = client.post(path + "/records/get", headers=headers(), json={"include": []})
        assert reduced.status_code == 200 and len(reduced.json()["items"]) == 2
        query = {"mode": "lexical", "query_text": "match", "limit": 1}
        assert client.post(path + "/search", headers=headers(), json={"searches": [query]}).status_code == 200
        rejected(client.post(path + "/search", headers=headers(), json={"searches": [query, query]}),
                 422, "collection_response_limit")
        query["select"] = ["score"]
        assert client.post(path + "/search", headers=headers(), json={"searches": [query, query]}).status_code == 200


def test_real_regex_deadline_fails_http_batch_without_partial_results(settings):
    app = create_app(replace(settings, collection_config=CollectionConfig(filter_timeout_seconds=0.01)))
    with TestClient(app) as client:
        cid = create(client, embedding={"mode": "external", "model": "fixture", "space_id": "fixture-v1", "dimensions": 2})["id"]
        add(client, cid, ids=["policy"], documents=["a" * 19_999 + "!"], embeddings=[[1, 0]], embedding_space_id="fixture-v1")
        response = client.post("/collections/" + cid + "/search", headers=headers(), json={"searches": [
            {"mode": "dense", "query_embedding": [1, 0], "embedding_space_id": "fixture-v1", "select": []},
            {"mode": "lexical", "query_text": "policy", "where_document": {"$regex": "(a|aa)+$"}},
        ]})
        rejected(response, 422, "collection_filter_timeout")
        assert "results" not in response.json()
        assert "(a|aa)" not in response.text


def test_provider_failure_and_execution_timeout_never_echo_private_exception(client, monkeypatch):
    cid = create(client)["id"]

    def failed(*args, **kwargs):
        raise RuntimeError("credential=private-provider-exception-sentinel")

    monkeypatch.setattr(client.app.state.collections.provider, "embedding", failed)
    response = client.post("/collections/" + cid + "/records/add", headers=headers(),
                           json={"ids": ["a"], "documents": ["retention policy"]})
    rejected(response, 502, "embedding_failed")
    assert "private-provider-exception-sentinel" not in response.text
    unchanged = client.get("/collections/" + cid + "/count", headers=headers()).json()
    assert unchanged["generation"] == 0 and unchanged["count"] == 0

    def expired(*args, **kwargs):
        raise CapabilityError("tool_timeout", "private-provider-exception-sentinel", 504)

    monkeypatch.setattr(client.app.state.collections.provider, "embedding", expired)
    response = client.post("/collections/" + cid + "/records/add", headers=headers(),
                           json={"ids": ["a"], "documents": ["retention policy"]})
    rejected(response, 504, "collection_timeout")
    assert "private-provider-exception-sentinel" not in response.text


def test_openapi_collection_contract_references_resolve_and_do_not_duplicate_on_repeated_calls(client):
    schema = client.app.openapi()
    frozen = json.dumps(schema, sort_keys=True)
    assert json.dumps(client.app.openapi(), sort_keys=True) == frozen

    def walk(value):
        if isinstance(value, dict):
            reference = value.get("$ref")
            if reference is not None:
                assert reference.startswith("#/")
                target = schema
                for part in reference[2:].split("/"):
                    target = target[part.replace("~1", "/").replace("~0", "~")]
                assert isinstance(target, dict)
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(schema)
    expected = {
        ("/collections", "post"): "CollectionCreate",
        ("/collections/{collection_id}", "patch"): "CollectionModify",
        ("/collections/{collection_id}/search", "post"): "SearchBatch",
        ("/collections/{collection_id}/ask", "post"): "CollectionAsk",
        ("/collections/{collection_id}/records/get", "post"): "GetRecords",
        ("/collections/{collection_id}/records/add", "post"): "RecordBatch",
    }
    for (path, method), model in expected.items():
        body = schema["paths"][path][method]["requestBody"]
        assert body["required"] is True
        assert body["content"]["application/json"]["schema"]["$ref"] == "#/components/schemas/" + model
    listing = schema["paths"]["/collections"]["get"]["parameters"]
    assert {item["name"] for item in listing if item["in"] == "query"} == {"name", "limit", "offset"}
    assert len({(item["name"], item["in"]) for item in listing}) == len(listing)
    deletion = schema["paths"]["/collections/{collection_id}"]["delete"]
    assert deletion["requestBody"]["required"] is False
    assert deletion["requestBody"]["content"]["application/json"]["schema"]["$ref"].endswith("/GenerationCheck")
    assert [item["name"] for item in deletion["parameters"] if item["in"] == "query"] == ["expected_generation"]
