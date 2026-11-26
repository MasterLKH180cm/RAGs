"""Authenticated workflow API: graph execution, human decision and isolation."""
from dataclasses import replace
import os

from fastapi.testclient import TestClient
import pytest

from rags.api import create_app
from rags.config import Settings

pytestmark = pytest.mark.integration

RUNNER={"Authorization":"Bearer runner-a"}
FOREIGN={"Authorization":"Bearer runner-b"}
READER={"Authorization":"Bearer reader-a"}
ADMIN={"Authorization":"Bearer admin-a"}


@pytest.fixture
def client(tmp_path,monkeypatch):
    for name in tuple(os.environ):
        if name.startswith(("RAGS_","OPENAI_","ANTHROPIC_")):
            monkeypatch.delenv(name,raising=False)
    monkeypatch.setenv("RAGS_ENV","test")
    monkeypatch.setenv("RAGS_ROUTER_ENABLED", "true")
    monkeypatch.setenv("RAGS_AGENT_ENABLED", "true")
    cfg=replace(Settings.from_env(),db_path=str(tmp_path/"workflows.db"),
        keys={"runner-a":"tenant-a","runner-b":"tenant-b",
              "reader-a":"tenant-a","admin-a":"tenant-a"},
        key_roles={"runner-a":("agent_runner",),"runner-b":("agent_runner",),
                   "reader-a":("reader",),"admin-a":("admin",)})
    with TestClient(create_app(cfg)) as c:
        yield c


def start(c, headers=RUNNER):
    return c.post("/agents/workflows/evidence_review/runs",headers=headers,json={
        "question":"What is the retention policy for record data?",
        "mode":"sparse","top_k":3,
    })


def test_graph_catalog_and_full_lifecycle_without_gateway_credentials(client):
    caps=client.get("/agents/capabilities",headers=RUNNER)
    assert caps.status_code==200
    assert caps.json()["router_ready"] is False
    catalog=client.get("/agents/workflows",headers=RUNNER)
    assert catalog.status_code==200
    template=catalog.json()["templates"][0]
    assert template["name"]=="evidence_review"
    assert [node["kind"] for node in template["nodes"]]==[
        "then","branch","parallel","then","then"]
    assert any(node["approval"] for node in template["nodes"])
    created=client.post("/documents",headers=ADMIN,json={
        "title":"Clinical retention policy","document_id":"policy",
        "text":"Retention policy for record data is ninety days."})
    assert created.status_code==201,created.text
    response=start(client)
    assert response.status_code==201,response.text
    data=response.json()
    assert data["status"]=="waiting_for_approval"
    assert data["state"]["outputs"]["cross_checks"]["dedup"]["unique_chunk_ids"] is True
    assert data["state"]["cursor"]==3
    identifier=data["run_id"]
    assert client.get("/agents/workflows/runs/"+identifier,headers=RUNNER).status_code==200
    assert client.get("/agents/workflows/runs/"+identifier,headers=FOREIGN).status_code==404
    assert client.get("/agents/workflows/runs/"+identifier,headers=READER).status_code==403
    assert client.post("/agents/workflows/runs/"+identifier+"/resume",headers=FOREIGN,json={
        "expected_revision":data["revision"],"approved":True}).status_code==404
    assert client.post("/agents/workflows/runs/"+identifier+"/resume",headers=RUNNER,json={
        "expected_revision":0,"approved":True}).status_code==409
    completed=client.post("/agents/workflows/runs/"+identifier+"/resume",headers=RUNNER,json={
        "expected_revision":data["revision"],"approved":True})
    assert completed.status_code==200,completed.text
    assert completed.json()["status"]=="completed"
    assert completed.json()["state"]["outputs"]["finalize"]["status"]=="approved"
    assert client.post("/agents/workflows/runs/"+identifier+"/resume",headers=RUNNER,json={
        "expected_revision":completed.json()["revision"],"approved":True}).status_code==409
    assert client.post("/agents/workflows/runs/"+identifier+"/recover",headers=RUNNER,json={
        "expected_revision":completed.json()["revision"]}).status_code==409


def test_workflow_denial_and_authorization(client):
    assert client.get("/agents/workflows").status_code==401
    assert client.get("/agents/workflows",headers=READER).status_code==403
    assert start(client,READER).status_code==403
    assert client.post("/agents/workflows/unknown/runs",headers=RUNNER,json={
        "question":"test question"}).status_code==404
    paused=start(client).json()
    rejected=client.post("/agents/workflows/runs/"+paused["run_id"]+"/resume",
        headers=RUNNER,json={"expected_revision":paused["revision"],"approved":False})
    assert rejected.status_code==200
    assert rejected.json()["status"]=="stopped"
    assert "finalize" not in rejected.json()["state"]["outputs"]
    assert client.get("/agents/workflows/runs/"+paused["run_id"],headers=FOREIGN).status_code==404


def test_workflow_unconfigured_router_and_no_memory_still_operational(client):
    response=start(client)
    assert response.status_code==201
    assert response.json()["status"]=="waiting_for_approval"


def test_workflow_approval_rejects_coerced_decisions_without_advancing(client):
    paused = start(client).json()
    run_path = "/agents/workflows/runs/" + paused["run_id"]
    decision = {"expected_revision": paused["revision"], "approved": True}
    malformed = [{**decision, "approved": value} for value in (1, 0, "yes", "false", None)]
    malformed += [{**decision, "expected_revision": value}
                  for value in (str(paused["revision"]), float(paused["revision"]), True)]
    for body in malformed:
        rejected = client.post(run_path + "/resume", headers=RUNNER, json=body)
        assert rejected.status_code == 422, rejected.text
        assert client.get(run_path, headers=RUNNER).json() == paused

    completed = client.post(run_path + "/resume", headers=RUNNER, json=decision)
    assert completed.status_code == 200, completed.text
    assert completed.json()["state"]["outputs"]["human_approval"]["human_reviewed"] is True


def test_workflow_recovery_requires_integer_revision_before_claiming_run(client):
    paused = start(client).json()
    state = client.app.state.agent_engine.state
    cp = state.get("tenant-a", paused["run_id"])
    interrupted = state.update(
        "tenant-a", cp.run_id, expected_revision=cp.revision,
        fingerprint=cp.fingerprint, status="needs_resume", state=cp.state,
    ).public()
    run_path = "/agents/workflows/runs/" + cp.run_id
    for value in (str(interrupted["revision"]), float(interrupted["revision"]), True):
        response = client.post(run_path + "/recover", headers=RUNNER,
                               json={"expected_revision": value})
        assert response.status_code == 422, response.text
        assert client.get(run_path, headers=RUNNER).json() == interrupted

    resumed = client.post(run_path + "/recover", headers=RUNNER,
                          json={"expected_revision": interrupted["revision"]})
    assert resumed.status_code == 200, resumed.text
    assert resumed.json()["status"] == "waiting_for_approval"


def test_workflow_run_with_disabled_feature_does_not_access_sources(tmp_path,monkeypatch):
    from rags.features import FeatureFlags
    for name in tuple(os.environ):
        if name.startswith(("RAGS_ROUTER_","RAGS_AGENT_","RAGS_MEMORY_")):
            monkeypatch.delenv(name,raising=False)
    monkeypatch.setenv("RAGS_ENV","test")
    monkeypatch.setenv("RAGS_ROUTER_ENABLED", "true")
    monkeypatch.setenv("RAGS_AGENT_ENABLED", "true")
    cfg=replace(Settings.from_env(),db_path=str(tmp_path/"denied.db"),
        keys={"operator":"tenant-a"},key_roles={"operator":("admin",)},
        features=FeatureFlags((("query",False),)))
    with TestClient(create_app(cfg)) as api:
        response=api.post("/agents/workflows/evidence_review/runs",
            headers={"Authorization":"Bearer operator"},json={"question":"What is the policy?"})
        assert response.status_code==403
        assert response.json()["detail"]["code"]=="feature_disabled"
