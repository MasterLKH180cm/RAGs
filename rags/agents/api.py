"""Opt-in tenant-authorized Agent Platform HTTP interface.

Registered operator tools only. The HTTP client cannot submit a Python tool,
model gateway URL, executable workflow or arbitrary tenant identifier.
"""
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from rags.agents.config import AgentPlatformConfig
from rags.agents.engine import AgentEngine
from rags.agents.router import ModelRouter
from rags.agents.state import StateStore
from rags.agents.templates import catalog as workflow_catalog, get_template, make_evidence_review
from rags.registry import CapabilityError


class RunIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    goal: str = Field(min_length=2, max_length=2000)
    model: str = Field(min_length=1, max_length=64)
    session_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")


class ResumeIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: int = Field(ge=0)
    approved: bool
    feedback: str = Field(default="", max_length=500)


class ObserveIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fact: str = Field(min_length=1, max_length=600)
    source_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
    confidence: float = Field(default=1., ge=0., le=1.)


class WorkflowRunIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=2, max_length=2000)
    mode: str = Field(default="sparse", pattern="^(sparse|dense|hybrid)$")
    top_k: int = Field(default=5, ge=1, le=10)


class WorkflowRecoverIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: int = Field(ge=0)


def mount_agent_api(app, *, store, rag, config: AgentPlatformConfig, auth,
                    features) -> AgentEngine | None:
    if not config.agent_enabled:
        return None
    state = StateStore(store)
    engine = AgentEngine(config=config, router=ModelRouter(config),
                         state=state, features=features, rag=rag)
    app.state.agent_engine = engine
    api = APIRouter(prefix="/agents", tags=["Agents"])

    def require(request: Request, *required: str):
        principal = getattr(request.state, "principal", None)
        if principal is None:
            raise HTTPException(401, detail={"code": "authentication_required"})
        principal.require(*required)
        features.require("agents")
        return principal

    @api.get("/capabilities")
    def capabilities(request: Request, tenant: Annotated[str, Depends(auth)]):
        principal = require(request, "rags:agent")
        return {"router_enabled": config.router_enabled,
                "router_ready": config.router_ready,
                "configuration_required": not config.router_ready,
                "models": sorted(config.models),
                "tools": sorted(engine.tools.allowed(principal)),
                "max_steps": config.max_steps, "max_tool_calls": config.max_tool_calls,
                "memory_enabled": config.memory_enabled,
                "state_backend": store.storage_backend}

    @api.post("/runs", status_code=201)
    def start_run(request: Request, data: RunIn,
                  tenant: Annotated[str, Depends(auth)]):
        principal = require(request, "rags:agent", "rags:generate")
        if not config.router_ready:
            raise CapabilityError("router_unconfigured",
                                  "Configure the model gateway before running an agent", 503)
        result = engine.start(principal, goal=data.goal, model=data.model,
                              session_id=data.session_id)
        return engine.summary(result)

    @api.get("/runs/{run_id}")
    def get_run(request: Request, run_id: str,
                tenant: Annotated[str, Depends(auth)]):
        principal = require(request, "rags:agent")
        return engine.summary(engine.get(principal, run_id))

    @api.post("/runs/{run_id}/resume")
    def resume_run(request: Request, run_id: str, data: ResumeIn,
                   tenant: Annotated[str, Depends(auth)]):
        principal = require(request, "rags:agent", "rags:generate")
        if not config.router_ready:
            raise CapabilityError("router_unconfigured",
                                  "Configure the model gateway before resuming an agent", 503)
        result = engine.resume(principal, run_id, expected_revision=data.expected_revision,
                               approved=data.approved, feedback=data.feedback)
        return engine.summary(result)

    @api.get("/sessions/{session_id}/messages")
    def history(request: Request, session_id: str,
                tenant: Annotated[str, Depends(auth)]):
        principal = require(request, "rags:agent", "rags:memory")
        if not config.memory_enabled:
            raise CapabilityError("feature_disabled", "Memory is disabled", 403)
        return {"messages": engine.history.last(principal.tenant, session_id,
                                                 limit=config.max_history_turns)}

    @api.get("/sessions/{session_id}/observations")
    def observations(request: Request, session_id: str,
                     tenant: Annotated[str, Depends(auth)]):
        principal = require(request, "rags:agent", "rags:memory")
        if not config.memory_enabled:
            raise CapabilityError("feature_disabled", "Memory is disabled", 403)
        return {"observations": engine.observations.recent(principal.tenant, session_id)}

    @api.post("/sessions/{session_id}/observations", status_code=201)
    def save_observation(request: Request, session_id: str, data: ObserveIn,
                         tenant: Annotated[str, Depends(auth)]):
        principal = require(request, "rags:agent", "rags:memory")
        if not config.memory_enabled:
            raise CapabilityError("feature_disabled", "Memory is disabled", 403)
        digest = engine.observations.observe(principal.tenant, session_id,
                                             fact=data.fact, source_id=data.source_id,
                                             confidence=data.confidence)
        return {"observation_id": digest}

    @api.get("/workflows")
    def list_workflows(request: Request, tenant: Annotated[str, Depends(auth)]):
        require(request, "rags:agent", "rags:query")
        return {"templates": workflow_catalog()}

    def workflow_identity(principal, run_id: str):
        cp = state.get(principal.tenant, run_id)
        if cp is None or cp.kind != "workflow" or cp.state["input"].get("owner_subject") != principal.subject:
            raise CapabilityError("not_found", "Workflow run not found", 404)
        if cp.fingerprint != make_evidence_review(rag, principal, features).fingerprint:
            raise CapabilityError("definition_mismatch", "Workflow definition changed", 409)
        return cp

    @api.post("/workflows/{workflow_name}/runs", status_code=201)
    def start_workflow(request: Request, workflow_name: str, data: WorkflowRunIn,
                       tenant: Annotated[str, Depends(auth)]):
        principal = require(request, "rags:agent", "rags:query")
        features.require("query")
        get_template(workflow_name)
        graph = make_evidence_review(rag, principal, features)
        cp = graph.start(state, principal.tenant, {
            "question": data.question, "mode": data.mode, "top_k": data.top_k,
            "owner_subject": principal.subject,
        })
        return cp.public()

    @api.get("/workflows/runs/{run_id}")
    def get_workflow(request: Request, run_id: str,
                     tenant: Annotated[str, Depends(auth)]):
        principal = require(request, "rags:agent", "rags:query")
        cp = workflow_identity(principal, run_id)
        return cp.public()

    @api.post("/workflows/runs/{run_id}/resume")
    def approve_workflow(request: Request, run_id: str, data: ResumeIn,
                         tenant: Annotated[str, Depends(auth)]):
        principal = require(request, "rags:agent", "rags:query")
        cp = workflow_identity(principal, run_id)
        graph = make_evidence_review(rag, principal, features)
        return graph.resume(state, principal.tenant, run_id,
                            expected_revision=data.expected_revision,
                            approved=data.approved).public()

    @api.post("/workflows/runs/{run_id}/recover")
    def recover_workflow(request: Request, run_id: str, data: WorkflowRecoverIn,
                         tenant: Annotated[str, Depends(auth)]):
        principal = require(request, "rags:agent", "rags:query")
        cp = workflow_identity(principal, run_id)
        graph = make_evidence_review(rag, principal, features)
        return graph.recover(state, principal.tenant, run_id,
                             expected_revision=data.expected_revision).public()

    app.include_router(api)
    return engine
