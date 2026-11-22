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
        return {"models": sorted(config.models),
                "tools": sorted(engine.tools.allowed(principal)),
                "max_steps": config.max_steps, "max_tool_calls": config.max_tool_calls,
                "memory_enabled": config.memory_enabled,
                "state_backend": store.storage_backend}

    @api.post("/runs", status_code=201)
    def start_run(request: Request, data: RunIn,
                  tenant: Annotated[str, Depends(auth)]):
        principal = require(request, "rags:agent", "rags:generate")
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

    app.include_router(api)
    return engine
