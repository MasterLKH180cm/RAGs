"""Pure, side-effect-free Agent Platform contracts and fail-closed validation."""
from dataclasses import replace

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rags.agents.config import AgentPlatformConfig
from rags.agents.engine import RememberArgs, SearchArgs, ToolRegistry, ToolSpec
from rags.agents.router import Message, ModelRouter, ToolCall
from rags.agents.workflow import Workflow
from rags.registry import CapabilityError


pytestmark = pytest.mark.unit


class StrictArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int = Field(ge=0, le=10)


@pytest.mark.parametrize("payload", [
    {"question": ""},
    {"question": "x"},
    {"question": "x" * 2001},
    {"question": "valid?", "mode": "exec"},
    {"question": "valid?", "top_k": 0},
    {"question": "valid?", "top_k": 6},
    {"question": "valid?", "unknown": "python-code"},
])
def test_search_tool_contract_rejects_invalid_inputs(payload):
    with pytest.raises(ValidationError):
        SearchArgs.model_validate(payload)


@pytest.mark.parametrize("payload", [
    {"fact": "", "source_id": "consent"},
    {"fact": "valid", "source_id": ""},
    {"fact": "valid", "source_id": "../../foreign"},
    {"fact": "valid", "source_id": "good", "extra": "unsupported"},
    {"fact": "x" * 601, "source_id": "consent"},
])
def test_observation_tool_contract_rejects_unapproved_shapes(payload):
    with pytest.raises(ValidationError):
        RememberArgs.model_validate(payload)


@pytest.mark.parametrize("payload", [
    {"role": "root", "content": "steal secrets"},
    {"role": "user", "content": "x" * 30001},
    {"role": "user", "content": "hello", "tenant": "other-tenant"},
])
def test_router_messages_have_closed_roles_and_fields(payload):
    with pytest.raises(ValidationError):
        Message.model_validate(payload)


@pytest.mark.parametrize("payload", [
    {"id": "bad/path", "name": "lookup", "arguments": {}},
    {"id": "call_1", "name": "os.system", "arguments": {}},
    {"id": "call_1", "name": "lookup", "arguments": []},
    {"id": "call_1", "name": "lookup", "arguments": {}, "extra": "forbidden"},
])
def test_tool_calls_have_strict_normalized_envelope(payload):
    with pytest.raises(ValidationError):
        ToolCall.model_validate(payload)


def test_workflow_builder_is_immutable_and_defaults_to_non_idempotent_branches():
    source = Workflow("case_review")
    first = source.then("load", lambda ctx: True)
    branch = first.branch(
        "route", lambda ctx: True,
        if_true=lambda ctx: "ok", if_false=lambda ctx: "review",
    )
    assert source._nodes == ()
    assert len(first._nodes) == 1
    assert len(branch._nodes) == 2
    assert branch._nodes[-1].idempotent is False
    assert branch.fingerprint != first.fingerprint
    declared = first.branch(
        "route", lambda ctx: True,
        if_true=lambda ctx: "ok", if_false=lambda ctx: "review",
        idempotent=True,
    )
    assert declared._nodes[-1].idempotent is True
    assert declared.fingerprint != branch.fingerprint


def test_parallel_and_approval_steps_require_explicit_replay_safety():
    with pytest.raises(ValueError, match="idempotency"):
        Workflow("safe").then("publish", lambda ctx: True, approval=True)
    with pytest.raises(ValueError, match="idempotent"):
        Workflow("safe").parallel("parallel", {"a": lambda ctx: 1, "b": lambda ctx: 2})
    registered = Workflow("safe").parallel(
        "parallel", {"a": lambda ctx: 1, "b": lambda ctx: 2}, idempotent=True,
    )
    assert registered._nodes[0].idempotent is True
    assert tuple(name for name, _ in registered._nodes[0].branches) == ("a", "b")


def test_tool_registry_enforces_unique_operator_registration_and_capacity():
    registry = ToolRegistry()
    for idx in range(12):
        registry.register(ToolSpec(
            f"safe_{idx}", StrictArgs,
            lambda tenant, session, args: {"value": args.value},
            permission="rags:query",
        ))
    with pytest.raises(ValueError, match="Duplicate"):
        registry.register(ToolSpec(
            "safe_0", StrictArgs, lambda *_: {}, permission="rags:query",
        ))
    with pytest.raises(ValueError, match="too many"):
        registry.register(ToolSpec(
            "safe_extra", StrictArgs, lambda *_: {}, permission="rags:query",
        ))


def test_router_default_on_requires_actual_gateway_before_any_inference(monkeypatch):
    config = AgentPlatformConfig()
    assert config.router_enabled and config.agent_enabled and config.memory_enabled
    assert config.router_ready is False
    router = ModelRouter(config)
    monkeypatch.setattr(router, "_client", lambda *_args: pytest.fail("No network on first boot"))
    with pytest.raises(CapabilityError) as failure:
        router.complete(model="unknown", messages=[{"role": "user", "content": "test"}])
    assert failure.value.code == "router_unconfigured"
    with pytest.raises(ValueError):
        replace(config, agent_enabled=True, router_enabled=False)


def test_idempotent_flag_cannot_be_a_string_or_integer():
    for unsafe in ("true", 1, None):
        with pytest.raises(ValueError, match="idempotency"):
            Workflow("reject").branch(
                "decision", lambda ctx: True,
                if_true=lambda ctx: 1, if_false=lambda ctx: 0,
                idempotent=unsafe,
            )


def test_context_deadline_caps_model_network_timeout():
    import time
    from rags.agents.deadline import execution_deadline, remaining_timeout

    assert remaining_timeout(30) == 30.0
    token = execution_deadline.set(time.monotonic() + 1.0)
    try:
        budget = remaining_timeout(30)
        assert 0.05 <= budget <= 1.0
        assert 0.05 <= remaining_timeout(0.5) <= 0.5
    finally:
        execution_deadline.reset(token)
    token = execution_deadline.set(time.monotonic() - 0.01)
    try:
        with pytest.raises(CapabilityError) as exhausted:
            remaining_timeout(30)
        assert exhausted.value.code == "tool_timeout"
    finally:
        execution_deadline.reset(token)


def test_timed_tool_requires_explicit_readonly_and_idempotency_declaration():
    with pytest.raises(ValueError, match="read_only"):
        ToolSpec("read", StrictArgs, lambda *_: {}, permission="rags:query", read_only=1)
    with pytest.raises(ValueError, match="idempotent"):
        ToolSpec("read", StrictArgs, lambda *_: {}, permission="rags:query",
                 idempotent=False, read_only=True)
    tool = ToolSpec("read", StrictArgs, lambda *_: {}, permission="rags:query",
                    read_only=True, idempotent=True)
    assert tool.read_only is True


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
def test_tool_and_workflow_replay_safety_require_boolean_declarations(value):
    for field in ("approval_required", "idempotent"):
        with pytest.raises(ValueError, match="explicit booleans"):
            ToolSpec("write", StrictArgs, lambda *_: {}, permission="rags:query", **{field: value})
    for field in ("approval", "idempotent"):
        with pytest.raises(ValueError, match="explicit booleans"):
            Workflow("safety").then("write", lambda _: None, **{field: value})
    with pytest.raises(ValueError, match="idempotent"):
        Workflow("safety").parallel("reads", {"one": lambda _: 1, "two": lambda _: 2},
                                     idempotent=value)
