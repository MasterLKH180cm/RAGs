"""Bounded agent tool loop, explicit approval, memory and privilege regression."""
from dataclasses import replace
from collections import deque

import pytest
from pydantic import BaseModel, ConfigDict, Field

from rags.agents.config import AgentPlatformConfig
from rags.agents.engine import AgentEngine, ToolRegistry, ToolSpec
from rags.agents.memory import scoped_session
from rags.agents.router import ModelReply, ToolCall
from rags.agents.state import StateStore
from rags.auth import Principal, PERMISSIONS
from rags.features import FeatureFlags
from rags.registry import CapabilityError
from rags.storage import Store

pytestmark = pytest.mark.integration


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")
    amount: int = Field(ge=0, le=10)


class FakeRouter:
    def __init__(self, *messages):
        self.replies = deque(messages)
        self.calls = []

    def complete(self, *, model, messages, tools, timeout_seconds=None):
        self.calls.append((model, messages, tools))
        if not self.replies:
            raise AssertionError("Agent exceeded scripted model responses")
        return self.replies.popleft()


def reply(content="", *calls):
    return ModelReply(content=content, tool_calls=list(calls),
                      finish_reason="tool_calls" if calls else "stop", model="openai/mock")


def call(name, arguments=None, identifier="call_1"):
    return ToolCall(id=identifier, name=name, arguments=arguments or {})


@pytest.fixture
def components(tmp_path):
    store = Store(str(tmp_path / "engine.db"))
    state = StateStore(store)
    config = AgentPlatformConfig(
        router_enabled=True, agent_enabled=True, memory_enabled=True,
        gateway_url="https://gateway.example/v1", gateway_key="hidden",
        models={"main": "openai/mock"}, max_steps=5, max_tool_calls=4,
    )
    return state, config


def principal(tenant="tenant-a", *, subject="operator-a", permissions=PERMISSIONS):
    return Principal(tenant, subject, frozenset(permissions), "api_key")


def engine(state, config, router, registry=None, features=None):
    return AgentEngine(config=config, router=router, state=state,
                       features=features or FeatureFlags(),
                       registry=registry)


def test_agent_completes_without_tool_and_persists_conversation(components):
    state, config = components
    bot = FakeRouter(reply("The final answer"))
    service = engine(state, config, bot)
    cp = service.start(principal(), goal="What is the retention policy?",
                       model="main", session_id="conversation1")
    assert cp.status == "completed" and cp.state["answer"] == "The final answer"
    assert len(bot.calls) == 1
    assert service.history.last("tenant-a", scoped_session(principal().subject, "conversation1")) == [
        {"role": "user", "content": "What is the retention policy?"},
        {"role": "assistant", "content": "The final answer"},
    ]
    assert service.history.last("tenant-b", scoped_session(principal().subject, "conversation1")) == []
    with pytest.raises(CapabilityError) as foreign:
        service.get(principal("tenant-b"), cp.run_id)
    assert foreign.value.status_code == 404
    with pytest.raises(CapabilityError):
        service.get(principal("tenant-a", subject="other-operator"), cp.run_id)


def test_agent_approval_gate_happens_before_tool_and_resumes_exactly_once(components):
    state, config = components
    invoked = []
    registry = ToolRegistry()
    registry.register(ToolSpec(
        "record", Args,
        handler=lambda tenant, session, arg: (
            invoked.append((tenant, session, arg.amount)) or {"saved": arg.amount}
        ),
        permission="rags:memory", approval_required=True,
    ))
    bot = FakeRouter(reply("", call("record", {"amount": 3})), reply("Approved and saved"))
    service = engine(state, config, bot, registry)
    waiting = service.start(principal(), goal="Record three consented observations",
                            model="main", session_id="conversation")
    assert waiting.status == "waiting_for_approval" and invoked == []
    assert service.summary(waiting)["pending_approval"]["name"] == "record"
    resumed = service.resume(principal(), waiting.run_id,
                             expected_revision=waiting.revision, approved=True)
    assert resumed.status == "completed"
    assert invoked == [("tenant-a", scoped_session(principal().subject, "conversation"), 3)]
    assert bot.calls[-1][1][-1]["role"] == "tool"
    assert bot.calls[-1][1][-1]["tool_call_id"] == "call_1"
    with pytest.raises(CapabilityError):
        service.resume(principal(), waiting.run_id,
                       expected_revision=waiting.revision, approved=True)
    assert invoked == [("tenant-a", "conversation", 3)]


def test_agent_human_denial_preserves_tool_trace_without_executing(components):
    state, config = components
    invoked = []
    registry = ToolRegistry()
    registry.register(ToolSpec("danger", Args,
                               lambda tenant, session, args: invoked.append(1),
                               permission="rags:memory", approval_required=True))
    bot = FakeRouter(reply("", call("danger", {"amount": 1})), reply("Action declined"))
    service = engine(state, config, bot, registry)
    waiting = service.start(principal(), goal="Should this be executed?",
                            model="main", session_id="session")
    final = service.resume(principal(), waiting.run_id, expected_revision=waiting.revision,
                           approved=False, feedback="Not permitted")
    assert final.status == "completed" and invoked == []
    assert "Not permitted" in bot.calls[-1][1][-1]["content"]


def test_agent_rejects_unknown_tools_before_executing_any(components):
    state, config = components
    bot = FakeRouter(reply("", call("python_exec", {"command": "rm -rf /"})))
    cp = engine(state, config, bot, ToolRegistry()).start(
        principal(), goal="Try to execute an unsafe tool", model="main", session_id="session")
    assert cp.status == "failed" and cp.state["stop_reason"] == "unauthorized_tool"


def test_agent_rejects_invalid_arguments_and_does_not_execute(components):
    state, config = components
    actions = []
    registry = ToolRegistry()
    registry.register(ToolSpec("safe", Args,
                               lambda tenant, session, args: actions.append(args.amount),
                               permission="rags:memory"))
    bot = FakeRouter(reply("", call("safe", {"amount": 999})))
    cp = engine(state, config, bot, registry).start(
        principal(), goal="Use a safe tool correctly", model="main", session_id="session")
    assert cp.status == "failed" and cp.state["stop_reason"] == "invalid_tool_arguments"
    assert actions == []


def test_agent_iteration_budget_and_repeated_call_guard(components):
    state, config = components
    registry = ToolRegistry()
    registry.register(ToolSpec("echo", Args,
                               lambda tenant, session, args: {"amount": args.amount},
                               permission="rags:query"))
    bot = FakeRouter(*(reply("", call("echo", {"amount": 1}, identifier="c" + str(i)))
                       for i in range(4)))
    cp = engine(state, config, bot, registry).start(
        principal(), goal="Repeat the same tool request", model="main", session_id="session")
    assert cp.status == "stopped" and cp.state["stop_reason"] == "repeated_tool_loop"
    assert cp.state["calls"] == 2


def test_agent_rejects_privileged_request_without_scopes(components):
    state, config = components
    bot = FakeRouter(reply("should not be called"))
    service = engine(state, config, bot)
    with pytest.raises(Exception) as unauthorized:
        service.start(principal(permissions={"rags:read", "rags:query"}),
                      goal="Find all documents", model="main", session_id="session")
    assert unauthorized.value.status_code == 403 and bot.calls == []


def test_agent_feature_flag_blocks_model_before_network(components):
    state, config = components
    bot = FakeRouter(reply("should not be called"))
    service = engine(state, config, bot, features=FeatureFlags((("model_generation", False),)))
    with pytest.raises(CapabilityError) as disabled:
        service.start(principal(), goal="Find docs", model="main", session_id="session")
    assert disabled.value.code == "feature_disabled" and bot.calls == []


def test_agent_memory_flag_prevents_persistent_conversation(components):
    state, config = components
    bot = FakeRouter(reply("Output without memory"))
    service = engine(state, replace(config, memory_enabled=False), bot)
    service.start(principal(), goal="Summarize this", model="main", session_id="session")
    assert service.history.last("tenant-a", scoped_session(principal().subject, "session")) == []


def test_agent_tool_registry_enforces_pydantic_extra_forbid():
    class WeakArgs(BaseModel):
        amount: int
    with pytest.raises(ValueError, match="forbid"):
        ToolSpec("weak", WeakArgs, lambda *_: None, permission="rags:query")


def test_memory_scope_required_to_read_or_store_agent_history(components):
    state, config = components
    from rags.agents.memory import ConversationStore
    memory = ConversationStore(state)
    memory.append("tenant-a", scoped_session(principal().subject, "shared"), "user", "Private prior question")
    fake = FakeRouter(reply("Generic answer"))
    service = engine(state, config, fake)
    permissions = {"rags:agent", "rags:generate", "rags:query"}
    result = service.start(principal(permissions=permissions), goal="Respond without history",
                           model="main", session_id="shared")
    assert result.status == "completed"
    assert "Private prior question" not in repr(fake.calls)
    assert len(memory.last("tenant-a", scoped_session(principal().subject, "shared"))) == 1


def test_gateway_hallucinated_citation_does_not_complete_agent(components):
    state, config = components
    registry = ToolRegistry()
    registry.register(ToolSpec("knowledge_search", Args,
        lambda *_: {"citations": [{"chunk_id": "authorized-id", "text": "source"}]},
        permission="rags:query"))
    fake = FakeRouter(reply("", call("knowledge_search", {"amount": 1})),
                      reply("Fabricated evidence [not-authorized]"))
    result = engine(state, config, fake, registry).start(
        principal(), goal="Cite the authorized evidence", model="main", session_id="evidence")
    assert result.status == "failed" and result.state["stop_reason"] == "invalid_citation"


def test_same_tenant_different_subjects_cannot_share_conversation_or_observations(components):
    from rags.agents.memory import ConversationStore, ObservationStore
    state, _config = components
    alice = scoped_session("alice", "shared")
    bob = scoped_session("bob", "shared")
    assert alice != bob and "alice" not in alice and "shared" not in alice
    history = ConversationStore(state)
    history.append("tenant-a", alice, "user", "Alice only")
    assert history.last("tenant-a", bob) == []
    observations = ObservationStore(state)
    observations.observe("tenant-a", alice, fact="Alice preference", source_id="consent")
    assert observations.recent("tenant-a", bob) == []
