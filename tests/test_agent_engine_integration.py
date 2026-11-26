"""Bounded agent tool loop, explicit approval, memory and privilege regression."""
from dataclasses import replace
from collections import deque
import json
import time

import httpx
import pytest
from pydantic import BaseModel, ConfigDict, Field

from rags.agents.config import AgentPlatformConfig
from rags.agents.engine import AgentEngine, ToolRegistry, ToolSpec
from rags.agents.memory import scoped_session
from rags.agents.router import MAX_RESPONSE_BYTES, ModelReply, ModelRouter, ToolCall
from rags.agents.state import MAX_STATE_BYTES, StateStore
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


def test_valid_gateway_reply_exceeding_checkpoint_budget_stops_before_tools(components, monkeypatch):
    state, config = components
    config = replace(config, max_tool_calls=16)
    router = ModelRouter(config)
    calls = [{"id": f"call_{index}", "type": "function", "function": {
        "name": "knowledge_search",
        "arguments": json.dumps({"question": str(index) + "界" * 1999, "top_k": 5}),
    }} for index in range(8)]
    payload = {"choices": [{"message": {
        "role": "assistant", "content": "界" * 30000,
        "reasoning_content": "界" * 30000, "tool_calls": calls,
    }, "finish_reason": "tool_calls"}]}
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    assert len(raw) < MAX_RESPONSE_BYTES
    assert all(len(item["function"]["arguments"].encode("utf-8")) < 16384 for item in calls)
    monkeypatch.setattr(router, "_post", lambda *_: httpx.Response(200, content=raw))
    service = engine(state, config, router)
    monkeypatch.setattr(service, "_invoke_tool",
                        lambda *_: pytest.fail("An unsaved tool batch must never execute"))

    result = service.start(principal(), goal="Search the source facts", model="main", session_id="budget")
    assert result.status == "stopped"
    assert result.state["stop_reason"] == "checkpoint_budget"
    assert result.state["pending"] == [] and result.state["calls"] == 0
    assert len(result.state["messages"]) == 2
    assert state.get("tenant-a", result.run_id) == result
    assert service.history.last("tenant-a", scoped_session(principal().subject, "budget")) == []


def test_tool_result_budget_stop_does_not_continue_or_replay_pending_actions(components):
    state, config = components
    executed = []
    registry = ToolRegistry()
    registry.register(ToolSpec(
        "action", Args,
        lambda _tenant, _session, args: executed.append(args.amount) or {"details": "x" * 8000},
        permission="rags:query",
    ))
    service = engine(state, config, FakeRouter(), registry)
    saved = {"subject": principal().subject, "session_id": "budget", "model": "main",
             "messages": [], "pending": [call("action", {"amount": 1}, "first").model_dump(),
                                          call("action", {"amount": 2}, "second").model_dump()],
             "approval": None, "evidence_ids": [], "calls": 0, "stop_reason": None,
             "padding": ""}
    encoded = json.dumps(saved, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    saved["padding"] = "x" * (MAX_STATE_BYTES - len(encoded) - 100)
    cp = state.create("tenant-a", kind="agent", fingerprint=service._fingerprint, state=saved)

    result = service._pending(cp, principal(), deadline=time.monotonic() + 2)
    assert result.status == "stopped" and result.state["stop_reason"] == "checkpoint_budget"
    assert executed == [1]
    assert result.state["pending"] == saved["pending"]
    assert state.get("tenant-a", cp.run_id) == result


@pytest.mark.parametrize("spare_bytes", [100, 0])
def test_approval_feedback_budget_is_checked_before_an_approved_tool_runs(components, spare_bytes):
    state, config = components
    executed = []
    registry = ToolRegistry()
    registry.register(ToolSpec("record", Args, lambda *_: executed.append("written"),
                               permission="rags:memory", approval_required=True))
    service = engine(state, config, FakeRouter(reply("", call("record", {"amount": 1}))), registry)
    paused = service.start(principal(), goal="Record the approved value", model="main", session_id="resume-budget")
    saved = {**paused.state, "padding": ""}
    encoded = json.dumps(saved, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    saved["padding"] = "x" * (MAX_STATE_BYTES - len(encoded) - spare_bytes)
    paused = state.update("tenant-a", paused.run_id, expected_revision=paused.revision,
                          fingerprint=paused.fingerprint, status="waiting_for_approval", state=saved)

    stopped = service.resume(principal(), paused.run_id, expected_revision=paused.revision,
                             approved=True, feedback="界" * 500)
    assert stopped.status == "stopped"
    assert stopped.state["stop_reason"] == ("checkpoint_budget" if spare_bytes else None)
    if not spare_bytes:
        # Legacy exact-cap state has no room even for the reason. Preserve it
        # byte-for-byte while still committing a terminal status outside JSON.
        assert stopped.state == saved
    assert stopped.state["approval"] is None and executed == []
    assert state.get("tenant-a", paused.run_id) == stopped


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
    assert invoked == [("tenant-a", scoped_session(principal().subject, "conversation"), 3)]


def test_agent_sdk_rejects_coercible_revisions_before_approved_tool_execution(components):
    state, config = components
    executed = []
    registry = ToolRegistry()
    registry.register(ToolSpec("record", Args, lambda *_: executed.append("written") or {},
                               permission="rags:memory", approval_required=True))
    service = engine(state, config, FakeRouter(reply("", call("record", {"amount": 1})), reply("Done")), registry)
    paused = service.start(principal(), goal="Record the approved value", model="main", session_id="sdk-revision")
    for invalid in (True, False, float(paused.revision), str(paused.revision), -1):
        with pytest.raises(ValueError, match="Invalid checkpoint revision"):
            service.resume(principal(), paused.run_id, expected_revision=invalid, approved=True)
        assert state.get("tenant-a", paused.run_id) == paused
        assert executed == []
    completed = service.resume(principal(), paused.run_id, expected_revision=paused.revision, approved=True)
    assert completed.status == "completed" and executed == ["written"]


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


def test_long_utf8_search_excerpts_fit_tool_budget_without_losing_all_evidence(components):
    state, config = components
    class CJKRag:
        def query(self, tenant, question, mode, *, top_k, generate):
            assert tenant == "tenant-a" and generate is False
            return {"outcome": "supported", "citations": [
                {"chunk_id": "chunk-" + str(i), "source": "政策文檔",
                 "text": "漢字內容" * 300}
                for i in range(5)]}
    bot = FakeRouter(
        reply("", call("knowledge_search", {"question": "Find policy"})),
        reply("This is supported by [chunk-0]"))
    service = AgentEngine(config=config, router=bot, state=state,
                          features=FeatureFlags(), rag=CJKRag())
    cp = service.start(principal(), goal="Find verified policy evidence",
                       model="main", session_id="long-cjk")
    assert cp.status == "completed", cp.state.get("stop_reason")
    tools = [m for m in cp.state["messages"] if m["role"] == "tool"]
    assert len(tools) == 1
    assert len(tools[0]["content"].encode("utf-8")) <= 8192
    import json
    payload = json.loads(tools[0]["content"])
    assert payload["citations"] and payload["citations"][0]["chunk_id"] == "chunk-0"
    assert "漢字" in payload["citations"][0]["text"]


def test_all_delivered_citations_remain_valid_across_multiple_search_batches(components):
    """9 tool calls can deliver 45 distinct citations, exceeding the old cap 40.

    Each reference was genuinely returned by a tenant-authorized tool. A
    hallucinated reference remains rejected by the separate negative test.
    """
    state, config = components
    config = replace(config, max_tool_calls=12, max_steps=5)
    registry = ToolRegistry()

    def search(_tenant, _session, args):
        return {"citations": [
            {"chunk_id": f"chunk-{args.amount * 5 + i}",
             "source": "authorized-fixture", "text": "Verified source"}
            for i in range(5)
        ]}

    registry.register(ToolSpec("knowledge_search", Args, search,
                               permission="rags:query"))
    batch = [call("knowledge_search", {"amount": n}, identifier=f"search_{n}")
             for n in range(8)]
    router = FakeRouter(
        reply("", *batch),
        reply("", call("knowledge_search", {"amount": 8}, identifier="search_8")),
        reply("Verified response [chunk-44]"),
    )
    finished = engine(state, config, router, registry).start(
        principal(), goal="Inspect all authorized references across tool batches",
        model="main", session_id="many-citations",
    )
    assert finished.status == "completed", finished.state.get("stop_reason")
    assert finished.state["answer"] == "Verified response [chunk-44]"
    assert finished.state["calls"] == 9
    assert len(finished.state["evidence_ids"]) == 45
    assert "chunk-44" in finished.state["evidence_ids"]


def test_capability_error_from_tool_persists_terminal_checkpoint(components):
    state, config = components
    router = FakeRouter(
        reply("", call("knowledge_search", {"question": "Find the policy"}, "search_1")),
        reply("This answer must never be generated"),
    )
    service = engine(state, config, router,
                     features=FeatureFlags((("query", False),)))
    run = service.start(principal(), goal="Review the query policy",
                        model="main", session_id="query_denied")
    assert run.status == "failed"
    assert run.state["stop_reason"] == "feature_disabled"
    assert run.state["answer"] is None
    stored = StateStore(Store(state.store.path)).get("tenant-a", run.run_id)
    assert stored is not None and stored.status == "failed"
    assert stored.state["stop_reason"] == "feature_disabled"
    assert len(router.calls) == 1
    with pytest.raises(CapabilityError) as invalid:
        service.resume(principal(), run.run_id,
                       expected_revision=run.revision, approved=True)
    assert invalid.value.code == "invalid_state"


def test_capability_error_from_trusted_tool_cannot_leak_exception_details(components):
    state, config = components
    registry = ToolRegistry()

    def failed(_tenant, _session, _args):
        raise CapabilityError("private_api_key_value",
                              "Example private data from upstream", 503)

    registry.register(ToolSpec("custom_lookup", Args, failed,
                               permission="rags:query"))
    router = FakeRouter(reply("", call("custom_lookup", {"amount": 1})))
    run = engine(state, config, router, registry).start(
        principal(), goal="Check external source",
        model="main", session_id="protected_failure",
    )
    assert run.status == "failed"
    assert run.state["stop_reason"] == "tool_execution_failed"
    saved = state.get("tenant-a", run.run_id)
    assert saved is not None
    assert "private_api_key_value" not in repr(saved.state)
    assert "Example private data" not in repr(saved.state)


def test_readonly_tool_deadline_returns_failed_checkpoint_without_waiting_for_slow_io(components):
    import threading
    import time

    state, config = components
    entered = threading.Event()
    release = threading.Event()

    def stalled(_tenant, _session, _args):
        entered.set()
        release.wait(timeout=2.0)
        return {"ok": True}

    tools = ToolRegistry()
    tools.register(ToolSpec("slow_read", Args, stalled,
                            permission="rags:query", read_only=True))
    service = engine(state, config, FakeRouter(), tools)
    request = call("slow_read", {"amount": 1}, identifier="request_1")
    checkpoint = state.create(
        "tenant-a", kind="agent", fingerprint=service._fingerprint,
        state={"subject": principal().subject, "session_id": "timeout",
               "messages": [], "pending": [request.model_dump()],
               "approval": None, "calls": 0, "steps": 1,
               "seen": [], "evidence_ids": [], "model": "main",
               "goal": "Test timeout", "answer": None, "stop_reason": None},
    )
    started = time.monotonic()
    try:
        result = service._pending(checkpoint, principal(),
                                  deadline=time.monotonic() + 0.10)
        assert entered.wait(timeout=0.1)
        assert time.monotonic() - started < 0.5
        assert result.status == "failed"
        assert result.state["stop_reason"] == "tool_timeout"
        assert result.state["calls"] == 0
    finally:
        release.set()
    stored = state.get("tenant-a", checkpoint.run_id)
    assert stored is not None and stored.status == "failed"
    assert stored.state["messages"] == []


def test_side_effecting_tools_do_not_run_on_unmanaged_background_workers(components):
    import threading
    import time

    state, config = components
    thread_names = []
    tools = ToolRegistry()

    def record(_tenant, _session, args):
        thread_names.append(threading.current_thread().name)
        return {"amount": args.amount}

    tools.register(ToolSpec("write_consented", Args, record,
                            permission="rags:memory", read_only=False))
    service = engine(state, config, FakeRouter(), tools)
    checkpoint = state.create(
        "tenant-a", kind="agent", fingerprint=service._fingerprint,
        state={"subject": principal().subject, "session_id": "write",
               "messages": [], "pending": [call("write_consented", {"amount": 2}).model_dump()],
               "approval": None, "calls": 0, "steps": 1,
               "seen": [], "evidence_ids": [], "model": "main",
               "goal": "Write once", "answer": None, "stop_reason": None},
    )
    updated = service._pending(checkpoint, principal(), deadline=time.monotonic() + 1.0)
    assert updated.status == "running" and updated.state["calls"] == 1
    assert thread_names == [threading.current_thread().name]


def test_verified_citation_survives_next_turn_and_process_restart(components):
    state, config = components
    registry = ToolRegistry()
    registry.register(ToolSpec(
        "knowledge_search", Args,
        lambda *_: {"citations": [{"chunk_id": "chunk_approved", "text": "source evidence"}]},
        permission="rags:query", read_only=True,
    ))
    actor = principal("tenant-a", subject="reader-1")
    first = engine(state, config, FakeRouter(
        reply("", call("knowledge_search", {"amount": 1})),
        reply("Document says yes [chunk_approved]"),
    ), registry)
    completed = first.start(actor, goal="Search for verified documentation",
                            model="main", session_id="shared")
    assert completed.status == "completed"
    assert completed.state["answer"] == "Document says yes [chunk_approved]"

    memory_id = scoped_session(actor.subject, "shared")
    assert first.history.last("tenant-a", memory_id) == [
        {"role": "user", "content": "Search for verified documentation"},
        {"role": "assistant", "content": "Document says yes [chunk_approved]"},
    ]
    last = first.history.last("tenant-a", memory_id, with_evidence=True)
    assert last[-1]["evidence_ids"] == ["chunk_approved"]

    # A fresh application instance must recover trusted provenance from durable
    # history, not from stale agent process memory or a prior request's variables.
    restarted = engine(StateStore(Store(state.store.path)), config,
                       FakeRouter(reply("Document says yes [chunk_approved]")), registry)
    followup = restarted.start(actor, goal="Repeat your last answer exactly",
                               model="main", session_id="shared")
    assert followup.status == "completed"
    assert followup.state["evidence_ids"] == []
    assert followup.state["prior_evidence_ids"] == ["chunk_approved"]
    assert restarted.history.last("tenant-a", memory_id, with_evidence=True)[-1][
        "evidence_ids"] == ["chunk_approved"]


def test_prior_verified_references_are_never_shared_across_subjects_or_tenants(components):
    state, config = components
    registry = ToolRegistry()
    registry.register(ToolSpec(
        "knowledge_search", Args,
        lambda *_: {"citations": [{"chunk_id": "restricted_source", "text": "private text"}]},
        permission="rags:query", read_only=True,
    ))
    owner = principal("tenant-a", subject="author-a")
    result = engine(state, config, FakeRouter(
        reply("", call("knowledge_search", {"amount": 1})),
        reply("Authorized reference [restricted_source]"),
    ), registry).start(owner, goal="Find source for the report",
                      model="main", session_id="shared")
    assert result.status == "completed"

    unauthorized = [
        principal("tenant-a", subject="other-user"),
        principal("tenant-b", subject="author-a"),
        principal("tenant-a", subject="author-a",
                  permissions={"rags:agent", "rags:generate", "rags:query"}),
    ]
    for actor in unauthorized:
        bot = FakeRouter(reply("Authorized reference [restricted_source]"))
        denied = engine(state, config, bot, registry).start(
            actor, goal="Repeat the last citation", model="main", session_id="shared")
        assert denied.status == "failed"
        assert denied.state["stop_reason"] == "invalid_citation"
        assert denied.state.get("prior_evidence_ids") == []


def test_citation_free_followup_after_sourced_answer_remains_valid(components):
    state, config = components
    registry = ToolRegistry()
    registry.register(ToolSpec(
        "knowledge_search", Args,
        lambda *_: {"citations": [{"chunk_id": "source_10", "text": "factual passage"}]},
        permission="rags:query", read_only=True,
    ))
    actor = principal()
    first = engine(state, config, FakeRouter(
        reply("", call("knowledge_search", {"amount": 1})),
        reply("Verified conclusion [source_10]"),
    ), registry).start(actor, goal="Find supporting facts",
                       model="main", session_id="same")
    assert first.status == "completed"
    next_run = engine(state, config, FakeRouter(reply("The requested format is a short paragraph.")),
                      registry).start(actor, goal="Change the formatting, without citing",
                                      model="main", session_id="same")
    assert next_run.status == "completed"
    assert next_run.state["prior_evidence_ids"] == ["source_10"]
    assert next_run.state["answer"] == "The requested format is a short paragraph."


def test_combined_fresh_and_previous_citations_over_eighty_are_persisted(components):
    """The persisted assistant provenance must handle new+historical IDs.

    15 authorized search tools each return five distinct source IDs, and six
    verified IDs come from a previous completed answer: 81 IDs total.
    """
    state, baseline = components
    config = replace(baseline, max_steps=5, max_tool_calls=16)

    class GroupArgs(BaseModel):
        model_config = ConfigDict(extra="forbid")
        index: int = Field(ge=0, le=14)

    def grouped_evidence(_tenant, _session, arguments):
        return {"citations": [
            {"chunk_id": f"fresh_{arguments.index}_{i}", "text": "verified"}
            for i in range(5)
        ]}

    registry = ToolRegistry()
    registry.register(ToolSpec(
        "knowledge_search", GroupArgs, grouped_evidence,
        permission="rags:query", read_only=True,
    ))
    session_id = scoped_session(principal().subject, "mixed_provenance")
    prior = [f"prior_{i}" for i in range(6)]
    memory = engine(state, config, FakeRouter(reply("unused")), registry).history
    memory.append("tenant-a", session_id, "assistant",
                  "Previous response " + " ".join(f"[{item}]" for item in prior),
                  evidence_ids=prior)

    new_sources = [f"fresh_{batch}_{i}" for batch in range(15) for i in range(5)]
    answer = "Authorized evidence " + " ".join(f"[{item}]" for item in prior + new_sources)
    responses = [
        reply("", *(call("knowledge_search", {"index":index}, f"call_{index}")
                    for index in range(8))),
        reply("", *(call("knowledge_search", {"index":index}, f"call_{index}")
                    for index in range(8, 15))),
        reply(answer),
    ]
    result = engine(state, config, FakeRouter(*responses), registry).start(
        principal(), goal="Compile all verified evidence references",
        model="main", session_id="mixed_provenance",
    )
    assert result.status == "completed"
    assert len(result.state["evidence_ids"]) == 75
    assert len(result.state["prior_evidence_ids"]) == 6
    persisted = memory.last("tenant-a", session_id, with_evidence=True)
    final = persisted[-1]
    assert final["role"] == "assistant"
    assert final["content"] == answer
    assert len(final["evidence_ids"]) == 81
    assert set(final["evidence_ids"]) == set(prior + new_sources)


@pytest.mark.parametrize("approved_tool", [True, False])
def test_truncated_tool_response_never_runs_or_requests_approval(components, approved_tool):
    state, config = components
    invocations = []
    tools = ToolRegistry()
    tools.register(ToolSpec("record", Args, lambda *_: invocations.append(True),
                            permission="rags:memory", approval_required=approved_tool))
    partial = ModelReply(content="", tool_calls=[call("record", {"amount": 1})],
                         finish_reason="length", model="thinkingmachines/Inkling-2-Mini")
    result = engine(state, config, FakeRouter(partial), tools).start(
        principal(), goal="Execute only a complete valid request", model="main", session_id="truncated")
    assert result.status == "stopped" and result.state["stop_reason"] == "model_output_limit"
    assert result.state["pending"] == [] and result.state["calls"] == 0
    assert invocations == []


@pytest.mark.parametrize("change", ["schema", "model", "effort", "limits", "idempotency", "runtime"])
def test_approval_cannot_resume_under_changed_execution_contract(components, change):
    state, config = components
    invoked = []
    registry = ToolRegistry()
    # A non-approval read tool also contributes its idempotency to the contract.
    registry.register(ToolSpec("read", Args, lambda *_: {"value": 1}, permission="rags:query"))
    registry.register(ToolSpec("record", Args, lambda *_: invoked.append(1),
                               permission="rags:memory", approval_required=True))
    service = engine(state, config, FakeRouter(reply("", call("record", {"amount": 1}))), registry)
    waiting = service.start(principal(), goal="Record one observation", model="main", session_id="update")
    assert waiting.status == "waiting_for_approval"

    if change == "schema":
        class ChangedArgs(BaseModel):
            model_config = ConfigDict(extra="forbid")
            amount: int = Field(ge=0, le=100)
        registry._tools["record"] = replace(registry._tools["record"], schema=ChangedArgs)
    elif change == "model":
        config = replace(config, models={"main": "thinkingmachines/Inkling-2-Mini"})
    elif change == "effort":
        config = replace(config, model_options={"main": {"protocol": "tinker", "reasoning_effort": "low"}})
    elif change == "limits":
        config = replace(config, max_tool_calls=8)
    elif change == "idempotency":
        registry._tools["read"] = replace(registry._tools["read"], idempotent=False)
    else:
        waiting = state.create(principal().tenant, kind="agent", fingerprint="0" * 64,
                               state=waiting.state, status="waiting_for_approval")
    restarted = engine(StateStore(Store(state.store.path)), config, FakeRouter(), registry)
    assert restarted.get(principal(), waiting.run_id).status == "waiting_for_approval"
    with pytest.raises(CapabilityError) as error:
        restarted.resume(principal(), waiting.run_id, expected_revision=waiting.revision, approved=True)
    assert error.value.code == "definition_mismatch"
    assert invoked == []
    assert state.get(principal().tenant, waiting.run_id).status == "waiting_for_approval"


@pytest.mark.parametrize("status", ["completed", "failed", "stopped"])
def test_upgrade_keeps_old_agent_history_readable_only_to_its_tenant_and_owner(components, status):
    state, config = components
    bot = FakeRouter(reply("Historical answer"))
    service = engine(state, config, bot)
    completed = service.start(principal(), goal="Read a historical answer", model="main", session_id="history")
    legacy = state.create(principal().tenant, kind="agent", fingerprint="0" * 64,
                          state=completed.state, status=status)
    read_only_owner = principal(permissions={"rags:agent"})
    restored = engine(StateStore(Store(state.store.path)), config, FakeRouter())
    historical = restored.get(read_only_owner, legacy.run_id)
    assert historical.status == status
    assert restored.summary(historical)["answer"] == "Historical answer"
    for unauthorized in (principal("tenant-b"), principal(subject="other-owner")):
        with pytest.raises(CapabilityError) as error:
            restored.get(unauthorized, legacy.run_id)
        assert error.value.status_code == 404
    assert len(bot.calls) == 1


def test_reordered_equivalent_model_configuration_preserves_contract(components):
    state, baseline = components
    first = replace(baseline, models={"a": "openai/a", "b": "openai/b"})
    second = replace(baseline, models={"b": "openai/b", "a": "openai/a"})
    assert engine(state, first, FakeRouter())._fingerprint == engine(state, second, FakeRouter())._fingerprint


@pytest.mark.parametrize("change", ["endpoint", "authentication"])
def test_local_gateway_contract_change_requires_new_approval(components, change):
    state, config = components
    config = replace(config, gateway_url="https://127.0.0.1:8080/v1",
                     model_options={"main": {"protocol": "openai"}})
    invoked = []
    registry = ToolRegistry()
    registry.register(ToolSpec("record", Args, lambda *_: invoked.append(True),
                               permission="rags:memory", approval_required=True))
    service = engine(state, config, FakeRouter(reply("", call("record", {"amount": 1}))), registry)
    waiting = service.start(principal(), goal="Review observation before saving", model="main", session_id="local")
    updated = (replace(config, gateway_url="https://127.0.0.1:8081/v1") if change == "endpoint" else
               replace(config, gateway_auth_mode="none", gateway_key=""))
    restarted = engine(state, updated, FakeRouter(), registry)
    assert restarted.get(principal(), waiting.run_id).status == "waiting_for_approval"
    with pytest.raises(CapabilityError) as error:
        restarted.resume(principal(), waiting.run_id, expected_revision=waiting.revision, approved=True)
    assert error.value.code == "definition_mismatch" and invoked == []


def test_gateway_credential_rotation_preserves_approval_contract(components):
    state, config = components
    first = engine(state, config, FakeRouter())
    rotated = engine(state, replace(config, gateway_key="rotated-secret"), FakeRouter())
    assert first._fingerprint == rotated._fingerprint


def test_tool_reasoning_continuation_is_private_across_approval_and_restart(components):
    state, config = components
    registry = ToolRegistry()
    registry.register(ToolSpec("record", Args, lambda *_: {"saved": True},
                               permission="rags:memory", approval_required=True))
    continuation = "PRIVATE upstream continuation fixture"
    first_reply = ModelReply(content="", tool_calls=[call("record", {"amount": 1})],
        finish_reason="tool_calls", model="thinkingmachines/Inkling-2-Mini", reasoning_content=continuation)
    assert continuation not in first_reply.model_dump_json() and continuation not in repr(first_reply)
    service = engine(state, config, FakeRouter(first_reply), registry)
    waiting = service.start(principal(), goal="Save an approved observation",
                            model="main", session_id="continuation")
    assert waiting.status == "waiting_for_approval"
    assert continuation not in str(service.summary(waiting))
    assert continuation not in str(waiting.public())
    assert waiting.state["messages"][-1]["reasoning_content"] == continuation

    following = FakeRouter(reply("Saved after approval"))
    restarted = engine(StateStore(Store(state.store.path)), config, following, registry)
    finished = restarted.resume(principal(), waiting.run_id,
                                 expected_revision=waiting.revision, approved=True)
    assert finished.status == "completed"
    assert following.calls[0][1][-2]["reasoning_content"] == continuation
    assert continuation not in str(restarted.summary(finished))
    assert continuation not in str(finished.public())
    assert continuation not in str(restarted.history.last("tenant-a", scoped_session(
        principal().subject, "continuation")))
