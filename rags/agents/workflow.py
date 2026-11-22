"""Deterministic code-defined workflows: .then(), .branch(), .parallel().

Workflow functions come only from an operator's Python registry, never user
prompt text. Each completed barrier is check-pointed in the tenant database.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Callable

from rags.agents.state import Checkpoint, StateStore
from rags.registry import CapabilityError


NAME = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
MAX_NODES = 32


@dataclass(frozen=True)
class WorkflowContext:
    tenant: str
    run_id: str
    input: dict
    outputs: dict


@dataclass(frozen=True)
class Node:
    name: str
    kind: str
    fn: Callable | None = None
    predicate: Callable | None = None
    if_true: Callable | None = None
    if_false: Callable | None = None
    branches: tuple[tuple[str, Callable], ...] = ()
    approval: bool = False
    idempotent: bool = False


class Workflow:
    """A bounded, ordered graph. Branch nodes choose one of two registered steps."""

    def __init__(self, name: str, version: str = "v1", nodes: tuple[Node, ...] = ()):
        if not NAME.fullmatch(name) or not NAME.fullmatch(version):
            raise ValueError("Workflow identity must be a simple operator-defined name")
        if len(nodes) > MAX_NODES:
            raise ValueError("Too many workflow nodes")
        self.name, self.version, self._nodes = name, version, nodes

    def _append(self, node: Node) -> "Workflow":
        if len(self._nodes) >= MAX_NODES or any(n.name == node.name for n in self._nodes):
            raise ValueError("Too many workflow nodes or duplicate step name")
        if not NAME.fullmatch(node.name):
            raise ValueError("Invalid workflow node name")
        return Workflow(self.name, self.version, (*self._nodes, node))

    def then(self, name: str, fn: Callable, *, approval: bool = False,
             idempotent: bool = False) -> "Workflow":
        if not callable(fn):
            raise ValueError("Workflow action must be an operator function")
        if approval and not idempotent:
            raise ValueError("Resumable approved actions require idempotency")
        return self._append(Node(name, "then", fn=fn, approval=approval, idempotent=idempotent))

    def branch(self, name: str, predicate: Callable, *, if_true: Callable,
               if_false: Callable) -> "Workflow":
        if not all(callable(fn) for fn in (predicate, if_true, if_false)):
            raise ValueError("Branch predicate and both actions must be registered functions")
        return self._append(Node(name, "branch", predicate=predicate,
                                 if_true=if_true, if_false=if_false, idempotent=True))

    def parallel(self, name: str, branches: dict[str, Callable], *,
                 idempotent: bool = False) -> "Workflow":
        if not idempotent:
            raise ValueError("Parallel branches must declare idempotent read-only behavior")
        if not isinstance(branches, dict) or not 2 <= len(branches) <= 4 or any(
            not NAME.fullmatch(key) or not callable(fn) for key, fn in branches.items()
        ):
            raise ValueError("Parallel work requires 2..4 named functions")
        return self._append(Node(name, "parallel", branches=tuple(sorted(branches.items())),
                                 idempotent=True))

    @property
    def fingerprint(self) -> str:
        # Explicit workflow version must change when operator changes function behavior.
        definition = [self.name, self.version, [
            [n.name, n.kind, n.approval, n.idempotent, [name for name, _ in n.branches]]
            for n in self._nodes]]
        return hashlib.sha256(json.dumps(definition, separators=(",", ":")).encode()).hexdigest()

    @staticmethod
    def _context(tenant: str, run_id: str, data: dict) -> WorkflowContext:
        return WorkflowContext(tenant, run_id, dict(data["input"]), dict(data["outputs"]))

    def start(self, state: StateStore, tenant: str, inputs: dict) -> Checkpoint:
        if not self._nodes or not isinstance(inputs, dict):
            raise ValueError("Workflow requires steps and JSON input")
        cp = state.create(tenant, kind="workflow", fingerprint=self.fingerprint,
                          state={"cursor": 0, "input": inputs, "outputs": {}, "approved": None})
        return self._advance(state, cp)

    def resume(self, state: StateStore, tenant: str, run_id: str, *,
               expected_revision: int, approved: bool) -> Checkpoint:
        cp = state.get(tenant, run_id)
        if cp is None:
            raise CapabilityError("not_found", "Workflow not found", 404)
        if cp.fingerprint != self.fingerprint:
            raise CapabilityError("definition_mismatch", "Workflow version changed", 409)
        if cp.status != "waiting_for_approval":
            raise CapabilityError("invalid_state", "Workflow is not waiting for approval", 409)
        if type(approved) is not bool:
            raise ValueError("Human decision must be explicit")
        if not approved:
            return state.update(tenant, run_id, expected_revision=expected_revision,
                                fingerprint=self.fingerprint, status="stopped", state={
                                    **cp.state, "approved": False,
                                })
        cp = state.update(tenant, run_id, expected_revision=expected_revision,
                          fingerprint=self.fingerprint, status="running", state={
                              **cp.state, "approved": cp.state["cursor"],
                          })
        return self._advance(state, cp)

    def _advance(self, state: StateStore, cp: Checkpoint) -> Checkpoint:
        steps = 0
        while cp.state["cursor"] < len(self._nodes):
            if steps >= MAX_NODES:
                return state.update(cp.tenant, cp.run_id, expected_revision=cp.revision,
                                    fingerprint=self.fingerprint, status="stopped", state=cp.state)
            node = self._nodes[cp.state["cursor"]]
            if node.approval and cp.state.get("approved") != cp.state["cursor"]:
                return state.update(cp.tenant, cp.run_id, expected_revision=cp.revision,
                                    fingerprint=self.fingerprint, status="waiting_for_approval",
                                    state={**cp.state, "waiting_for": node.name})
            context = self._context(cp.tenant, cp.run_id, cp.state)
            try:
                if node.kind == "then":
                    value = node.fn(context)
                elif node.kind == "branch":
                    choice = node.predicate(context)
                    if type(choice) is not bool:
                        raise ValueError("Workflow branch predicate must return bool")
                    value = {"selected": "true" if choice else "false",
                             "result": (node.if_true if choice else node.if_false)(context)}
                else:
                    with ThreadPoolExecutor(max_workers=len(node.branches)) as pool:
                        futures = {name: pool.submit(fn, context) for name, fn in node.branches}
                        value = {name: future.result() for name, future in futures.items()}
                # All output is validated as JSON-safe by StateStore.update.
                new_state = {**cp.state, "cursor": cp.state["cursor"] + 1,
                             "approved": None, "waiting_for": None,
                             "outputs": {**cp.state["outputs"], node.name: value}}
                status = "completed" if new_state["cursor"] >= len(self._nodes) else "running"
                cp = state.update(cp.tenant, cp.run_id, expected_revision=cp.revision,
                                  fingerprint=self.fingerprint, status=status, state=new_state)
                steps += 1
            except CapabilityError:
                raise
            except Exception as exc:
                state.update(cp.tenant, cp.run_id, expected_revision=cp.revision,
                             fingerprint=self.fingerprint, status="failed",
                             state={**cp.state, "error": "workflow_step_failed"})
                raise CapabilityError("workflow_step_failed",
                                      "A registered workflow step failed", 502) from exc
        return cp
