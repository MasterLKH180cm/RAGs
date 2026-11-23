"""Actual HTTP -> fixed worker -> pinned upstream/BGE gate, without paid models.

Run with one isolated engine environment. The local OpenAI-protocol endpoint
serves explicitly deterministic generation fixtures; it is not model validation.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import secrets
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import ExitStack
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

from rags.research_protocol import UPSTREAMS

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
from research_fixture_provider import PASSAGES, QUESTION, fixture_completion  # noqa: E402 -- explicit test-only fixture


def completion_fixture(engine):
    if engine == "full_graphrag":
        from graphrag_fixture_provider import fixture_completion as completion
    elif engine == "hipporag":
        from hipporag_fixture_provider import hipporag_completion as completion
    elif engine in {"raptor", "lightrag"}:
        completion = fixture_completion
    else:
        raise ValueError("Unknown research generation fixture")
    return completion


class GenerationFixture:
    def __init__(self, engine="raptor"):
        self.calls = 0
        self.token = secrets.token_urlsafe(32)
        self._lock = threading.Lock()
        fixture = self
        completion = completion_fixture(engine)

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                if self.path != "/v1/chat/completions" or self.headers.get("Authorization") != "Bearer " + fixture.token:
                    self.send_error(401)
                    return
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 1 <= size <= 250_000:
                        raise ValueError("Invalid fixture body")
                    body = json.loads(self.rfile.read(size))
                    messages = body["messages"]
                    system = next(item["content"] for item in messages if item["role"] == "system")
                    user = next(item["content"] for item in messages if item["role"] == "user")
                    text = completion(system, user)
                    value = {"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}
                    encoded = json.dumps(value).encode()
                except (AssertionError, KeyError, StopIteration, ValueError, TypeError, IndexError):
                    self.send_error(422, "Fixture input shape not supported")
                    return
                with fixture._lock:
                    fixture.calls += 1
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def free_port() -> int:
    with socket.socket() as channel:
        channel.bind(("127.0.0.1", 0))
        return channel.getsockname()[1]


def start_app(stack: ExitStack, target: str, env: dict, directory: Path, name: str) -> str:
    port = free_port()
    log = stack.enter_context((directory / f"{name}.log").open("wb"))
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", target, "--factory", "--host", "127.0.0.1",
         "--port", str(port), "--log-level", "warning"],
        cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )

    def stop():
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)

    stack.callback(stop)
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 40
    with httpx.Client(timeout=1, trust_env=False) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise AssertionError(f"{name} exited before readiness; inspect its local test log")
            try:
                if client.get(url + "/live").status_code == 200:
                    return url
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
    raise AssertionError(f"{name} did not become ready")


def request(client, method, url, token, expected=200, **kwargs):
    response = client.request(method, url, headers={"Authorization": "Bearer " + token}, **kwargs)
    if response.status_code != expected:
        raise AssertionError(f"{method} {url.rsplit('/', 1)[-1]} returned {response.status_code}; expected {expected}: {response.text[:300]}")
    return response.json()


def chunk_snapshot(database: Path, tenant: str, revisions: dict) -> dict:
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT id,document_id,revision,content FROM chunks WHERE tenant=? AND role='leaf'", (tenant,),
        ).fetchall()
    return {row["id"]: dict(row) for row in rows if revisions.get(row["document_id"]) == row["revision"]}


def clean(work: Path) -> None:
    if list(work.iterdir()):
        raise AssertionError("The process runner retained a request-local index")


def verify_query(value: dict, engine: str, expected: dict, work: Path, *, empty=False) -> dict:
    assert value["mode_requested"] == value["mode_used"] == engine
    assert value["engine"] == engine + ":authorized-evidence-v1"
    event = next(row for row in value["trace"] if row.get("operation") == "external_backend")
    assert event["evidence_level"] == "pinned_upstream_reported"
    assert event["implementation"] == UPSTREAMS[engine]
    diagnostics = event["diagnostics"]
    assert diagnostics["source_count"] == len(expected)
    assert diagnostics["engine_executed"] is (not empty)
    actual_ids = {row["chunk_id"] for row in value["citations"]}
    assert actual_ids <= set(expected)
    if empty:
        assert value["outcome"] == "insufficient_evidence" and value["supported"] is False
        assert value["refusal_reason"] == "no_candidates" and not actual_ids
    else:
        assert value["outcome"] == "success" and value["supported"] is None and actual_ids
        assert diagnostics["embedding_model"] == "fastembed:BAAI/bge-small-en-v1.5"
        assert diagnostics["embedding_token_limit"] == 512
        assert diagnostics["embedding_window_policy"] == "native_tokenizer_reject_oversize"
        for row in value["citations"]:
            source = expected[row["chunk_id"]]
            assert row["document_id"] == source["document_id"] and row["revision"] == source["revision"]
            assert row["text"] == source["content"]
    assert value["answer"] is None
    clean(work)
    return {"implementation": event["implementation"], "diagnostics": diagnostics,
            "cited_chunk_ids": sorted(actual_ids), "outcome": value["outcome"],
            "request_index_removed": True}


def run(args) -> dict:
    started = time.monotonic()
    base = {key: value for key, value in os.environ.items()
            if not key.startswith(("RAGS_", "OPENAI_", "ANTHROPIC_"))}
    base.update(
        RAGS_ENV="test", ORT_DISABLE_TELEMETRY="1", HF_HUB_DISABLE_TELEMETRY="1",
        OMP_NUM_THREADS="2", NUMBA_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
        TOKENIZERS_PARALLELISM="false", NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost",
        LITELLM_LOCAL_MODEL_COST_MAP="True",
        RAGS_MODEL_CACHE=str(Path(args.model_cache).resolve()), RAGS_ADMISSION_POLICY="compatibility",
        RAGS_EMBEDDING_PROVIDER="fastembed", RAGS_FASTEMBED_MODEL="BAAI/bge-small-en-v1.5",
        RAGS_TOKENIZER_JSON=json.dumps({"backend": "tiktoken", "encoding": "cl100k_base"}),
        RAGS_GENERATION_TOKENIZER_JSON=json.dumps({"backend": "tiktoken", "encoding": "cl100k_base"}),
        RAGS_CHUNKING_JSON=json.dumps({"strategy": "fixed", "max_tokens": 128, "overlap_tokens": 0}),
    )
    if args.raptor_source:
        base["RAGS_RAPTOR_SOURCE"] = str(Path(args.raptor_source).resolve())
    checks = {}
    with ExitStack() as stack:
        root = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="rags-http-v03-")))
        work = root / "research-work"
        work.mkdir()
        fixture = GenerationFixture(args.engine)
        stack.callback(fixture.close)
        alpha_side, beta_side, alpha_api, beta_api = [secrets.token_urlsafe(32) for _ in range(4)]
        side_env = dict(base, RAGS_RESEARCH_ENGINE=args.engine,
                        RAGS_RESEARCH_KEYS_JSON=json.dumps({alpha_side: "alpha", beta_side: "beta"}),
                        RAGS_RESEARCH_WORK_DIR=str(work), RAGS_RESEARCH_TIMEOUT_SECONDS="120",
                        RAGS_MODEL_PROVIDER="openai", OPENAI_API_KEY=fixture.token,
                        OPENAI_BASE_URL=fixture.url, OPENAI_CHAT_MODEL="fixture-deterministic-generation-v1")
        side = start_app(stack, "rags.research_service:create_app", side_env, root, "research")
        database = root / "rags.sqlite"
        api_env = dict(base, RAGS_MODEL_PROVIDER="extractive", RAGS_DB_PATH=str(database),
                       RAGS_KEYS_JSON=json.dumps({alpha_api: "alpha", beta_api: "beta"}),
                       RAGS_RESEARCH_BACKENDS_JSON=json.dumps({args.engine: {
                           "url": side + "/retrieve", "timeout_seconds": 130, "require_pinned_upstream": True,
                           "tenant_tokens": {"alpha": alpha_side, "beta": beta_side},
                       }}))
        api = start_app(stack, "rags.api:create_app", api_env, root, "api")
        client = stack.enter_context(httpx.Client(timeout=150, trust_env=False))
        body = {"protocol": "authorized-evidence-v1", "engine": args.engine, "namespace": "alpha",
                "question": QUESTION, "top_k": 4, "passages": PASSAGES[:4]}
        request(client, "POST", side + "/retrieve", "wrong", expected=401, json=body)
        request(client, "POST", side + "/retrieve", beta_side, expected=403, json=body)
        request(client, "POST", api + "/query", "wrong", expected=401,
                json={"question": QUESTION, "mode": args.engine, "generate": False})
        assert fixture.calls == 0
        clean(work)
        checks["authentication"] = {"invalid_token": 401, "tenant_mismatch": 403, "no_model_calls": True}

        revisions = {}
        passages = list(PASSAGES)
        if args.engine in {"full_graphrag", "hipporag"}:
            passages.append({"id": "water-duplicate", "text": PASSAGES[0]["text"]})
        for item in passages:
            document = {"document_id": item["id"], "title": item["id"], "text": item["text"],
                        "metadata": {"group": "lifecycle" if item["id"].startswith("water-") else "other"},
                        "valid_from": "2020-01-01T00:00:00Z"}
            value = request(client, "POST", api + "/documents", alpha_api, expected=201, json=document)
            assert value["chunks"] == 1
            revisions[item["id"]] = value["revision"]
        private_text = "PrivateLab develops SecretSensor for protected tenant storage."
        private = request(client, "POST", api + "/documents", beta_api, expected=201, json={
            "document_id": "water-1", "title": "Private document", "text": private_text,
            "valid_from": "2020-01-01T00:00:00Z",
        })
        assert request(client, "GET", api + "/documents/water-1", beta_api)["body"] == private_text
        initial = request(client, "POST", api + "/query", alpha_api, json={
            "question": QUESTION, "mode": args.engine, "top_k": 20, "generate": False,
        })
        checks["actual_upstream_http"] = verify_query(initial, args.engine, chunk_snapshot(database, "alpha", revisions), work)
        diagnostic = checks["actual_upstream_http"]["diagnostics"]
        if args.engine == "raptor":
            assert diagnostic["summary_layers"] >= 1 and diagnostic["selected_summary_nodes"] >= 1
        elif args.engine in {"lightrag", "full_graphrag"}:
            assert diagnostic["graph_nodes"] >= 4 and diagnostic["graph_edges"] >= 4
            assert diagnostic["retrieved_entities"] > 0 and diagnostic["retrieved_relationships"] > 0
            if args.engine == "full_graphrag":
                assert diagnostic["pipeline_stages"] == 9
                assert diagnostic["community_reports"] == diagnostic["communities"] > 0
                assert diagnostic["query_mode"] == "local_graph_context"
        else:
            assert diagnostic["graph_nodes"] > 0 and diagnostic["graph_edges"] > 0
            assert diagnostic["ppr_calls"] == 1 and diagnostic["linked_facts"] > 0
            assert diagnostic["retrieval_path"] == "personalized_pagerank"
        if args.engine in {"full_graphrag", "hipporag"}:
            assert {"water-1", "water-duplicate"} <= {row["document_id"] for row in initial["citations"]}
            checks["duplicate_source_identity"] = {"both_original_ids_preserved": True}
        catalog = request(client, "GET", api + "/strategies", alpha_api)
        capability = next(item for item in catalog["capabilities"] if item["id"] == args.engine)
        assert capability["evidence_level"] == "pinned_upstream_reported" and capability["upstream"] == UPSTREAMS[args.engine]

        if args.agent_python:
            agent_env = dict(os.environ, AGENT_E2E_RAG_URL=api, AGENT_E2E_RAG_KEY=alpha_api,
                             AGENT_E2E_RAG_MODE=args.engine, AGENT_E2E_RAG_QUESTION=QUESTION,
                             AGENT_E2E_RAG_DOCUMENT_IDS_JSON=json.dumps(list(revisions)),
                             ORT_DISABLE_TELEMETRY="1", HF_HUB_DISABLE_TELEMETRY="1",
                             NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost")
            completed = subprocess.run([args.agent_python, args.agent_script], env=agent_env,
                                       cwd=Path(args.agent_script).resolve().parents[1], timeout=240, check=False,
                                       capture_output=True, text=True)
            assert completed.returncode == 0, "Agent upstream HTTP/MCP chain failed: " + completed.stderr[-2000:]
            agent_report = json.loads(completed.stdout)
            assert agent_report["status"] == "passed" and agent_report["external_research_rag"] is True
            checks["companion_agent_chain"] = {"completed": True, "mode": args.engine,
                                                "checks": agent_report["checks"]}
            clean(work)

        scoped = {key: revision for key, revision in revisions.items() if key.startswith("water-")}
        query = {"question": QUESTION, "mode": args.engine,
                 "top_k": 20 if args.engine in {"full_graphrag", "hipporag"} else 4, "generate": False,
                 "filters": {"metadata": {"group": "lifecycle"}}}
        updated = request(client, "POST", api + "/documents", alpha_api, expected=201, json={
            "document_id": "water-1", "title": "water-1", "expected_revision": scoped["water-1"],
            "text": "OrionLab develops HarborSensor for monitoring industrial river pollution with weekly alerts.",
            "metadata": {"group": "lifecycle"}, "valid_from": "2024-01-01T00:00:00Z",
        })
        historic = dict(scoped)
        scoped["water-1"] = updated["revision"]
        for label, parameters, expected in [
            ("updated_snapshot", query, scoped),
            ("historical_snapshot", dict(query, as_of="2022-01-01T00:00:00Z"), historic),
        ]:
            value = request(client, "POST", api + "/query", alpha_api, json=parameters)
            checks[label] = verify_query(value, args.engine, chunk_snapshot(database, "alpha", expected), work)
            matching = [row for row in value["citations"] if row["document_id"] == "water-1"]
            assert matching and {row["revision"] for row in matching} == {expected["water-1"]}

        private_query = request(client, "POST", api + "/query", beta_api, json={
            "question": "How does PrivateLab use SecretSensor?", "mode": args.engine, "generate": False,
        })
        checks["tenant_source_isolation"] = verify_query(
            private_query, args.engine, chunk_snapshot(database, "beta", {"water-1": private["revision"]}), work,
        )
        for identifier in ["water-1", "water-2"]:
            request(client, "DELETE", api + "/documents/" + identifier, alpha_api)
            scoped.pop(identifier)
        assert request(client, "GET", api + "/documents/water-1", beta_api)["body"] == private_text
        for label, parameters in [
            ("deleted_current_snapshot", query),
            ("deleted_historical_snapshot", dict(query, as_of="2022-01-01T00:00:00Z")),
        ]:
            value = request(client, "POST", api + "/query", alpha_api, json=parameters)
            checks[label] = verify_query(value, args.engine, chunk_snapshot(database, "alpha", scoped), work)
        empty = request(client, "POST", api + "/query", alpha_api, json=dict(
            query, filters={"metadata": {"group": "does-not-exist"}},
        ))
        checks["empty_scope"] = verify_query(empty, args.engine, {}, work, empty=True)

        timeout_work = root / "timeout-work"
        timeout_work.mkdir()
        timeout_side = start_app(stack, "rags.research_service:create_app", dict(
            side_env, RAGS_RESEARCH_WORK_DIR=str(timeout_work), RAGS_RESEARCH_TIMEOUT_SECONDS="0.1",
        ), root, "timeout-sidecar")
        expired = request(client, "POST", timeout_side + "/retrieve", alpha_side, expected=504, json=body)
        assert expired["detail"]["code"] == "upstream_timeout"
        clean(timeout_work)
        checks["deadline_cleanup"] = {"http_status": 504, "code": "upstream_timeout", "request_index_removed": True}
        assert fixture.calls > 0
        report = {
            "schema": "research-http-e2e-v03", "verified_at": datetime.now(UTC).isoformat(), "engine": args.engine,
            "implementation": UPSTREAMS[args.engine], "embedding_library": "fastembed:" + importlib.metadata.version("fastembed"),
            "generation": "explicit_local_openai_protocol_deterministic_fixture",
            "trained_generation_quality_verified": False, "fixture_http_calls": fixture.calls,
            "checks": checks, "passed": True,
        }
    report["elapsed_seconds"] = time.monotonic() - started
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["raptor", "lightrag", "full_graphrag", "hipporag"], required=True)
    parser.add_argument("--model-cache", default=os.getenv("RAGS_MODEL_CACHE", str(REPO.parent / "model-cache-v03")))
    parser.add_argument("--raptor-source", default=os.getenv("RAGS_RAPTOR_SOURCE"))
    parser.add_argument("--report", required=True)
    parser.add_argument("--agent-python")
    parser.add_argument("--agent-script", default=str(REPO.parent / "agentic-system/tests/e2e_v03.py"))
    args = parser.parse_args()
    report = run(args)
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"engine": args.engine, "passed": True, "checks": list(report["checks"]), "report": str(destination)}))


if __name__ == "__main__":
    main()
