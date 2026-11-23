"""Real CRAG evaluator through HTTP and the fixed subprocess worker.

Generation is an explicitly deterministic local OpenAI-protocol fixture. This
gate proves the trained evaluator, source lifecycle and answer preservation;
it makes no learned-generation or live external-search quality claim.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import secrets
import sys
import tempfile
import threading
import time
from contextlib import ExitStack
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
from e2e_research_v03 import chunk_snapshot, clean, request, start_app

from rags.research_protocol import UPSTREAMS

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
CRAG_FIXTURE = importlib.import_module("crag_fixture_provider")
HAMLET_ANSWER = CRAG_FIXTURE.HAMLET_ANSWER
HAMLET_QUESTION = CRAG_FIXTURE.HAMLET_QUESTION
HAMLET_TEXT = CRAG_FIXTURE.HAMLET_TEXT


class GenerationFixture:
    """Test-only authenticated fixture, with no hosted provider fallback."""

    def __init__(self):
        self.calls = 0
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.Lock()
        fixture = self

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
                    completion = CRAG_FIXTURE.crag_completion(system, user)
                    encoded = json.dumps({"choices": [{"message": {"role": "assistant", "content": completion},
                                                       "finish_reason": "stop"}]}).encode()
                except (AssertionError, KeyError, StopIteration, ValueError, TypeError, IndexError):
                    self.send_error(422, "Unsupported deterministic fixture input")
                    return
                with fixture.lock:
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


def verify_query(value: dict, expected: dict, work: Path, *, title: str | None = None) -> dict:
    assert value["mode_requested"] == value["mode_used"] == "crag"
    assert value["engine"] == "crag:authorized-evidence-v2"
    event = next(row for row in value["trace"] if row.get("operation") == "external_backend")
    assert event["evidence_level"] == "pinned_upstream_reported"
    assert event["implementation"] == UPSTREAMS["crag"]
    diagnostics = event["diagnostics"]
    assert diagnostics["source_count"] == len(expected)
    assert diagnostics["engine_executed"] is bool(expected)
    actual_ids = {row["chunk_id"] for row in value["citations"]}
    assert actual_ids == set(expected)
    if expected:
        assert value["outcome"] == "success" and value["supported"] is None
        assert diagnostics["correction_branch"] == "correct"
        assert diagnostics["evaluator_calls"] == 2 and diagnostics["generation_calls"] == 1
        assert diagnostics["search_calls"] == diagnostics["external_sources"] == 0
        assert diagnostics["execution_path"] == "bounded_crag_port"
        assert value["answer"] == HAMLET_ANSWER + "\n\n" + " ".join("[" + row["chunk_id"] + "]" for row in value["citations"])
        for row in value["citations"]:
            source = expected[row["chunk_id"]]
            assert row["document_id"] == source["document_id"] and row["revision"] == source["revision"]
            assert row["text"] == source["content"] == HAMLET_TEXT
            assert row["title"] == title and row["origin_source"] == "local"
    else:
        assert value["outcome"] == "insufficient_evidence" and value["supported"] is False
        assert value["refusal_reason"] == "no_candidates"
        assert value["answer"] == "Insufficient evidence in the authorized knowledge base."
    clean(work)
    return {"implementation": event["implementation"], "diagnostics": diagnostics,
            "cited_chunk_ids": sorted(actual_ids), "outcome": value["outcome"],
            "answer_preserved": bool(expected), "request_index_removed": True}


def run(args) -> dict:
    started = time.monotonic()
    # This executable is a fixture gate, so ambient provider/tenant credentials
    # and proxies cannot influence either local process or test HTTP traffic.
    base = {key: value for key, value in os.environ.items()
            if not key.startswith(("RAGS_", "OPENAI_", "ANTHROPIC_"))
            and key.lower() not in {"http_proxy", "https_proxy", "all_proxy"}}
    base.update(
        RAGS_ENV="test", HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_OFFLINE="1",
        OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false",
        NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost",
        RAGS_CRAG_MODEL_PATH=str(Path(args.model_path).resolve()),
        RAGS_ADMISSION_POLICY="compatibility", RAGS_EMBEDDING_PROVIDER="local",
        RAGS_TOKENIZER_JSON=json.dumps({"backend": "unicode"}),
        RAGS_GENERATION_TOKENIZER_JSON=json.dumps({"backend": "unicode"}),
        RAGS_CHUNKING_JSON=json.dumps({"strategy": "fixed", "max_tokens": 128, "overlap_tokens": 0}),
    )
    checks = {}
    with ExitStack() as stack:
        root = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="rags-crag-http-v10-")))
        work = root / "research-work"
        work.mkdir()
        fixture = GenerationFixture()
        stack.callback(fixture.close)
        alpha_side, beta_side, alpha_api, beta_api = [secrets.token_urlsafe(32) for _ in range(4)]
        side_env = dict(base, RAGS_RESEARCH_ENGINE="crag",
                        RAGS_RESEARCH_KEYS_JSON=json.dumps({alpha_side: "alpha", beta_side: "beta"}),
                        RAGS_RESEARCH_WORK_DIR=str(work), RAGS_RESEARCH_TIMEOUT_SECONDS="120",
                        RAGS_MODEL_PROVIDER="openai", OPENAI_API_KEY=fixture.token,
                        OPENAI_BASE_URL=fixture.url, OPENAI_CHAT_MODEL="fixture-crag-generation-v1")
        side = start_app(stack, "rags.research_service:create_app", side_env, root, "research")
        database = root / "rags.sqlite"
        api_env = dict(base, RAGS_MODEL_PROVIDER="extractive", RAGS_DB_PATH=str(database),
                       RAGS_KEYS_JSON=json.dumps({alpha_api: "alpha", beta_api: "beta"}),
                       RAGS_RESEARCH_BACKENDS_JSON=json.dumps({"crag": {
                           "url": side + "/retrieve", "timeout_seconds": 130, "require_pinned_upstream": True,
                           "tenant_tokens": {"alpha": alpha_side, "beta": beta_side},
                       }}))
        api = start_app(stack, "rags.api:create_app", api_env, root, "api")
        client = stack.enter_context(httpx.Client(timeout=150, trust_env=False))
        body = {"protocol": "authorized-evidence-v2", "engine": "crag", "namespace": "alpha",
                "question": HAMLET_QUESTION, "top_k": 5, "passages": [{"id": "authorized", "text": HAMLET_TEXT}]}
        query = {"question": HAMLET_QUESTION, "mode": "crag", "top_k": 5, "generate": True}
        request(client, "POST", side + "/retrieve", "wrong", expected=401, json=body)
        request(client, "POST", side + "/retrieve", beta_side, expected=403, json=body)
        request(client, "POST", api + "/query", "wrong", expected=401, json=query)
        request(client, "POST", side + "/retrieve", alpha_side, expected=422,
                json=dict(body, protocol="authorized-evidence-v1"))
        forbidden = request(client, "POST", side + "/retrieve", alpha_side, expected=403,
                            json=dict(body, external_hosts=["reference.example.org"]))
        assert forbidden["detail"]["code"] == "external_source_forbidden"
        assert fixture.calls == 0
        clean(work)
        checks["authorization_before_inference"] = {
            "invalid_token": 401, "tenant_mismatch": 403, "protocol_mismatch": 422,
            "unapproved_external_hosts": 403, "no_model_calls": True,
        }

        def ingest(token, title, *, valid_from="2020-01-01T00:00:00Z", expected_revision=None):
            document = {"document_id": "hamlet", "title": title, "text": HAMLET_TEXT, "valid_from": valid_from}
            if expected_revision is not None:
                document["expected_revision"] = expected_revision
            result = request(client, "POST", api + "/documents", token, expected=201, json=document)
            assert result["chunks"] == 1
            return result["revision"]

        old_revision = ingest(alpha_api, "Original Hamlet reference")
        beta_revision = ingest(beta_api, "Private Hamlet reference")

        def check_query(label, token, tenant, revision, title, parameters=query):
            before = fixture.calls
            call_started = time.monotonic()
            value = request(client, "POST", api + "/query", token, json=parameters)
            elapsed = time.monotonic() - call_started
            expected = chunk_snapshot(database, tenant, {"hamlet": revision} if revision else {})
            check = verify_query(value, expected, work, title=title)
            assert fixture.calls - before == int(bool(expected))
            checks[label] = dict(check, elapsed_seconds=elapsed, fixture_http_calls=fixture.calls - before)
            return value

        initial = check_query("actual_evaluator_and_preserved_answer", alpha_api, "alpha", old_revision,
                              "Original Hamlet reference")
        catalog = request(client, "GET", api + "/strategies", alpha_api)
        capability = next(item for item in catalog["capabilities"] if item["id"] == "crag")
        assert capability["evidence_level"] == "pinned_upstream_reported" and capability["upstream"] == UPSTREAMS["crag"]
        checks["observed_capability"] = {"evidence_level": capability["evidence_level"]}
        revision = ingest(alpha_api, "Updated Hamlet reference", valid_from="2024-01-01T00:00:00Z",
                          expected_revision=old_revision)
        assert revision != old_revision
        current = check_query("updated_snapshot", alpha_api, "alpha", revision, "Updated Hamlet reference")
        historical = check_query("historical_snapshot", alpha_api, "alpha", old_revision, "Original Hamlet reference",
                                 dict(query, as_of="2022-01-01T00:00:00Z"))
        private = check_query("tenant_source_isolation", beta_api, "beta", beta_revision, "Private Hamlet reference")
        assert initial["citations"][0]["chunk_id"] == historical["citations"][0]["chunk_id"]
        assert current["citations"][0]["chunk_id"] != historical["citations"][0]["chunk_id"]
        assert {row["chunk_id"] for row in private["citations"]}.isdisjoint(
            row["chunk_id"] for row in initial["citations"] + current["citations"])
        request(client, "DELETE", api + "/documents/hamlet", alpha_api)
        check_query("deleted_current_snapshot", alpha_api, "alpha", None, None)
        check_query("deleted_historical_snapshot", alpha_api, "alpha", None, None,
                    dict(query, as_of="2022-01-01T00:00:00Z"))
        assert request(client, "GET", api + "/documents/hamlet", beta_api)["body"] == HAMLET_TEXT
        checks["other_tenant_retained_after_delete"] = {"retained": True}

        timeout_work = root / "timeout-work"
        timeout_work.mkdir()
        timeout_side = start_app(stack, "rags.research_service:create_app", dict(
            side_env, RAGS_RESEARCH_WORK_DIR=str(timeout_work), RAGS_RESEARCH_TIMEOUT_SECONDS="0.1",
        ), root, "timeout-research")
        expired = request(client, "POST", timeout_side + "/retrieve", alpha_side, expected=504, json=body)
        assert expired["detail"]["code"] == "upstream_timeout"
        clean(timeout_work)
        checks["deadline_cleanup"] = {"http_status": 504, "code": "upstream_timeout", "request_index_removed": True}
        assert fixture.calls == 4
        report = {
            "schema": "crag-http-e2e-v10", "verified_at": datetime.now(UTC).isoformat(), "engine": "crag",
            "implementation": UPSTREAMS["crag"],
            "runtime_versions": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "sentencepiece", "safetensors"]},
            "generation": "explicit_local_openai_protocol_deterministic_fixture",
            "trained_evaluator_executed": True, "trained_generation_quality_verified": False,
            "live_external_acquisition_executed": False, "fixture_http_calls": fixture.calls,
            "checks": checks, "passed": True,
        }
    report["elapsed_seconds"] = time.monotonic() - started
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.getenv("RAGS_CRAG_MODEL_PATH"), required=not os.getenv("RAGS_CRAG_MODEL_PATH"))
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report = run(args)
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"engine": "crag", "passed": True, "checks": list(report["checks"]), "report": str(destination)}))


if __name__ == "__main__":
    main()
