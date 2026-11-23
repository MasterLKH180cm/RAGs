"""Actual trained Self-RAG -> isolated worker -> HTTP RAG lifecycle gate.

Only synthetic passages are used. No fixture stands in for generation: the
ordinary RAG provider is extractive and the research worker loads the exact
trained Self-RAG Q4_K_M artifact for each authorized source snapshot.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import sys
import tempfile
import time
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rags.research.selfrag_runtime import CHECKPOINT_REVISION, MODEL_SHA256, RUNTIME_REVISION
from rags.research_protocol import UPSTREAMS
from scripts.e2e_research_v03 import chunk_snapshot, clean, request, start_app

QUESTION = "What is the refund period?"


def create_audited_sidecar():
    """Test-only ASGI wrapper retaining the original sidecar status on failures.

    The real production app and worker are unchanged. Never record authorization
    headers, request bodies, or model source text. The operator-supplied audit
    path belongs to this harness's private temporary directory.
    """
    from rags.research_service import create_app

    application = create_app()
    destination = Path(os.environ["RAGS_SELFRAG_AUDIT_PATH"])

    def append_record(record):
        with destination.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

    async def audited(scope, receive, send):
        if scope["type"] != "http" or scope["path"] != "/retrieve":
            return await application(scope, receive, send)
        started, status, parts = time.monotonic(), None, bytearray()

        async def capture(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            elif message["type"] == "http.response.body":
                parts.extend(message.get("body", b""))
            await send(message)

        try:
            await application(scope, receive, capture)
        finally:
            try:
                value = json.loads(parts) if parts else {}
            except (ValueError, UnicodeError):
                value = {"detail": {"code": "non_json_response"}}
            record = {"http_status": status, "elapsed_seconds": time.monotonic() - started,
                      "detail": value.get("detail"), "diagnostics": value.get("diagnostics"),
                      "usage": value.get("usage")}
            await asyncio.to_thread(append_record, record)

    return audited


def verify_query(value: dict, expected: dict, work: Path, *, days: int | None = None,
                 title: str | None = None, refusal: bool = False) -> dict:
    assert value["mode_requested"] == value["mode_used"] == "self_rag"
    assert value["engine"] == "self_rag:authorized-evidence-v2"
    event = next(row for row in value["trace"] if row.get("operation") == "external_backend")
    assert event["evidence_level"] == "pinned_upstream_reported"
    assert event["implementation"] == UPSTREAMS["self_rag"]
    diagnostics = event["diagnostics"]
    assert diagnostics["source_count"] == len(expected)
    assert diagnostics["engine_executed"] is bool(expected)
    assert value["usage"]["calls"] == 0  # No ordinary generator or embeddings.
    actual_ids = {row["chunk_id"] for row in value["citations"]}
    if refusal or not expected:
        assert not actual_ids and value["outcome"] == "insufficient_evidence" and value["supported"] is False
        assert value["answer"] == "Insufficient evidence in the authorized knowledge base."
        if refusal:
            assert diagnostics["reflection_abstentions"] == diagnostics["reflection_calls"] == 1
    else:
        assert actual_ids == set(expected) and value["outcome"] == "success" and value["supported"] is None
        assert diagnostics["reflection_calls"] == 1 and diagnostics["reflection_abstentions"] == 0
        assert diagnostics["checkpoint_sha256"] == MODEL_SHA256
        assert f"{days} days" in value["answer"]
        assert not value["answer"].startswith("Retrieved source evidence")
        references = " ".join("[" + row["chunk_id"] + "]" for row in value["citations"])
        assert value["answer"].endswith("\n\n" + references)
        for row in value["citations"]:
            source = expected[row["chunk_id"]]
            assert row["document_id"] == source["document_id"] and row["revision"] == source["revision"]
            assert row["text"] == source["content"] == f"The refund period is {days} days after purchase."
            assert row["title"] == title and row["origin_source"] == "local"
    clean(work)
    return {"implementation": event["implementation"], "diagnostics": diagnostics,
            "cited_chunk_ids": sorted(actual_ids), "outcome": value["outcome"],
            "answer": value["answer"], "ordinary_provider_calls": 0, "request_index_removed": True}


def run(args, checks: dict, sidecar_responses: list) -> dict:
    started = time.monotonic()
    base = {key: value for key, value in os.environ.items()
            if not key.startswith(("RAGS_", "OPENAI_", "ANTHROPIC_"))
            and key.lower() not in {"http_proxy", "https_proxy", "all_proxy"}}
    base.update(
        RAGS_ENV="test", HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
        OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false",
        NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost",
        RAGS_SELFRAG_MODEL_PATH=str(Path(args.model_path).resolve()),
        RAGS_SELFRAG_LLAMA_DIR=str(Path(args.llama_dir).resolve()), RAGS_SELFRAG_THREADS=str(args.threads),
        RAGS_ADMISSION_POLICY="compatibility", RAGS_EMBEDDING_PROVIDER="local", RAGS_MODEL_PROVIDER="extractive",
        RAGS_TOKENIZER_JSON=json.dumps({"backend": "unicode"}),
        RAGS_GENERATION_TOKENIZER_JSON=json.dumps({"backend": "unicode"}),
        RAGS_CHUNKING_JSON=json.dumps({"strategy": "fixed", "max_tokens": 128, "overlap_tokens": 0}),
    )
    with ExitStack() as stack:
        root = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="rags-selfrag-http-v10-")))
        audit = root / "sidecar-responses.jsonl"

        def preserve_sidecar_responses():
            if audit.exists():
                sidecar_responses.extend(json.loads(row) for row in audit.read_text().splitlines())

        stack.callback(preserve_sidecar_responses)
        work = root / "research-work"
        work.mkdir()
        alpha_side, beta_side, alpha_api, beta_api = [secrets.token_urlsafe(32) for _ in range(4)]
        side_env = dict(base, RAGS_RESEARCH_ENGINE="self_rag",
                        RAGS_RESEARCH_KEYS_JSON=json.dumps({alpha_side: "alpha", beta_side: "beta"}),
                        RAGS_RESEARCH_WORK_DIR=str(work), RAGS_RESEARCH_TIMEOUT_SECONDS="120",
                        RAGS_SELFRAG_AUDIT_PATH=str(audit))
        side = start_app(stack, "scripts.e2e_selfrag_v10:create_audited_sidecar", side_env, root, "research")
        database = root / "rags.sqlite"
        api_env = dict(base, RAGS_DB_PATH=str(database), RAGS_KEYS_JSON=json.dumps({alpha_api: "alpha", beta_api: "beta"}),
                       RAGS_RESEARCH_BACKENDS_JSON=json.dumps({"self_rag": {
                           "url": side + "/retrieve", "timeout_seconds": 130, "require_pinned_upstream": True,
                           "tenant_tokens": {"alpha": alpha_side, "beta": beta_side},
                       }}))
        api = start_app(stack, "rags.api:create_app", api_env, root, "api")
        client = stack.enter_context(httpx.Client(timeout=150, trust_env=False))
        body = {"protocol": "authorized-evidence-v2", "engine": "self_rag", "namespace": "alpha",
                "question": QUESTION, "top_k": 5, "passages": [{"id": "authorized", "text": "The refund period is 14 days after purchase."}]}
        query = {"question": QUESTION, "mode": "self_rag", "top_k": 5, "generate": True,
                 "filters": {"source": "fixture:refund"}}
        request(client, "POST", side + "/retrieve", "wrong", expected=401, json=body)
        request(client, "POST", side + "/retrieve", beta_side, expected=403, json=body)
        request(client, "POST", api + "/query", "wrong", expected=401, json=query)
        request(client, "POST", side + "/retrieve", alpha_side, expected=422, json=dict(body, protocol="authorized-evidence-v1"))
        clean(work)
        checks["authorization_and_protocol"] = {"invalid_token": 401, "tenant_mismatch": 403, "protocol_mismatch": 422}

        def ingest(token, title, days, *, valid_from="2020-01-01T00:00:00Z", expected_revision=None):
            document = {"document_id": "refund", "title": title, "source": "fixture:refund",
                        "text": f"The refund period is {days} days after purchase.", "valid_from": valid_from}
            if expected_revision is not None:
                document["expected_revision"] = expected_revision
            result = request(client, "POST", api + "/documents", token, expected=201, json=document)
            assert result["chunks"] == 1
            return result["revision"]

        old_revision = ingest(alpha_api, "Original refund policy", 14)
        beta_revision = ingest(beta_api, "Private refund policy", 60)

        def check_query(label, token, tenant, revision, title, days, parameters=query):
            call_started = time.monotonic()
            value = request(client, "POST", api + "/query", token, json=parameters)
            expected = chunk_snapshot(database, tenant, {"refund": revision} if revision else {})
            checks[label] = {**verify_query(value, expected, work, days=days, title=title),
                             "elapsed_seconds": time.monotonic() - call_started}
            print(json.dumps({"check": label, "passed": True, "elapsed_seconds": checks[label]["elapsed_seconds"]}), flush=True)
            return value

        initial = check_query("trained_answer_preserved", alpha_api, "alpha", old_revision, "Original refund policy", 14)
        capability = next(item for item in request(client, "GET", api + "/strategies", alpha_api)["capabilities"]
                          if item["id"] == "self_rag")
        assert capability["evidence_level"] == "pinned_upstream_reported" and capability["upstream"] == UPSTREAMS["self_rag"]
        checks["observed_capability"] = {"evidence_level": capability["evidence_level"]}
        revision = ingest(alpha_api, "Updated refund policy", 30, valid_from="2024-01-01T00:00:00Z", expected_revision=old_revision)
        assert revision != old_revision
        current = check_query("updated_source_content", alpha_api, "alpha", revision, "Updated refund policy", 30)
        historical = check_query("historical_source_content", alpha_api, "alpha", old_revision, "Original refund policy", 14,
                                 dict(query, as_of="2022-01-01T00:00:00Z"))
        private = check_query("tenant_source_isolation", beta_api, "beta", beta_revision, "Private refund policy", 60)
        assert initial["citations"][0]["chunk_id"] == historical["citations"][0]["chunk_id"]
        assert current["citations"][0]["chunk_id"] != historical["citations"][0]["chunk_id"]
        assert {row["chunk_id"] for row in private["citations"]}.isdisjoint(
            row["chunk_id"] for row in initial["citations"] + current["citations"])
        negative = request(client, "POST", api + "/documents", alpha_api, expected=201, json={
            "document_id": "unrelated", "title": "Unrelated synthetic source", "source": "fixture:unrelated",
            "text": "Bananas are yellow fruit and dolphins swim in the ocean."})
        negative_result = request(client, "POST", api + "/query", alpha_api,
                                  json=dict(query, filters={"source": "fixture:unrelated"}))
        checks["trained_refusal_no_fallback"] = verify_query(
            negative_result, chunk_snapshot(database, "alpha", {"unrelated": negative["revision"]}), work, refusal=True)
        request(client, "DELETE", api + "/documents/refund", alpha_api)
        check_query("deleted_current_snapshot", alpha_api, "alpha", None, None, None)
        check_query("deleted_historical_snapshot", alpha_api, "alpha", None, None, None,
                    dict(query, as_of="2022-01-01T00:00:00Z"))
        assert "60 days" in request(client, "GET", api + "/documents/refund", beta_api)["body"]
        checks["other_tenant_retained_after_delete"] = {"retained": True}
        timeout_work = root / "timeout-work"
        timeout_work.mkdir()
        timeout_side = start_app(stack, "scripts.e2e_selfrag_v10:create_audited_sidecar", dict(
            side_env, RAGS_RESEARCH_WORK_DIR=str(timeout_work), RAGS_RESEARCH_TIMEOUT_SECONDS="0.1"), root, "timeout-research")
        expired = request(client, "POST", timeout_side + "/retrieve", alpha_side, expected=504, json=body)
        assert expired["detail"]["code"] == "upstream_timeout"
        clean(timeout_work)
        checks["deadline_cleanup"] = {"http_status": 504, "code": "upstream_timeout", "request_index_removed": True}
        report = {"schema": "selfrag-http-e2e-v10", "verified_at": datetime.now(UTC).isoformat(), "engine": "self_rag",
                  "implementation": UPSTREAMS["self_rag"], "checkpoint_revision": CHECKPOINT_REVISION,
                  "checkpoint_sha256": MODEL_SHA256, "runtime_revision": RUNTIME_REVISION,
                  "native_threads": args.threads,
                  "trained_checkpoint_executed": True, "ordinary_generation_provider": "extractive",
                  "evidence_kind": "actual_trained_checkpoint_with_synthetic_source_lifecycle",
                  "limitations": ["No paper benchmark, domain quality or full-precision equivalence claim.",
                                  "Latency includes checkpoint verification, fresh model load and complete request lifecycle."],
                  "checks": checks, "sidecar_responses": sidecar_responses, "passed": True}
    report["elapsed_seconds"] = time.monotonic() - started
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.getenv("RAGS_SELFRAG_MODEL_PATH"), required=not os.getenv("RAGS_SELFRAG_MODEL_PATH"))
    parser.add_argument("--llama-dir", default=os.getenv("RAGS_SELFRAG_LLAMA_DIR"), required=not os.getenv("RAGS_SELFRAG_LLAMA_DIR"))
    parser.add_argument("--threads", type=int, choices=range(1, 9), default=int(os.getenv("RAGS_SELFRAG_THREADS", "4")))
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    checks, sidecar_responses = {}, []
    started = time.monotonic()
    try:
        report = run(args, checks, sidecar_responses)
    except Exception as error:
        failure = {"schema": "selfrag-http-e2e-v10", "passed": False, "engine": "self_rag",
                   "error_type": type(error).__name__, "error": str(error), "partial_checks": checks,
                   "sidecar_responses": sidecar_responses, "elapsed_seconds": time.monotonic() - started}
        failed_target = destination.with_name(destination.stem + "_failure" + destination.suffix)
        failed_target.write_text(json.dumps(failure, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"engine": "self_rag", "passed": False, "report": str(failed_target)}), flush=True)
        raise
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"engine": "self_rag", "passed": True, "checks": list(report["checks"]), "report": str(destination)}))


if __name__ == "__main__":
    main()
