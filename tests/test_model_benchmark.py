"""Synthetic contract fixtures exercise the real router; no paid model calls."""
from copy import deepcopy
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import threading
import time

import httpx
import pytest

from rags import model_benchmark as benchmark
from rags.agents.router import ModelReply, ModelRouter, ToolCall
from rags.model_profiles import INKLING_MODEL, TINKER_BASE_URL

pytestmark = pytest.mark.unit


def envelope(case, *, reason=None, usage=True):
    message = {"content": json.dumps(case.expected)}
    if case.kind == "tool":
        message = {"content": None, "tool_calls": [{"id": "call_synthetic", "type": "function",
                   "function": {"name": "knowledge_search", "arguments": json.dumps(case.expected)}}]}
    result = {"model": INKLING_MODEL, "choices": [{"message": message,
              "finish_reason": reason or ("tool_calls" if case.kind == "tool" else "stop")}]}
    if usage:
        result["usage"] = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                           "prompt_tokens_details": {"cached_tokens": 10},
                           "completion_tokens_details": {"reasoning_tokens": 5}}
    return result


def install_transport(monkeypatch, response):
    real_client = httpx.Client
    def respond(request):
        return response(request) if callable(response) else response
    monkeypatch.setattr(ModelRouter, "_client", lambda self, *_args: real_client(
        transport=httpx.MockTransport(respond)))


def fixture_run(monkeypatch, **options):
    cases = benchmark.default_cases()
    by_request = {case.request: case for case in cases}
    install_transport(monkeypatch, lambda req: httpx.Response(
        200, json=envelope(by_request[json.loads(req.content)["messages"][-1]["content"]])))
    return benchmark.run_benchmark(
        benchmark.Options(repetitions=1, warmup_rounds=0, **options), cases,
        environment={"TINKER_API_KEY": "not-a-real-secret"}, evidence_kind="http_contract_fixture")


def test_default_dataset_snapshot_is_identical():
    path = Path(__file__).parents[1] / "examples" / "benchmark_cases.json"
    assert benchmark.load_cases(path) == benchmark.default_cases()


def test_dataset_snapshots_do_not_share_mutable_expectations():
    cases = benchmark.default_cases()
    snapshot = benchmark.validate_cases(cases)
    cases[1].expected["cited_ids"].append("extra")
    assert snapshot[1].expected["cited_ids"] == ["retention-01"]
    assert benchmark.default_cases()[1].expected["cited_ids"] == ["retention-01"]


@pytest.mark.parametrize("changes", [
    {"repetitions": True}, {"repetitions": 0}, {"repetitions": 11},
    {"warmup_rounds": -1}, {"warmup_rounds": 4}, {"output_tokens": True}, {"output_tokens": 4097},
    {"concurrency": (True,)}, {"concurrency": (1, 1)}, {"concurrency": (8,)}, {"concurrency": ()},
    {"timeout_seconds": math.nan}, {"timeout_seconds": math.inf}, {"timeout_seconds": True},
    {"max_duration_seconds": 3601}, {"effort": math.nan}, {"effort": True}, {"effort": 1},
    {"model": "unknown"}, {"key_env": "KEY=value"}, {"allow_loopback_http": 1},
    {"base_url": "https://user:password@example.test/v1"},
    {"base_url": "https://example.test/v1?key=secret"},
    {"base_url": "http://example.test/v1", "allow_loopback_http": True},
    {"base_url": "http://127.0.0.1:8000/v1"},
])
def test_workload_and_endpoint_bounds_fail_closed(changes):
    with pytest.raises(ValueError):
        benchmark.Options(**changes)


@pytest.mark.parametrize("values", [
    (True, 1), (1, math.nan), (1, math.inf), (-1, 1), (None, 1), (1, None), (1, 2, math.nan),
])
def test_price_schedule_rejects_ambiguous_values(values):
    with pytest.raises(ValueError):
        benchmark.Prices(*values)


def test_no_credentials_blocks_before_router_or_network(monkeypatch):
    monkeypatch.setattr(benchmark, "ModelRouter", lambda *_: pytest.fail("No router without credentials"))
    result = benchmark.run_benchmark(benchmark.Options(), environment={})
    assert result["status"] == "blocked" and result["block_reason"] == "missing_model_credential"
    assert result["warmup"] is None and result["phases"] == []
    assert result["target"]["model"] == INKLING_MODEL
    assert result["paid_request_upper_bound"] == 42
    assert TINKER_BASE_URL not in json.dumps(result)


def test_credential_file_conflict_and_symlink_are_redacted(tmp_path):
    path = tmp_path / "private-key"
    path.write_text("private-credential-do-not-log", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(path)
    for env in ({"TINKER_API_KEY": "private-credential-do-not-log", "TINKER_API_KEY_FILE": str(path)},
                {"TINKER_API_KEY_FILE": str(link)}):
        result = benchmark.run_benchmark(benchmark.Options(), environment=env)
        assert result["block_reason"] == "credential_configuration_invalid"
        assert str(path) not in json.dumps(result) and "private-credential-do-not-log" not in json.dumps(result)


def test_credential_file_is_supported(tmp_path):
    path = tmp_path / "key"
    path.write_text("not-a-real-secret\n", encoding="utf-8")
    assert benchmark._read_key("TINKER_API_KEY", {"TINKER_API_KEY_FILE": str(path)}) == "not-a-real-secret"


def test_json_schema_and_behavior_are_separate():
    case = benchmark.default_cases()[0]
    for text, expected_code in (("{broken", "invalid_json"),
                               ('{"value":true,"unit":"GB"}', "schema_mismatch"),
                               ('{"value":32,"unit":"GB","extra":1}', "schema_mismatch"),
                               ('{"value":31,"unit":"GB"}', "behavior_mismatch"),
                               ('{"value":32,"value":31,"unit":"GB"}', "invalid_json"),
                               ('{"value":NaN,"unit":"GB"}', "invalid_json")):
        reply = ModelReply(content=text, model=INKLING_MODEL, finish_reason="stop")
        schema, correct, code = benchmark.validate_reply(case, reply)
        assert not correct and code == expected_code
        assert schema == (expected_code == "behavior_mismatch")


def test_valid_json_with_length_finish_reason_is_failure():
    case = benchmark.default_cases()[0]
    reply = ModelReply(content=json.dumps(case.expected), model=INKLING_MODEL, finish_reason="length")
    assert benchmark.validate_reply(case, reply) == (False, False, "truncated")


def test_citations_and_abstention_are_validated_exactly():
    case = benchmark.default_cases()[1]
    altered = {**case.expected, "cited_ids": ["invented-source"]}
    reply = ModelReply(content=json.dumps(altered), model=INKLING_MODEL, finish_reason="stop")
    assert benchmark.validate_reply(case, reply) == (True, False, "behavior_mismatch")
    refusal = benchmark.default_cases()[2]
    reply.content = json.dumps(refusal.expected)
    assert benchmark.validate_reply(refusal, reply) == (True, True, None)


@pytest.mark.parametrize(("name", "arguments", "code"), [
    ("run_shell", {"query": "sample retention policy", "top_k": 2}, "unapproved_tool_selection"),
    ("knowledge_search", {"query": "sample retention policy", "top_k": True}, "schema_mismatch"),
    ("knowledge_search", {"query": "sample retention policy", "top_k": 2, "tenant_id": "foreign"}, "schema_mismatch"),
    ("knowledge_search", {"query": "other", "top_k": 2}, "behavior_mismatch"),
])
def test_tool_selection_cannot_grant_authority(name, arguments, code):
    case = benchmark.default_cases()[4]
    reply = ModelReply(model=INKLING_MODEL, finish_reason="tool_calls",
                       tool_calls=[ToolCall(id="call_safe", name=name, arguments=arguments)])
    assert benchmark.validate_reply(case, reply)[2] == code


def test_percentiles_and_throughput_use_correct_denominator():
    assert benchmark.percentile([10, 20, 30, 40], .5) == 25
    assert benchmark.percentile([10, 20, 30, 40], .95) == pytest.approx(38.5)
    assert benchmark.percentile([], .95) is None
    sample = {"status": "passed", "latency_ms": 10, "schema_valid": True,
              "usage": {"prompt_tokens": 10, "completion_tokens": 5}, "estimated_cost_usd": None}
    failed = {**sample, "status": "failed", "latency_ms": 90, "schema_valid": False, "usage": {}}
    result = benchmark.summarize([sample, failed], 2.)
    assert result["validated_requests_per_second"] == .5
    assert result["output_tokens_per_second"] == 2.5
    assert result["p95_ms"] == 10 and result["successful_latency_sample_count"] == 1
    assert result["usage_complete"] is False and result["estimated_total_cost_usd"] is None
    with pytest.raises(ValueError):
        benchmark.percentile([True], .95)
    with pytest.raises(ValueError):
        benchmark.summarize([], math.nan)


def test_latency_percentiles_accept_full_ninety_second_request_in_milliseconds():
    assert benchmark.percentile([89000., 90000.], .95) == 89950.
    sample = {"status": "passed", "latency_ms": 90000., "schema_valid": True,
              "usage": {}, "estimated_cost_usd": None}
    report = benchmark.summarize([sample], 90.)
    assert report["p95_ms"] == 90000.
    assert report["validated_requests_per_second"] == pytest.approx(1 / 90)


def test_huge_numbers_rejected_without_overflow():
    for value in (10**1000, -(10**1000)):
        with pytest.raises(ValueError):
            benchmark.Options(timeout_seconds=value)
        with pytest.raises(ValueError):
            benchmark.Prices(value, 1)


def test_usage_sanitizes_nonfinite_bools_and_impossible_counts():
    result = benchmark.clean_usage({"prompt_tokens": 5, "completion_tokens": 2, "cached_prompt_tokens": 6,
                                    "reasoning_tokens": 3, "total_tokens": 100, "secret": 123})
    assert result == {"prompt_tokens": 5, "completion_tokens": 2}
    assert benchmark.clean_usage({"prompt_tokens": True, "completion_tokens": math.inf}) == {}


def test_cost_is_operator_supplied_and_cache_aware():
    usage = {"prompt_tokens": 100, "completion_tokens": 20, "cached_prompt_tokens": 10}
    assert benchmark._cost(usage, None) == (None, "no_operator_price_schedule")
    cost, reason = benchmark._cost(usage, benchmark.Prices(1, 2, .1))
    assert cost == pytest.approx(.000131) and reason == "operator_estimate_with_cache"
    assert benchmark._cost({}, benchmark.Prices(1, 2))[0] is None


def test_fixture_warmup_is_excluded_and_report_has_no_content(monkeypatch):
    cases = benchmark.default_cases()
    by_request = {case.request: case for case in cases}
    calls = []
    def respond(request):
        body = json.loads(request.content)
        calls.append(body)
        assert body["max_completion_tokens"] == 512 and "max_tokens" not in body
        assert body["reasoning_effort"] == "low"
        return httpx.Response(200, json=envelope(by_request[body["messages"][-1]["content"]]))
    install_transport(monkeypatch, respond)
    report = benchmark.run_benchmark(benchmark.Options(repetitions=1, warmup_rounds=1), cases,
        environment={"TINKER_API_KEY": "not-a-real-secret"}, evidence_kind="http_contract_fixture")
    assert report["status"] == "passed" and len(calls) == 24
    assert report["warmup"]["summary"]["passed"] == 6
    assert [p["summary"]["passed"] for p in report["phases"]] == [6, 6, 6]
    assert report["phases"][0]["summary"]["reported_token_totals"]["cached_prompt_tokens"] == 60
    assert report["phases"][0]["samples"][0]["usage"]["reasoning_tokens"] == 5
    serialized = json.dumps(report)
    for private in ("not-a-real-secret", TINKER_BASE_URL, cases[0].request, "sample retention policy",
                    "retention-01", "tool_synthetic"):
        assert private not in serialized
    with pytest.raises(ValueError, match="live-model"):
        benchmark.compare_reports(report, report)


@pytest.mark.parametrize("response", [
    httpx.Response(401, text="private-credential-do-not-log"),
    httpx.Response(429, text="private-credential-do-not-log"),
    httpx.Response(200, json={"choices": []}),
])
def test_provider_failures_remain_failures_without_retry_or_secrets(monkeypatch, response):
    count = 0
    def respond(request):
        nonlocal count
        count += 1
        return response
    install_transport(monkeypatch, respond)
    result = benchmark.run_benchmark(benchmark.Options(concurrency=(1,), repetitions=1, warmup_rounds=0),
        benchmark.default_cases()[:1], environment={"TINKER_API_KEY": "private-credential-do-not-log"},
        evidence_kind="http_contract_fixture")
    assert result["status"] == "failed" and count == 1
    assert result["phases"][0]["summary"]["passed"] == 0
    assert result["phases"][0]["summary"]["p95_ms"] is None
    assert "private-credential-do-not-log" not in json.dumps(result)


def test_truncated_http_response_cannot_pass_benchmark(monkeypatch):
    case = benchmark.default_cases()[0]
    install_transport(monkeypatch, httpx.Response(200, json=envelope(case, reason="length")))
    result = benchmark.run_benchmark(benchmark.Options(concurrency=(1,), repetitions=1, warmup_rounds=0),
        [case], environment={"TINKER_API_KEY": "fixture"}, evidence_kind="http_contract_fixture")
    sample = result["phases"][0]["samples"][0]
    assert result["status"] == "failed" and sample["failure_code"] == "truncated"
    assert sample["usage"]["completion_tokens"] == 20


def test_reasoning_and_unexpected_exception_text_are_not_recorded(monkeypatch):
    case = benchmark.default_cases()[0]
    response = envelope(case)
    response["choices"][0]["message"]["reasoning_content"] = "private-reasoning-token-12345"
    install_transport(monkeypatch, httpx.Response(200, json=response))
    options = benchmark.Options(concurrency=(1,), repetitions=1, warmup_rounds=0)
    report = benchmark.run_benchmark(options, [case], environment={"TINKER_API_KEY": "fixture"},
                                     evidence_kind="http_contract_fixture")
    assert report["status"] == "passed" and "private-reasoning" not in json.dumps(report)
    def fail(self, *_args):
        raise RuntimeError("private-provider-secret must never be logged")
    monkeypatch.setattr(ModelRouter, "_client", fail)
    failed = benchmark.run_benchmark(options, [case], environment={"TINKER_API_KEY": "fixture"},
                                     evidence_kind="http_contract_fixture")
    assert failed["status"] == "failed" and "private-provider-secret" not in json.dumps(failed)
    assert failed["phases"][0]["samples"][0]["failure_code"] == "provider_error"


def test_deadline_exhaustion_prevents_any_request(monkeypatch):
    config = benchmark.AgentPlatformConfig(gateway_url=TINKER_BASE_URL, gateway_key="fixture",
                                           models={"benchmark": INKLING_MODEL})
    router = ModelRouter(config)
    monkeypatch.setattr(router, "complete", lambda **_: pytest.fail("No request after deadline"))
    result = benchmark._sample(router, benchmark.default_cases()[0], 0, benchmark.Options(), time.monotonic() - 1)
    assert result["status"] == "not_run" and result["failure_code"] == "run_deadline_exhausted"


def test_actual_loopback_http_contract_path_is_explicit(monkeypatch):
    case = benchmark.default_cases()[0]
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body))
            encoded = json.dumps(envelope(case)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *_args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        report = benchmark.run_benchmark(benchmark.Options(
            base_url=f"http://127.0.0.1:{server.server_port}/v1", allow_loopback_http=True,
            concurrency=(1, 2, 4), repetitions=2, warmup_rounds=0), [case],
            environment={"TINKER_API_KEY": "loopback-fixture"}, evidence_kind="http_contract_fixture")
        assert report["status"] == "passed" and len(requests) == 6
        assert all(path == "/v1/chat/completions" for path, _ in requests)
        assert report["evidence_kind"] == "http_contract_fixture"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _live_shaped_fixture(monkeypatch):
    report = fixture_run(monkeypatch)
    # For comparator validation only; never persisted as live evidence.
    report["evidence_kind"] = "live_model"
    return report


def test_comparison_enforces_dataset_and_control_parity(monkeypatch):
    original = _live_shaped_fixture(monkeypatch)
    target = deepcopy(original)
    target["target"]["model"] = "thinkingmachines/Inkling-2-Small"
    target["target"]["endpoint_sha256"] = "f" * 64
    compared = benchmark.compare_reports(original, target)
    assert compared["status"] == "comparable" and compared["different_endpoints"]
    assert compared["phases"][0]["p50_latency_ratio_baseline_over_target"] == 1
    for field, new_value in (("dataset_sha256", "e" * 64), ("max_output_tokens", 1024),
                             ("reasoning_effort", "high"), ("warmup_rounds", 2)):
        changed = deepcopy(target)
        changed["measurement"][field] = new_value
        changed["measurement_sha256"] = benchmark._hash(changed["measurement"])
        with pytest.raises(ValueError):
            benchmark.compare_reports(original, changed)


@pytest.mark.parametrize("alteration", [
    lambda r: r.update(status="blocked"),
    lambda r: r.update(evidence_kind="http_contract_fixture"),
    lambda r: r.update(measurement_sha256="bad"),
    lambda r: r["phases"][0]["summary"].update(p50_ms=math.nan),
    lambda r: r["phases"][0]["summary"].update(p50_ms=True),
    lambda r: r["phases"][0]["summary"].update(failed=1),
    lambda r: r["phases"][0].update(concurrency=True),
    lambda r: r["phases"][0]["summary"].update(passed=True),
    lambda r: r["phases"][0]["summary"].update(p50_ms=0),
    lambda r: r["phases"][0]["samples"].pop(),
    lambda r: r["phases"][0]["samples"].__setitem__(0, r["phases"][0]["samples"][1]),
    lambda r: r["phases"][0]["samples"][0].update(behavior_valid=False),
    lambda r: r.update(warmup=None),
    lambda r: r["warmup"]["summary"].update(passed=1),
])
def test_comparison_rejects_non_live_or_malformed_metrics(monkeypatch, alteration):
    original = _live_shaped_fixture(monkeypatch)
    changed = deepcopy(original)
    alteration(changed)
    with pytest.raises(ValueError):
        benchmark.compare_reports(original, changed)


def test_cli_writes_blocked_artifact_and_uses_nonzero_status(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    monkeypatch.delenv("TINKER_API_KEY_FILE", raising=False)
    output = tmp_path / "evidence" / "blocked.json"
    assert benchmark.main(["run", "--output", str(output)]) == 2
    artifact = json.loads(output.read_text())
    assert artifact["status"] == "blocked"
    assert artifact["block_reason"] == "missing_model_credential"
    assert "Traceback" not in capsys.readouterr().out
    assert os.stat(output).st_mode & 0o777 == 0o600


def test_cli_invalid_config_does_not_echo_secret_url(tmp_path, capsys):
    status = benchmark.main(["run", "--base-url", "https://private-secret@example.test/v1",
                             "--output", str(tmp_path / "invalid.json")])
    output = capsys.readouterr().out
    assert status == 2 and "private-secret" not in output and "Traceback" not in output


def test_cli_parse_errors_do_not_echo_values(tmp_path, capsys):
    result = benchmark.main(["run", "--timeout-seconds", "private-credential",
                              "--output", str(tmp_path / "unused.json")])
    captured = capsys.readouterr()
    assert result == 2
    assert "private-credential" not in captured.out + captured.err
    assert "Traceback" not in captured.out + captured.err


def test_dataset_bounds_duplicates_and_schema_are_rejected(tmp_path):
    case = benchmark.default_cases()[0]
    for cases in ([], [case] * 13, [case, case]):
        with pytest.raises(ValueError):
            benchmark.validate_cases(cases)
    path = tmp_path / "cases.json"
    path.write_text('{"version":"unknown","cases":[]}', encoding="utf-8")
    with pytest.raises(ValueError):
        benchmark.load_cases(path)


def test_options_hide_endpoint_from_repr():
    assert TINKER_BASE_URL not in repr(benchmark.Options())
    with pytest.raises(ValueError):
        replace(benchmark.Options(), protocol="openai")
