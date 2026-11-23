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
    real_client = httpx.AsyncClient
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


def deployment_manifest(**changes):
    return benchmark.DeploymentManifest.model_validate({
        "schema_version": "model-deployment-v1", "source": "operator_supplied",
        "runtime_name": "llama.cpp", "runtime_version": "b12631",
        "runtime_revision": "e300919cc704ddde3a8a8346a8714f57eeeee71b",
        "model_repository": "ggml-org/Inkling-2-Mini-GGUF",
        "model_revision": "e8a2cecf0e78914f07cc27b9be711a9b2c7561d9",
        "model_artifact": "Inkling-2-Mini-IQ4_NL.gguf",
        "model_sha256": "cbdf3fa9dc50a4124977980f37da4f1fa764b0cfe800d275cd43d26ce2a81af4",
        "model_bytes": 17627940768, "quantization": "IQ4_NL", "context_tokens": 4096,
        "server_parallel_slots": 1, "server_threads": 8, "hardware_class": "cpu", "hardware_label": "x86_64",
        "host_memory_bytes": 32 * 2**30, "accelerator_count": 0, "accelerator_memory_bytes": 0,
        **changes,
    })


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
    assert result["request_upper_bound"] == 42
    assert "paid_request_upper_bound" not in result
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


def test_local_preset_is_explicit_openai_no_auth_and_unset_effort():
    local = benchmark.Options.for_preset("inkling-local")
    assert local.base_url == "http://127.0.0.1:8080/v1"
    assert local.model == INKLING_MODEL and local.auth_mode == "none"
    assert local.protocol == "openai" and local.effort is None and local.allow_loopback_http
    assert local.key_env == "INKLING_LOCAL_API_KEY"
    hosted = benchmark.Options.for_preset("inkling-tinker")
    assert hosted == benchmark.Options()
    assert hosted.auth_mode == "required" and hosted.effort == "low"
    assert hosted.key_env == "TINKER_API_KEY"


def test_authenticated_local_default_never_reads_hosted_credentials(monkeypatch):
    monkeypatch.setattr(benchmark, "ModelRouter", lambda *_: pytest.fail("No router without dedicated local key"))
    class HostedSecretsMustNotBeRead(dict):
        def get(self, name, default=None):
            assert name in {"INKLING_LOCAL_API_KEY", "INKLING_LOCAL_API_KEY_FILE"}
            return super().get(name, default)
    environment = HostedSecretsMustNotBeRead({
        "TINKER_API_KEY": "unused-hosted-key", "TINKER_API_KEY_FILE": "/private/tinker-key",
        "OPENAI_API_KEY": "unused-openai-key", "OPENAI_API_KEY_FILE": "/private/openai-key",
    })
    report = benchmark.run_benchmark(
        benchmark.Options.for_preset("inkling-local", auth_mode="required",
                                     deployment_manifest=deployment_manifest()), environment=environment)
    assert report["status"] == "blocked" and report["block_reason"] == "missing_model_credential"
    assert report["warmup"] is None and report["phases"] == []


def test_qwen_local_preset_has_dedicated_credentials_and_explicit_protocol(monkeypatch):
    options = benchmark.Options.for_preset("qwen-local")
    assert options.model == "qwen/Qwen3-1.7B" and options.base_url == benchmark.LOCAL_BASE_URL
    assert options.protocol == "openai" and options.auth_mode == "none" and options.effort is None
    assert options.key_env == "QWEN_LOCAL_API_KEY" and options.allow_loopback_http
    class DedicatedKeyOnly(dict):
        def get(self, name, default=None):
            assert name in {"QWEN_LOCAL_API_KEY", "QWEN_LOCAL_API_KEY_FILE"}
            return super().get(name, default)
    monkeypatch.setattr(benchmark, "ModelRouter", lambda *_: pytest.fail("No router without Qwen credential"))
    report = benchmark.run_benchmark(replace(options, auth_mode="required", deployment_manifest=deployment_manifest()),
                                     environment=DedicatedKeyOnly({"TINKER_API_KEY": "private", "OPENAI_API_KEY": "private"}))
    assert report["status"] == "blocked" and report["block_reason"] == "missing_model_credential"
    assert report["warmup"] is None and report["phases"] == []


@pytest.mark.parametrize("key_name,arguments", [
    ("INKLING_LOCAL_API_KEY", []),
    ("DEDICATED_LOCAL_TEST_KEY", ["--key-env", "DEDICATED_LOCAL_TEST_KEY"]),
])
def test_cli_authenticated_local_selects_only_dedicated_or_explicit_key(
    tmp_path, monkeypatch, capsys, key_name, arguments,
):
    path = tmp_path / "deployment.json"
    path.write_text(deployment_manifest().model_dump_json())
    output = tmp_path / "result.json"
    for name in ("TINKER_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(name, "unused-hosted-key")
        monkeypatch.setenv(name + "_FILE", "/private/never-read")
    monkeypatch.setenv(key_name, "dedicated-local-fixture-key")
    monkeypatch.delenv(key_name + "_FILE", raising=False)
    calls = []
    def respond(request):
        calls.append(request)
        return httpx.Response(503, text="synthetic provider failure")
    install_transport(monkeypatch, respond)
    assert benchmark.main([
        "run", "--preset", "inkling-local", "--auth-mode", "required", *arguments,
        "--deployment-manifest", str(path), "--warmup-rounds", "0", "--repetitions", "1",
        "--concurrency", "1", "--output", str(output),
    ]) == 1
    # A deliberate HTTP failure verifies credential selection without producing
    # passing fixture timings labeled as live-model evidence.
    report = json.loads(output.read_text())
    assert report["status"] == "failed" and len(calls) == 6
    assert {request.headers.get("Authorization") for request in calls} == {"Bearer dedicated-local-fixture-key"}
    assert report["target"]["auth_mode"] == "required"
    serialized = output.read_text() + capsys.readouterr().out
    for secret in ("dedicated-local-fixture-key", "unused-hosted-key", "/private/never-read"):
        assert secret not in serialized


@pytest.mark.parametrize("changes", [
    {"base_url": "http://localhost:8080/v1"},
    {"base_url": "https://example.test/v1"},
    {"base_url": "http://192.168.1.1:8080/v1"},
    {"base_url": "http://127.0.0.1:8080/v1?secret=private-value"},
    {"base_url": "http://127.0.0.1:8080/v1#secret"},
    {"base_url": "http://key@127.0.0.1:8080/v1"},
    {"base_url": "http://127.0.0.1:8080/v1", "allow_loopback_http": False},
    {"protocol": "auto"}, {"protocol": "tinker"}, {"effort": "low"}, {"effort": 0},
    {"auth_mode": True}, {"auth_mode": "optional"},
])
def test_no_auth_requires_clean_literal_loopback_and_explicit_wire_contract(changes):
    with pytest.raises(ValueError):
        benchmark.Options.for_preset("inkling-local", **changes)


@pytest.mark.parametrize("url", ["http://127.0.0.2:8080/v1", "http://[::1]:8080/v1"])
def test_other_literal_loopback_addresses_are_explicitly_classified(url, monkeypatch):
    monkeypatch.setattr(benchmark, "ModelRouter", lambda *_: pytest.fail("No request without manifest"))
    report = benchmark.run_benchmark(benchmark.Options.for_preset("inkling-local", base_url=url), environment={})
    assert report["block_reason"] == "missing_deployment_manifest"
    assert report["target"]["endpoint_kind"] == "loopback"


def test_local_live_without_manifest_blocks_without_reading_secrets_or_network(monkeypatch):
    monkeypatch.setattr(benchmark, "_read_key", lambda *_: pytest.fail("No local key lookup"))
    monkeypatch.setattr(benchmark, "ModelRouter", lambda *_: pytest.fail("No router before provenance"))
    report = benchmark.run_benchmark(benchmark.Options.for_preset("inkling-local"),
        environment={"TINKER_API_KEY": "private-unused-secret", "TINKER_API_KEY_FILE": "/private/unused-secret"})
    assert report["status"] == "blocked" and report["block_reason"] == "missing_deployment_manifest"
    assert report["request_upper_bound"] == 42 and report["warmup"] is None and report["phases"] == []
    assert report["target"]["auth_mode"] == "none" and "private-unused-secret" not in json.dumps(report)


@pytest.mark.parametrize("preset,model", [("inkling-local", INKLING_MODEL), ("qwen-local", "qwen/Qwen3-1.7B")])
def test_real_local_no_auth_fixture_omits_auth_and_ignores_ambient_secrets_and_proxies(monkeypatch, preset, model):
    cases = benchmark.default_cases()
    by_request = {case.request: case for case in cases}
    requests = []
    monkeypatch.setattr(benchmark, "_read_key", lambda *_: pytest.fail("No local credential lookup"))
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    monkeypatch.setenv("TINKER_API_KEY", "private-ambient-key")
    monkeypatch.setenv("TINKER_API_KEY_FILE", "/private/not-read")
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, self.headers.get("Authorization"), body))
            reply = envelope(by_request[body["messages"][-1]["content"]])
            reply["model"] = body["model"]
            encoded = json.dumps(reply).encode()
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
        options = benchmark.Options.for_preset(preset, base_url=f"http://127.0.0.1:{server.server_port}/v1",
            concurrency=(1, 2, 4), repetitions=1, warmup_rounds=0)
        report = benchmark.run_benchmark(options, cases, evidence_kind="http_contract_fixture")
        assert report["status"] == "passed" and len(requests) == 18
        for path, auth, body in requests:
            assert path == "/v1/chat/completions" and auth is None
            assert body["model"] == model and body["max_tokens"] == 512
            assert "max_completion_tokens" not in body and "reasoning_effort" not in body
        assert report["evidence_kind"] == "http_contract_fixture" and report["deployment_manifest"] is None
        serialized = json.dumps(report)
        assert "private-ambient-key" not in serialized and "/private/not-read" not in serialized
        with pytest.raises(ValueError, match="live-model"):
            benchmark.compare_reports(report, report)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("changes", [
    {"source": "verified"}, {"schema_version": "unknown"}, {"secret": "never-log-me"},
    {"runtime_revision": "main"}, {"model_revision": "e8a2cec"},
    {"model_sha256": "a" * 63}, {"model_sha256": "A" * 64},
    {"model_repository": "https://huggingface.co/example/model"},
    {"model_artifact": "/private/weights.gguf"}, {"model_artifact": "../weights.gguf"},
    {"hardware_label": "https://private.example"}, {"hardware_label": "x" * 97},
    {"hardware_label": "host\nsecret"}, {"hardware_class": "unknown"},
    {"model_bytes": True}, {"model_bytes": 0}, {"model_bytes": 2**51},
    {"context_tokens": 0}, {"context_tokens": True}, {"context_tokens": 1048577},
    {"server_parallel_slots": 0}, {"server_parallel_slots": True}, {"server_parallel_slots": 257},
    {"server_threads": 0}, {"server_threads": True}, {"server_threads": 1025},
    {"host_memory_bytes": math.inf}, {"accelerator_count": 1},
    {"hardware_class": "cuda"}, {"accelerator_memory_bytes": 1},
])
def test_deployment_manifest_is_strict_and_bounded(changes):
    with pytest.raises(ValueError):
        deployment_manifest(**changes)


def test_deployment_file_rejects_duplicate_keys_oversize_and_unknown_fields_without_echo(tmp_path, capsys):
    path = tmp_path / "operator.json"
    output = tmp_path / "report.json"
    original = deployment_manifest().model_dump_json()
    for text in (original[:-1] + ',"source":"operator_supplied"}', " " * 16385,
                 original[:-1] + ',"api_key":"private-never-log"}'):
        path.write_text(text)
        assert benchmark.main(["run", "--preset", "inkling-local", "--deployment-manifest", str(path),
                               "--output", str(output)]) == 2
        assert not output.exists()
    captured = capsys.readouterr()
    assert "private-never-log" not in captured.out + captured.err and "Traceback" not in captured.out + captured.err
    path.write_text(original)
    assert benchmark.load_deployment_manifest(path) == deployment_manifest()


def test_cli_local_preset_preserves_manifest_and_reaches_router_without_key(tmp_path, monkeypatch, capsys):
    path = tmp_path / "deployment.json"
    path.write_text(deployment_manifest().model_dump_json())
    output = tmp_path / "result.json"
    monkeypatch.setattr(benchmark, "_read_key", lambda *_: pytest.fail("No local key lookup"))
    calls = []
    def respond(request):
        calls.append(request)
        return httpx.Response(503, text="synthetic-private-provider-body")
    install_transport(monkeypatch, respond)
    assert benchmark.main(["run", "--preset", "inkling-local", "--deployment-manifest", str(path),
                           "--warmup-rounds", "0", "--repetitions", "1", "--concurrency", "1",
                           "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    # Deliberately failing transport: no passing live-model fixture evidence is written.
    assert report["status"] == "failed" and len(calls) == 6
    assert report["target"]["auth_mode"] == "none" and report["measurement"]["reasoning_effort"] is None
    assert report["deployment_manifest"] == deployment_manifest().model_dump()
    assert report["deployment_sha256"] == benchmark._hash(report["deployment_manifest"])
    assert "synthetic-private-provider-body" not in output.read_text() + capsys.readouterr().out


def _local_live_shaped_fixture(monkeypatch):
    report = fixture_run(monkeypatch, base_url=benchmark.LOCAL_BASE_URL, allow_loopback_http=True,
                         auth_mode="none", protocol="openai", effort=None, deployment_manifest=deployment_manifest())
    # Comparator input only, never saved or presented as a real model measurement.
    report["evidence_kind"] = "live_model"
    return report


def test_comparison_discloses_valid_deployment_differences(monkeypatch):
    original = _local_live_shaped_fixture(monkeypatch)
    target = deepcopy(original)
    target["deployment_manifest"] = deployment_manifest(
        runtime_version="b12632", runtime_revision="f" * 40, quantization="Q8_0",
        model_artifact="Inkling-2-Mini-Q8_0.gguf", model_sha256="e" * 64, model_bytes=27800000000,
        context_tokens=8192, server_parallel_slots=2, server_threads=16, hardware_label="x86_64-avx512",
    ).model_dump()
    target["deployment_sha256"] = benchmark._hash(target["deployment_manifest"])
    report = benchmark.compare_reports(original, target)
    assert report["status"] == "comparable" and report["different_deployments"]
    assert report["deployment_information_complete"]
    assert {"quantization", "model_sha256", "context_tokens", "server_parallel_slots", "server_threads", "hardware_label"} <= set(
        report["deployment_fields_differ"])
    assert report["baseline_deployment"]["source"] == report["target_deployment"]["source"] == "operator_supplied"
    assert report["reasoning_effort_unset"]
    assert "not parallel decoding" in " ".join(report["limitations"])


@pytest.mark.parametrize("alteration", [
    lambda r: r.update(schema_version="model-contract-benchmark-v1"),
    lambda r: r.update(deployment_manifest=None, deployment_sha256=None),
    lambda r: r.update(deployment_sha256="0" * 64),
    lambda r: r["deployment_manifest"].update(server_parallel_slots=True),
    lambda r: r["deployment_manifest"].update(unknown="private-value"),
    lambda r: r["target"].update(endpoint_kind="operator_endpoint"),
    lambda r: r["target"].update(protocol="auto"),
    lambda r: r["target"].update(auth_mode="optional"),
    lambda r: r.update(request_upper_bound=True),
    lambda r: r.update(request_upper_bound=0),
])
def test_local_comparison_rejects_missing_or_tampered_provenance(monkeypatch, alteration):
    original = _local_live_shaped_fixture(monkeypatch)
    target = deepcopy(original)
    alteration(target)
    with pytest.raises(ValueError):
        benchmark.compare_reports(original, target)


def test_comparison_discloses_authenticated_hosted_vs_local_conditions(monkeypatch):
    local = _local_live_shaped_fixture(monkeypatch)
    hosted = deepcopy(local)
    hosted["target"].update(endpoint_kind="tinker_sampling", protocol="tinker", auth_mode="required",
                            endpoint_sha256="f" * 64)
    hosted.update(deployment_manifest=None, deployment_sha256=None)
    result = benchmark.compare_reports(hosted, local)
    assert result["different_deployments"] and not result["deployment_information_complete"]
    assert result["different_protocols"] and result["different_auth_modes"]


@pytest.mark.parametrize("preset,model", [("inkling-litellm", "gateway/inkling-2-mini-iq4nl-v1"),
                                         ("qwen-litellm", "gateway/qwen3-1.7b-q8_0-v1")])
def test_litellm_preset_requires_explicit_deployment_before_credentials_or_network(monkeypatch, preset, model):
    options = benchmark.Options.for_preset(preset)
    assert options.model == model
    assert options.auth_mode == "required" and options.protocol == "openai" and options.effort is None
    assert options.require_deployment_manifest and not options.allow_loopback_http
    assert options.key_env == "LITELLM_BENCHMARK_API_KEY"
    monkeypatch.setattr(benchmark, "_read_key", lambda *_: pytest.fail("No secret lookup without manifest"))
    report = benchmark.run_benchmark(options)
    assert report["status"] == "blocked" and report["block_reason"] == "missing_deployment_manifest"


def test_benchmark_passes_explicit_ca_without_disclosing_its_path(monkeypatch, tmp_path):
    import certifi
    ca = tmp_path / "private-ca-path.pem"
    ca.write_bytes(Path(certifi.where()).read_bytes())
    captured = []
    real = benchmark.ModelRouter
    def router(config):
        captured.append(config)
        return real(config)
    monkeypatch.setattr(benchmark, "ModelRouter", router)
    report = fixture_run(monkeypatch, ca_file=str(ca))
    assert report["status"] == "passed"
    assert captured and captured[0].gateway_ca_sha256
    assert str(ca) not in json.dumps(report)


@pytest.mark.parametrize("mode", ["missing", "symlink", "invalid", "http"])
def test_benchmark_custom_ca_fails_closed(tmp_path, mode):
    import certifi
    ca = tmp_path / "ca.pem"
    if mode == "symlink":
        ca.symlink_to(certifi.where())
    elif mode == "invalid":
        ca.write_text("not a certificate")
    elif mode == "http":
        ca.write_bytes(Path(certifi.where()).read_bytes())
    with pytest.raises(ValueError):
        benchmark.Options.for_preset("inkling-local", ca_file=str(ca))


@pytest.mark.parametrize("change", ["none", "weights", "missing_manifest", "client", "protocol"])
def test_gateway_comparison_requires_same_deployment_protocol_and_client(
    monkeypatch, tmp_path, capsys, change,
):
    original = _local_live_shaped_fixture(monkeypatch)
    target = deepcopy(original)
    target["target"].update(model="gateway/inkling-2-mini-iq4nl-v1", endpoint_kind="operator_endpoint",
                             auth_mode="required", endpoint_sha256="f" * 64)
    if change == "weights":
        target["deployment_manifest"]["model_sha256"] = "b" * 64
        target["deployment_sha256"] = benchmark._hash(target["deployment_manifest"])
    elif change == "missing_manifest":
        target.update(deployment_manifest=None, deployment_sha256=None)
    elif change == "client":
        target["runtime"]["python"] = "3.13.0"
    elif change == "protocol":
        target["target"]["protocol"] = "tinker"
    left, right, output = (tmp_path / name for name in ("direct.json", "proxy.json", "comparison.json"))
    left.write_text(json.dumps(original))
    right.write_text(json.dumps(target))
    result = benchmark.main(["compare", str(left), str(right), "--require-same-deployment", "--output", str(output)])
    if change == "none":
        assert result == 0
        assert json.loads(output.read_text())["comparison_scope"] == "same_deployment_gateway_path"
    else:
        assert result == 2 and not output.exists()
    assert "Traceback" not in capsys.readouterr().out


@pytest.mark.parametrize("alteration", [
    lambda report: report.pop("runtime"),
    lambda report: report.update(runtime=None),
    lambda report: report.update(runtime={}),
    lambda report: report.update(runtime=[]),
    lambda report: report["runtime"].pop("implementation"),
    lambda report: report["runtime"].update(extra="private-runtime-value"),
    lambda report: report["runtime"].update(python=None),
    lambda report: report["runtime"].update(python="not-a-python-version"),
    lambda report: report["runtime"].update(implementation=""),
    lambda report: report["runtime"].update(implementation="x" * 257),
    lambda report: report["runtime"].update(os=True),
    lambda report: report["runtime"].update(machine="private-runtime-value\n"),
    lambda report: report["runtime"].update(package_version=None),
    lambda report: report["runtime"].update(package_version=""),
    lambda report: report["runtime"].update(cpu_count=True),
    lambda report: report["runtime"].update(cpu_count=0),
    lambda report: report["runtime"].update(cpu_count=-1),
    lambda report: report["runtime"].update(cpu_count=1.5),
    lambda report: report["runtime"].update(cpu_count=2**63),
    lambda report: report["runtime"].update(adapter_sha256="private-runtime-value"),
    lambda report: report["runtime"].update(harness_sha256=None),
])
def test_gateway_comparison_rejects_matching_malformed_runtime_evidence(
    monkeypatch, tmp_path, capsys, alteration,
):
    # Identical missing/invalid evidence on both sides must not count as parity.
    original = _local_live_shaped_fixture(monkeypatch)
    alteration(original)
    target = deepcopy(original)
    with pytest.raises(ValueError):
        benchmark.compare_reports(original, target)
    left, right, output = (tmp_path / name for name in ("direct.json", "proxy.json", "comparison.json"))
    left.write_text(json.dumps(original))
    right.write_text(json.dumps(target))
    result = benchmark.main(["compare", str(left), str(right), "--require-same-deployment", "--output", str(output)])
    assert result == 2 and not output.exists()
    captured = capsys.readouterr()
    assert "private-runtime-value" not in captured.out + captured.err
    assert "Traceback" not in captured.out + captured.err


@pytest.mark.parametrize("source", ["adapter_sha256", "harness_sha256"])
@pytest.mark.parametrize("changed_section", ["runtime", "measurement"])
def test_comparison_binds_runtime_source_identity_to_measurement(monkeypatch, source, changed_section):
    original = _local_live_shaped_fixture(monkeypatch)
    original[changed_section][source] = "0" * 64
    original["measurement_sha256"] = benchmark._hash(original["measurement"])
    # Each side remains internally inconsistent even when both copies agree.
    with pytest.raises(ValueError):
        benchmark.compare_reports(original, deepcopy(original))


def test_comparison_accepts_generated_runtime_when_platform_information_is_unknown(monkeypatch):
    monkeypatch.setattr(benchmark.platform, "python_version", lambda: "3.13.0rc1")
    monkeypatch.setattr(benchmark.platform, "system", lambda: "")
    monkeypatch.setattr(benchmark.platform, "machine", lambda: "")
    monkeypatch.setattr(benchmark.os, "cpu_count", lambda: None)
    original = _local_live_shaped_fixture(monkeypatch)
    assert original["runtime"]["cpu_count"] is None
    result = benchmark.compare_reports(original, deepcopy(original))
    assert result["status"] == "comparable" and not result["different_client_runtime"]
