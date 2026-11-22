"""Telemetry and HTTP OTLP privacy, bounds and protocol unit contracts."""
import json
import math
import os
from types import SimpleNamespace

import httpx
import pytest

from rags.config import Settings
from rags.monitoring import MAX_QUEUE, MAX_TENANTS, Monitor, percentile, route_name

pytestmark = pytest.mark.unit


def test_route_templates_cannot_leak_user_supplied_paths():
    assert route_name({"route": SimpleNamespace(path="/documents/{document_id}")}) == "/documents/{document_id}"
    for value in ("/documents/private-patient-123?query=medical",
                  "/patients/日本語", "", None, "https://collector.internal/private"):
        assert route_name({"route": SimpleNamespace(path=value)}) == "/unmatched"
    assert route_name({"path": "/documents/sensitive"}) == "/unmatched"


def test_per_tenant_metrics_are_bounded_and_private():
    mon = Monitor()
    mon.record(tenant="tenant-a", route="/query", method="POST", status=200,
               duration_ms=12., start_ns=10, end_ns=20)
    mon.record(tenant="tenant-a", route="/query", method="POST", status=503,
               duration_ms=90., start_ns=20, end_ns=30)
    mon.record(tenant="tenant-b", route="/documents/{document_id}", method="GET", status=200,
               duration_ms=5., start_ns=40, end_ns=50)
    a, b = mon.snapshot("tenant-a"), mon.snapshot("tenant-b")
    assert a["requests_5m"] == 2 and a["errors_5m"] == 1 and a["error_rate_5m"] == 50.
    assert b["requests_5m"] == 1 and b["errors_5m"] == 0
    assert a["p50_ms_5m"] == 12. and a["p95_ms_5m"] == 90.
    assert sum(p["requests"] for p in a["series"]) == 2
    assert a["otel"]["enabled"] is False and a["otel"]["scope"] == "single_process"
    assert percentile([], .95) is None


def test_otlp_json_shapes_include_traces_and_cumulative_metrics_without_secrets(monkeypatch):
    mon = Monitor(enabled=True, endpoint="https://collector.example.com", start_worker=False)
    mon.record(tenant="secret-tenant", route="/documents/{document_id}", method="GET", status=404,
               duration_ms=31.2, start_ns=1900000000000000000, end_ns=1900000000031200000)
    mon.record(tenant="secret-tenant", route="/query", method="POST", status=200,
               duration_ms=12.2, start_ns=1900000001000000000, end_ns=1900000001012200000)
    seen = []
    mon._post = lambda path, payload: seen.append((path, payload))
    event1 = mon._queue.get_nowait()
    event2 = mon._queue.get_nowait()
    mon.flush([event1, event2])
    assert [path for path, _ in seen] == ["/v1/traces", "/v1/metrics"]
    payload = seen[0][1]
    group = payload["resourceSpans"][0]
    assert group["resource"]["attributes"][0]["value"]["stringValue"] == "rags-platform"
    spans = group["scopeSpans"][0]["spans"]
    assert len(spans) == 2
    assert len(spans[0]["traceId"]) == 32 and len(spans[0]["spanId"]) == 16
    assert spans[0]["name"] == "GET /documents/{document_id}"
    assert spans[0]["startTimeUnixNano"] == "1900000000000000000"
    metrics = seen[1][1]["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]
    assert metrics[0]["name"] == "rags.http.requests"
    assert metrics[0]["sum"]["isMonotonic"] is True
    assert metrics[0]["sum"]["aggregationTemporality"] == 2
    assert metrics[1]["name"] == "rags.http.duration"
    assert metrics[1]["histogram"]["aggregationTemporality"] == 2
    assert len(metrics[1]["histogram"]["dataPoints"][0]["bucketCounts"]) == 10
    serialized = json.dumps(seen)
    for forbidden in ("secret-tenant", "Bearer ", "api-key", "private-patient", "password"):
        assert forbidden not in serialized
    assert mon.snapshot("secret-tenant")["otel"]["exported"] == 2


def test_otlp_collector_error_does_not_fail_request_or_expose_exception():
    mon = Monitor(enabled=True, endpoint="https://collector.invalid", start_worker=False)
    mon.record(tenant="tenant", route="/query", method="POST", status=502, duration_ms=10,
               start_ns=1, end_ns=2)
    event = mon._queue.get_nowait()
    mon._post = lambda *_a, **_k: (_ for _ in ()).throw(httpx.ConnectError("private"))
    mon.flush([event])
    assert mon.export_failures == 1
    assert mon.snapshot("tenant")["otel"]["exported"] == 0
    assert "private" not in json.dumps(mon.snapshot("tenant"))


def test_otel_queue_and_tenant_memory_have_finite_bounds():
    mon = Monitor(enabled=True, endpoint="https://collector.invalid", start_worker=False)
    for i in range(MAX_QUEUE + 9):
        mon.record(tenant="tenant-a", route="/query", method="POST", status=200,
                   duration_ms=1., start_ns=10, end_ns=20)
    assert mon.dropped == 9
    assert mon._queue.qsize() == MAX_QUEUE
    for i in range(MAX_TENANTS + 12):
        mon.record(tenant="tenant-" + str(i), route="/query", method="GET", status=200,
                   duration_ms=1., start_ns=10, end_ns=20)
    assert len(mon._per_tenant) <= MAX_TENANTS
    assert mon.snapshot("tenant-a")["requests_5m"] <= 600
    mon.shutdown()


@pytest.fixture(autouse=True)
def no_env_leak(monkeypatch):
    for key in ("RAGS_OTEL_ENABLED", "RAGS_OTEL_ENDPOINT", "RAGS_OTEL_ALLOW_INSECURE",
                "RAGS_OTEL_SERVICE_NAME", "RAGS_ADMIN_ENABLED", "RAGS_ADMIN_DEV_HTTP"):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("name,value", [
    ("RAGS_ADMIN_ENABLED", "yes"), ("RAGS_ADMIN_DEV_HTTP", "1"),
    ("RAGS_OTEL_ENABLED", "banana"), ("RAGS_OTEL_ALLOW_INSECURE", "YES"),
])
def test_boolean_controls_fail_closed(monkeypatch, name, value):
    monkeypatch.setenv("RAGS_ENV", "test")
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        Settings.from_env()


@pytest.mark.parametrize(("endpoint", "allow"), [
    ("", False), ("http://collector.example:4318", False),
    ("ftp://collector.example", False),
    ("https://user:password@collector.example", False),
    ("https://collector.example/?token=secret", False),
])
def test_otel_endpoint_validation_rejects_unsafe_values(monkeypatch, endpoint, allow):
    monkeypatch.setenv("RAGS_ENV", "test")
    monkeypatch.setenv("RAGS_OTEL_ENABLED", "true")
    monkeypatch.setenv("RAGS_OTEL_ENDPOINT", endpoint)
    monkeypatch.setenv("RAGS_OTEL_ALLOW_INSECURE", "true" if allow else "false")
    with pytest.raises(ValueError, match="OTLP endpoint"):
        Settings.from_env()


def test_otel_insecure_optin_and_production_admin_http_prohibited(monkeypatch):
    monkeypatch.setenv("RAGS_ENV", "test")
    monkeypatch.setenv("RAGS_OTEL_ENABLED", "true")
    monkeypatch.setenv("RAGS_OTEL_ENDPOINT", "http://otel-collector:4318")
    monkeypatch.setenv("RAGS_OTEL_ALLOW_INSECURE", "true")
    values = Settings.from_env()
    assert values.otel_enabled and values.otel_endpoint == "http://otel-collector:4318"
    monkeypatch.setenv("RAGS_ENV", "production")
    monkeypatch.setenv("RAGS_ADMIN_ENABLED", "true")
    monkeypatch.setenv("RAGS_ADMIN_DEV_HTTP", "true")
    monkeypatch.setenv("RAGS_KEYS_JSON", '{"abcdefghijklmnopqrstuvwxyzzzz":"local"}')
    with pytest.raises(ValueError, match="only permitted"):
        Settings.from_env()
