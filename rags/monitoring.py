"""Bounded, privacy-preserving OTLP/HTTP JSON tracing and metrics.

This exporter uses the vendor-neutral OTLP JSON wire format with existing
httpx rather than claiming to be the full OpenTelemetry Python SDK. The
collector endpoint is operator-configured; no request can change it.
"""
from collections import Counter, deque
from dataclasses import dataclass
from datetime import UTC, datetime
import math
import queue
import re
import secrets
from statistics import median
from threading import Event, Lock, Thread
import time

import httpx


BUCKETS_MS = (5, 10, 25, 50, 100, 250, 500, 1000, 2500)
MAX_TENANTS = 64
MAX_EVENTS_PER_TENANT = 600
MAX_SERIES = 120
MAX_QUEUE = 512
ROUTE = re.compile(r"^/[a-z0-9_{}./-]{1,100}$")
SCOPE = "rags.http"
VERSION = "0.7.0"


def route_name(scope: dict) -> str:
    """Only use application-defined route templates; never actual URL IDs."""
    route = scope.get("route")
    path = getattr(route, "path", "")
    if isinstance(path, str) and ROUTE.fullmatch(path):
        return path
    return "/unmatched"


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return round(values[math.ceil(fraction * len(values)) - 1], 2)


def _attribute(key: str, value) -> dict:
    if isinstance(value, bool):
        encoded = {"boolValue": value}
    elif isinstance(value, int):
        encoded = {"intValue": str(value)}
    elif isinstance(value, float):
        encoded = {"doubleValue": value}
    else:
        encoded = {"stringValue": str(value)}
    return {"key": key, "value": encoded}


@dataclass(frozen=True)
class RequestEvent:
    time: float
    route: str
    method: str
    status: int
    duration_ms: float
    start_ns: int
    end_ns: int


class Monitor:
    """Process-local dashboard pulse and asynchronous OpenTelemetry collector."""

    def __init__(self, *, enabled: bool = False, endpoint: str = "", service_name: str = "rags-platform",
                 start_worker: bool = True):
        self.enabled = enabled
        self.endpoint = endpoint
        self.service_name = service_name
        self._lock = Lock()
        self._per_tenant: dict[str, deque[RequestEvent]] = {}
        self._counts: Counter[tuple[str, str, str]] = Counter()
        self._durations: dict[tuple[str, str, str], tuple[int, float, list[int]]] = {}
        self._start_ns = time.time_ns()
        self._instance = secrets.token_hex(8)
        self._queue: queue.Queue[RequestEvent] = queue.Queue(maxsize=MAX_QUEUE)
        self._stopping = Event()
        self.exported = 0
        self.dropped = 0
        self.export_failures = 0
        self._thread: Thread | None = None
        if enabled and start_worker:
            self._thread = Thread(target=self._work, name="rags-otlp", daemon=True)
            self._thread.start()

    def record(self, *, tenant: str | None, route: str, method: str, status: int,
               duration_ms: float, start_ns: int, end_ns: int) -> None:
        # No raw path, tokens, headers, prompt text, document IDs or exception messages.
        if not ROUTE.fullmatch(route):
            route = "/unmatched"
        method = method if method in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"} else "OTHER"
        status = status if isinstance(status, int) and 100 <= status <= 599 else 500
        duration_ms = max(0., min(float(duration_ms), 120_000.))
        event = RequestEvent(time.time(), route, method, status, duration_ms, start_ns, end_ns)
        group = f"{status // 100}xx"
        key = (route, method, group)
        with self._lock:
            if tenant is not None:
                if tenant not in self._per_tenant and len(self._per_tenant) >= MAX_TENANTS:
                    # Do not let unbounded attacker-controlled tenants exhaust memory.
                    self._per_tenant.pop(next(iter(self._per_tenant)))
                self._per_tenant.setdefault(tenant, deque(maxlen=MAX_EVENTS_PER_TENANT)).append(event)
            if key not in self._counts and len(self._counts) >= MAX_SERIES:
                key = ("/other", "OTHER", "other")
            self._counts[key] += 1
            count, total, buckets = self._durations.get(key, (0, 0., [0] * (len(BUCKETS_MS) + 1)))
            bucket = next((i for i, limit in enumerate(BUCKETS_MS) if duration_ms <= limit), len(BUCKETS_MS))
            buckets[bucket] += 1
            self._durations[key] = (count + 1, total + duration_ms, buckets)
        if self.enabled:
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                with self._lock:
                    self.dropped += 1

    def snapshot(self, tenant: str) -> dict:
        now = time.time()
        with self._lock:
            recent = [e for e in self._per_tenant.get(tenant, ()) if e.time >= now - 3600]
            stats = {"enabled": self.enabled, "exported": self.exported,
                     "dropped": self.dropped, "export_failures": self.export_failures,
                     "scope": "single_process"}
        five_minutes = [e for e in recent if e.time >= now - 300]
        errors = sum(e.status >= 500 for e in five_minutes)
        series = []
        base = int(now // 300) * 300
        for step in range(11, -1, -1):
            start = base - step * 300
            sample = [e for e in recent if start <= e.time < start + 300]
            series.append({"at": datetime.fromtimestamp(start, UTC).isoformat(timespec="seconds"),
                           "requests": len(sample), "errors": sum(e.status >= 500 for e in sample)})
        return {"requests_5m": len(five_minutes), "errors_5m": errors,
                "error_rate_5m": round(errors / len(five_minutes) * 100, 1) if five_minutes else 0.,
                "p50_ms_5m": percentile([e.duration_ms for e in five_minutes], 0.5),
                "p95_ms_5m": percentile([e.duration_ms for e in five_minutes], 0.95),
                "series": series, "otel": stats}

    def _resource(self) -> dict:
        return {"attributes": [
            _attribute("service.name", self.service_name),
            _attribute("service.version", VERSION),
            _attribute("service.instance.id", self._instance),
        ]}

    def _traces(self, events: list[RequestEvent]) -> dict:
        spans = []
        for item in events:
            spans.append({
                "traceId": secrets.token_hex(16), "spanId": secrets.token_hex(8),
                "name": item.method + " " + item.route, "kind": 2,
                "startTimeUnixNano": str(item.start_ns), "endTimeUnixNano": str(item.end_ns),
                "attributes": [_attribute("http.request.method", item.method),
                               _attribute("http.route", item.route),
                               _attribute("http.response.status_code", item.status)],
                "status": {"code": 2 if item.status >= 500 else 1},
            })
        return {"resourceSpans": [{"resource": self._resource(),
                                   "scopeSpans": [{"scope": {"name": SCOPE, "version": VERSION},
                                                   "spans": spans}]}]}

    def _metrics(self) -> dict:
        with self._lock:
            counts = dict(self._counts)
            histograms = {key: (count, total, list(buckets))
                          for key, (count, total, buckets) in self._durations.items()}
        at = str(time.time_ns())
        start = str(self._start_ns)
        requests, durations = [], []
        for key, count in counts.items():
            attrs = [_attribute("http.route", key[0]), _attribute("http.request.method", key[1]),
                     _attribute("http.response.status_class", key[2])]
            requests.append({"attributes": attrs, "startTimeUnixNano": start,
                             "timeUnixNano": at, "asInt": str(count)})
            n, total, buckets = histograms[key]
            durations.append({"attributes": attrs, "startTimeUnixNano": start,
                              "timeUnixNano": at, "count": str(n),
                              "sum": total, "explicitBounds": list(BUCKETS_MS),
                              "bucketCounts": [str(i) for i in buckets]})
        return {"resourceMetrics": [{"resource": self._resource(),
                                     "scopeMetrics": [{
                                         "scope": {"name": SCOPE, "version": VERSION},
                                         "metrics": [
                                             {"name": "rags.http.requests", "unit": "1",
                                              "sum": {"aggregationTemporality": 2,
                                                      "isMonotonic": True, "dataPoints": requests}},
                                             {"name": "rags.http.duration", "unit": "ms",
                                              "histogram": {"aggregationTemporality": 2,
                                                            "dataPoints": durations}},
                                         ],
                                     }]}]}

    def _post(self, suffix: str, payload: dict) -> None:
        with httpx.Client(timeout=2., follow_redirects=False, trust_env=False) as client:
            response = client.post(self.endpoint + suffix, json=payload,
                                   headers={"Content-Type": "application/json"})
            if not 200 <= response.status_code < 300:
                raise ValueError("Collector rejected telemetry")

    def flush(self, events: list[RequestEvent]) -> None:
        if not self.enabled or not events:
            return
        try:
            self._post("/v1/traces", self._traces(events))
            self._post("/v1/metrics", self._metrics())
        except (httpx.HTTPError, OSError, ValueError):
            with self._lock:
                self.export_failures += 1
        else:
            with self._lock:
                self.exported += len(events)

    def _work(self) -> None:
        while not self._stopping.is_set() or not self._queue.empty():
            batch = []
            try:
                batch.append(self._queue.get(timeout=0.3))
            except queue.Empty:
                continue
            while len(batch) < 50:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            self.flush(batch)
            for _ in batch:
                self._queue.task_done()

    def shutdown(self) -> None:
        self._stopping.set()
        if self._thread:
            self._thread.join(timeout=3.)
