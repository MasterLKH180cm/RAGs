"""Real process acceptance for Agent Platform; fixture replies are not live-model evidence."""
import argparse
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid

import httpx

ROOT = Path(__file__).resolve().parents[1]
MODEL = "thinkingmachines/Inkling-2-Mini"
QUESTION = "What is the session timeout?"
EXPECTED = ["inkling_wire_contract", "agent_rag_tool_roundtrip", "truncated_tool_blocked",
            "tenant_isolation", "workflow_restart_and_revision", "admin_cookie_csrf"]
BROWSER_CHECKS = ["browser_agent_workflow_logout"]


def require(value, message):
    if not value:
        raise AssertionError(message)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextmanager
def model_fixture():
    """Fixed local response server: never resolves an external destination."""
    calls, failures = [], []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            status = 200
            try:
                require(self.path == "/v1/chat/completions", "Unexpected fixture route")
                size = int(self.headers.get("content-length", "0"))
                require(0 < size <= 150_000, "Unbounded model request")
                request = json.loads(self.rfile.read(size))
                require(request["model"] == MODEL, "Incorrect model identity")
                require(request.get("max_completion_tokens") == 512, "Incorrect output limit")
                require("max_tokens" not in request and "response_format" not in request,
                        "Unsupported provider parameter was sent")
                require(request.get("reasoning_effort") == "low", "Reasoning effort was lost")
                calls.append({"model": request["model"], "max_completion_tokens": 512,
                              "reasoning_effort": "low"})
                messages = request["messages"]
                goal = next(item["content"] for item in messages if item["role"] == "user")
                results = [item for item in messages if item["role"] == "tool"]
                reason = "tool_calls"
                if "truncate-probe" in goal:
                    reason = "length"
                    name, arguments = "remember_observation", {"fact": "Must not be saved", "source_id": "probe"}
                elif results:
                    citation = json.loads(results[-1]["content"])["citations"][0]["chunk_id"]
                    message = {"role": "assistant", "content": f"The session timeout is 30 minutes. [{citation}]"}
                    reason = "stop"
                else:
                    name, arguments = "knowledge_search", {"question": QUESTION, "mode": "sparse", "top_k": 3}
                if reason != "stop":
                    message = {"role": "assistant", "content": None, "tool_calls": [
                        {"id": "call_fixture", "type": "function", "function": {
                            "name": name, "arguments": json.dumps(arguments)}}]}
                payload = {"model": MODEL, "choices": [{"finish_reason": reason, "message": message}],
                           "usage": {"prompt_tokens": 30, "completion_tokens": 15, "total_tokens": 45}}
            except (AssertionError, ValueError, KeyError, IndexError, StopIteration) as exc:
                failures.append(type(exc).__name__)
                status, payload = 400, {"error": {"message": "Local fixture contract failed"}}
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .05}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", calls, failures
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        require(not thread.is_alive(), "Fixture did not stop")


class Application:
    def __init__(self, env, directory):
        self.env, self.directory = env, directory
        self.url = "http://127.0.0.1:" + str(free_port())
        self.process = None
        self.log = None

    def start(self):
        self.log = open(self.directory / "api.log", "a")
        self.process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "rags.api:app", "--host", "127.0.0.1",
             "--port", self.url.rsplit(":", 1)[1]], cwd=ROOT, env=self.env,
            stdout=self.log, stderr=self.log,
        )
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            require(self.process.poll() is None, "API exited before readiness")
            try:
                if httpx.get(self.url + "/live", timeout=1, trust_env=False).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(.05)
        raise AssertionError("API readiness timed out")

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
                raise AssertionError("API required forced cleanup") from None
        if self.log:
            self.log.close()


def check_http(app, token, other, checks):
    with httpx.Client(base_url=app.url, timeout=20, trust_env=False) as client:
        headers = {"Authorization": "Bearer " + token}
        foreign = {"Authorization": "Bearer " + other}

        def request(method, path, *, body=None, expected=200, auth=headers):
            response = client.request(method, path, json=body, headers=auth)
            require(response.status_code == expected, f"Unexpected HTTP status for {method} {path}")
            return response.json() if response.content else None

        request("POST", "/documents", expected=201, body={"document_id": "session-policy",
                "title": "Session policy", "text": "Session timeout is 30 minutes after inactivity."})
        run = request("POST", "/agents/runs", expected=201,
                      body={"goal": QUESTION, "model": "inkling-mini", "session_id": "process-test"})
        require(run["status"] == "completed" and "30 minutes" in run["answer"], "Agent did not finish")
        require(run["tool_calls"] == 1 and run["steps"] == 2, "Real retrieval tool did not execute exactly once")
        history = request("GET", "/agents/sessions/process-test/messages")["messages"]
        require([item["role"] for item in history] == ["user", "assistant"], "Conversation was not persisted")
        checks.extend([{"id": EXPECTED[0], "passed": True}, {"id": EXPECTED[1], "passed": True}])
        truncated = request("POST", "/agents/runs", expected=201,
                            body={"goal": "truncate-probe", "model": "inkling-mini", "session_id": "truncated"})
        require(truncated["status"] in {"failed", "stopped"} and truncated["tool_calls"] == 0,
                "Truncated tool call reached execution or approval")
        require(not request("GET", "/agents/sessions/truncated/observations")["observations"],
                "Truncated output mutated memory")
        checks.append({"id": EXPECTED[2], "passed": True})
        request("GET", "/agents/runs/" + run["run_id"], expected=404, auth=foreign)
        require(not request("GET", "/agents/sessions/process-test/messages", auth=foreign)["messages"],
                "Memory crossed tenant boundary")
        checks.append({"id": EXPECTED[3], "passed": True})
        workflow = request("POST", "/agents/workflows/evidence_review/runs", expected=201,
                           body={"question": QUESTION, "mode": "sparse", "top_k": 3})
        require(workflow["status"] == "waiting_for_approval", "Workflow did not pause for approval")
        path = "/agents/workflows/runs/" + workflow["run_id"]
        app.stop()
        app.start()
        persisted = request("GET", path)
        require(persisted["revision"] == workflow["revision"], "Checkpoint changed across restart")
        request("POST", path + "/resume", expected=409,
                body={"expected_revision": 0, "approved": True})
        completed = request("POST", path + "/resume", body={
            "expected_revision": persisted["revision"], "approved": True})
        require(completed["status"] == "completed", "Approved workflow did not complete")
        request("POST", path + "/resume", expected=409, body={
            "expected_revision": completed["revision"], "approved": True})
        checks.append({"id": EXPECTED[4], "passed": True})
        login = client.post("/admin/api/session", headers={**headers, "Origin": app.url})
        require(login.status_code == 200, "Admin sign-in failed")
        session = client.get("/admin/api/session").json()
        require(client.delete("/admin/api/session", headers={"Origin": app.url}).status_code == 403,
                "Admin CSRF guard failed")
        require(client.delete("/admin/api/session", headers={"Origin": app.url,
                    "X-CSRF-Token": session["csrf"]}).status_code == 204, "Admin sign-out failed")
        require(client.get("/admin/api/session").status_code == 401, "Revoked session still worked")
        checks.append({"id": EXPECTED[5], "passed": True})


def check_browser(app, token, executable, checks):
    from playwright.sync_api import expect, sync_playwright
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=executable)
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.goto(app.url + "/admin/")
            page.locator("#admin-token").fill(token)
            page.locator("#login-submit").click()
            expect(page.locator("#workspace")).to_be_visible()
            page.locator('[data-view="agents"]').click()
            expect(page.locator("#agent-model")).to_have_value("inkling-mini")
            page.locator("#agent-goal").fill(QUESTION)
            page.locator("#agent-session").fill("browser-test")
            page.locator("#agent-submit").click()
            expect(page.locator("#agent-status")).to_have_text("completed", timeout=15000)
            expect(page.locator("#agent-answer")).to_contain_text("30 minutes")
            expect(page.locator("#workflow-template")).to_have_value("evidence_review")
            expect(page.locator("#workflow-start")).to_be_enabled()
            page.locator("#workflow-question").fill(QUESTION)
            page.locator("#workflow-start").click()
            expect(page.locator("#workflow-status")).to_have_text("waiting_for_approval", timeout=15000)
            page.locator("#workflow-approve").click()
            expect(page.locator("#workflow-status")).to_have_text("completed", timeout=15000)
            stored = page.evaluate("JSON.stringify([Object.entries(localStorage),Object.entries(sessionStorage)])")
            require(token not in stored and token not in page.url, "Credential persisted in browser storage or URL")
            page.locator("#logout-button").click()
            expect(page.locator("#auth-screen")).to_be_visible()
            expect(page.locator("#agent-output")).to_be_hidden()
            require(page.request.get(app.url + "/admin/api/session").status == 401,
                    "Browser logout did not revoke the server session")
            checks.append({"id": BROWSER_CHECKS[0], "passed": True, "browser_version": browser.version})
        finally:
            browser.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--browser", action="store_true", help="Require the actual Chromium UI gate")
    parser.add_argument("--browser-executable")
    parser.add_argument("--postgres", action="store_true", help="Use disposable RAGS_E2E_POSTGRES_URL")
    args = parser.parse_args()
    report = {"schema_version": "agent-platform-process-v09", "status": "failed",
              "model_evidence": "local_http_contract_fixture", "live_model": False,
              "storage": "postgresql" if args.postgres else "sqlite", "checks": [],
              "expected_checks": EXPECTED + (BROWSER_CHECKS if args.browser else []), "cleanup_completed": False}
    app = None
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="rags-platform-e2e-") as temporary, model_fixture() as fixture:
            directory = Path(temporary)
            endpoint, calls, failures = fixture
            token, other = "admin-" + uuid.uuid4().hex, "other-" + uuid.uuid4().hex
            allowed = {"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT", "VIRTUAL_ENV", "LD_LIBRARY_PATH"}
            env = {key: value for key, value in os.environ.items() if key in allowed}
            env.update(RAGS_ENV="test", RAGS_DB_PATH=str(directory / "state.db"),
                       RAGS_KEYS_JSON=json.dumps({token: "fixture-a", other: "fixture-b"}),
                       RAGS_KEY_ROLES_JSON=json.dumps({token: ["admin"], other: ["admin"]}),
                       RAGS_MODEL_PROVIDER="extractive", RAGS_EMBEDDING_PROVIDER="local",
                       RAGS_ADMIN_ENABLED="true", RAGS_ADMIN_DEV_HTTP="true", RAGS_ADMISSION_POLICY="compatibility",
                       RAGS_ROUTER_GATEWAY_URL=endpoint, RAGS_ROUTER_API_KEY="disposable-fixture-key",
                       RAGS_ROUTER_ALLOW_HTTP="true", RAGS_ROUTER_MODELS_JSON=json.dumps({"inkling-mini": MODEL}),
                       RAGS_ROUTER_MODEL_OPTIONS_JSON='{"inkling-mini":{"reasoning_effort":"low"}}',
                       RAGS_ROUTER_MAX_OUTPUT_TOKENS="512")
            if args.postgres:
                dsn = os.getenv("RAGS_E2E_POSTGRES_URL")
                require(dsn and dsn.startswith(("postgresql://", "postgres://")), "Disposable PostgreSQL URL is required")
                env["RAGS_DATABASE_URL"] = dsn
                # Keep this fixture isolated from any prior run in the same test database.
                tenant = "fixture-" + uuid.uuid4().hex
                env["RAGS_KEYS_JSON"] = json.dumps({token: tenant, other: tenant + "-other"})
            app = Application(env, directory)
            try:
                app.start()
                check_http(app, token, other, report["checks"])
                if args.browser:
                    check_browser(app, token, args.browser_executable, report["checks"])
                require(not failures and len(calls) >= 3, "Model fixture contract failed")
                report["model_http_requests"] = len(calls)
                require([item["id"] for item in report["checks"]] == report["expected_checks"], "Incomplete acceptance gate")
                report["status"] = "passed"
            finally:
                app.stop()
        report["cleanup_completed"] = True
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": "Acceptance gate failed; inspect the command log"}
        raise
    finally:
        report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        destination = Path(args.report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"status": report["status"], "checks": len(report["checks"]),
                          "expected": len(report["expected_checks"]), "live_model": False}))


if __name__ == "__main__":
    main()
