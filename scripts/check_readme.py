"""Check documentation navigation and execute the README's local onboarding.

Run with an installed RAGs package. --live uses temporary data and an ephemeral
loopback port, with model/operator environment values removed. It executes the
marked README examples without fetching packages, weights or external services.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import ProxyHandler, Request, build_opener


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def prose(text: str) -> str:
    """Remove fenced examples while preserving line numbers for diagnostics."""
    lines, fence = [], None
    for line in text.splitlines(keepends=True):
        match = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if match and (fence is None or match[1][0] == fence):
            fence = match[1][0] if fence is None else None
            lines.append("\n")
        else:
            lines.append(line if fence is None else "\n")
    return "".join(lines)


def anchors(path: Path) -> set[str]:
    text = prose(path.read_text(encoding="utf-8"))
    found = set(re.findall(r'\b(?:id|name)=["\']([^"\']+)["\']', text))
    occurrences: dict[str, int] = {}
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", text, re.MULTILINE):
        heading = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", heading)
        heading = html.unescape(re.sub(r"<[^>]*>", "", heading)).lower()
        slug = re.sub(r"[^\w\- ]", "", heading).replace(" ", "-")
        count = occurrences.get(slug, 0)
        occurrences[slug] = count + 1
        found.add(slug + (f"-{count}" if count else ""))
    return found


def check_links() -> dict:
    files = sorted([*ROOT.glob("*.md"), *(ROOT / "docs").rglob("*.md")])
    cache: dict[Path, set[str]] = {}
    problems, checked = [], 0
    for source in files:
        text = prose(source.read_text(encoding="utf-8"))
        for match in re.finditer(r"\]\(([^)\s]+)\)", text):
            target = match[1].strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc:
                continue
            checked += 1
            destination = (source.parent / unquote(parsed.path)).resolve() if parsed.path else source
            label = f"{source.relative_to(ROOT)}:{text[:match.start()].count(chr(10)) + 1}: {target}"
            if not destination.exists():
                problems.append(label + " (missing path)")
            elif parsed.fragment and destination.suffix == ".md":
                if destination not in cache:
                    cache[destination] = anchors(destination)
                if unquote(parsed.fragment) not in cache[destination]:
                    problems.append(label + " (missing heading)")
    if problems:
        raise AssertionError("Broken documentation links:\n" + "\n".join(problems))
    return {"id": "internal_markdown_links", "passed": True, "files": len(files), "links": checked}


def examples() -> dict[str, str]:
    text = README.read_text(encoding="utf-8")
    found = re.findall(r"<!-- rags-smoke: (\w+) -->\s*```bash\n(.*?)\n```", text, re.DOTALL)
    result = dict(found)
    expected = {"setup", "serve", "client_env", "sdk", "document_add", "document_query"}
    if len(result) != len(found) or set(result) != expected:
        raise AssertionError("README executable example markers are missing or duplicated")
    return result


def check_onboarding() -> dict:
    blocks = examples()
    prefixes = ("RAGS_", "OPENAI_", "ANTHROPIC_", "TINKER_", "AGENT_", "OTEL_")
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(prefixes) and key not in {"PYTHONPATH", "PYTHONHOME"}
                   and key.lower() not in {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}}
    environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
    environment["PYTHONNOUSERSITE"] = "1"
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    opener = build_opener(ProxyHandler({}))

    with tempfile.TemporaryDirectory(prefix="rags-readme-") as directory:
        work = Path(directory)
        process = None

        def run_example(name: str, *, client: bool = False) -> str:
            code = (blocks["client_env"] + "\n" if client else "") + blocks[name]
            code = code.replace("http://127.0.0.1:8000", base)
            completed = subprocess.run(["bash", "-e", "-u", "-o", "pipefail", "-c", code],
                                       cwd=work, env=environment, capture_output=True, text=True, timeout=45)
            if completed.returncode:
                raise AssertionError(f"README {name} example failed (exit {completed.returncode})")
            return completed.stdout

        def request(path: str, *, token: str | None = None, data: dict | None = None) -> dict:
            headers = {"Authorization": "Bearer " + token} if token else {}
            body = None
            if data is not None:
                headers["Content-Type"] = "application/json"
                body = json.dumps(data).encode()
            with opener.open(Request(base + path, data=body, headers=headers), timeout=5) as response:
                raw = response.read(2 * 1024 * 1024 + 1)
            if len(raw) > 2 * 1024 * 1024:
                raise AssertionError("README onboarding response exceeded its bound")
            return json.loads(raw)

        def stop() -> None:
            nonlocal process
            if process is not None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                process = None

        with (work / "server.log").open("wb") as server_log:
            def start() -> None:
                nonlocal process
                process = subprocess.Popen(shlex.split(blocks["serve"]) + ["--port", str(port)],
                                           cwd=work, env=environment, stdout=server_log, stderr=server_log)
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise AssertionError("README server exited before becoming ready")
                    try:
                        if request("/live").get("status") == "ok":
                            return
                    except (URLError, OSError):
                        pass
                    time.sleep(0.05)
                raise AssertionError("README server did not become ready")

            try:
                run_example("setup")
                if (work / ".env").stat().st_mode & 0o077:
                    raise AssertionError("Generated credential file is not private")
                values = {}
                for line in (work / ".env").read_text(encoding="utf-8").splitlines():
                    if line and not line.startswith("#"):
                        key, raw = line.split("=", 1)
                        values[key] = shlex.split(raw)[0] if raw else ""
                token = values["RAGS_API_KEY"]
                start()
                health = request("/health", token=token)
                if health["storage"]["backend"] != "sqlite":
                    raise AssertionError("README offline profile did not select SQLite")
                if (health["embedding_model"] != "offline-hash-v1"
                        or request("/capabilities", token=token)["generation"]["provider"] != "extractive"):
                    raise AssertionError("README offline profile selected an unexpected model provider")
                try:
                    request("/collections")
                except HTTPError as error:
                    if error.code != 401:
                        raise AssertionError("Unauthenticated collection access returned an unexpected status") from None
                else:
                    raise AssertionError("Unauthenticated collection access succeeded")
                output = run_example("sdk", client=True)
                if "Backups are encrypted at rest." not in output or "30 days" not in output:
                    raise AssertionError("README SDK example did not return its expected evidence")
                added = json.loads(run_example("document_add", client=True))
                queried = json.loads(run_example("document_query", client=True))
                if added["chunks"] < 1 or "30 minutes" not in queried["answer"] or not queried["citations"]:
                    raise AssertionError("README document example did not return its expected evidence")
                collection = request("/collections?name=product-notes", token=token)["items"][0]
                stop()
                start()
                if request("/collections/" + collection["id"] + "/count", token=token)["count"] != 2:
                    raise AssertionError("README collection did not survive restart")
                if request("/documents/session-policy", token=token)["revision"] != added["revision"]:
                    raise AssertionError("README source revision did not survive restart")
                repeated = json.loads(run_example("document_query", client=True))
                if "30 minutes" not in repeated["answer"] or not repeated["citations"]:
                    raise AssertionError("README source retrieval failed after restart")
                return {"id": "readme_onboarding", "passed": True, "storage": "sqlite",
                        "unauthenticated_status": 401, "collection_records": 2,
                        "document_chunks": added["chunks"], "data_survived_restart": True,
                        "installed_package_used": True, "model_profile": "offline",
                        "model_credentials_required": False,
                        "port_rebound_for_isolation": True}
            finally:
                stop()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Execute README examples with the installed package")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = {"status": "failed", "checks": [], "measured_at": datetime.now(timezone.utc).isoformat()}
    try:
        report["checks"].append(check_links())
        if args.live:
            report["checks"].append(check_onboarding())
        report["status"] = "passed"
        return 0
    except (AssertionError, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        # Report only our bounded assertions; transport/process errors can contain configuration values.
        message = str(error) if isinstance(error, AssertionError) else "Documentation verification could not complete"
        report["error"] = {"type": type(error).__name__, "message": message}
        print(message, file=sys.stderr)
        return 1
    finally:
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
