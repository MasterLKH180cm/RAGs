"""Minimal installed command-line interface for the RAG service.

All diagnostics are local/read-only. This module does not start or migrate a
database until the operator explicitly chooses "serve".
"""
import argparse
from importlib import resources
import ipaddress
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import sqlite3
import sys

from rags.config import Settings
from rags.features import MANIFEST


PROFILES = ("offline", "postgres", "openai", "ollama", "anthropic")
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
MAX_ENV_SIZE = 65_536
PROFILE_DISABLED = {
    "offline": ("embedding.openai", "embedding.fastembed", "model_generation", "research",
                "federation", "multimodal", "image_captioning", "original_images"),
    "postgres": ("embedding.openai", "embedding.fastembed", "model_generation", "research",
                 "federation", "multimodal", "image_captioning", "original_images"),
    "openai": ("embedding.local", "embedding.fastembed", "research", "federation",
               "multimodal", "original_images"),
    "ollama": ("embedding.local", "embedding.fastembed", "research", "federation",
               "multimodal", "original_images"),
    "anthropic": ("embedding.openai", "embedding.fastembed", "research", "federation",
                  "multimodal", "original_images"),
}


def load_env_file(path: str | Path | None) -> None:
    """Read a small shell-quoted file without executing arbitrary shell code.

    Existing process variables take precedence; files never replace secrets
    supplied by a shell, container orchestrator, or secret-management system.
    """
    if path is None:
        return
    filename = Path(path)
    if filename.is_symlink() or not filename.is_file():
        raise ValueError("Environment file must be an existing regular file, not a symlink")
    if filename.stat().st_size > MAX_ENV_SIZE:
        raise ValueError("Environment file exceeds 64 KiB")
    try:
        raw = filename.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError("Cannot read environment file") from exc
    seen = set()
    parsed = {}
    for number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        name, separator, value = line.partition("=")
        if not separator or not ENV_NAME.fullmatch(name) or name in seen:
            raise ValueError(f"Malformed or repeated environment name on line {number}")
        seen.add(name)
        try:
            tokens = shlex.split(value, comments=True, posix=True)
        except ValueError as exc:
            raise ValueError(f"Malformed shell quoting on line {number}") from exc
        if len(tokens) > 1:
            raise ValueError(f"Environment line {number} must contain one value")
        parsed[name] = tokens[0] if tokens else ""
    for name, value in parsed.items():
        os.environ.setdefault(name, value)


def init_config(directory: Path, profile: str) -> tuple[Path, Path]:
    """Generate private auth values, fail closed and never overwrite/symlink-follow."""
    if profile not in PROFILES:
        raise ValueError("Unsupported profile")
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Target must be an existing non-symlink directory")
    env_path = directory / ".env"
    flags_path = directory / "rags.features.json"
    if any(path.exists() or path.is_symlink() for path in (env_path, flags_path)):
        raise FileExistsError("Existing configuration was preserved")
    template = resources.files("rags").joinpath("templates", profile + ".env").read_text(encoding="utf-8")
    token = secrets.token_urlsafe(32)
    env_body = template.replace("__FEATURES_PATH__", shlex.quote(str(flags_path.absolute())))
    env_body = env_body.replace("__DB_PATH__", shlex.quote(str((directory / "data" / "rags.db").absolute())))
    if "__FEATURES_PATH__" not in template or "RAGS_KEYS_JSON" in template or "RAGS_API_KEY" in template:
        raise ValueError("Packaged configuration template is invalid")
    env_body += "\nRAGS_KEYS_JSON=" + shlex.quote(json.dumps({token: "local"})) + "\n"
    env_body += "RAGS_API_KEY=" + shlex.quote(token) + "\n"
    flags = {name: False for name in PROFILE_DISABLED[profile]}
    if any(name not in MANIFEST for name in flags):
        raise ValueError("Packaged feature manifest is invalid")
    content = json.dumps(flags, indent=2, sort_keys=True) + "\n"
    created = []
    try:
        for path, body, mode in ((env_path, env_body, 0o600), (flags_path, content, 0o644)):
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), mode)
            created.append(path)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                os.fchmod(stream.fileno(), mode)
                stream.write(body)
    except BaseException:
        for path in reversed(created):
            path.unlink()
        raise
    return env_path, flags_path


def _check(name: str, status: str, detail: str) -> dict:
    return {"name": name, "status": status, "detail": detail}


def database_check(settings: Settings) -> dict:
    """Inspect the selected DB with SELECT only and a bounded connection."""
    if settings.db_path.startswith(("postgresql://", "postgres://")):
        if "REPLACE_ME" in settings.db_path:
            return _check("database", "warning", "Replace the PostgreSQL DSN placeholder")
        try:
            import psycopg
        except ImportError:
            return _check("database", "error", "Install rags-platform[postgres] for PostgreSQL")
        try:
            with psycopg.connect(settings.db_path, connect_timeout=3) as connection:
                connection.execute("SET TRANSACTION READ ONLY")
                connection.execute("SET LOCAL statement_timeout='3s'")
                extension = connection.execute(
                    "SELECT extversion FROM pg_extension WHERE extname='vector'"
                ).fetchone()
                if extension is None:
                    return _check("database", "error", "PostgreSQL vector extension is missing")
                rows = connection.execute(
                    "SELECT table_name,column_name,udt_name FROM information_schema.columns "
                    "WHERE table_schema=current_schema() "
                    "AND ((table_name='chunks' AND column_name='embedding') "
                    "OR (table_name='image_assets' AND column_name='vector'))"
                ).fetchall()
                types = {(row[0], row[1]): row[2] for row in rows}
            if ("chunks", "embedding") not in types:
                return _check("database", "warning", "pgvector is installed; application schema is not initialized")
            if types[("chunks", "embedding")] != "vector":
                return _check("database", "error", "chunks.embedding is not a native vector column")
            if "image_assets" in {key[0] for key in types} and types.get(("image_assets", "vector")) != "vector":
                return _check("database", "error", "image_assets.vector is not a native vector column")
            return _check("database", "ok", "PostgreSQL with pgvector native columns is available")
        except Exception:
            # Psycopg errors can contain credentials, hosts, SQL parameters, etc.
            return _check("database", "error", "PostgreSQL connection or read-only inspection failed")
    filename = Path(settings.db_path)
    if not filename.is_file():
        return _check("database", "warning", "SQLite file not initialized; serving will create it")
    try:
        with sqlite3.connect(filename.resolve().as_uri() + "?mode=ro", uri=True, timeout=3) as conn:
            tables = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('documents','chunks')"
            )}
        if tables != {"documents", "chunks"}:
            return _check("database", "warning", "SQLite application schema is not initialized")
    except sqlite3.Error:
        return _check("database", "error", "SQLite read-only inspection failed")
    return _check("database", "ok", "Existing SQLite application database is readable")


def diagnose(*, check_db: bool = True) -> dict:
    checks = []
    try:
        settings = Settings.from_env()
    except Exception:
        # Never serialize invalid user-supplied input or a secret path.
        return {"status": "error", "checks": [
            _check("configuration", "error", "Settings validation failed; check RAGS_* environment values")
        ]}
    storage = "postgresql" if settings.db_path.startswith(("postgresql://", "postgres://")) else "sqlite"
    checks.append(_check("configuration", "ok", "Validated " + settings.auth_mode + " configuration"))
    checks.append(_check("storage", "ok", storage))
    checks.append(_check("features", "ok",
                         str(len([x for x in settings.features.as_dict().values() if x])) +
                         " enabled / " +
                         str(len([x for x in settings.features.as_dict().values() if not x])) + " disabled"))
    from urllib.parse import urlsplit

    providers = (
        ("embedding", settings.embedding_provider,
         "offline-hash-v1" if settings.embedding_provider == "local" else
         settings.fastembed_model if settings.embedding_provider == "fastembed" else settings.embedding_model,
         settings.embedding_endpoint,
         settings.embedding_credential if settings.embedding_provider == "openai" else ""),
        ("generation", settings.model_provider,
         settings.anthropic_model if settings.model_provider == "anthropic" else
         settings.chat_model if settings.model_provider == "openai" else "extractive",
         settings.generation_endpoint if settings.model_provider != "extractive" else "local",
         settings.generation_credential if settings.model_provider != "extractive" else ""),
    )
    for name, provider, model, url, credential in providers:
        host = urlsplit(url).hostname or "local"
        details = f"{provider}; model={model}; endpoint_host={host}"
        if (provider == "openai" or provider == "anthropic") and (
            not credential or credential == "REPLACE_ME"
        ):
            checks.append(_check(name, "warning", details + "; credentials missing"))
        else:
            checks.append(_check(name, "ok", details))
    if check_db:
        checks.append(database_check(settings))
    status = ("error" if any(check["status"] == "error" for check in checks)
              else "warning" if any(check["status"] == "warning" for check in checks) else "ok")
    return {"status": status, "checks": checks}


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _run_serve(args) -> int:
    if not _is_loopback(args.host) and not args.allow_public:
        raise ValueError("Non-loopback binding requires --allow-public")
    Settings.from_env()  # Fail closed before the ASGI process opens a socket.
    import uvicorn
    uvicorn.run("rags.api:app", host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rags", description="RAGs platform tooling")
    parser.add_argument("--version", action="version", version="rags-platform 0.8.0")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Create private credentials and a deployment profile")
    init.add_argument("--profile", choices=PROFILES, default="offline")
    init.add_argument("--directory", type=Path, default=Path("."))
    doctor = sub.add_parser("doctor", help="Read-only configuration and storage checks")
    doctor.add_argument("--env-file", type=Path)
    doctor.add_argument("--no-db", action="store_true", help="Skip database connection")
    doctor.add_argument("--json", action="store_true", help="Emit machine-readable results")
    serve = sub.add_parser("serve", help="Start the existing FastAPI service")
    serve.add_argument("--env-file", type=Path)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--allow-public", action="store_true", help="Explicitly allow non-loopback bind")
    agents = sub.add_parser("agent", help="Inspect and run the opt-in Agent Platform")
    agent_actions = agents.add_subparsers(dest="agent_command", required=True)
    agent_doctor = agent_actions.add_parser("doctor", help="Offline router/agent readiness report")
    agent_doctor.add_argument("--env-file", type=Path)
    agent_doctor.add_argument("--json", action="store_true")
    agent_run = agent_actions.add_parser("run", help="Start a tenant-scoped run through local RAG API")
    agent_run.add_argument("--env-file", type=Path)
    agent_run.add_argument("--goal-file", type=Path, required=True)
    agent_run.add_argument("--model", required=True)
    agent_run.add_argument("--session", required=True)
    agent_run.add_argument("--port", type=int, default=8000)
    agent_review = agent_actions.add_parser("review", help="Approve or deny a pending tool call")
    agent_review.add_argument("--env-file", type=Path)
    agent_review.add_argument("--run-id", required=True)
    agent_review.add_argument("--revision", type=int, required=True)
    verdict = agent_review.add_mutually_exclusive_group(required=True)
    verdict.add_argument("--approve", action="store_true")
    verdict.add_argument("--deny", action="store_true")
    agent_review.add_argument("--port", type=int, default=8000)
    workflow = sub.add_parser("workflow", help="Execute registered, human-reviewed workflow graphs")
    wf_actions = workflow.add_subparsers(dest="workflow_command", required=True)
    wf_list = wf_actions.add_parser("list", help="List operator-registered workflow graphs")
    wf_run = wf_actions.add_parser("run", help="Start an approved workflow template")
    wf_run.add_argument("--question-file", type=Path, required=True)
    wf_run.add_argument("--template", default="evidence_review")
    wf_run.add_argument("--mode", choices=("sparse", "dense", "hybrid"), default="sparse")
    wf_run.add_argument("--top-k", type=int, choices=tuple(range(1, 11)), default=5)
    wf_status = wf_actions.add_parser("status", help="Inspect a workflow run")
    wf_status.add_argument("--run-id", required=True)
    wf_review = wf_actions.add_parser("review", help="Approve or deny a workflow checkpoint")
    wf_review.add_argument("--run-id", required=True)
    wf_review.add_argument("--revision", type=int, required=True)
    wf_decision = wf_review.add_mutually_exclusive_group(required=True)
    wf_decision.add_argument("--approve", action="store_true")
    wf_decision.add_argument("--deny", action="store_true")
    wf_recover = wf_actions.add_parser("recover", help="Recover interrupted workflow from checkpoint")
    wf_recover.add_argument("--run-id", required=True)
    wf_recover.add_argument("--revision", type=int, required=True)
    for command in (wf_list, wf_run, wf_status, wf_review, wf_recover):
        command.add_argument("--env-file", type=Path)
        command.add_argument("--port", type=int, default=8000)
    bench = sub.add_parser("benchmark", help="Synthetic read-path PostgreSQL comparison (temporary tables)")
    bench.add_argument("--env-file", type=Path)
    bench.add_argument("--documents", type=int, default=200)
    bench.add_argument("--chunks-per-document", type=int, default=8)
    bench.add_argument("--iterations", type=int, default=20)
    bench.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            paths = init_config(args.directory, args.profile)
            for path in paths:
                print("Created " + str(path))
            print("Start with: rags serve --env-file " + str(paths[0]))
            return 0
        load_env_file(args.env_file)
        if args.command == "agent":
            from rags.agents.config import AgentPlatformConfig
            import httpx

            settings = AgentPlatformConfig.from_env()
            if args.agent_command == "doctor":
                report = {"router_enabled": settings.router_enabled,
                          "agent_enabled": settings.agent_enabled,
                          "memory_enabled": settings.memory_enabled,
                          "model_aliases": sorted(settings.models),
                          "gateway_configured": settings.router_ready,
                          "status": ("ready" if settings.router_ready else "configuration_required")}
                if args.json:
                    print(json.dumps(report, sort_keys=True))
                else:
                    print("Agent Platform: " + report["status"])
                    print("Model aliases: " + ", ".join(sorted(settings.models)))
                return 0 if settings.agent_enabled and settings.router_ready else 1
            if not 1 <= args.port <= 65535:
                raise ValueError("Agent API port must be 1..65535")
            token = os.getenv("RAGS_API_KEY", "")
            if not token or len(token) > 4096:
                raise ValueError("Set RAGS_API_KEY to a scoped local API token")
            if args.agent_command == "run":
                path = args.goal_file
                if path.is_symlink() or not path.is_file() or path.stat().st_size > 8192:
                    raise ValueError("Provide a regular small goal file")
                goal = path.read_text(encoding="utf-8").strip()
                payload = {"goal": goal, "model": args.model, "session_id": args.session}
                endpoint = "/agents/runs"
            else:
                payload = {"expected_revision": args.revision, "approved": args.approve}
                endpoint = "/agents/runs/" + args.run_id + "/resume"
            with httpx.Client(timeout=90, follow_redirects=False, trust_env=False) as client:
                response = client.post(
                    f"http://127.0.0.1:{args.port}" + endpoint,
                    headers={"Authorization": "Bearer " + token}, json=payload)
                response.raise_for_status()
                result = response.json()
            print(json.dumps({key: result.get(key) for key in (
                "run_id", "status", "revision", "answer", "pending_approval", "stop_reason")},
                ensure_ascii=False))
            return 0
        if args.command == "workflow":
            from rags.agents.state import _id
            import httpx

            if not 1 <= args.port <= 65535:
                raise ValueError("Workflow API port must be 1..65535")
            token = os.getenv("RAGS_API_KEY", "")
            if not 1 <= len(token) <= 4096:
                raise ValueError("Set RAGS_API_KEY to a scoped loopback API token")
            method = "GET"
            payload = None
            if args.workflow_command == "list":
                route = "/agents/workflows"
            elif args.workflow_command == "run":
                filename = args.question_file
                if filename.is_symlink() or not filename.is_file() or filename.stat().st_size > 8192:
                    raise ValueError("Workflow question must be a regular file of at most 8 KiB")
                question = filename.read_text("utf-8").strip()
                if not 2 <= len(question) <= 2000:
                    raise ValueError("Workflow question must be 2..2000 chars")
                from rags.agents.templates import get_template
                get_template(args.template)
                route = "/agents/workflows/" + args.template + "/runs"
                payload = {"question": question, "mode": args.mode, "top_k": args.top_k}
                method = "POST"
            else:
                run_id = _id(args.run_id, "run")
                route = "/agents/workflows/runs/" + run_id
                if args.workflow_command == "review":
                    route += "/resume"
                    payload = {"expected_revision": args.revision, "approved": args.approve}
                    method = "POST"
                elif args.workflow_command == "recover":
                    route += "/recover"
                    payload = {"expected_revision": args.revision}
                    method = "POST"
            with httpx.Client(timeout=90, follow_redirects=False, trust_env=False) as client:
                response = client.request(method, f"http://127.0.0.1:{args.port}" + route,
                    headers={"Authorization": "Bearer " + token},
                    json=payload)
                response.raise_for_status()
                result = response.json()
            print(json.dumps(result, ensure_ascii=False))
            return 0
        if args.command == "doctor":
            result = diagnose(check_db=not args.no_db)
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                for check in result["checks"]:
                    print(f"{check['status'].upper():7s} {check['name']}: {check['detail']}")
            return 0 if result["status"] == "ok" else 1
        if args.command == "serve":
            if not 1 <= args.port <= 65535:
                raise ValueError("Port must be between 1 and 65535")
            return _run_serve(args)
        if args.command == "benchmark":
            from rags.benchmark import benchmark
            settings = Settings.from_env()
            result = benchmark(settings.db_path, documents=args.documents,
                               chunks_per_document=args.chunks_per_document, iterations=args.iterations)
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                for key, value in result["measurements"].items():
                    print(f"{key}: p50={value['p50_ms']:.3f}ms p95={value['p95_ms']:.3f}ms "
                          f"rows={value['rows']}")
            return 0
        parser.error("Unknown command")
    except Exception as exc:
        # CLI failures must not leak credentials embedded in provider/DB exceptions.
        # Never echo arbitrary configuration content from exception strings.
        print("rags: " + (str(exc) if args.command == "init" else
                          "Unable to complete command; validate inputs and selected dependencies"), file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
