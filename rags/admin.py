"""Self-hosted, tenant-scoped admin UI and credential-safe same-origin BFF.

The browser receives an opaque short-lived HttpOnly session and a CSRF nonce,
never a reusable API key or OAuth access token. Sessions are process-local;
each request revalidates the stored credential to honor OAuth2 revocation.
"""
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
from importlib import resources
import re
import secrets
from threading import Lock
import time
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from rags.auth import Authenticator, Principal
from rags.config import Settings
from rags.features import strategy_features
from rags.registry import CapabilityError
from rags.storage import Store

SESSION_TTL = 1800
MAX_SESSIONS = 256
DOC_ID = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
STATIC = {"app.js": "text/javascript; charset=utf-8",
          "style.css": "text/css; charset=utf-8"}
CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
       "connect-src 'self'; form-action 'self'; base-uri 'none'; object-src 'none'; "
       "frame-ancestors 'none'")
ADMIN_HEADERS = {
    "Cache-Control": "no-store", "Pragma": "no-cache", "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY",
    "Content-Security-Policy": CSP,
}


@dataclass(frozen=True)
class AdminSession:
    bearer: str  # Secret, server memory only. Do not serialize, repr or log.
    tenant: str
    subject: str
    csrf: str
    expires: float

    def __repr__(self) -> str:
        return f"AdminSession(tenant={self.tenant!r}, expires={int(self.expires)})"


class SessionStore:
    def __init__(self, *, clock=time.time, capacity: int = MAX_SESSIONS):
        self._clock = clock
        self._capacity = capacity
        self._sessions: dict[str, AdminSession] = {}
        self._lock = Lock()

    @staticmethod
    def _key(cookie: str) -> str:
        return hashlib.sha256(cookie.encode("ascii")).hexdigest()

    def create(self, bearer: str, principal: Principal) -> tuple[str, AdminSession]:
        cookie = secrets.token_urlsafe(32)
        session = AdminSession(bearer, principal.tenant, principal.subject, secrets.token_urlsafe(32),
                               self._clock() + SESSION_TTL)
        with self._lock:
            now = self._clock()
            for key in [k for k, s in self._sessions.items() if s.expires <= now]:
                del self._sessions[key]
            if len(self._sessions) >= self._capacity:
                raise HTTPException(429, detail={"code": "session_capacity"})
            self._sessions[self._key(cookie)] = session
        return cookie, session

    def get(self, cookie: str | None) -> AdminSession:
        if not isinstance(cookie, str) or not 32 <= len(cookie) <= 128 or not cookie.isascii():
            raise HTTPException(401, detail={"code": "session_required"})
        with self._lock:
            session = self._sessions.get(self._key(cookie))
            if session is None or session.expires <= self._clock():
                self._sessions.pop(self._key(cookie), None)
                raise HTTPException(401, detail={"code": "session_expired"})
        return session

    def revoke(self, cookie: str | None) -> None:
        if isinstance(cookie, str) and cookie.isascii() and 32 <= len(cookie) <= 128:
            with self._lock:
                self._sessions.pop(self._key(cookie), None)


class SourceIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=200_000)
    source: str = Field(default="dashboard", min_length=1, max_length=500)
    document_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.:-]{1,128}$")


def _origin(request: Request) -> None:
    """Same-origin guard for every state change, including session creation."""
    expected = f"{request.url.scheme}://{request.headers.get('host', '')}"
    if request.headers.get("origin") != expected:
        raise HTTPException(403, detail={"code": "invalid_origin"})


def _cookie(request: Request, settings: Settings) -> tuple[str, bool]:
    if request.url.scheme == "https":
        return "__Host-rags_admin", True
    if (settings.admin_dev_http and request.url.scheme == "http" and
            request.url.hostname in {"localhost", "127.0.0.1", "::1", "testserver"}):
        return "rags_admin_dev", False
    raise HTTPException(403, detail={"code": "https_required"})


def _static_file(name: str) -> bytes:
    return resources.files("rags").joinpath("admin_static", name).read_bytes()


def _equality_params(search: str) -> str:
    # ! is the SQL LIKE escape char for both PostgreSQL and SQLite.
    escaped = search.casefold().replace("!", "!!").replace("%", "!%").replace("_", "!_")
    return "%" + escaped + "%"


def install_admin(app: FastAPI, *, settings: Settings, authenticator: Authenticator,
                  store: Store, service, monitor) -> SessionStore:
    sessions = SessionStore()
    app.state.admin_sessions = sessions
    if not settings.admin_enabled:
        return sessions

    @app.middleware("http")
    async def secure_dashboard(request: Request, call_next):
        response = await call_next(request)
        if request.url.path == "/admin" or request.url.path.startswith("/admin/"):
            for key, value in ADMIN_HEADERS.items():
                response.headers[key] = value
            if request.url.scheme == "https":
                response.headers["Strict-Transport-Security"] = "max-age=31536000"
        return response

    def authenticate(request: Request) -> tuple[Principal, AdminSession, str]:
        name, _ = _cookie(request, settings)
        cookie = request.cookies.get(name)
        session = sessions.get(cookie)
        try:
            principal = authenticator.authenticate("Bearer " + session.bearer)
            principal.require("rags:admin")
            if principal.tenant != session.tenant or principal.subject != session.subject:
                raise HTTPException(401, detail={"code": "session_identity_changed"})
        except HTTPException:
            sessions.revoke(cookie)
            raise
        request.state.principal = principal
        return principal, session, name

    def csrf_check(request: Request, session: AdminSession) -> None:
        _origin(request)
        supplied = request.headers.get("x-csrf-token", "")
        if (not supplied or not supplied.isascii()
                or not secrets.compare_digest(supplied, session.csrf)):
            raise HTTPException(403, detail={"code": "invalid_csrf"})

    def require(request: Request, *scopes: str, mutation: bool = False) -> tuple[Principal, AdminSession]:
        principal, session, _ = authenticate(request)
        principal.require(*scopes)
        if mutation:
            csrf_check(request, session)
        return principal, session

    @app.get("/admin", include_in_schema=False)
    def admin_redirect():
        return RedirectResponse("/admin/", status_code=308, headers=ADMIN_HEADERS)

    @app.get("/admin/", include_in_schema=False)
    def admin_index():
        return Response(_static_file("index.html"), media_type="text/html; charset=utf-8",
                        headers=ADMIN_HEADERS)

    @app.get("/admin/assets/{filename}", include_in_schema=False)
    def admin_asset(filename: str):
        if filename not in STATIC:
            raise HTTPException(404, detail={"code": "not_found"})
        return Response(_static_file(filename), media_type=STATIC[filename],
                        headers={**ADMIN_HEADERS, "Cache-Control": "private, max-age=300"})

    @app.post("/admin/api/session", include_in_schema=False)
    def login(request: Request, authorization: Annotated[str | None, Header()] = None):
        _origin(request)
        cookie_name, secure = _cookie(request, settings)
        principal = authenticator.authenticate(authorization)
        principal.require("rags:admin", "rags:read")
        # A new browser login revokes the previous session cookie, preventing
        # unnecessary parallel privileges and session fixation.
        sessions.revoke(request.cookies.get(cookie_name))
        session_cookie, session = sessions.create(authorization.partition(" ")[2], principal)
        request.state.principal = principal
        response = Response(content='{"status":"authenticated"}', media_type="application/json",
                            headers=ADMIN_HEADERS)
        response.set_cookie(
            cookie_name, session_cookie, max_age=SESSION_TTL, httponly=True, secure=secure,
            samesite="strict", path="/",
        )
        return response

    @app.get("/admin/api/session", include_in_schema=False)
    def get_session(request: Request):
        principal, session = require(request, "rags:read")
        return {"tenant": principal.tenant, "subject": principal.subject,
                "expires_at": datetime.fromtimestamp(session.expires, UTC).isoformat(),
                "csrf": session.csrf, "permissions": sorted(principal.permissions)}

    @app.delete("/admin/api/session", include_in_schema=False)
    def logout(request: Request):
        _, session, name = authenticate(request)
        csrf_check(request, session)
        cookie_name, secure = _cookie(request, settings)
        sessions.revoke(request.cookies.get(name))
        response = Response(status_code=204, headers=ADMIN_HEADERS)
        response.delete_cookie(cookie_name, path="/", secure=secure, httponly=True, samesite="strict")
        return response

    @app.get("/admin/api/overview", include_in_schema=False)
    def overview(request: Request):
        principal, _ = require(request, "rags:read")
        settings.features.require("documents")
        with store.connect() as conn:
            documents = conn.execute("SELECT count(*) AS n FROM documents WHERE tenant=?",
                                     (principal.tenant,)).fetchone()["n"]
            revisions = conn.execute("SELECT count(*) AS n FROM document_revisions WHERE tenant=?",
                                     (principal.tenant,)).fetchone()["n"]
            chunks = conn.execute(
                "SELECT count(*) AS n FROM chunks c JOIN documents d "
                "ON d.tenant=c.tenant AND d.id=c.document_id AND d.revision=c.revision "
                "WHERE c.tenant=? AND c.role='leaf'", (principal.tenant,),
            ).fetchone()["n"]
            images = conn.execute("SELECT count(*) AS n FROM image_assets WHERE tenant=?",
                                  (principal.tenant,)).fetchone()["n"]
            events = conn.execute(
                "SELECT timestamp,action,document_id FROM audit WHERE tenant=? "
                "ORDER BY id DESC LIMIT 8", (principal.tenant,),
            ).fetchall()
        return {
            "tenant": principal.tenant,
            "counts": {"documents": documents, "revisions": revisions, "chunks": chunks,
                       "images": images},
            "activity": [dict(row) for row in events],
            "telemetry": monitor.snapshot(principal.tenant),
        }

    @app.get("/admin/api/documents", include_in_schema=False)
    def documents(request: Request, search: str = Query(default="", max_length=120),
                  limit: int = Query(default=20, ge=1, le=50),
                  offset: int = Query(default=0, ge=0, le=1_000_000)):
        principal, _ = require(request, "rags:read")
        settings.features.require("documents")
        pattern = _equality_params(search.strip())
        predicate = "tenant=? AND (lower(title) LIKE ? ESCAPE '!' OR lower(id) LIKE ? ESCAPE '!')"
        params = (principal.tenant, pattern, pattern)
        with store.connect() as conn:
            count = conn.execute(f"SELECT count(*) AS n FROM documents WHERE {predicate}", params
                                 ).fetchone()["n"]
            rows = conn.execute(
                "SELECT id,title,source,revision,updated_at FROM documents "
                f"WHERE {predicate} ORDER BY updated_at DESC,id LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
        return {"items": [dict(item) for item in rows], "total": count,
                "offset": offset, "limit": limit}

    @app.post("/admin/api/documents", status_code=201, include_in_schema=False)
    def add_document(request: Request, data: SourceIn):
        principal, _ = require(request, "rags:write", mutation=True)
        settings.features.require("ingestion")
        if settings.model_provider != "extractive" and settings.chunking_config.get("strategy") == "contextual":
            principal.require("rags:generate")
        try:
            return service.ingest(principal.tenant, title=data.title, text=data.text,
                                  source=data.source, document_id=data.document_id)
        except CapabilityError:
            raise
        except ValueError as exc:
            raise HTTPException(400, detail={"code": "invalid_document"}) from exc

    @app.delete("/admin/api/documents/{document_id}", include_in_schema=False)
    def remove_document(request: Request, document_id: str,
                        if_match: Annotated[str | None, Header()] = None):
        principal, _ = require(request, "rags:delete", mutation=True)
        settings.features.require("deletion")
        if not DOC_ID.fullmatch(document_id) or not if_match or not DOC_ID.fullmatch(if_match):
            raise HTTPException(422, detail={"code": "revision_required"})
        deleted = store.delete(principal.tenant, document_id, time.strftime("%Y-%m-%dT%H:%M:%S+00:00",
                                                                            time.gmtime()),
                               expected_revision=if_match)
        if not deleted:
            raise HTTPException(404, detail={"code": "document_not_found"})
        return {"deleted": True}

    @app.post("/admin/api/query", include_in_schema=False)
    def query(request: Request, payload: dict):
        # Import the existing typed contract at request time, not module import
        # time, to avoid a circular dependency with the FastAPI application.
        from pydantic import ValidationError
        from rags.api import QueryIn

        principal, _ = require(request, "rags:query", mutation=True)
        settings.features.require("query")
        try:
            data = QueryIn.model_validate(payload)
        except ValidationError as exc:
            raise HTTPException(422, detail={"code": "invalid_query"}) from exc
        if ((data.generate and settings.model_provider != "extractive")
                or "model_generation" in strategy_features(data.mode)):
            principal.require("rags:generate")
        try:
            return service.query(principal.tenant, **data.model_dump(exclude_none=True))
        except CapabilityError:
            raise
        except ValueError as exc:
            raise HTTPException(400, detail={"code": "query_failed"}) from exc

    def _agent_engine():
        settings.features.require("agents")
        runtime = getattr(app.state, "agent_engine", None)
        if runtime is None:
            raise HTTPException(404, detail={"code": "agents_disabled"})
        return runtime

    @app.get("/admin/api/agents/capabilities", include_in_schema=False)
    def agent_capabilities(request: Request):
        principal, _ = require(request, "rags:agent")
        runtime = _agent_engine()
        return {"models": sorted(runtime.config.models),
                "tools": sorted(runtime.tools.allowed(principal)),
                "max_steps": runtime.config.max_steps,
                "memory_enabled": runtime.config.memory_enabled}

    @app.post("/admin/api/agents/runs", status_code=201, include_in_schema=False)
    def agent_start(request: Request, payload: dict):
        from pydantic import ValidationError
        from rags.agents.api import RunIn

        principal, _ = require(request, "rags:agent", "rags:generate", mutation=True)
        runtime = _agent_engine()
        try:
            data = RunIn.model_validate(payload)
        except ValidationError as exc:
            raise HTTPException(422, detail={"code": "invalid_agent_run"}) from exc
        cp = runtime.start(principal, goal=data.goal, model=data.model, session_id=data.session_id)
        return runtime.summary(cp)

    @app.get("/admin/api/agents/runs/{run_id}", include_in_schema=False)
    def agent_status(request: Request, run_id: str):
        principal, _ = require(request, "rags:agent")
        runtime = _agent_engine()
        return runtime.summary(runtime.get(principal, run_id))

    @app.post("/admin/api/agents/runs/{run_id}/resume", include_in_schema=False)
    def agent_approve(request: Request, run_id: str, payload: dict):
        from pydantic import ValidationError
        from rags.agents.api import ResumeIn

        principal, _ = require(request, "rags:agent", "rags:generate", mutation=True)
        runtime = _agent_engine()
        try:
            data = ResumeIn.model_validate(payload)
        except ValidationError as exc:
            raise HTTPException(422, detail={"code": "invalid_agent_approval"}) from exc
        cp = runtime.resume(principal, run_id, expected_revision=data.expected_revision,
                            approved=data.approved, feedback=data.feedback)
        return runtime.summary(cp)

    @app.get("/admin/api/agents/workflows", include_in_schema=False)
    def admin_workflow_catalog(request: Request):
        require(request, "rags:agent", "rags:query")
        _agent_engine()
        from rags.agents.templates import catalog

        return {"templates": catalog()}

    def _workflow_owner(principal, run_id):
        runtime = _agent_engine()
        cp = runtime.state.get(principal.tenant, run_id)
        if cp is None or cp.kind != "workflow" or cp.state["input"].get("owner_subject") != principal.subject:
            raise HTTPException(404, detail={"code": "not_found"})
        return cp

    @app.post("/admin/api/agents/workflows/{workflow_name}/runs",
              status_code=201, include_in_schema=False)
    def admin_workflow_start(request: Request, workflow_name: str, payload: dict):
        from pydantic import ValidationError
        from rags.agents.api import WorkflowRunIn
        from rags.agents.templates import get_template, make_evidence_review

        principal, _ = require(request, "rags:agent", "rags:query", mutation=True)
        runtime = _agent_engine()
        settings.features.require("query")
        get_template(workflow_name)
        try:
            data = WorkflowRunIn.model_validate(payload)
        except ValidationError as exc:
            raise HTTPException(422, detail={"code": "invalid_workflow"}) from exc
        graph = make_evidence_review(service, principal, settings.features)
        return graph.start(runtime.state, principal.tenant,
                           {"owner_subject": principal.subject, **data.model_dump()}).public()

    @app.get("/admin/api/agents/workflows/runs/{run_id}", include_in_schema=False)
    def admin_workflow_status(request: Request, run_id: str):
        principal, _ = require(request, "rags:agent", "rags:query")
        return _workflow_owner(principal, run_id).public()

    @app.post("/admin/api/agents/workflows/runs/{run_id}/resume", include_in_schema=False)
    def admin_workflow_approval(request: Request, run_id: str, payload: dict):
        from pydantic import ValidationError
        from rags.agents.api import ResumeIn
        from rags.agents.templates import make_evidence_review

        principal, _ = require(request, "rags:agent", "rags:query", mutation=True)
        cp = _workflow_owner(principal, run_id)
        try:
            data = ResumeIn.model_validate(payload)
        except ValidationError as exc:
            raise HTTPException(422, detail={"code": "invalid_workflow_approval"}) from exc
        graph = make_evidence_review(service, principal, settings.features)
        return graph.resume(_agent_engine().state, principal.tenant, cp.run_id,
                            expected_revision=data.expected_revision,
                            approved=data.approved).public()

    @app.post("/admin/api/agents/workflows/runs/{run_id}/recover", include_in_schema=False)
    def admin_workflow_recovery(request: Request, run_id: str, payload: dict):
        from pydantic import ValidationError
        from rags.agents.api import WorkflowRecoverIn
        from rags.agents.templates import make_evidence_review

        principal, _ = require(request, "rags:agent", "rags:query", mutation=True)
        cp = _workflow_owner(principal, run_id)
        try:
            data = WorkflowRecoverIn.model_validate(payload)
        except ValidationError as exc:
            raise HTTPException(422, detail={"code": "invalid_workflow_recovery"}) from exc
        graph = make_evidence_review(service, principal, settings.features)
        return graph.recover(_agent_engine().state, principal.tenant, cp.run_id,
                             expected_revision=data.expected_revision).public()

    @app.get("/admin/api/operations", include_in_schema=False)
    def operations(request: Request):
        principal, _ = require(request, "rags:read")
        from rags.model_catalog import catalog as model_catalog
        flags = settings.features.as_dict()
        rights = {
            "documents": "rags:read", "query": "rags:query", "ingestion": "rags:write",
            "reindex": "rags:reindex", "deletion": "rags:delete",
            "model_generation": "rags:generate", "mcp": "rags:mcp",
            "agents": "rags:agent",
        }
        for flag, permission in rights.items():
            flags[flag] = flags[flag] and permission in principal.permissions
        return {"auth_mode": settings.auth_mode, "storage": store.storage_backend,
                "embedding_provider": settings.embedding_provider,
                "embedding_model": service.provider.embedding_model,
                "model_selection": model_catalog(settings.features),
                "generation_provider": settings.model_provider,
                "features": flags, "permissions": sorted(principal.permissions),
                "telemetry": monitor.snapshot(principal.tenant)["otel"]}

    return sessions
