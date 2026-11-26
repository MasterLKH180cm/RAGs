"""Thin authenticated HTTP transport for the shared collection application service."""
from fastapi import APIRouter, Depends, Request
from starlette.concurrency import run_in_threadpool

from rags.collection_models import MAX_REQUEST_BYTES
from rags.collection_service import CollectionService, REQUEST_MODELS
from rags.model_profiles import strict_json
from rags.registry import CapabilityError


def _invalid() -> CapabilityError:
    return CapabilityError("invalid_collection_input", "Invalid collection request", 422)


async def _body(request: Request, *, optional: bool = False) -> dict:
    raw = await request.body()
    if len(raw) > MAX_REQUEST_BYTES:
        raise CapabilityError("collection_request_limit", "Collection request exceeds 2 MiB", 413)
    if optional and not raw:
        return {}
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
        raise _invalid()
    try:
        value = strict_json(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Request must be an object")
        return value
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise _invalid() from exc


def _query(request: Request, allowed: set[str]) -> dict:
    result = {}
    for key, value in request.query_params.multi_items():
        if key not in allowed or key in result:
            raise _invalid()
        if key in {"limit", "offset", "expected_generation"}:
            if not value.isascii() or not value.isdecimal() or len(value) > 19:
                raise _invalid()
            value = int(value)
        result[key] = value
    return result


def mount_collection_api(app, *, service: CollectionService, auth) -> None:
    router = APIRouter(prefix="/collections", tags=["Collections"])

    async def invoke(request, operation, collection_id=None, *, body=False, optional=False, query=()):
        # Dependencies authenticate and authorize before JSON parsing. The same
        # service then applies feature gates, contracts and tenant boundaries for
        # both the HTTP SDK and the local SDK.
        service.authorize(request.state.principal, operation)
        parameters = _query(request, set(query))
        if body:
            payload = await _body(request, optional=optional)
            if set(payload) & set(parameters):
                raise _invalid()
            parameters.update(payload)
        return await run_in_threadpool(service.dispatch, request.state.principal,
                                       operation, collection_id, parameters)

    @router.post("", status_code=201)
    async def create(request: Request, tenant: str = Depends(auth)):
        return await invoke(request, "create", body=True)

    @router.get("")
    async def list_collections(request: Request, tenant: str = Depends(auth)):
        return await invoke(request, "list", query=("name", "limit", "offset"))

    @router.get("/{collection_id}")
    async def get(request: Request, collection_id: str, tenant: str = Depends(auth)):
        return await invoke(request, "get", collection_id)

    @router.patch("/{collection_id}")
    async def modify(request: Request, collection_id: str, tenant: str = Depends(auth)):
        return await invoke(request, "modify", collection_id, body=True)

    @router.delete("/{collection_id}")
    async def delete_collection(request: Request, collection_id: str, tenant: str = Depends(auth)):
        return await invoke(request, "delete_collection", collection_id, body=True, optional=True,
                            query=("expected_generation",))

    @router.get("/{collection_id}/count")
    async def count(request: Request, collection_id: str, tenant: str = Depends(auth)):
        return await invoke(request, "count", collection_id)

    @router.get("/{collection_id}/indexing-status")
    async def indexing_status(request: Request, collection_id: str, tenant: str = Depends(auth)):
        return await invoke(request, "indexing_status", collection_id)

    @router.post("/{collection_id}/fork", status_code=201)
    async def fork(request: Request, collection_id: str, tenant: str = Depends(auth)):
        return await invoke(request, "fork", collection_id, body=True)

    def record_route(operation):
        async def records(request: Request, collection_id: str, tenant: str = Depends(auth)):
            return await invoke(request, operation, collection_id, body=True)
        return records

    for suffix, operation in (("add", "add"), ("update", "update"), ("upsert", "upsert"),
                              ("get", "get_records"), ("delete", "delete_records")):
        router.add_api_route("/{collection_id}/records/" + suffix, record_route(operation),
                             methods=["POST"], name="collection_records_" + suffix)

    @router.post("/{collection_id}/search")
    async def search(request: Request, collection_id: str, tenant: str = Depends(auth)):
        return await invoke(request, "search", collection_id, body=True)

    @router.post("/{collection_id}/ask")
    async def ask(request: Request, collection_id: str, tenant: str = Depends(auth)):
        return await invoke(request, "ask", collection_id, body=True)

    app.include_router(router)

    # Request bodies are parsed after authentication, so register their ordinary
    # Pydantic schemas explicitly for interactive docs and generated clients.
    original_openapi = app.openapi
    def collection_openapi():
        schema = original_openapi()
        components = schema.setdefault("components", {}).setdefault("schemas", {})
        for model in sorted(set(REQUEST_MODELS.values()), key=lambda item: item.__name__):
            model_schema = model.model_json_schema(ref_template="#/components/schemas/{model}")
            components.update(model_schema.pop("$defs", {}))
            components[model.__name__] = model_schema
        paths = {
            ("/collections", "post"): "create",
            ("/collections/{collection_id}", "patch"): "modify",
            ("/collections/{collection_id}/fork", "post"): "fork",
            ("/collections/{collection_id}/search", "post"): "search",
            ("/collections/{collection_id}/ask", "post"): "ask",
            **{("/collections/{collection_id}/records/" + suffix, "post"): operation
               for suffix, operation in (("add", "add"), ("update", "update"), ("upsert", "upsert"),
                                         ("get", "get_records"), ("delete", "delete_records"))},
        }
        for (path, method), operation in paths.items():
            schema["paths"][path][method]["requestBody"] = {
                "required": True, "content": {"application/json": {
                    "schema": {"$ref": "#/components/schemas/" + REQUEST_MODELS[operation].__name__}}}}
        schema["paths"]["/collections/{collection_id}"]["delete"]["requestBody"] = {
            "required": False, "content": {"application/json": {
                "schema": {"$ref": "#/components/schemas/GenerationCheck"}}}}
        for path, method, name, kind, maximum in (
            ("/collections", "get", "name", "string", None),
            ("/collections", "get", "limit", "integer", 100),
            ("/collections", "get", "offset", "integer", 1_000_000),
            ("/collections/{collection_id}", "delete", "expected_generation", "integer", 2 ** 63 - 1),
        ):
            parameters = schema["paths"][path][method].setdefault("parameters", [])
            if not any(item["name"] == name and item["in"] == "query" for item in parameters):
                parameters.append({"name": name, "in": "query", "required": False,
                                   "schema": {"type": kind, **({"minimum": 1 if name == "limit" else 0,
                                                                 "maximum": maximum} if maximum is not None else {})}})
        return schema
    app.openapi = collection_openapi
