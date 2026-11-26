"""Authenticated collection clients with side-effect-free imports.

HTTP and local persistence share the application request contracts. Importing
this module never creates an ASGI app, opens a database or loads a model.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import ipaddress
import json
import math
from pathlib import Path
import re
from time import monotonic
from urllib.parse import urlsplit

import httpx
from rags.config import Settings
from rags.model_profiles import strict_json

MAX_BYTES = 2 * 1024 * 1024
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}\Z")
_SAFE_CODES = frozenset({
    "authentication_required", "invalid_token", "insufficient_scope", "permission_denied",
    "identity_provider_unavailable", "feature_disabled", "collection_not_found",
    "collection_exists", "collection_conflict", "collection_generation_conflict",
    "generation_conflict", "revision_conflict", "record_exists", "record_not_found",
    "record_conflict", "invalid_request", "invalid_collection_input", "invalid_collection",
    "invalid_filter", "filter_timeout", "regex_timeout", "collection_limit_exceeded",
    "batch_limit_exceeded", "query_limit_exceeded", "scan_limit_exceeded",
    "response_limit_exceeded", "request_too_large", "invalid_embedding",
    "embedding_space_mismatch", "embedding_profile_mismatch", "embedding_dimension_mismatch",
    "embedding_dimensions_mismatch", "sparse_space_mismatch", "requires_reindex",
    "requires_provider", "provider_failed", "invalid_model_input", "invalid_model_output",
    "context_budget_exceeded", "unsupported_operation", "collection_input_invalid",
    "invalid_collection_filter", "invalid_collection_search", "invalid_collection_data",
    "collection_filter_timeout", "collection_response_limit", "collection_timeout",
    "collection_integrity_error", "collection_request_limit", "embedding_failed",
})


class ClientError(ValueError):
    """A content-free error carrying a bounded, documented status and code."""

    def __init__(self, code: str, status_code: int = 400):
        self.code, self.status_code = code, status_code
        super().__init__(f"{code} (HTTP {status_code})")


def _safe_error(payload: object, status: int) -> ClientError:
    detail = payload.get("detail", payload) if isinstance(payload, dict) else None
    code = detail.get("code") if isinstance(detail, dict) else None
    return ClientError(code if isinstance(code, str) and code in _SAFE_CODES else "request_failed", status)


def _token(value: str) -> str:
    if (not isinstance(value, str) or not 1 <= len(value) <= 4096
            or any(not 33 <= ord(c) <= 126 for c in value)):
        raise ClientError("invalid_token", 401)
    return value


def _identifier(value: object) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ClientError("invalid_collection_id", 422)
    return value


def _json_bytes(value: object) -> bytes:
    body = bytearray()
    try:
        encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        for part in encoder.iterencode(value):
            if len(part) > MAX_BYTES - len(body):
                raise ClientError("request_too_large", 413)
            encoded = part.encode("utf-8")
            if len(encoded) > MAX_BYTES - len(body):
                raise ClientError("request_too_large", 413)
            body.extend(encoded)
    except ClientError:
        raise
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise ClientError("invalid_request", 422) from None
    return bytes(body)


def _parse_json(raw: bytes) -> dict:
    try:
        value = strict_json(raw)
    except (ValueError, UnicodeError, RecursionError):
        raise ClientError("invalid_response", 502) from None
    if not isinstance(value, dict):
        raise ClientError("invalid_response", 502)
    return value


def _payload(operation: str, data: dict | None) -> dict:
    # Importing these contracts does not import the API or initialize services.
    from rags import collection_models as contracts

    models = {
        "create": contracts.CollectionCreate, "list": contracts.CollectionList,
        "modify": contracts.CollectionModify, "fork": contracts.CollectionFork,
        "delete_collection": contracts.GenerationCheck,
        "add": contracts.RecordBatch, "update": contracts.RecordBatch, "upsert": contracts.RecordBatch,
        "get_records": contracts.GetRecords, "delete_records": contracts.DeleteRecords,
        "search": contracts.SearchBatch, "ask": contracts.CollectionAsk,
    }
    value = {} if data is None else data
    _json_bytes(value)  # Bound the original request before normalization or work.
    model = models.get(operation)
    if model is None:
        if value or operation not in {"get", "count", "indexing_status"}:
            raise ClientError("invalid_request", 422)
        return {}
    try:
        result = model.model_validate(value).model_dump(mode="json", exclude_unset=True)
    except (ValueError, TypeError, RecursionError):
        raise ClientError("invalid_request", 422) from None
    _json_bytes(result)
    return result


class SearchResult(dict):
    """Search envelope; rows(index=0) returns a copy of one query's result rows."""

    def rows(self, index: int = 0) -> list[dict]:
        results = self.get("results")
        if type(index) is not int or not isinstance(results, list) or not 0 <= index < len(results):
            raise ClientError("invalid_result_index", 422)
        result = results[index]
        if not isinstance(result, dict) or not isinstance(result.get("items"), list):
            raise ClientError("invalid_response", 502)
        return deepcopy(result["items"])


class _CollectionClient:
    _closed = False

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *_):
        self.close()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(closed={self._closed})"

    def _check_open(self) -> None:
        if self._closed:
            raise ClientError("client_closed", 400)

    def close(self) -> None:
        self._closed = True

    def _dispatch(self, operation: str, collection_id: str | None = None, data: dict | None = None) -> dict:
        raise NotImplementedError

    def create_collection(self, name: str, **configuration) -> Collection:
        return Collection(self, self._dispatch("create", data={"name": name, **configuration}))

    def get_or_create_collection(self, name: str, **configuration) -> Collection:
        return self.create_collection(name, **{**configuration, "get_or_create": True})

    def list_collections_page(self, *, limit: int = 100, offset: int = 0) -> dict:
        result = self._dispatch("list", data={"limit": limit, "offset": offset})
        if not isinstance(result.get("items"), list):
            raise ClientError("invalid_response", 502)
        for item in result["items"]:
            Collection(self, item)
        return result

    def list_collections(self, *, limit: int = 100, offset: int = 0) -> list[Collection]:
        page = self.list_collections_page(limit=limit, offset=offset)
        return [Collection(self, item) for item in page["items"]]

    def get_collection(self, name: str | None = None, *, collection_id: str | None = None) -> Collection:
        if (name is None) == (collection_id is None):
            raise ClientError("invalid_request", 422)
        if collection_id is not None:
            found = Collection(self, self._dispatch("get", _identifier(collection_id)))
            if found.id != collection_id:
                raise ClientError("invalid_response", 502)
            return found
        result = self._dispatch("list", data={"name": name, "limit": 1, "offset": 0})
        items = result.get("items")
        if not isinstance(items, list) or len(items) > 1:
            raise ClientError("invalid_response", 502)
        if not items:
            raise ClientError("collection_not_found", 404)
        found = Collection(self, items[0])
        if found.name != name:
            raise ClientError("invalid_response", 502)
        return found

    def delete_collection(self, name: str | None = None, *, collection_id: str | None = None,
                          expected_generation: int | None = None) -> dict:
        if (name is None) == (collection_id is None):
            raise ClientError("invalid_request", 422)
        key = _identifier(collection_id) if collection_id is not None else self.get_collection(name).id
        data = {} if expected_generation is None else {"expected_generation": expected_generation}
        return self._dispatch("delete_collection", key, data)


class Collection:
    """A collection handle bound to its immutable ID, independent of renames."""

    def __init__(self, client: _CollectionClient, info: dict):
        if not isinstance(info, dict) or not isinstance(info.get("name"), str):
            raise ClientError("invalid_response", 502)
        try:
            key = _identifier(info.get("id"))
            from rags.collection_models import validate_collection_name
            validate_collection_name(info["name"])
        except ValueError:
            raise ClientError("invalid_response", 502) from None
        self._client, self._id, self._info = client, key, deepcopy(info)

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._info["name"]

    @property
    def metadata(self) -> dict:
        return deepcopy(self._info.get("metadata", {}))

    def to_dict(self) -> dict:
        return deepcopy(self._info)

    def refresh(self) -> Collection:
        updated = self._client.get_collection(collection_id=self.id)
        self._info = updated.to_dict()
        return self

    def modify(self, **changes) -> dict:
        result = self._client._dispatch("modify", self.id, changes)
        updated = Collection(self._client, result)
        if updated.id != self.id:
            raise ClientError("invalid_response", 502)
        self._info = updated.to_dict()
        return self.to_dict()

    def fork(self, name: str, **options) -> Collection:
        return Collection(self._client, self._client._dispatch("fork", self.id, {"name": name, **options}))

    def add(self, *, ids: list[str], **fields) -> dict:
        return self._client._dispatch("add", self.id, {"ids": ids, **fields})

    def update(self, *, ids: list[str], **fields) -> dict:
        return self._client._dispatch("update", self.id, {"ids": ids, **fields})

    def upsert(self, *, ids: list[str], **fields) -> dict:
        return self._client._dispatch("upsert", self.id, {"ids": ids, **fields})

    def get(self, *, ids: list[str] | None = None, **options) -> dict:
        data = dict(options)
        if ids is not None:
            data["ids"] = ids
        return self._client._dispatch("get_records", self.id, data)

    def delete(self, *, ids: list[str] | None = None, **options) -> dict:
        data = dict(options)
        if ids is not None:
            data["ids"] = ids
        return self._client._dispatch("delete_records", self.id, data)

    def count(self) -> int:
        result = self._client._dispatch("count", self.id)
        count = result.get("count")
        if type(count) is not int or count < 0:
            raise ClientError("invalid_response", 502)
        return count

    def indexing_status(self) -> dict:
        return self._client._dispatch("indexing_status", self.id)

    get_indexing_status = indexing_status

    def search(self, *, searches: list[dict] | None = None, query_texts: list[str] | None = None,
               query_embeddings: list[list[float]] | None = None, **options) -> SearchResult:
        if searches is not None:
            if query_texts is not None or query_embeddings is not None or options:
                raise ClientError("invalid_request", 422)
            batch = searches
        elif query_texts is not None or query_embeddings is not None:
            inputs = [x for x in (query_texts, query_embeddings) if x is not None]
            if (any(not isinstance(x, list) or not 1 <= len(x) <= 16 for x in inputs)
                    or len({len(x) for x in inputs}) != 1
                    or "query_text" in options or "query_embedding" in options):
                raise ClientError("invalid_request", 422)
            batch = []
            for index in range(len(inputs[0])):
                item = dict(options)
                if query_texts is not None:
                    item["query_text"] = query_texts[index]
                if query_embeddings is not None:
                    item["query_embedding"] = query_embeddings[index]
                batch.append(item)
        else:
            batch = [options]
        result = self._client._dispatch("search", self.id, {"searches": batch})
        if not isinstance(result.get("results"), list) or len(result["results"]) != len(batch):
            raise ClientError("invalid_response", 502)
        output = SearchResult(result)
        for index in range(len(batch)):
            output.rows(index)
        return output

    def query(self, **options) -> SearchResult:
        """Alias of search; uses this package's contract rather than Chroma wire syntax."""
        return self.search(**options)

    def ask(self, question: str, **options) -> dict:
        return self._client._dispatch("ask", self.id, {"question": question, **options})


class HttpClient(_CollectionClient):
    """Synchronous authenticated HTTP client with a whole-request deadline.

    HTTPS is required outside loopback unless allow_insecure_http is explicitly
    enabled by the caller. Each call owns and closes its async HTTP resources.
    """

    def __init__(self, base_url: str, api_key: str, *, timeout: float = 30.0,
                 max_response_bytes: int = MAX_BYTES, allow_insecure_http: bool = False):
        self._api_key = _token(api_key)
        if (type(timeout) not in (int, float) or not 0 < timeout <= 360 or not math.isfinite(timeout)
                or type(max_response_bytes) is not int or not 1 <= max_response_bytes <= MAX_BYTES
                or type(allow_insecure_http) is not bool):
            raise ClientError("invalid_client_configuration", 422)
        try:
            if (not isinstance(base_url, str) or len(base_url) > 2048
                    or any(c.isspace() or ord(c) < 33 or ord(c) == 127 for c in base_url)
                    or "\\" in base_url):
                raise ValueError
            parsed = urlsplit(base_url)
            if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                    or parsed.username is not None or parsed.password is not None
                    or parsed.query or parsed.fragment or (parsed.port is not None and not 1 <= parsed.port <= 65535)
                    or any(part in {".", ".."} for part in parsed.path.split("/"))
                    or "%" in parsed.path):
                raise ValueError
            try:
                loopback = ipaddress.ip_address(parsed.hostname).is_loopback
            except ValueError:
                loopback = parsed.hostname.lower() == "localhost"
            if parsed.scheme == "http" and not loopback and not allow_insecure_http:
                raise ValueError
            self._base_url = str(httpx.URL(base_url)).rstrip("/")
        except (ValueError, TypeError, httpx.InvalidURL):
            raise ClientError("invalid_endpoint", 422) from None
        self._timeout, self._max_response_bytes, self._closed = timeout, max_response_bytes, False

    def close(self) -> None:
        super().close()
        self._api_key = ""

    def _open_http(self, remaining: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=remaining, follow_redirects=False, trust_env=False)

    async def _exchange(self, method: str, path: str, data: dict, deadline: float) -> dict:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError
        encoded = _json_bytes(data) if method in {"POST", "PATCH"} else None
        headers = {"Authorization": "Bearer " + self._api_key, "Accept": "application/json",
                   "Accept-Encoding": "identity"}
        if encoded is not None:
            headers["Content-Type"] = "application/json"
        params = data if method in {"GET", "DELETE"} else None
        async with asyncio.timeout(remaining):
            async with self._open_http(remaining) as client:
                async with client.stream(method, self._base_url + path, headers=headers,
                                         params=params, content=encoded) as response:
                    status = response.status_code
                    if 300 <= status < 400:
                        raise ClientError("redirect_rejected", 502)
                    if response.headers.get("content-encoding", "identity").strip().lower() != "identity":
                        raise ClientError("unsupported_response_encoding", 502)
                    declared = response.headers.get("content-length")
                    if declared is not None and (not declared.isascii() or not declared.isdecimal()
                                                 or len(declared) > 20):
                        raise ClientError("invalid_response", 502)
                    if declared is not None and int(declared) > self._max_response_bytes:
                        raise ClientError("response_too_large", 502)
                    body = bytearray()
                    if response.is_stream_consumed:
                        if len(response.content) > self._max_response_bytes:
                            raise ClientError("response_too_large", 502)
                        body.extend(response.content)
                    else:
                        async for part in response.aiter_raw(chunk_size=16 * 1024):
                            if len(part) > self._max_response_bytes - len(body):
                                raise ClientError("response_too_large", 502)
                            body.extend(part)
                    if monotonic() >= deadline:
                        raise TimeoutError
                    if declared is not None and len(body) != int(declared):
                        raise ClientError("invalid_response", 502)
                    try:
                        payload = _parse_json(bytes(body))
                    except ClientError:
                        if status >= 400:
                            raise ClientError("request_failed", status) from None
                        raise
                    if not 200 <= status < 300:
                        raise _safe_error(payload, status)
                    return payload

    def _dispatch(self, operation: str, collection_id: str | None = None, data: dict | None = None) -> dict:
        self._check_open()
        prepared = _payload(operation, data)
        if operation in {"create", "list"}:
            method, path = ("POST" if operation == "create" else "GET"), "/collections"
        else:
            key = _identifier(collection_id)
            path = "/collections/" + key
            method = {"get": "GET", "modify": "PATCH", "delete_collection": "DELETE",
                      "count": "GET", "indexing_status": "GET"}.get(operation, "POST")
            suffix = {"count": "/count", "indexing_status": "/indexing-status",
                      "fork": "/fork", "search": "/search", "ask": "/ask",
                      "get_records": "/records/get", "delete_records": "/records/delete",
                      "add": "/records/add", "update": "/records/update", "upsert": "/records/upsert"}
            path += suffix.get(operation, "")
        deadline = monotonic() + self._timeout

        def run():
            return asyncio.run(self._exchange(method, path, prepared, deadline))

        try:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return run()
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="rags-collection-client") as pool:
                return pool.submit(run).result()
        except ClientError:
            raise
        except (TimeoutError, httpx.TimeoutException):
            raise ClientError("request_timeout", 504) from None
        except (httpx.HTTPError, ValueError, UnicodeError, OSError):
            raise ClientError("transport_failed", 502) from None


class PersistentClient(_CollectionClient):
    """Authenticated local access to the same collection application service.

    With no settings, the file owner explicitly initializes a local-tenant,
    offline configuration using the supplied credential. It does not load model
    credentials or authentication mappings from the environment. Use from_env
    or explicit settings for an existing multi-tenant/operator configuration.
    """

    def __init__(self, path: str | Path, api_key: str, *, settings: Settings | None = None):
        from fastapi import HTTPException
        from rags.auth import Authenticator
        from rags.config import Settings

        self._api_key, self._closed = _token(api_key), False
        if not isinstance(path, (str, Path)) or not str(path) or "\x00" in str(path):
            raise ClientError("invalid_client_configuration", 422)
        try:
            configured = replace(settings, db_path=str(path)) if settings is not None else Settings(
                db_path=str(path), keys={self._api_key: "local"}, embedding_provider="local",
                model_provider="extractive", openai_key="", openai_base_url="https://api.openai.com/v1",
                embedding_model="offline-hash-v1", chat_model="extractive", anthropic_key="",
                anthropic_model="", key_roles={self._api_key: ("admin",)},
            )
            self._authenticator = Authenticator(mode=configured.auth_mode, keys=configured.keys,
                                               key_roles=configured.key_roles, oauth2=configured.oauth2)
            self._authenticator.authenticate("Bearer " + self._api_key)
        except HTTPException as exc:
            raise _safe_error({"detail": exc.detail}, exc.status_code) from None
        except (TypeError, ValueError, KeyError):
            raise ClientError("invalid_client_configuration", 422) from None
        try:
            from rags.collection_service import CollectionService
            from rags.providers import Providers
            from rags.storage import Store

            self._service = CollectionService(Store(configured.db_path), Providers(configured))
        except Exception:
            raise ClientError("initialization_failed", 503) from None

    @classmethod
    def from_env(cls, api_key: str) -> PersistentClient:
        from rags.config import Settings

        try:
            settings = Settings.from_env()
        except Exception:
            raise ClientError("invalid_client_configuration", 422) from None
        return cls(settings.db_path, api_key, settings=settings)

    def close(self) -> None:
        super().close()
        self._api_key = ""
        self._authenticator = None
        self._service = None

    def _dispatch(self, operation: str, collection_id: str | None = None, data: dict | None = None) -> dict:
        from fastapi import HTTPException
        from rags.registry import CapabilityError

        self._check_open()
        try:
            # Authenticate each operation, before models/providers/storage reads.
            principal = self._authenticator.authenticate("Bearer " + self._api_key)
            prepared = _payload(operation, data)
            if collection_id is not None:
                _identifier(collection_id)
            result = self._service.dispatch(principal, operation, collection_id, prepared)
            if not isinstance(result, dict):
                raise ClientError("invalid_response", 502)
            return result
        except ClientError:
            raise
        except HTTPException as exc:
            raise _safe_error({"detail": exc.detail}, exc.status_code) from None
        except CapabilityError as exc:
            raise _safe_error({"code": exc.code}, exc.status_code) from None
        except (ValueError, TypeError, RecursionError):
            raise ClientError("invalid_request", 422) from None
        except Exception:
            raise ClientError("request_failed", 500) from None
