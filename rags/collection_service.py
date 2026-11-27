"""Authenticated application service shared by collection HTTP and local clients."""
from __future__ import annotations

from dataclasses import asdict
from functools import cached_property
import hashlib
import json
import os
from time import monotonic

from pydantic import ValidationError

from rags.agents.deadline import execution_deadline, remaining_timeout
from rags.auth import Principal, TENANT
from rags.collection_filters import compile_filters
from rags.collection_models import (
    CollectionAsk, CollectionConfig, CollectionCreate, CollectionFork, CollectionList,
    CollectionModify, DeleteRecords, GenerationCheck, GetRecords, RecordBatch,
    SearchBatch, SearchQuery, canonical_dense, valid_text,
)
from rags.collection_search import execute_get, execute_search, select_matching_records
from rags.collection_store import CollectionStore
from rags.providers import Providers
from rags.registry import CapabilityError
from rags.retrieval import RAGService
from rags.storage import Store
from rags.tokenization import resolve_tokenizer


OPERATIONS = {
    "create": ("rags:write",), "list": ("rags:read",), "get": ("rags:read",),
    "modify": ("rags:write",), "delete_collection": ("rags:delete",),
    "count": ("rags:read",), "indexing_status": ("rags:read",),
    "fork": ("rags:read", "rags:write"), "add": ("rags:write",),
    "update": ("rags:write",), "upsert": ("rags:write",),
    "get_records": ("rags:read",), "delete_records": ("rags:delete",),
    "search": ("rags:query",), "ask": ("rags:query",),
}
REQUEST_MODELS = {
    "create": CollectionCreate, "list": CollectionList, "modify": CollectionModify,
    "delete_collection": GenerationCheck, "fork": CollectionFork,
    "add": RecordBatch, "update": RecordBatch, "upsert": RecordBatch,
    "get_records": GetRecords, "delete_records": DeleteRecords,
    "search": SearchBatch, "ask": CollectionAsk,
}
OPERATION_FEATURES = {
    "create": ("ingestion",), "list": ("documents",), "get": ("documents",),
    "modify": ("ingestion",), "delete_collection": ("deletion",),
    "count": ("documents",), "indexing_status": ("documents",),
    "fork": ("documents", "ingestion", "collections.fork"),
    "add": ("ingestion",), "update": ("ingestion",), "upsert": ("ingestion",),
    "get_records": ("documents",), "delete_records": ("deletion",),
    "search": ("query",), "ask": ("query",),
}


def _invalid(reason: str = "Invalid collection request") -> CapabilityError:
    return CapabilityError("invalid_collection_input", reason, 422)


def _expected(info: dict, expected: int | None) -> None:
    if expected is not None and expected != info["generation"]:
        raise CapabilityError("collection_generation_conflict", "Collection generation changed; refresh and retry", 409)


def _fingerprint(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class CollectionService:
    def __init__(self, store: Store, provider: Providers, config: CollectionConfig | None = None):
        self.store = store
        self.provider = provider
        self.settings = provider.settings
        self.config = config if config is not None else self.settings.collection_config
        if not isinstance(self.config, CollectionConfig):
            raise ValueError("Collection service needs validated limits")

    @cached_property
    def database(self) -> CollectionStore:
        # A disabled feature or denied principal never initializes collection tables.
        return CollectionStore(self.store, max_records=self.config.max_records,
                               max_scan_bytes=self.config.max_scan_bytes,
                               result_validator=self._response)

    def capabilities(self, principal: Principal) -> dict:
        enabled = self.settings.features.enabled("collections")
        methods = [name for name, scopes in OPERATIONS.items()
                   if enabled and set(scopes) <= principal.permissions
                   and all(self.settings.features.enabled(flag) for flag in OPERATION_FEATURES[name])]
        return {"enabled": enabled, "methods": methods, "limits": asdict(self.config),
                "record_unit": "one_vector_per_record", "consistency": "transactional",
                "dense_index": "pgvector_exact" if self.store.is_postgres else "float32_cosine_exact",
                "lexical_ranking": "bounded_bm25", "sparse_ranking": "explicit_space_dot_product",
                "hybrid_ranking": "positive_weighted_rrf", "score_order": "descending",
                "fork": "shared_immutable_payloads_linear_mapping_copy",
                "features": {key: enabled and self.settings.features.enabled("collections." + key)
                             for key in ("fork", "regex", "sparse_vectors", "batch", "grouping")}}

    def _response(self, result):
        raw = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        if len(raw) > self.config.max_response_bytes:
            raise CapabilityError("collection_response_limit", "Collection response exceeds its byte limit; reduce batch, projection or page size", 422)
        return result

    def authorize(self, principal: Principal, operation: str) -> None:
        if (not isinstance(principal, Principal) or not TENANT.fullmatch(principal.tenant)
                or operation not in OPERATIONS):
            raise CapabilityError("permission_denied", "Collection operation is not authorized", 403)
        principal.require(*OPERATIONS[operation])
        self.settings.features.require("collections", *OPERATION_FEATURES[operation])

    def dispatch(self, principal: Principal, operation: str, collection_id: str | None = None,
                 data: dict | None = None):
        self.authorize(principal, operation)
        try:
            if not isinstance(data, (dict, type(None))):
                raise ValueError("Request must be an object")
            body = data if data is not None else {}
            model = REQUEST_MODELS.get(operation)
            request = model.model_validate(body) if model is not None else None
            if model is None and body:
                raise ValueError("This operation does not accept body fields")
            if operation not in {"create", "list"}:
                valid_text(collection_id, 128)
        except (ValidationError, ValueError, TypeError) as exc:
            raise _invalid() from exc
        parent_deadline = execution_deadline.get()
        deadline = min(parent_deadline, monotonic() + 60) if parent_deadline is not None else monotonic() + 60
        token = execution_deadline.set(deadline)
        try:
            return self._response(self._dispatch(principal, operation, collection_id, request))
        except CapabilityError as exc:
            if exc.code == "tool_timeout":
                raise CapabilityError("collection_timeout", "Collection execution deadline was exhausted", 504) from exc
            raise
        except (ValueError, TypeError) as exc:
            raise _invalid() from exc
        finally:
            execution_deadline.reset(token)

    def _dispatch(self, principal: Principal, operation: str, cid: str | None, request):
        tenant = principal.tenant
        if operation == "create":
            return self._create(tenant, request)
        if operation == "list":
            if request.name is not None:
                found = self.database.find_by_name(tenant, request.name)
                rows = [found] if found is not None else []
                return {"items": rows[request.offset:request.offset + request.limit], "total": len(rows),
                        "limit": request.limit, "offset": request.offset}
            return {"items": self.database.list(tenant, request.limit, request.offset),
                    "total": self.database.collection_count(tenant), "limit": request.limit, "offset": request.offset}
        if operation == "get":
            return self.database.get(tenant, cid)
        if operation == "modify":
            return self.database.modify(tenant, cid, **request.model_dump())
        if operation == "delete_collection":
            return self.database.delete_collection(tenant, cid, expected_generation=request.expected_generation)
        if operation == "fork":
            return self.database.fork(tenant, cid, request.name, metadata=request.metadata,
                                      expected_generation=request.expected_generation)
        if operation in {"count", "indexing_status"}:
            info = self.database.get(tenant, cid)
            result = {"collection_id": cid, "generation": info["generation"], "count": info["record_count"]}
            if operation == "indexing_status":
                result.update(state="ready", consistency="transactional", record_count=info["record_count"],
                              vector_count=info["record_count"], pending_operations=0,
                              embedding=info["configuration"]["embedding"],
                              text_query_profile_matches=self._configured_matches(info),
                              index="pgvector_exact" if self.store.is_postgres else "float32_cosine_exact")
            return result
        if operation in {"add", "update", "upsert"}:
            return self._write(tenant, cid, operation, request)
        if operation == "get_records":
            self._preflight_filters(request)
            if "sparse_embedding" in request.include:
                self.settings.features.require("collections.sparse_vectors")
            with self.database.read_snapshot(tenant, cid, record_ids=request.ids) as snapshot:
                result = execute_get(snapshot, request, deadline=monotonic() + self.config.filter_timeout_seconds)
                return {"collection_id": cid, "generation": snapshot.collection["generation"], **result}
        if operation == "delete_records":
            return self._delete_records(tenant, cid, request)
        if operation == "search":
            return self._search(tenant, cid, request)[0]
        if operation == "ask":
            return self._ask(principal, cid, request)
        raise _invalid()

    def _configured_identity(self) -> dict:
        model = valid_text(self.provider.embedding_model, 160)
        endpoint = valid_text(self.provider.embedding_endpoint, 4096)
        space = self.settings.embedding_space_id or ("configured:" + _fingerprint({"model": model, "endpoint": endpoint}))
        return {"mode": "configured", "model": model, "endpoint": endpoint, "space_id": space,
                "dimensions": None, "normalization": "l2-f32-v1"}

    def _configured_matches(self, info: dict) -> bool:
        binding = info["configuration"]["embedding"]
        if binding["mode"] != "configured":
            return False
        active = self._configured_identity()
        return all(binding[key] == active[key] for key in ("model", "endpoint", "space_id"))

    def _create(self, tenant: str, request: CollectionCreate) -> dict:
        if request.sparse_space_id is not None:
            self.settings.features.require("collections.sparse_vectors")
        if request.get_or_create:
            existing = self.database.find_by_name(tenant, request.name)
            if existing is not None:
                return existing
        if request.embedding is None:
            embedding = self._configured_identity()
        else:
            embedding = {**request.embedding.model_dump(), "endpoint": "external", "normalization": "l2-f32-v1"}
        configuration = {"embedding": embedding, "sparse_space_id": request.sparse_space_id}
        return self.database.create(tenant, request.name, request.metadata, configuration,
                                    get_or_create=request.get_or_create)

    def _preflight_filters(self, request) -> None:
        compile_filters(request.where, request.where_document,
                        allow_regex=self.settings.features.enabled("collections.regex"))

    def _preflight_embeddings(self, texts: list[str], *, query: bool = False) -> None:
        """Count every complete input before encoding any member of a batch."""
        if not texts:
            return
        self.settings.features.require("embedding." + self.settings.embedding_provider)
        for text in texts:
            valid_text(text, 20_000)
        validate = getattr(self.provider, "validate_embedding_inputs", None)
        if callable(validate):
            # The production provider owns model-specific preprocessing and its
            # complete-input window; batches validate before the first call.
            validate(texts, query=query)
            return
        callback = getattr(self.provider, "embedding_token_count", None)
        if callable(callback):
            maximum = getattr(self.provider, "embedding_token_limit", None)
            if type(maximum) is not int or not 1 <= maximum <= 100_000:
                raise CapabilityError("requires_provider", "An exact embedding token window is required")
            count = callback
        elif self.settings.embedding_provider == "local":
            return
        elif self.settings.embedding_provider == "fastembed":
            try:
                from rags.providers import _semantic_encoder
                from rags.research.common import disable_encoder_telemetry, fastembed_token_window
                disable_encoder_telemetry()
                encoder = _semantic_encoder(self.settings.fastembed_model, os.getenv("RAGS_MODEL_CACHE") or None)
                count, maximum = fastembed_token_window(encoder)
            except ImportError as exc:
                raise CapabilityError("requires_provider", "Configured embedding runtime is unavailable") from exc
        else:
            self.settings.features.require("tokenizer.tiktoken")
            name = self.settings.embedding_model
            if self.settings.embedding_provider == "router":
                resolved = self.provider.model_router.config.models.get(name, "")
                name = resolved.removeprefix("openai/") if resolved.startswith("openai/") else ""
            if name not in {"text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"}:
                raise CapabilityError("requires_provider", "Configured model needs an explicit supported embedding token window; precomputed vectors remain available")
            tokenizer = resolve_tokenizer({"backend": "tiktoken", "model": name})
            count, maximum = lambda text, query=False: tokenizer.count(text), 8191
        for text in texts:
            remaining_timeout(30)
            units = count(text, query=query)
            if type(units) is not int or units < 0:
                raise CapabilityError("invalid_embedding", "Embedding token counter returned an invalid value", 502)
            if units > maximum:
                raise CapabilityError("context_budget_exceeded", "Complete collection input exceeds the embedding token window", 422)

    def _embed(self, text: str, *, query: bool = False) -> list[float]:
        remaining_timeout(30)
        try:
            vector = self.provider.query_embedding(text) if query else self.provider.embedding(text)
            result = canonical_dense(vector)
        except CapabilityError:
            raise
        except Exception as exc:
            raise CapabilityError("embedding_failed", "Configured embedding provider failed or returned an invalid vector", 502) from exc
        remaining_timeout(30)
        return result

    @staticmethod
    def _vector_binding(info: dict, vector: list[float], space: str | None) -> None:
        binding = info["configuration"]["embedding"]
        if space != binding["space_id"]:
            raise CapabilityError("embedding_space_mismatch", "Vector space does not match this collection", 409)
        if binding["dimensions"] is not None and len(vector) != binding["dimensions"]:
            raise CapabilityError("embedding_dimension_mismatch", "Vector dimensions do not match this collection", 409)

    @staticmethod
    def _sparse_binding(info: dict, vector: dict | None) -> None:
        if vector is not None and vector["space_id"] != info["configuration"]["sparse_space_id"]:
            raise CapabilityError("embedding_space_mismatch", "Sparse vector space does not match this collection", 409)

    def _write(self, tenant: str, cid: str, mode: str, request: RecordBatch) -> dict:
        if len(request.ids) > self.config.max_batch:
            raise CapabilityError("collection_limit_exceeded", "Record batch exceeds the configured limit", 422)
        if len(request.ids) > 1:
            self.settings.features.require("collections.batch")
        if request.sparse_embeddings is not None:
            self.settings.features.require("collections.sparse_vectors")
        with self.database.read_snapshot(tenant, cid, include_embeddings=True, record_ids=request.ids) as snapshot:
            info = snapshot.collection
            _expected(info, request.expected_generation)
            existing = {record["id"]: record for record in snapshot.records}
        if mode == "add" and existing:
            raise CapabilityError("collection_conflict", "At least one record already exists", 409)
        if mode == "update" and len(existing) != len(request.ids):
            raise CapabilityError("record_not_found", "At least one record does not exist", 404)
        if info["record_count"] + len(set(request.ids) - set(existing)) > self.config.max_records:
            raise CapabilityError("collection_limit_exceeded", "Collection record limit would be exceeded", 422)
        prepared, needs_embedding = [], []
        for index, identifier in enumerate(request.ids):
            old = existing.get(identifier, {})
            document = request.documents[index] if request.documents is not None else old.get("document")
            metadata = request.metadatas[index] if request.metadatas is not None else old.get("metadata", {})
            sparse = request.sparse_embeddings[index] if request.sparse_embeddings is not None else old.get("sparse_embedding")
            self._sparse_binding(info, sparse)
            if request.embeddings is not None:
                vector = request.embeddings[index]
                self._vector_binding(info, vector, request.embedding_space_id)
            elif old and (document is None or document == old["document"]):
                vector = old["embedding"]
            else:
                if not document or not document.strip():
                    raise _invalid("New records require nonempty text or an explicit vector")
                if not self._configured_matches(info):
                    raise CapabilityError("embedding_space_mismatch", "Text encoding requires this collection's configured embedding profile", 409)
                vector = None
                needs_embedding.append((index, document))
            prepared.append({"id": identifier, "document": document, "metadata": metadata,
                             "embedding": vector, "sparse_embedding": sparse})
        self._preflight_embeddings([text for _, text in needs_embedding])
        self.provider.begin_usage()
        for index, text in needs_embedding:
            vector = self._embed(text)
            self._vector_binding(info, vector, info["configuration"]["embedding"]["space_id"])
            prepared[index]["embedding"] = vector
        if len({len(record["embedding"]) for record in prepared}) != 1:
            raise CapabilityError("embedding_dimension_mismatch", "Batch contains inconsistent embedding dimensions", 409)
        usage = self.provider.usage()
        # Provider usage can grow with every item. Check a conservative complete
        # envelope before committing; never report a failed write after it has
        # already become visible because the response byte cap was exceeded.
        self._response({"collection_id": cid, "generation": max(info["generation"] + 1, 2 ** 63 - 1),
                        "changed": len(prepared), "added": len(prepared), "updated": len(prepared),
                        "usage": usage})
        remaining_timeout(30)
        result = self.database.write(tenant, cid, mode, prepared, expected_generation=info["generation"])
        return {**result, "usage": usage}

    def _delete_records(self, tenant: str, cid: str, request: DeleteRecords) -> dict:
        self._preflight_filters(request)
        with self.database.read_snapshot(tenant, cid, record_ids=request.ids) as snapshot:
            info = snapshot.collection
            _expected(info, request.expected_generation)
            rows = select_matching_records(snapshot, request.where, request.where_document, request.ids,
                                           deadline=monotonic() + self.config.filter_timeout_seconds)
            ids = [row["id"] for row in rows]
        if len(ids) > self.config.max_batch:
            raise CapabilityError("collection_limit_exceeded", "Delete matches more than the configured batch limit", 422)
        if len(ids) > 1:
            self.settings.features.require("collections.batch")
        remaining_timeout(30)
        return self.database.delete_records(tenant, cid, ids, expected_generation=info["generation"])

    def _preflight_search(self, request: SearchQuery, info: dict) -> None:
        self._preflight_filters(request)
        if request.group_by is not None:
            self.settings.features.require("collections.grouping")
        if request.query_sparse is not None or "sparse_embedding" in request.select:
            self.settings.features.require("collections.sparse_vectors")
        if request.mode == "hybrid":
            self.settings.features.require("strategy.hybrid")
        if request.query_embedding is not None:
            self._vector_binding(info, request.query_embedding, request.embedding_space_id)
        if request.uses_dense:
            self.settings.features.require("strategy.dense")
            if request.query_embedding is None and not self._configured_matches(info):
                raise CapabilityError("embedding_space_mismatch", "Text query requires the collection's configured embedding profile", 409)
        if request.mode == "lexical" or request.uses_sparse:
            self.settings.features.require("strategy.sparse")
        if request.query_sparse is not None:
            self._sparse_binding(info, request.query_sparse)

    def _search(self, tenant: str, cid: str, request: SearchBatch) -> tuple[dict, dict]:
        if len(request.searches) > self.config.max_queries:
            raise CapabilityError("collection_limit_exceeded", "Search batch exceeds the configured limit", 422)
        if len(request.searches) > 1:
            self.settings.features.require("collections.batch")
        info = self.database.get(tenant, cid)
        for search in request.searches:
            self._preflight_search(search, info)
        to_encode = [search.query_text for search in request.searches
                     if search.uses_dense and search.query_embedding is None and info["record_count"]]
        self._preflight_embeddings(to_encode, query=True)
        self.provider.begin_usage()
        cache, vectors = {}, []
        for search in request.searches:
            vector = search.query_embedding
            if search.uses_dense and vector is None and info["record_count"]:
                if search.query_text not in cache:
                    cache[search.query_text] = self._embed(search.query_text, query=True)
                vector = cache[search.query_text]
                self._vector_binding(info, vector, info["configuration"]["embedding"]["space_id"])
            vectors.append(vector)
        remaining_timeout(30)
        with self.database.read_snapshot(tenant, cid) as snapshot:
            current = snapshot.collection
            # An initially empty collection may acquire data during provider-free
            # preparation. Require a fresh request instead of ranking without a vector.
            if current["record_count"] and any(s.uses_dense and v is None for s, v in zip(request.searches, vectors, strict=True)):
                raise CapabilityError("collection_generation_conflict", "Collection changed during query preparation; retry", 409)
            for search, vector in zip(request.searches, vectors, strict=True):
                if vector is not None and search.uses_dense:
                    self._vector_binding(current, vector, current["configuration"]["embedding"]["space_id"])
            deadline = monotonic() + self.config.filter_timeout_seconds
            results = [execute_search(snapshot, search, query_embedding=vector, deadline=deadline)
                       for search, vector in zip(request.searches, vectors, strict=True)]
            return ({"collection_id": cid, "generation": current["generation"], "results": results,
                     "evidence_status": "ranked_unassessed", "usage": self.provider.usage()}, current)

    def _ask(self, principal: Principal, cid: str, request: CollectionAsk) -> dict:
        if request.generate:
            self.settings.features.require("tokenizer." + self.settings.generation_tokenizer_config.get("backend", "unicode"))
            if self.settings.model_provider != "extractive":
                principal.require("rags:generate")
                self.settings.features.require("model_generation")
        result, info = self._search(principal.tenant, cid,
                                    SearchBatch(searches=[request.search_query().model_dump()]))
        ranked = result["results"][0]
        profile = {"algorithm": "collection-record-v1", **info["configuration"]["embedding"]}
        profile_id = _fingerprint(profile)
        chunks = []
        for row in ranked["items"]:
            document = row.get("document")
            if not document or not document.strip():
                continue
            identity = [principal.tenant, cid, row["id"]]
            chunk_id = "collection:" + _fingerprint([*identity, row["version"]])
            chunks.append({"id": chunk_id, "document_id": "collection:" + _fingerprint(identity),
                           "collection_id": cid, "record_id": row["id"],
                           "revision": row["version"], "profile_id": profile_id,
                           "title": row["id"], "source": "collection:" + cid, "section": "record",
                           "content": document, "start_char": 0, "end_char": len(document),
                           "role": "leaf", "score": row["score"], "metadata": row.get("metadata", {}),
                           "origin_source": "collection", "origin_chunk_id": chunk_id})
        remaining_timeout(30)
        rag = RAGService(self.store, self.provider)
        answer = rag.answer_result(request.question, request.mode, request.generate,
                                   {"chunks": chunks, "mode_used": request.mode,
                                    "engine": "collection:" + ranked["score_kind"],
                                    "trace": [{"action": "collection_search", "collection_id": cid,
                                               "generation": result["generation"],
                                               "candidate_count": ranked["candidate_count"]}]})
        return {"collection_id": cid, "generation": result["generation"], **answer}
