"""Optional real encoders and an explicit authorized-evidence backend protocol.

The HTTP adapter is an integration boundary, not a reimplementation of the
named research systems. Only the operator selects endpoints and credentials.
"""
import asyncio
import hashlib
import importlib.util
import json
import math
from numbers import Real
import os
from threading import Lock
import time
from urllib.parse import urlparse

import httpx

from rags.features import FeatureFlags, strategy_features
from rags.research_protocol import ResearchRequest, UPSTREAMS, validate_result


PAPER_BACKENDS = {
    "colbert": "ColBERT", "raptor": "RAPTOR", "full_graphrag": "Microsoft GraphRAG",
    "lightrag": "LightRAG", "hipporag": "HippoRAG", "self_rag": "Self-RAG",
    "crag": "CRAG", "multimodal_native": "Aligned multimodal retrieval",
}


class BackendUnavailable(ValueError):
    """A configured capability cannot run, without a silent substitute."""


def maxsim(query, document, query_mask=None, document_mask=None) -> float:
    """Sum of per-query maximum cosine similarities, ignoring masked rows."""
    def normalize(rows, mask):
        if mask is not None and len(mask) != len(rows):
            raise ValueError("Token mask length differs from matrix")
        if mask is not None and any(type(x) not in (bool, int) or x not in (0, 1) for x in mask):
            raise ValueError("Token masks require boolean or zero/one entries")
        vectors = []
        dimension = None
        for i, row in enumerate(rows):
            if mask is not None and not mask[i]:
                continue
            if any(isinstance(x, bool) or not isinstance(x, Real) for x in row):
                raise ValueError("Token vectors require real numeric entries")
            values = [float(x) for x in row]
            if not values or not all(math.isfinite(x) for x in values):
                raise ValueError("Token vectors must be finite and nonempty")
            if dimension is None:
                dimension = len(values)
            if len(values) != dimension:
                raise ValueError("Inconsistent token dimensions")
            scale = max(abs(x) for x in values)
            if scale <= 0:
                raise ValueError("Unmasked token vectors must be nonzero")
            scaled = [x / scale for x in values]
            norm = math.sqrt(sum(x * x for x in scaled))
            vectors.append([x / norm for x in scaled])
        if not vectors:
            raise ValueError("Token matrix has no unmasked vectors")
        return vectors

    q = normalize(query, query_mask)
    d = normalize(document, document_mask)
    if len(q[0]) != len(d[0]):
        raise ValueError("Query and document dimensions differ")
    return sum(max(sum(x * y for x, y in zip(qv, dv, strict=True)) for dv in d) for qv in q)


class BackendManager:
    def __init__(self, settings=None):
        self.features = settings.features if settings is not None else FeatureFlags.from_env()
        self.model_name = os.getenv("RAGS_LATE_INTERACTION_MODEL", "")
        self.cache_dir = os.getenv("RAGS_MODEL_CACHE", "") or None
        self.config = json.loads(os.getenv("RAGS_RESEARCH_BACKENDS_JSON") or "{}")
        if not isinstance(self.config, dict):
            raise ValueError("RAGS_RESEARCH_BACKENDS_JSON must be an object")
        for name, config in self.config.items():
            if name not in PAPER_BACKENDS or not isinstance(config, dict):
                raise ValueError("Unknown research backend or invalid configuration")
            parsed = urlparse(config.get("url", ""))
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username is not None or parsed.password is not None:
                raise ValueError("Research backends require operator-configured HTTP URLs")
            if os.getenv("RAGS_ENV", "production") == "production" and parsed.scheme != "https":
                raise ValueError("Production research backends require HTTPS")
            tokens = config.get("tenant_tokens", {})
            if not isinstance(tokens, dict) or any(
                not isinstance(k, str) or not k or not isinstance(v, str) or not v for k, v in tokens.items()
            ):
                raise ValueError("Backend tenant_tokens must map tenants to credentials")
            if len(set(tokens.values())) != len(tokens):
                raise ValueError("Research backends require distinct tenant credentials")
            timeout = config.get("timeout_seconds", 30)
            if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 1 <= timeout <= 360:
                raise ValueError("Research backend timeout_seconds must be finite and between 1 and 360")
            required = config.get("require_pinned_upstream", False)
            if type(required) is not bool or (required and name not in UPSTREAMS):
                raise ValueError("Pinned upstream identity is supported only for RAPTOR and LightRAG")
            model = config.get("model")
            if model is not None and (not isinstance(model, str) or not 1 <= len(model) <= 200):
                raise ValueError("Research backend model must be a bounded operator label")
        self._encoder = None
        self._lock = Lock()
        self._vectors = {}
        self._observed = {}

    def _requirements(self, mode: str) -> tuple[str, ...]:
        local_encoder = mode in {"multivector", "colbert"} and bool(self.model_name)
        return (*strategy_features(mode), "embedding.fastembed" if local_encoder else "research")

    def catalog(self, tenant: str | None = None) -> list[dict]:
        entries = []
        encoder_installed = importlib.util.find_spec("fastembed") is not None
        supported = False
        if encoder_installed and self.model_name and self.features.enabled("embedding.fastembed"):
            from fastembed import LateInteractionTextEmbedding
            supported = self.model_name in {m["model"] for m in LateInteractionTextEmbedding.list_supported_models()}
        for name in ("multivector", *PAPER_BACKENDS):
            local = name in {"multivector", "colbert"} and bool(self.model_name)
            configured = name in self.config and (tenant is None or tenant in self.config[name].get("tenant_tokens", {}))
            available = (local and supported) or (not local and configured)
            with self._lock:
                observed = self._observed.get((tenant, name)) if tenant is not None else None
            entries.append({
                "id": name, "family": PAPER_BACKENDS.get(name, "Multi-vector retrieval"),
                "implementation": "fastembed_colbert_maxsim" if local else "authorized_evidence_http_v1",
                "requirements": ["encoders extra and checkpoint"] if local else ["operator research backend"],
                "status": "available" if available else "requires_backend",
                "reason": ("Checkpoint loads on first query; availability is not a quality verification"
                           if local and available and self._encoder is None else None if available
                           else "Configure a supported encoder or tenant-authorized research backend"),
                "evidence_level": (("runtime_encoder_loaded" if self._encoder else "runtime_not_yet_exercised") if local
                                   else "pinned_upstream_reported" if observed else "external_unverified"),
                "model": self.model_name if local else self.config.get(name, {}).get("model"),
                "upstream": dict(observed) if observed else None,
            })
            disabled = [feature for feature in self._requirements(name) if not self.features.enabled(feature)]
            entries[-1].update(enabled=not disabled, disabled_by=disabled)
            if disabled:
                entries[-1].update(status="feature_disabled", reason="Disabled features: " + ", ".join(disabled))
        return entries

    def _load_encoder(self):
        self.features.require("embedding.fastembed")
        if not self.model_name:
            raise BackendUnavailable("Set RAGS_LATE_INTERACTION_MODEL for actual token embeddings")
        if self._encoder is None:
            with self._lock:
                if self._encoder is None:
                    try:
                        from fastembed import LateInteractionTextEmbedding
                        self._encoder = LateInteractionTextEmbedding(
                            model_name=self.model_name, cache_dir=self.cache_dir, threads=2,
                        )
                    except Exception as exc:
                        raise BackendUnavailable("Late-interaction encoder/checkpoint could not load") from exc
        return self._encoder

    def encode_tokens(self, text: str, *, is_query: bool = False) -> list[list[float]]:
        encoder = self._load_encoder()
        operation = encoder.query_embed if is_query else encoder.embed
        try:
            return next(iter(operation([text]))).tolist()
        except Exception as exc:
            raise BackendUnavailable("Late-interaction encoding failed") from exc

    @staticmethod
    async def _request(config, token: str, encoded: bytes) -> bytes:
        timeout = config.get("timeout_seconds", 30)
        deadline = time.monotonic() + timeout
        # One wall deadline spans connect, headers and every body read. A fresh
        # synchronous request facade owns this loop; no background task survives.
        async with asyncio.timeout(timeout):
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
                async with client.stream("POST", config["url"], headers={"Authorization": "Bearer " + token,
                                         "Accept-Encoding": "identity", "Content-Type": "application/json"},
                                         content=encoded) as response:
                    response.raise_for_status()
                    if response.headers.get("content-encoding", "identity").lower() != "identity":
                        raise BackendUnavailable("Compressed research responses are not accepted")
                    if int(response.headers.get("content-length", "0")) > 1_000_000:
                        raise BackendUnavailable("Research backend response exceeds limit")
                    body = bytearray()
                    async for part in response.aiter_bytes():
                        if time.monotonic() > deadline:
                            raise BackendUnavailable("Research backend response deadline exceeded")
                        if len(body) + len(part) > 1_000_000:
                            raise BackendUnavailable("Research backend response exceeds limit")
                        body.extend(part)
                    if time.monotonic() > deadline:
                        raise BackendUnavailable("Research backend response deadline exceeded")
                    return bytes(body)

    def search(self, mode, *, tenant, question, chunks, top_k, options=None) -> dict:
        self.features.require(*self._requirements(mode))
        if len(chunks) > 1000:
            raise BackendUnavailable("Research query candidate limit is 1000 authorized chunks")
        if sum(len(c["content"].encode()) for c in chunks) > 2_000_000:
            raise BackendUnavailable("Research query source content exceeds 2MB")
        if mode in {"multivector", "colbert"} and self.model_name:
            query = self.encode_tokens(question, is_query=True)
            hits = []
            for chunk in chunks:
                # Content-addressed cache; no cross-tenant document identifiers are cached.
                digest = hashlib.sha256(chunk["content"].encode()).hexdigest()
                with self._lock:
                    vectors = self._vectors.get(digest)
                if vectors is None:
                    vectors = self.encode_tokens(chunk["content"])
                    with self._lock:
                        if len(self._vectors) >= 1000:
                            self._vectors.clear()
                        self._vectors[digest] = vectors
                hits.append({"chunk_id": chunk["id"], "score": maxsim(query, vectors)})
            hits.sort(key=lambda hit: (-hit["score"], hit["chunk_id"]))
            return {"hits": hits[:top_k], "trace": [{"operation": "token_maxsim", "candidates": len(chunks)}],
                    "engine": "fastembed:" + self.model_name, "usage": {"provider": "local_onnx"}}
        config = self.config.get(mode)
        if config is None:
            raise BackendUnavailable("Actual upstream backend is not configured for " + mode)
        token = config.get("tenant_tokens", {}).get(tenant)
        if not token:
            raise BackendUnavailable("No tenant credential is configured for this research backend")
        authorized = {c["id"] for c in chunks}
        payload = {
            "protocol": "authorized-evidence-v1", "question": question, "top_k": top_k,
            "namespace": tenant, "engine": mode, "model": config.get("model"),
            "passages": [{"id": c["id"], "text": c["content"]} for c in chunks],
            "options": {k: v for k, v in (options or {}).items() if k in {"max_iterations"}},
        }
        encoded = json.dumps(payload, allow_nan=False).encode()
        if len(encoded) > 3_000_000:
            raise BackendUnavailable("Research backend request exceeds 3MB")
        try:
            request_contract = ResearchRequest.model_validate(payload) if mode in UPSTREAMS else None
            body = asyncio.run(self._request(config, token, encoded))
            data = json.loads(body)
            if not isinstance(data, dict):
                raise ValueError("Invalid backend response")
            provenance = None
            diagnostics = None
            if "implementation" in data or "diagnostics" in data or config.get("require_pinned_upstream"):
                if request_contract is None:
                    raise ValueError("Unsupported pinned upstream identity")
                verified = validate_result(data, request_contract)
                provenance, diagnostics = verified["implementation"], verified["diagnostics"]
            raw = data.get("hits")
            if not isinstance(raw, list) or len(raw) > 100:
                raise ValueError("Invalid backend hits")
            hits = []
            seen = set()
            for hit in raw:
                if not isinstance(hit, dict) or type(hit.get("score")) not in (int, float):
                    raise ValueError("Backend scores require real numbers")
                identifier, score = hit.get("chunk_id"), float(hit.get("score"))
                if identifier not in authorized or not math.isfinite(score):
                    raise ValueError("Backend returned unauthorized or invalid evidence")
                if identifier not in seen:
                    hits.append({"chunk_id": identifier, "score": score})
                    seen.add(identifier)
            hits.sort(key=lambda hit: (-hit["score"], hit["chunk_id"]))
            usage = data.get("usage")
            if usage is not None and (not isinstance(usage, dict) or any(
                key not in {"input_tokens", "output_tokens"} or type(value) is not int or not 0 <= value <= 10_000_000
                for key, value in usage.items()
            )):
                raise ValueError("Backend usage must contain bounded nonnegative token counts")
            event = {"operation": "external_backend", "backend": mode,
                     "evidence_level": "pinned_upstream_reported" if provenance else "external_unverified"}
            if provenance:
                event.update(implementation=provenance, diagnostics=diagnostics)
                if diagnostics.get("engine_executed"):
                    with self._lock:
                        self._observed[tenant, mode] = dict(provenance)
            return {"hits": hits[:top_k], "trace": [event],
                    "engine": mode + ":authorized-evidence-v1", "usage": {"reported": usage}}
        except BackendUnavailable:
            raise
        except TimeoutError as exc:
            raise BackendUnavailable("Research backend response deadline exceeded") from exc
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as exc:
            raise BackendUnavailable("Research backend failed protocol or evidence validation") from exc
