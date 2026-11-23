"""Pinned Microsoft GraphRAG standard indexing and local graph-context retrieval.

The upstream graph, Leiden communities, reports and context builder execute in
one private request directory. Only verified original text units become hits;
generated entity descriptions and community reports never become source text.
"""
from __future__ import annotations

import asyncio
import hashlib
import html
import logging
import math
import os
import re
import shutil
from contextvars import ContextVar
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from threading import Lock

from rags.model_profiles import strict_json
from rags.registry import CapabilityError
from rags.research.common import ProviderBridge, private_directory, validate_request, verify_source

SOURCE_COMMIT = "5faaaf4f5685fa2056fa8c3bf9342cb089f2942f"
SOURCE_FILES = {
    "__init__.py": "1971e2f23c84b1bf807a986a0498d8568e54bb4396bc06e1efcbf944facfdc64",
    "api/index.py": "f3c45fc18b0f7a375b31acabdd856414cc14d576369486155a9958cf18d5b864",
    "index/workflows/factory.py": "61a8f11d7567f04b65753c75cc5f8d8543f6a8dd37297d63cfc248a9119bb4da",
    "index/run/run_pipeline.py": "d9a75a91c36688c4fffff0794442ada02d6e1d3a5058716ba3a0b6c6716c6676",
    "index/workflows/create_base_text_units.py": "e5d5e08a5f0cce617004c72b3c222325d802144a8de8e1e7a8a1d6ee4f426318",
    "index/workflows/create_final_text_units.py": "4973551ddc8741099daad74e4ff05797129f7b3f13026268266a1900f4f70797",
    "index/workflows/extract_graph.py": "69cbb245fec857c4939b44a42e9de7d0cbb4ad243d07b8666950af97f41afdee",
    "index/operations/extract_graph/extract_graph.py": "5ccfb0fe91171623b7973d498dd4c744037d8eca79c1d0b276eafd04bc09e6ee",
    "index/operations/extract_graph/graph_extractor.py": "300825a90fdf53c43d017d9a5f0146cb56e0265f9594424bb4b4f7fd251242df",
    "index/utils/string.py": "aadd91d186fc3c72181d9ea8e152d37dc17865c5dbc3ac5635f175868ebeeeb1",
    "index/operations/cluster_graph.py": "93c51b3f562c5176170d226ae26234360bc80feee799e06feccea923e22e10be",
    "index/workflows/create_community_reports.py": "ae1cd8ccd4fff439ca5480439b5f09ea9330680f39abbea53e91a288992cc724",
    "index/operations/summarize_communities/community_reports_extractor.py": "7382df03b42eae7b43572af5682a7d8f94516615b34c8a6ea849a7cfdc62b279",
    "query/structured_search/local_search/mixed_context.py": "93c019de4fbce62e7c4031eeb161d06f58818ed1660c6b4122770fba26c08287",
    "query/context_builder/entity_extraction.py": "9d8d4a22d55165b3a3dc440b28cc02df91ddeee30593275033782820bb05739c",
    "query/context_builder/source_context.py": "01cfc554b5833996020db85e8b9e2576d1a8b89e7390a95d87b9ab7ace6ed752",
    "query/factory.py": "a702434d6a6c8df9867f3b1e143de96ea6cb10ed177137fa0c9d095fb5a9c734",
    "query/indexer_adapters.py": "d330af291d827941fe904c6d1532913249e9a8aa05702bf8f3fa9eb6cbc28270",
}
PIPELINE_STAGES = (
    "create_base_text_units", "create_final_documents", "extract_graph",
    "finalize_graph", "extract_covariates", "create_communities",
    "create_final_text_units", "create_community_reports", "generate_text_embeddings",
)
_BUDGET_ERRORS = {
    "context_budget_exceeded": "GraphRAG provider input exceeds its configured token budget",
    "research_limit_exceeded": "GraphRAG provider call budget was exhausted",
}
_ACTIVE_STATE: ContextVar = ContextVar("rags_graphrag_request")


def group_sources(sources: dict[str, str]) -> tuple[list[dict], dict[str, list[str]]]:
    """Deduplicate upstream content IDs while retaining every original identity."""
    documents, groups, texts = [], {}, {}
    for identifier, text in sources.items():
        key = "source-" + hashlib.sha256(text.encode()).hexdigest()
        if key in texts and texts[key] != text:
            raise CapabilityError("invalid_source_lineage", "Conflicting GraphRAG source digest", 502)
        if key not in texts:
            texts[key] = text
            documents.append({"id": key, "title": key, "text": text,
                              "creation_date": "", "raw_data": ""})
            groups[key] = []
        groups[key].append(identifier)
    return documents, groups


def project_sources(selected: list[dict], text_units: list[dict], documents: list[dict],
                    groups: dict[str, list[str]], sources: dict[str, str], top_k: int) -> list[dict]:
    """Resolve human-readable context IDs through exact, complete source lineage."""
    if (not isinstance(selected, list) or len(selected) > 64
            or not isinstance(text_units, list) or len(text_units) != len(groups)
            or not isinstance(documents, list) or len(documents) != len(groups)):
        raise CapabilityError("invalid_source_lineage", "Invalid GraphRAG source tables", 502)
    doc_lookup = {}
    for doc in documents:
        if not isinstance(doc, dict) or not isinstance(doc.get("id"), str) or doc["id"] not in groups:
            raise CapabilityError("invalid_source_lineage", "Unknown GraphRAG original document", 502)
        key = doc["id"]
        if key in doc_lookup or any(identifier not in sources or sources[identifier] != doc.get("text")
                                    for identifier in groups[key]):
            raise CapabilityError("invalid_source_lineage", "GraphRAG original document changed", 502)
        doc_lookup[key] = doc
    units, internal_ids, document_ids = {}, set(), set()
    for unit in text_units:
        if not isinstance(unit, dict):
            raise CapabilityError("invalid_source_lineage", "Invalid GraphRAG text unit", 502)
        internal, short, doc_id = unit.get("id"), unit.get("human_readable_id"), unit.get("document_id")
        if (not isinstance(internal, str) or not internal or type(short) is not int or short < 0
                or internal in internal_ids or str(short) in units or not isinstance(doc_id, str) or doc_id not in doc_lookup
                or doc_id in document_ids or unit.get("text") != doc_lookup[doc_id]["text"]):
            raise CapabilityError("invalid_source_lineage", "GraphRAG text-unit lineage changed", 502)
        internal_ids.add(internal)
        document_ids.add(doc_id)
        units[str(short)] = unit
    hits, seen = {}, set()
    for rank, item in enumerate(selected):
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise CapabilityError("invalid_source_lineage", "Invalid GraphRAG selected source", 502)
        key = item["id"]
        if key in seen or key not in units or item.get("text") != units[key]["text"]:
            raise CapabilityError("invalid_source_lineage", "GraphRAG selected source changed", 502)
        seen.add(key)
        for original in groups[units[key]["document_id"]]:
            hits[original] = 1 / (rank + 1)
    return [{"chunk_id": key, "score": score} for key, score in
            sorted(hits.items(), key=lambda item: (-item[1], item[0]))[:top_k]]


def _extracted_text(value: str) -> str:
    # Match the pinned upstream clean_str exactly. Quotes remain part of an
    # entity identity; HTML escapes and controls must not disguise new endpoints.
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", html.unescape(value.strip()))


def validate_extraction(result: str) -> str:
    """Validate and serialize exactly the records the pinned parser will read.

    Upstream does not strip individual record-type fields or separate its final
    completion marker. Passing the original response after permissive validation
    can drop entities or silently replace the last relationship weight with 1.
    Canonical field separators preserve the checked records and numeric values.
    """
    if not isinstance(result, str) or not result.strip().endswith("<|COMPLETE|>"):
        raise CapabilityError("invalid_model_output", "GraphRAG extraction is incomplete", 502)
    body = result.strip()[:-len("<|COMPLETE|>")].strip()
    rows = [row.strip() for row in body.split("##") if row.strip()]
    if not 1 <= len(rows) <= 64:
        raise CapabilityError("invalid_model_output", "GraphRAG extraction record limit exceeded", 502)
    entities, relationships, canonical = set(), [], []
    for row in rows:
        if not row.startswith("(") or not row.endswith(")"):
            raise CapabilityError("invalid_model_output", "Invalid GraphRAG extraction row", 502)
        fields = [value.strip() for value in row[1:-1].split("<|>")]
        if any(not value or len(value) > 8000 for value in fields):
            raise CapabilityError("invalid_model_output", "Invalid GraphRAG extraction field", 502)
        if fields[0] == '"entity"' and len(fields) == 4 and len(fields[1]) <= 200:
            name = _extracted_text(fields[1].upper())
            if (not name.strip().strip('"').strip() or len(name) > 200
                    or not _extracted_text(fields[2].upper()).strip()
                    or not _extracted_text(fields[3]).strip()):
                raise CapabilityError("invalid_model_output", "Invalid GraphRAG entity fields", 502)
            entities.add(name)
        elif fields[0] == '"relationship"' and len(fields) == 5:
            try:
                weight = float(fields[4])
            except ValueError as exc:
                raise CapabilityError("invalid_model_output", "Invalid GraphRAG relationship weight", 502) from exc
            if not math.isfinite(weight) or not 0 <= weight <= 10:
                raise CapabilityError("invalid_model_output", "Invalid GraphRAG relationship weight", 502)
            if not _extracted_text(fields[3]).strip():
                raise CapabilityError("invalid_model_output", "Invalid GraphRAG relationship description", 502)
            relationships.append((_extracted_text(fields[1].upper()), _extracted_text(fields[2].upper())))
        else:
            raise CapabilityError("invalid_model_output", "Invalid GraphRAG extraction row", 502)
        canonical.append("(" + "<|>".join(fields) + ")")
    if len(entities) > 32 or len(relationships) > 32 or any(
        source not in entities or target not in entities for source, target in relationships
    ):
        raise CapabilityError("invalid_model_output", "GraphRAG extraction has unknown entities", 502)
    return "##".join([*canonical, "<|COMPLETE|>"])


def validate_pipeline(results, errors: list) -> None:
    if (errors or not isinstance(results, list)
            or tuple(getattr(item, "workflow", None) for item in results) != PIPELINE_STAGES
            or any(getattr(item, "error", None) is not None for item in results)):
        raise CapabilityError("invalid_upstream_output", "GraphRAG indexing did not complete successfully", 502)


def validate_report(value) -> None:
    """Community rows need actual bounded content, not merely a parsed object."""
    fields = {"title", "summary", "findings", "rating", "rating_explanation"}
    if (not isinstance(value, dict) or set(value) != fields
            or any(not isinstance(value.get(key), str) or not value[key].strip() or len(value[key]) > 8000
                   for key in ("title", "summary", "rating_explanation"))
            or type(value.get("rating")) not in (int, float) or not 0 <= value["rating"] <= 10
            or not isinstance(value.get("findings"), list) or not 1 <= len(value["findings"]) <= 16):
        raise CapabilityError("invalid_model_output", "Invalid GraphRAG community report", 502)
    for finding in value["findings"]:
        if (not isinstance(finding, dict) or set(finding) != {"summary", "explanation"}
                or any(not isinstance(text, str) or not text.strip() or len(text) > 8000 for text in finding.values())):
            raise CapabilityError("invalid_model_output", "Invalid GraphRAG community finding", 502)


def _messages(messages) -> tuple[str, str, list]:
    if isinstance(messages, str):
        return "", messages, []
    if not isinstance(messages, list) or not 1 <= len(messages) <= 16:
        raise CapabilityError("invalid_model_input", "Invalid GraphRAG message list", 422)
    normalized = []
    for message in messages:
        if (not isinstance(message, dict) or set(message) - {"role", "content"}
                or message.get("role") not in {"system", "user", "assistant"}
                or not isinstance(message.get("content"), str)):
            raise CapabilityError("invalid_model_input", "Invalid GraphRAG message", 422)
        normalized.append(message)
    if normalized[-1]["role"] != "user":
        raise CapabilityError("invalid_model_input", "GraphRAG request must end with a user message", 422)
    system = "\n".join(item["content"] for item in normalized[:-1] if item["role"] == "system")
    history = [item for item in normalized[:-1] if item["role"] != "system"]
    return system, normalized[-1]["content"], history


@dataclass
class _ProviderState:
    bridge: ProviderBridge
    lock: Lock = field(default_factory=Lock)
    errors: list = field(default_factory=list)
    stores: list = field(default_factory=list)
    budget_error: str | None = None

    def raise_for_failure(self):
        if self.budget_error is not None:
            raise CapabilityError(self.budget_error, _BUDGET_ERRORS[self.budget_error], 422)
        if self.errors:
            raise CapabilityError("invalid_upstream_output", "GraphRAG provider previously failed", 502)

    def call(self, operation, *args, **kwargs):
        with self.lock:
            self.raise_for_failure()
            try:
                return operation(*args, **kwargs)
            except Exception as exc:
                # Upstream extraction/report helpers catch model exceptions. Keep
                # a request-local latch so partial indexes cannot become success.
                self.errors.append(type(exc).__name__)
                if isinstance(exc, CapabilityError) and exc.code in _BUDGET_ERRORS:
                    self.budget_error = exc.code
                raise


def _register_providers():
    """Register stateless factories; ContextVar binds instances to this request."""
    import tiktoken
    from graphrag_llm.completion import LLMCompletion, register_completion
    from graphrag_llm.embedding import LLMEmbedding, register_embedding
    from graphrag_llm.tokenizer import Tokenizer
    from graphrag_llm.types import LLMCompletionResponse, LLMEmbeddingResponse
    from graphrag_vectors.lancedb import LanceDBVectorStore
    from graphrag_vectors.vector_store_factory import register_vector_store

    class Counter(Tokenizer):
        def __init__(self):
            self.encoding = tiktoken.get_encoding("cl100k_base")

        def encode(self, text):
            return self.encoding.encode(text, disallowed_special=())

        def decode(self, tokens):
            return self.encoding.decode(tokens)

    class Model:
        def __init__(self, *, model_config, metrics_store, **kwargs):
            self.state = _ACTIVE_STATE.get()
            self.purpose = model_config.model_extra["rags_purpose"]
            self._metrics_store = metrics_store
            self._tokenizer = Counter()

        @property
        def metrics_store(self):
            return self._metrics_store

        @property
        def tokenizer(self):
            return self._tokenizer

    class Completion(Model, LLMCompletion):
        def completion(self, **kwargs):
            def run():
                if kwargs.get("stream") or kwargs.get("tools") or kwargs.get("n", 1) != 1:
                    raise CapabilityError("invalid_model_input", "Unsupported GraphRAG completion mode", 422)
                system, prompt, history = _messages(kwargs.get("messages"))
                content = self.state.bridge.complete(prompt, system=system, history=history)
                if self.purpose == "extract":
                    content = validate_extraction(content)
                response_format = kwargs.get("response_format")
                formatted = None
                if response_format is not None:
                    try:
                        value = strict_json(content)
                        if self.purpose == "report":
                            validate_report(value)
                        formatted = response_format.model_validate(value, strict=True)
                    except Exception as exc:
                        raise CapabilityError("invalid_model_output", "Invalid GraphRAG structured output", 502) from exc
                return LLMCompletionResponse(
                    id="rags-graphrag", object="chat.completion", created=0,
                    model=self.state.bridge.generation_provider,
                    choices=[{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                    usage=None, formatted_response=formatted,
                )
            return self.state.call(run)

        async def completion_async(self, **kwargs):
            return await asyncio.to_thread(self.completion, **kwargs)

    class Embedding(Model, LLMEmbedding):
        def embedding(self, **kwargs):
            def run():
                texts = kwargs.get("input")
                if not isinstance(texts, list) or not 1 <= len(texts) <= 64:
                    raise CapabilityError("invalid_model_input", "Invalid GraphRAG embedding batch", 422)
                vectors = [self.state.bridge.embed(text, query=self.purpose == "query") for text in texts]
                # The public RAGs usage report remains authoritative. This local
                # response carries no measured tokens; zero is not exported.
                return LLMEmbeddingResponse(
                    object="list", model=self.state.bridge.embedding_model,
                    data=[{"index": index, "object": "embedding", "embedding": vector}
                          for index, vector in enumerate(vectors)],
                    usage={"prompt_tokens": 0, "total_tokens": 0},
                )
            return self.state.call(run)

        async def embedding_async(self, **kwargs):
            return await asyncio.to_thread(self.embedding, **kwargs)

    class TrackedLanceDB(LanceDBVectorStore):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            _ACTIVE_STATE.get().stores.append(self)

    register_completion("rags-provider-v10", Completion)
    register_embedding("rags-provider-v10", Embedding)
    register_vector_store("rags-lancedb-v10", TrackedLanceDB)


def _config(directory: Path, bridge, dimension):
    from graphrag.config.models.graph_rag_config import GraphRagConfig

    def model(purpose, identity):
        return {"type": "rags-provider-v10", "model_provider": "rags", "model": identity,
                "metrics": None, "rags_purpose": purpose}

    return GraphRagConfig.model_validate({
        "completion_models": {key: model(key, bridge.generation_provider) for key in ("extract", "summary", "report")},
        "embedding_models": {key: model(key, bridge.embedding_model) for key in ("document", "query")},
        "concurrent_requests": 1, "async_mode": "asyncio",
        "input_storage": {"type": "file", "base_dir": str(directory / "input")},
        "output_storage": {"type": "file", "base_dir": str(directory / "output")},
        "update_output_storage": {"type": "file", "base_dir": str(directory / "updates")},
        "reporting": {"type": "file", "base_dir": str(directory / "logs")},
        "cache": {"type": "none", "storage": None},
        "chunking": {"type": "tokens", "size": 1_000_000, "overlap": 0, "encoding_model": "cl100k_base"},
        "vector_store": {"type": "rags-lancedb-v10", "db_uri": str(directory / "vectors"), "vector_size": dimension},
        "extract_graph": {"completion_model_id": "extract", "max_gleanings": 0},
        "summarize_descriptions": {"completion_model_id": "summary", "max_length": 128,
                                   "max_input_tokens": min(4000, bridge.input_budget // 2)},
        "cluster_graph": {"max_cluster_size": 8, "use_lcc": False, "seed": 42},
        "extract_claims": {"enabled": False},
        "community_reports": {"completion_model_id": "report", "max_length": 256,
                              "max_input_length": min(4000, bridge.input_budget // 2)},
        "embed_text": {"embedding_model_id": "document", "batch_size": 8, "batch_max_tokens": 1_000_000,
                       "names": ["entity_description", "text_unit_text"]},
        "local_search": {"completion_model_id": "summary", "embedding_model_id": "query",
                         "text_unit_prop": 0.5, "community_prop": 0.2,
                         "top_k_entities": 10, "top_k_relationships": 10,
                         "max_context_tokens": min(8000, bridge.input_budget)},
    })


def _cleanup(state, directory):
    # The pinned LanceDB synchronous facades expose no close method. Their
    # underlying AsyncTable/AsyncConnection do; close those before unlinking.
    errors = []
    for store in state.stores:
        for field_name, inner_name in (("document_collection", "_table"), ("db_connection", "_conn")):
            handle = getattr(store, field_name, None)
            close = getattr(handle, "close", None) or getattr(getattr(handle, inner_name, None), "close", None)
            if callable(close):
                try:
                    close()
                except (OSError, RuntimeError, ValueError) as exc:
                    errors.append(type(exc).__name__)
            setattr(store, field_name, None)
    state.stores.clear()
    for name in ("graphrag", "graphrag_llm"):
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).is_relative_to(directory):
                handler.close()
                logger.removeHandler(handler)
    try:
        shutil.rmtree(directory)
    finally:
        if errors:
            raise CapabilityError("upstream_cleanup_failed", "GraphRAG vector handles did not close cleanly", 502)


async def search(question: str, passages: list[dict], top_k: int, workdir: Path, provider) -> dict:
    sources = validate_request(question, passages, top_k)
    bridge = ProviderBridge(provider, engine="full_graphrag")
    bridge.features.require("tokenizer.tiktoken")
    # GraphRAG imports LiteLLM types; its public cost-map download is unrelated
    # to this bounded provider bridge and must not create an extra network call.
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    verify_source("graphrag", SOURCE_FILES)
    try:
        import pandas as pd
        from graphrag.api import build_index
        from graphrag.config.embeddings import entity_description_embedding
        from graphrag.query.factory import get_local_search_engine
        from graphrag.query.indexer_adapters import (
            read_indexer_entities,
            read_indexer_relationships,
            read_indexer_reports,
            read_indexer_text_units,
        )
        from graphrag.utils.api import get_embedding_store
        from graphrag_storage import create_storage
        from graphrag_storage.tables.table_provider_factory import create_table_provider
    except ImportError as exc:
        raise CapabilityError("requires_backend", "Pinned GraphRAG dependencies are unavailable") from exc
    if any(version(name) != "3.3.0" for name in (
        "graphrag", "graphrag-llm", "graphrag-common", "graphrag-storage", "graphrag-vectors",
        "graphrag-cache", "graphrag-chunking", "graphrag-input",
    )):
        raise CapabilityError("requires_backend", "GraphRAG distributions must match the pinned version")
    directory = private_directory(workdir, "graphrag")
    state = _ProviderState(bridge)
    token = _ACTIVE_STATE.set(state)
    try:
        documents, groups = group_sources(sources)
        dimension = len(await asyncio.to_thread(bridge.embed, documents[0]["text"]))
        _register_providers()
        config = _config(directory, bridge, dimension)
        try:
            results = await build_index(config, method="standard", input_documents=pd.DataFrame(documents))
        except Exception:
            state.raise_for_failure()
            raise
        state.raise_for_failure()
        validate_pipeline(results, state.errors)
        tables = create_table_provider(config.table_provider, create_storage(config.output_storage))
        data = {name: await tables.read_dataframe(name) for name in (
            "documents", "text_units", "entities", "relationships", "communities", "community_reports",
        )}
        if (not 1 <= len(data["entities"]) <= 2048 or not 1 <= len(data["relationships"]) <= 2048
                or not 1 <= len(data["communities"]) <= 4096 or len(data["community_reports"]) != len(data["communities"])):
            raise CapabilityError("invalid_upstream_output", "GraphRAG graph or reports are incomplete or oversized", 502)
        # Validate the complete source table even if query returns no text units.
        units, docs = data["text_units"].to_dict("records"), data["documents"].to_dict("records")
        project_sources([], units, docs, groups, sources, top_k)
        store = get_embedding_store(config.vector_store, entity_description_embedding)
        if store.count() != len(data["entities"]):
            raise CapabilityError("invalid_upstream_output", "GraphRAG entity embeddings are incomplete", 502)
        engine = get_local_search_engine(
            config=config, reports=read_indexer_reports(data["community_reports"], data["communities"], 99),
            text_units=read_indexer_text_units(data["text_units"]),
            entities=read_indexer_entities(data["entities"], data["communities"], 99),
            relationships=read_indexer_relationships(data["relationships"]), covariates={},
            description_embedding_store=store, response_type="source evidence",
        )
        context = await asyncio.to_thread(engine.context_builder.build_context, question, **engine.context_builder_params)
        records = context.context_records
        selected = records["sources"].to_dict("records") if "sources" in records else []
        hits = project_sources(selected, units, docs, groups, sources, top_k)
        state.raise_for_failure()
        return {
            "hits": hits,
            "implementation": {"library": "graphrag", "version": "3.3.0",
                               "source_commit": SOURCE_COMMIT, "adapter_version": "v10"},
            "diagnostics": {**bridge.diagnostics(), "score_kind": "reciprocal_local_source_rank",
                            "query_mode": "local_graph_context", "source_count": len(sources),
                            "pipeline_stages": len(results), "text_units": len(units),
                            "graph_nodes": len(data["entities"]), "graph_edges": len(data["relationships"]),
                            "communities": len(data["communities"]), "community_reports": len(data["community_reports"]),
                            "retrieved_chunks": len(hits), "retrieved_entities": len(records.get("entities", [])),
                            "retrieved_relationships": len(records.get("relationships", []))},
        }
    finally:
        _ACTIVE_STATE.reset(token)
        _cleanup(state, directory)
