"""Actual pinned LightRAG graph extraction and hybrid source retrieval."""
from __future__ import annotations

import json
import secrets
from importlib.metadata import version
from pathlib import Path

from rags.registry import CapabilityError
from rags.research.common import ProviderBridge, private_directory, validate_request, verify_source

SOURCE_COMMIT = "28ff1b05f2ac3f3e6fa14dd2cd33656579bd0c9c"
SOURCE_FILES = {
    "__init__.py": "fa653125d5ad2d215ffabea25d24be0d8ffb732959c73c91ac6ba635a47355b7",
    "lightrag.py": "ff118d8d197b7f5ab899a2190157fa88782dc6be897c59ab47a42cab52ac45b9",
    "base.py": "53a1628afb3e2b440aa03eb5a0590c365a04ae4974b432ae79914e8927f92a96",
    "operate.py": "fecb70d27b860efacb57d7b1bc16a706d7d68bd18de9ac80571fcd4628305047",
    "utils.py": "70fe8389a462a3ada44df7d2dd7c49ed85a5c959ca32422f78ac8fdb2dafd5fc",
    "prompt.py": "db0cf2d0f5d420a8372fac2b33043d8b4f14c84ec0a60653862e4e53a27047e1",
    "kg/networkx_impl.py": "087e226f292d6837edf5de95e077a8fa65c35abe64b3956039a78d4120538057",
    "kg/nano_vector_db_impl.py": "a6894c6048cac279ac15bf09ae9f38f70c5c1bcc6104ac0761cb02141343df73",
    "kg/json_kv_impl.py": "04e9941d766943ad073a8ff64a092db78464a3ba74effddf424703d3b8682053",
    "kg/json_doc_status_impl.py": "fd1496c765e32d700ff16d7a99dc074853e3acbdbeea1ad8853a3bbaecbe2ec0",
    "kg/shared_storage.py": "f8e80fa821b6c27701c3ac6afff859dcbbb83ba5c8e2448fee57d6d3e66f40bb",
}


def validate_keywords(result: str) -> None:
    """Invalid generation is an execution error, never an empty evidence result."""
    try:
        value = json.loads(result)
    except (TypeError, ValueError) as exc:
        raise CapabilityError("invalid_model_output", "LightRAG keyword output is invalid", 502) from exc
    keys = {"high_level_keywords", "low_level_keywords"}
    if (not isinstance(value, dict) or set(value) != keys
            or any(not isinstance(value[key], list) or len(value[key]) > 32
                   or any(not isinstance(item, str) or not item.strip() or len(item) > 200
                          for item in value[key]) for key in keys)):
        raise CapabilityError("invalid_model_output", "LightRAG keyword output is invalid", 502)


def validate_extraction(result: str) -> None:
    """Bound the format required by the pinned upstream extraction prompt."""
    rows = [row.strip() for row in result.strip().splitlines() if row.strip()]
    if not rows or rows[-1] != "<|COMPLETE|>" or len(rows) > 33:
        raise CapabilityError("invalid_model_output", "LightRAG extraction output is invalid", 502)
    entities, relations = set(), []
    for row in rows[:-1]:
        fields = row.split("<|#|>")
        if fields[0] == "entity" and len(fields) == 4 and all(part.strip() for part in fields[1:]):
            entities.add(fields[1].strip())
        elif fields[0] == "relation" and len(fields) == 5 and all(part.strip() for part in fields[1:]):
            relations.append((fields[1].strip(), fields[2].strip()))
        else:
            raise CapabilityError("invalid_model_output", "LightRAG extraction output is invalid", 502)
    if len(entities) > 16 or any(source not in entities or target not in entities for source, target in relations):
        raise CapabilityError("invalid_model_output", "LightRAG extraction references unknown entities", 502)


async def project_chunks(chunks: list[dict], store, sources: dict[str, str], top_k: int) -> list[dict]:
    if not isinstance(chunks, list) or len(chunks) > 64:
        raise CapabilityError("invalid_source_lineage", "Invalid LightRAG retrieval result", 502)
    identifiers = []
    for chunk in chunks:
        if not isinstance(chunk, dict) or not isinstance(chunk.get("chunk_id"), str):
            raise CapabilityError("invalid_source_lineage", "Invalid LightRAG retrieval result", 502)
        identifiers.append(chunk["chunk_id"])
    if len(set(identifiers)) != len(identifiers):
        raise CapabilityError("invalid_source_lineage", "Duplicate LightRAG source chunks", 502)
    records = await store.get_by_ids(identifiers)
    if not isinstance(records, list) or len(records) != len(chunks):
        raise CapabilityError("invalid_source_lineage", "LightRAG source records are unavailable", 502)
    hits = {}
    for rank, (chunk, record) in enumerate(zip(chunks, records, strict=True)):
        if not isinstance(record, dict) or record.get("full_doc_id") not in sources:
            raise CapabilityError("invalid_source_lineage", "Unknown LightRAG original source", 502)
        identifier = record["full_doc_id"]
        if record.get("content") != sources[identifier] or chunk.get("content") != sources[identifier]:
            raise CapabilityError("invalid_source_lineage", "LightRAG source content changed", 502)
        hits[identifier] = max(hits.get(identifier, 0), 1 / (rank + 1))
    return [{"chunk_id": identifier, "score": score} for identifier, score in
            sorted(hits.items(), key=lambda item: (-item[1], item[0]))[:top_k]]


async def search(question: str, passages: list[dict], top_k: int, workdir: Path, provider) -> dict:
    sources = validate_request(question, passages, top_k)
    bridge = ProviderBridge(provider)
    directory = private_directory(workdir, "lightrag")
    verify_source("lightrag", SOURCE_FILES)
    try:
        import numpy as np
        from lightrag import LightRAG, QueryParam
        from lightrag.utils import EmbeddingFunc, sanitize_text_for_encoding
    except ImportError as exc:
        raise CapabilityError("requires_backend", "Pinned LightRAG dependencies are unavailable") from exc
    if version("lightrag-hku") != "1.5.7":
        raise CapabilityError("requires_backend", "LightRAG distribution must match the pinned version")
    if any(sanitize_text_for_encoding(text) != text for text in sources.values()):
        raise CapabilityError("invalid_research_input", "LightRAG would change the original source encoding", 422)
    dimension = len(bridge.embed(next(iter(sources.values()))))

    async def embed(texts, context="document"):
        if context not in {"query", "document"} or not isinstance(texts, list) or len(texts) > 128:
            raise CapabilityError("invalid_model_input", "Invalid LightRAG embedding batch", 422)
        return np.asarray([bridge.embed(text, query=context == "query") for text in texts], dtype=np.float32)

    async def complete(prompt, system_prompt=None, history_messages=None, **kwargs):
        result = bridge.complete(prompt, system=system_prompt or "", history=history_messages)
        if kwargs.get("response_format") == {"type": "json_object"}:
            validate_keywords(result)
        elif prompt.startswith("---Task---\nExtract entities and relationships"):
            validate_extraction(result)
        return result

    rag = LightRAG(
        working_dir=str(directory), workspace="request-" + secrets.token_hex(12),
        kv_storage="JsonKVStorage", vector_storage="NanoVectorDBStorage",
        graph_storage="NetworkXStorage", doc_status_storage="JsonDocStatusStorage",
        embedding_func=EmbeddingFunc(embedding_dim=dimension, func=embed,
                                     model_name=bridge.embedding_model, supports_asymmetric=True),
        llm_model_func=complete, llm_model_name=bridge.generation_provider,
        embedding_batch_num=8, embedding_func_max_async=1, llm_model_max_async=1,
        entity_extract_max_gleaning=0, entity_extract_max_records=32, entity_extract_max_entities=16,
        summary_max_tokens=256, summary_length_recommended=128,
        summary_context_size=min(bridge.input_budget, 8000),
        max_parallel_insert=1, max_graph_nodes=1024,
        max_source_ids_per_entity=64, max_source_ids_per_relation=64,
        related_chunk_number=min(len(sources), 64), kg_chunk_pick_method="WEIGHT",
        enable_llm_cache=False, enable_llm_cache_for_entity_extract=False,
        cosine_better_than_threshold=0.2, max_total_tokens=12000,
    )
    try:
        await rag.initialize_storages()
        for identifier, content in sources.items():
            await rag.ainsert_custom_chunks(content, [content], doc_id=identifier)
        nodes = await rag.chunk_entity_relation_graph.get_all_nodes()
        edges = await rag.chunk_entity_relation_graph.get_all_edges()
        result = await rag.aquery_data(question, QueryParam(
            mode="hybrid", top_k=min(max(top_k * 2, 10), 64), chunk_top_k=top_k,
            max_entity_tokens=3000, max_relation_tokens=3000, max_total_tokens=12000,
            enable_rerank=False,
        ))
        if not isinstance(result, dict):
            raise CapabilityError("invalid_upstream_output", "Invalid LightRAG query result", 502)
        if result.get("status") == "failure" and result.get("metadata", {}).get("failure_reason") == "no_results":
            data = {"chunks": [], "entities": [], "relationships": []}
        elif result.get("status") == "success" and isinstance(result.get("data"), dict):
            data = result["data"]
        else:
            raise CapabilityError("invalid_upstream_output", "LightRAG query failed", 502)
        hits = await project_chunks(data.get("chunks"), rag.text_chunks, sources, top_k)
        entities, relationships = data.get("entities"), data.get("relationships")
        if not isinstance(entities, list) or not isinstance(relationships, list):
            raise CapabilityError("invalid_upstream_output", "Invalid LightRAG graph result", 502)
        return {
            "hits": hits,
            "implementation": {"library": "lightrag-hku", "version": "1.5.7",
                               "source_commit": SOURCE_COMMIT, "adapter_version": "v03"},
            "diagnostics": {**bridge.diagnostics(), "score_kind": "reciprocal_hybrid_chunk_rank",
                            "source_count": len(sources), "graph_nodes": len(nodes), "graph_edges": len(edges),
                            "retrieved_chunks": len(hits), "retrieved_entities": len(entities),
                            "retrieved_relationships": len(relationships)},
        }
    finally:
        await rag.finalize_storages()
