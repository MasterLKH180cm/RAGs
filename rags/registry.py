"""Capability names describe mechanisms, never silent fallback aliases."""
from dataclasses import dataclass


class CapabilityError(ValueError):
    def __init__(self, code: str, reason: str, status_code: int = 503):
        super().__init__(reason)
        self.code = code
        self.reason = reason
        self.status_code = status_code


@dataclass(frozen=True)
class Strategy:
    id: str
    family: str
    implementation: str
    requirements: tuple[str, ...] = ()


NATIVE = (
    Strategy("naive", "Naive RAG", "single dense retrieval and evidence assembly"),
    Strategy("sparse", "Sparse RAG", "SQLite FTS5 BM25"),
    Strategy("dense", "Dense RAG", "cosine similarity with declared encoder"),
    Strategy("hybrid", "Hybrid RAG", "BM25 and cosine reciprocal rank fusion"),
    Strategy("fusion", "RAG-Fusion", "validated generated queries and rank fusion", ("generation",)),
    Strategy("hyde", "HyDE", "hypothetical-document embedding centroid", ("generation",)),
    Strategy("contextual", "Contextual RAG", "generated context in sparse and vector indexes", ("contextual_index",)),
    Strategy("parent_child", "Hierarchical RAG", "leaf search with persisted parent expansion", ("parent_index",)),
    Strategy("hierarchical", "Hierarchical / RAPTOR", "recursive vector clustering and source-linked summaries", ("hierarchy_index",)),
    Strategy("graph", "GraphRAG", "entity graph traversal and label-propagation communities", ("graph_index",)),
    Strategy("corrective", "CRAG", "bounded evidence evaluation and corrective retrieval", ("generation",)),
    Strategy("adaptive", "Adaptive RAG", "validated model routing among available strategies", ("generation",)),
    Strategy("agentic", "Agentic RAG", "bounded search-plan and evidence-gap loop", ("generation",)),
    Strategy("self_reflective", "Self-RAG", "bounded answer critique and evidence refinement", ("generation",)),
    Strategy("structured", "Structured RAG", "typed metadata and row predicates with safe aggregation"),
    Strategy("federated", "Federated RAG", "tenant-authenticated source fan-out and rank fusion", ("federation",)),
    Strategy("temporal", "Temporal / real-time RAG", "version validity, as-of selection and recency ranking"),
    Strategy("caption", "Multimodal RAG", "vision-generated text-caption retrieval", ("caption_index",)),
)
UPSTREAM_IDS = ("multivector", "colbert", "raptor", "full_graphrag", "lightrag", "hipporag", "self_rag", "crag", "multimodal_native")
MODES = tuple(item.id for item in NATIVE) + UPSTREAM_IDS
MODEL_MODES = frozenset(item.id for item in NATIVE if "generation" in item.requirements)


def strategy(mode: str) -> Strategy:
    for item in NATIVE:
        if item.id == mode:
            return item
    raise CapabilityError("unknown_strategy", "Unknown native retrieval strategy", 422)
