"""Safe operator-registered workflow templates for API and dashboard.

Workflow definitions are owned by trusted Python code; HTTP callers can only
choose an approved name and bounded input. No generated code is evaluated.
"""
from dataclasses import dataclass

from rags.agents.workflow import Workflow
from rags.auth import Principal
from rags.features import FeatureFlags
from rags.registry import CapabilityError


@dataclass(frozen=True)
class Template:
    name: str
    title: str
    description: str
    version: str
    nodes: tuple[tuple[str, str, bool], ...]

    def public(self) -> dict:
        return {"name": self.name, "title": self.title, "description": self.description,
                "version": self.version,
                "nodes": [{"name": name, "kind": kind, "approval": approval}
                          for name, kind, approval in self.nodes]}


EVIDENCE_REVIEW = Template(
    name="evidence_review", title="Evidence review / 證據審核",
    description="檢索來源、判斷證據、平行檢查，最後交由人工核准。",
    version="v1",
    nodes=(("retrieve", "then", False), ("quality_gate", "branch", False),
           ("cross_checks", "parallel", False), ("human_approval", "then", True),
           ("finalize", "then", False)),
)


def make_evidence_review(rag, principal: Principal, flags: FeatureFlags) -> Workflow:
    """Construct the same versioned registry graph on every start/resume."""

    def retrieve(ctx):
        principal.require("rags:query")
        flags.require("agents", "query")
        result = rag.query(ctx.tenant, ctx.input["question"],
                           mode=ctx.input.get("mode", "sparse"),
                           top_k=ctx.input.get("top_k", 5), generate=False)
        rows = result.get("citations") or []
        if not isinstance(rows, list):
            raise CapabilityError("invalid_evidence", "Retrieval evidence is invalid", 502)
        # Never store raw source excerpts in workflow checkpoint/UI graph.
        citations = [{"chunk_id": row["chunk_id"],
                      "document_id": row["document_id"], "source": row["source"]}
                     for row in rows[:10] if isinstance(row, dict)
                     and all(isinstance(row.get(k), str) for k in ("chunk_id", "document_id", "source"))]
        return {"outcome": str(result.get("outcome", "retrieved"))[:80],
                "citations": citations}

    def gate(ctx):
        return bool(ctx.outputs["retrieve"]["citations"])

    def yes(ctx):
        return {"status": "evidence_available"}

    def no(ctx):
        return {"status": "no_evidence"}

    def provenance(ctx):
        citations = ctx.outputs["retrieve"]["citations"]
        return {"document_count": len({item["document_id"] for item in citations})}

    def uniqueness(ctx):
        ids = [item["chunk_id"] for item in ctx.outputs["retrieve"]["citations"]]
        return {"unique_chunk_ids": len(set(ids)) == len(ids)}

    def approve(ctx):
        return {"human_reviewed": True, "run_id": ctx.run_id}

    def finalize(ctx):
        return {"status": "approved",
                "source_count": len(ctx.outputs["retrieve"]["citations"]),
                "evidence_gate": ctx.outputs["quality_gate"]["selected"]}

    return (Workflow("evidence_review", version=EVIDENCE_REVIEW.version)
            .then("retrieve", retrieve, idempotent=True)
            .branch("quality_gate", gate, if_true=yes, if_false=no)
            .parallel("cross_checks",
                      {"provenance": provenance, "dedup": uniqueness}, idempotent=True)
            .then("human_approval", approve, approval=True, idempotent=True)
            .then("finalize", finalize, idempotent=True))


def catalog() -> list[dict]:
    return [EVIDENCE_REVIEW.public()]


def get_template(name: str) -> Template:
    if name != EVIDENCE_REVIEW.name:
        raise CapabilityError("not_found", "Workflow template not found", 404)
    return EVIDENCE_REVIEW
