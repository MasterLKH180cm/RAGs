"""Bounded public contract for the pinned, original-source research workers."""
import math
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, field_validator, model_validator


SOURCE_ID = r"^[A-Za-z0-9_.:-]{1,200}$"
MAX_BODY_BYTES = 1_500_000
MAX_RESPONSE_BYTES = 100_000
UPSTREAMS = {
    "raptor": {"library": "raptor", "version": "git-7da1d48",
               "source_commit": "7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767", "adapter_version": "v03"},
    "lightrag": {"library": "lightrag-hku", "version": "1.5.7",
                 "source_commit": "28ff1b05f2ac3f3e6fa14dd2cd33656579bd0c9c", "adapter_version": "v03"},
}
COUNT_FIELDS = {"source_count", "selected_nodes", "summary_layers", "summary_nodes", "tree_nodes",
                "graph_nodes", "graph_edges", "retrieved_chunks", "generation_calls", "embedding_calls",
                "embedding_dimensions", "selected_summary_nodes", "retrieved_entities", "retrieved_relationships",
                "query_embedding_calls", "embedding_token_limit"}
LABEL_FIELDS = {"score_kind", "embedding_model", "generation_provider", "embedding_window_policy"}


class Passage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = Field(pattern=SOURCE_ID)
    text: str = Field(min_length=1, max_length=200_000)

    @field_validator("text")
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError("Passages must contain source text")
        return value


class ResearchOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    max_iterations: int | None = Field(default=None, ge=1, le=3)


class ResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    protocol: Literal["authorized-evidence-v1"]
    engine: Literal["raptor", "lightrag"]
    namespace: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(ge=1, le=20)
    passages: list[Passage] = Field(max_length=64)
    model: str | None = Field(default=None, max_length=200)
    options: ResearchOptions = Field(default_factory=ResearchOptions)

    @model_validator(mode="after")
    def source_bounds(self):
        if not self.question.strip():
            raise ValueError("Question is required")
        if len({p.id for p in self.passages}) != len(self.passages):
            raise ValueError("Source IDs must be unique")
        if sum(len(p.text) for p in self.passages) > 200_000:
            raise ValueError("Authorized source characters exceed limit")
        return self


class ResearchHit(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    chunk_id: str = Field(pattern=SOURCE_ID)
    score: StrictFloat | StrictInt

    @field_validator("score")
    @classmethod
    def finite_score(cls, value):
        try:
            valid = math.isfinite(value)
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("Scores must be finite")
        return value


class Implementation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    library: Literal["raptor", "lightrag-hku"]
    version: str = Field(min_length=1, max_length=100)
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    adapter_version: Literal["v03"]


def validate_implementation(value, engine: str) -> dict:
    parsed = Implementation.model_validate(value).model_dump()
    if engine not in UPSTREAMS or parsed != UPSTREAMS[engine]:
        raise ValueError("Unexpected upstream implementation identity")
    return parsed


def validate_diagnostics(value) -> dict:
    if not isinstance(value, dict) or set(value) - (COUNT_FIELDS | LABEL_FIELDS | {"engine_executed"}):
        raise ValueError("Unknown research diagnostic fields")
    for key, item in value.items():
        if key in COUNT_FIELDS and (type(item) is not int or not 0 <= item <= 1_000_000):
            raise ValueError("Invalid research count")
        if key in LABEL_FIELDS and (not isinstance(item, str) or not re.fullmatch(r"[A-Za-z0-9_.:/+@-]{1,200}", item)):
            raise ValueError("Invalid research model or score label")
        if key == "engine_executed" and type(item) is not bool:
            raise ValueError("Invalid research execution flag")
    return value


def validate_result(value, request: ResearchRequest) -> dict:
    if not isinstance(value, dict) or set(value) - {"hits", "implementation", "diagnostics", "usage"}:
        raise ValueError("Invalid research response")
    if not isinstance(value.get("hits"), list) or len(value["hits"]) > request.top_k:
        raise ValueError("Research hit count exceeds requested limit")
    hits = [ResearchHit.model_validate(hit).model_dump() for hit in value["hits"]]
    identifiers = [hit["chunk_id"] for hit in hits]
    if len(set(identifiers)) != len(identifiers) or not set(identifiers) <= {p.id for p in request.passages}:
        raise ValueError("Research results must identify supplied original sources")
    implementation = validate_implementation(value.get("implementation"), request.engine)
    diagnostics = validate_diagnostics(value.get("diagnostics", {}))
    if diagnostics.get("source_count") != len(request.passages):
        raise ValueError("Research source count does not match the request")
    if diagnostics.get("engine_executed") is not bool(request.passages):
        raise ValueError("Research execution state does not match the request")
    usage = value.get("usage")
    if usage is not None and (not isinstance(usage, dict) or set(usage) - {"input_tokens", "output_tokens"}
                              or any(type(n) is not int or not 0 <= n <= 10_000_000 for n in usage.values())):
        raise ValueError("Invalid research usage")
    return {"hits": hits, "implementation": implementation, "diagnostics": diagnostics, "usage": usage}
