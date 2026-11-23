"""Bounded public contract for the pinned, original-source research workers."""
import math
import re
import hashlib
from datetime import datetime
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, field_validator, model_validator


SOURCE_ID = r"^[A-Za-z0-9_.:-]{1,200}$"
MAX_BODY_BYTES = 1_500_000
MAX_RESPONSE_BYTES = 100_000
UPSTREAMS = {
    "raptor": {"library": "raptor", "version": "git-7da1d48",
               "source_commit": "7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767", "adapter_version": "v03"},
    "lightrag": {"library": "lightrag-hku", "version": "1.5.7",
                 "source_commit": "28ff1b05f2ac3f3e6fa14dd2cd33656579bd0c9c", "adapter_version": "v03"},
    "full_graphrag": {"library": "graphrag", "version": "3.3.0",
                      "source_commit": "5faaaf4f5685fa2056fa8c3bf9342cb089f2942f", "adapter_version": "v10"},
    "hipporag": {"library": "hipporag", "version": "2.0.0a5",
                 "source_commit": "d5c8329422e0a0b834a15874545cb6a74b4f9b26", "adapter_version": "v10"},
    "self_rag": {"library": "self-rag", "version": "git-1fcdc42",
                 "source_commit": "1fcdc420e48f50a7d7ab1ece5494221b93252e99", "adapter_version": "v10"},
    "crag": {"library": "crag", "version": "git-de7c296",
             "source_commit": "de7c2961ae624a1483a138c5798e1f6d0c4fb0e0", "adapter_version": "v10"},
}
COUNT_FIELDS = {"source_count", "selected_nodes", "summary_layers", "summary_nodes", "tree_nodes",
                "graph_nodes", "graph_edges", "retrieved_chunks", "generation_calls", "embedding_calls",
                "embedding_dimensions", "selected_summary_nodes", "retrieved_entities", "retrieved_relationships",
                "query_embedding_calls", "embedding_token_limit", "communities", "community_reports",
                "text_units", "pipeline_stages", "ppr_calls", "extracted_triples", "linked_facts",
                "unique_passages", "reflection_calls", "evaluator_calls", "refined_strips", "external_sources",
                "search_calls", "input_tokens", "output_tokens", "generated_tokens",
                "reflection_relevance_tokens", "reflection_support_tokens", "reflection_utility_tokens",
                "reflection_abstentions"}
LABEL_FIELDS = {"score_kind", "embedding_model", "generation_provider", "embedding_window_policy",
                "query_mode", "retrieval_path", "execution_path", "checkpoint_sha256", "runtime",
                "correction_branch", "threshold_profile", "evaluator_sha256", "runtime_revision",
                "inference_mode", "checkpoint_quantization", "checkpoint_revision", "scoring"}
ANSWER_ENGINES = frozenset({"self_rag", "crag"})


def validate_external_hosts(value: list[str]) -> list[str]:
    if (not isinstance(value, list) or len(value) > 8
            or any(not isinstance(host, str) or len(host) > 253
                   or not re.fullmatch(r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}", host)
                   or host.endswith((".localhost", ".local", ".internal")) for host in value)
            or len(set(value)) != len(value)):
        raise ValueError("External hosts must be distinct exact public DNS hostnames")
    return value


def external_source_id(namespace: str, url: str, content_sha256: str) -> str:
    return "web:" + hashlib.sha256((namespace + "\0" + url + "\0" + content_sha256).encode()).hexdigest()


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
    protocol: Literal["authorized-evidence-v1", "authorized-evidence-v2"]
    engine: Literal["raptor", "lightrag", "full_graphrag", "hipporag", "self_rag", "crag"]
    namespace: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(ge=1, le=20)
    passages: list[Passage] = Field(max_length=64)
    model: str | None = Field(default=None, max_length=200)
    options: ResearchOptions = Field(default_factory=ResearchOptions)
    external_hosts: list[str] = Field(default_factory=list, max_length=8)

    @field_validator("external_hosts")
    @classmethod
    def valid_external_hosts(cls, value):
        return validate_external_hosts(value)

    @model_validator(mode="after")
    def source_bounds(self):
        if (self.engine in ANSWER_ENGINES) != (self.protocol == "authorized-evidence-v2"):
            raise ValueError("The selected engine requires its versioned research protocol")
        if self.external_hosts and self.engine != "crag":
            raise ValueError("External source acquisition is exclusive to explicitly authorized CRAG")
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
    library: Literal["raptor", "lightrag-hku", "graphrag", "hipporag", "self-rag", "crag"]
    version: str = Field(min_length=1, max_length=100)
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    adapter_version: Literal["v03", "v10"]


class GroundedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str = Field(min_length=1, max_length=60_000)
    cited_ids: list[str] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def valid_answer(self):
        if (not self.text.strip() or len(self.text.encode("utf-8")) > 60_000
                or len(set(self.cited_ids)) != len(self.cited_ids)
                or any(not re.fullmatch(SOURCE_ID, identifier) for identifier in self.cited_ids)):
            raise ValueError("Invalid grounded answer or source references")
        return self


class ExternalSource(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = Field(pattern=r"^web:[0-9a-f]{64}$")
    url: str = Field(min_length=1, max_length=2048)
    title: str = Field(max_length=200)
    text: str = Field(min_length=1, max_length=16_000)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fetched_at: str = Field(max_length=40)


def validate_external_sources(value, request: ResearchRequest) -> list[dict]:
    if not isinstance(value, list) or len(value) > 5:
        raise ValueError("Invalid external source count")
    if value and (request.engine != "crag" or not request.external_hosts):
        raise ValueError("External source acquisition was not authorized")
    results, seen, size = [], set(), 0
    internal_ids = {passage.id for passage in request.passages}
    for item in value:
        source = ExternalSource.model_validate(item)
        if ("\\" in source.url
                or any(character.isspace() or ord(character) < 32 or ord(character) == 127
                       for character in source.url)):
            raise ValueError("External evidence URL must not contain whitespace or controls")
        url = urlsplit(source.url)
        if (url.scheme != "https" or url.hostname not in request.external_hosts
                or url.username is not None or url.password is not None or url.fragment
                or url.port not in (None, 443)):
            raise ValueError("External evidence URL is outside the approved hosts")
        content = source.text.encode("utf-8")
        if not source.text.strip() or hashlib.sha256(content).hexdigest() != source.content_sha256:
            raise ValueError("External source content does not match its snapshot hash")
        if (source.id != external_source_id(request.namespace, source.url, source.content_sha256)
                or source.id in seen or source.id in internal_ids):
            raise ValueError("External source identity does not match tenant and snapshot")
        when = datetime.fromisoformat(source.fetched_at)
        if when.tzinfo is None:
            raise ValueError("External source time must include its timezone")
        size += len(content)
        if size > 48_000:
            raise ValueError("External evidence exceeds its total byte budget")
        seen.add(source.id)
        results.append(source.model_dump())
    return results


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
    allowed = {"hits", "implementation", "diagnostics", "usage"}
    if request.protocol == "authorized-evidence-v2":
        allowed.update(("grounded_answer", "external_sources"))
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError("Invalid research response")
    if not isinstance(value.get("hits"), list) or len(value["hits"]) > request.top_k:
        raise ValueError("Research hit count exceeds requested limit")
    hits = [ResearchHit.model_validate(hit).model_dump() for hit in value["hits"]]
    identifiers = [hit["chunk_id"] for hit in hits]
    external = validate_external_sources(value.get("external_sources", []), request)
    authorized = {p.id for p in request.passages} | {p["id"] for p in external}
    if len(set(identifiers)) != len(identifiers) or not set(identifiers) <= authorized:
        raise ValueError("Research results must identify supplied original sources")
    if not {p["id"] for p in external} <= set(identifiers):
        raise ValueError("Unselected external source content cannot cross the research boundary")
    implementation = validate_implementation(value.get("implementation"), request.engine)
    diagnostics = validate_diagnostics(value.get("diagnostics", {}))
    if diagnostics.get("source_count") != len(request.passages):
        raise ValueError("Research source count does not match the request")
    if diagnostics.get("engine_executed") is not bool(request.passages or request.external_hosts):
        raise ValueError("Research execution state does not match the request")
    usage = value.get("usage")
    if usage is not None and (not isinstance(usage, dict) or set(usage) - {"input_tokens", "output_tokens"}
                              or any(type(n) is not int or not 0 <= n <= 10_000_000 for n in usage.values())):
        raise ValueError("Invalid research usage")
    result = {"hits": hits, "implementation": implementation, "diagnostics": diagnostics, "usage": usage}
    if request.protocol == "authorized-evidence-v2":
        answer = value.get("grounded_answer")
        if answer is not None:
            answer = GroundedAnswer.model_validate(answer).model_dump()
            if not set(answer["cited_ids"]) <= set(identifiers):
                raise ValueError("Grounded answer references unselected sources")
        if hits and answer is None:
            raise ValueError("A trained/corrective answer cannot be replaced by ordinary generation")
        result.update(grounded_answer=answer, external_sources=external)
    return result
