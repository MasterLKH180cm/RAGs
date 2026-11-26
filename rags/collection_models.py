"""Side-effect-free contracts for collection records, search and local/HTTP SDKs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import ipaddress
import json
import math
import os
import re
import struct
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_DIMENSIONS = 4096
MAX_DOCUMENT_CHARS = 20_000
RECORD_FIELDS = frozenset({"document", "metadata", "embedding", "sparse_embedding"})
SEARCH_FIELDS = RECORD_FIELDS | {"score", "distance", "id", "version"}


def valid_text(value: object, maximum: int, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or len(value) > maximum or (not allow_empty and not value.strip()):
        raise ValueError("Expected a bounded text value")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError("Text must be valid UTF-8") from exc
    if "\x00" in value:
        raise ValueError("Text cannot contain NUL")
    return value


def validate_collection_name(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError("Collection name requires 1..128 letters, digits, dots, underscores or dashes")
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return value
    raise ValueError("Collection names cannot be IP addresses")


def _scalar(value: object) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "boolean"
    if type(value) is int:
        if not -(2 ** 63) <= value < 2 ** 63:
            raise ValueError("Metadata integers require signed 64-bit values")
        return "number"
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Metadata numbers must be finite")
        return "number"
    if isinstance(value, str):
        valid_text(value, 4096, allow_empty=True)
        return "string"
    raise ValueError("Metadata accepts scalars and homogeneous scalar arrays")


def canonical_metadata(value: object) -> dict:
    if not isinstance(value, dict) or len(value) > 64:
        raise ValueError("Metadata requires at most 64 fields")
    for key, item in value.items():
        valid_text(key, 128)
        if key.startswith(("$", "_rags_")):
            raise ValueError("Metadata field uses a reserved prefix")
        if isinstance(item, list):
            if len(item) > 100:
                raise ValueError("Metadata arrays require at most 100 values")
            kinds = {_scalar(part) for part in item}
            if len(kinds) > 1 or "null" in kinds:
                raise ValueError("Metadata arrays must have a single non-null scalar type")
        else:
            _scalar(item)
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if len(raw.encode("utf-8")) > 16_384:
        raise ValueError("Metadata exceeds 16 KiB")
    return json.loads(raw)


def canonical_dense(value: object) -> list[float]:
    """Finite float32, safely normalized for cosine, with idempotent unit vectors."""
    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= MAX_DIMENSIONS:
        raise ValueError("Dense vectors require 1..4096 values")
    rounded = []
    try:
        for item in value:
            if type(item) not in (int, float) or not math.isfinite(item):
                raise ValueError("Dense vector values must be finite numbers")
            number = struct.unpack("!f", struct.pack("!f", item))[0]
            if not math.isfinite(number):
                raise ValueError("Dense vector values exceed float32")
            rounded.append(number)
        maximum = max(abs(item) for item in rounded)
        if maximum == 0:
            raise ValueError("Cosine vectors cannot be zero")
        scaled = [item / maximum for item in rounded]
        length = math.sqrt(math.fsum(item * item for item in scaled))
        norm = maximum * length
        if abs(norm - 1.0) <= 1e-6:
            return rounded
        return [struct.unpack("!f", struct.pack("!f", item / length))[0] for item in scaled]
    except (OverflowError, struct.error) as exc:
        raise ValueError("Dense vector values exceed float32") from exc


def canonical_sparse(value: object) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"indices", "values", "space_id"}:
        raise ValueError("Sparse vectors require indices, values and space_id")
    indices, values = value["indices"], value["values"]
    space = valid_text(value["space_id"], 160)
    if (not isinstance(indices, list) or not isinstance(values, list)
            or len(indices) != len(values) or not 1 <= len(indices) <= 2048
            or any(type(i) is not int or not 0 <= i <= 2_147_483_647 for i in indices)
            or indices != sorted(set(indices))):
        raise ValueError("Sparse indices must be 1..2048 sorted unique nonnegative integers")
    result = []
    for item in values:
        if type(item) not in (int, float) or not 0 < item <= 1_000_000:
            raise ValueError("Sparse values must be positive finite numbers up to 1000000")
        try:
            number = struct.unpack("!f", struct.pack("!f", item))[0]
        except (OverflowError, struct.error) as exc:
            raise ValueError("Invalid sparse float32 value") from exc
        if not math.isfinite(number) or number <= 0:
            raise ValueError("Sparse values must remain positive float32 numbers")
        result.append(number)
    return {"indices": list(indices), "values": result, "space_id": space}


def record_ids(value: object) -> list[str]:
    if not isinstance(value, list) or not 1 <= len(value) <= 100:
        raise ValueError("Provide 1..100 record IDs")
    result = [valid_text(item, 128) for item in value]
    if len(set(result)) != len(result):
        raise ValueError("Record IDs must be unique within a request")
    return result


def _encoded_bound(value: object, maximum: int = MAX_REQUEST_BYTES) -> None:
    try:
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError("Request requires finite, valid JSON") from exc
    if len(raw) > maximum:
        raise ValueError("Request exceeds its encoded byte limit")


@dataclass(frozen=True)
class CollectionConfig:
    max_records: int = 10_000
    max_batch: int = 100
    max_queries: int = 16
    max_scan_bytes: int = 16 * 1024 * 1024
    max_response_bytes: int = 2 * 1024 * 1024
    filter_timeout_seconds: float = 1.0

    def __post_init__(self):
        maxima = {"max_records": 10_000, "max_batch": 100, "max_queries": 16,
                  "max_scan_bytes": 16 * 1024 * 1024, "max_response_bytes": 2 * 1024 * 1024}
        for key, maximum in maxima.items():
            value = getattr(self, key)
            minimum = 32768 if key == "max_response_bytes" else 1
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"Collection {key} must be {minimum}..{maximum}")
        if (type(self.filter_timeout_seconds) not in (int, float)
                or not 0.001 <= self.filter_timeout_seconds <= 1.0):
            raise ValueError("Collection filter timeout must be 0.001..1 seconds")

    @classmethod
    def from_env(cls) -> "CollectionConfig":
        raw = os.getenv("RAGS_COLLECTION_LIMITS_JSON") or "{}"
        if len(raw) > 2048:
            raise ValueError("Collection limits configuration is too large")
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate collection limit")
                result[key] = value
            return result
        try:
            data = json.loads(raw, object_pairs_hook=unique)
            if not isinstance(data, dict) or not set(data) <= set(asdict(cls())):
                raise ValueError("Unknown collection limit")
            return cls(**data)
        except (TypeError, ValueError) as exc:
            raise ValueError("RAGS_COLLECTION_LIMITS_JSON requires known bounded numeric limits") from exc


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_default=True)

    @model_validator(mode="before")
    @classmethod
    def json_size(cls, data):
        if isinstance(data, dict):
            _encoded_bound(data)
        return data


class ExternalEmbedding(Contract):
    mode: Literal["external"] = "external"
    model: str = Field(min_length=1, max_length=160)
    space_id: str = Field(min_length=1, max_length=160)
    dimensions: int = Field(ge=1, le=MAX_DIMENSIONS)

    @field_validator("model", "space_id")
    @classmethod
    def text(cls, value):
        return valid_text(value, 160)


class CollectionCreate(Contract):
    name: str
    metadata: dict = Field(default_factory=dict)
    embedding: ExternalEmbedding | None = None
    sparse_space_id: str | None = Field(default=None, min_length=1, max_length=160)
    get_or_create: bool = False

    _name = field_validator("name")(validate_collection_name)
    _metadata = field_validator("metadata")(canonical_metadata)

    @field_validator("sparse_space_id")
    @classmethod
    def sparse_space(cls, value):
        return valid_text(value, 160) if value is not None else None


class CollectionModify(Contract):
    name: str | None = None
    metadata: dict | None = None
    expected_generation: int | None = Field(default=None, ge=0, le=2 ** 63 - 1)

    @field_validator("name")
    @classmethod
    def name_value(cls, value):
        return validate_collection_name(value) if value is not None else None

    @field_validator("metadata")
    @classmethod
    def metadata_value(cls, value):
        return canonical_metadata(value) if value is not None else None

    @model_validator(mode="after")
    def changes(self):
        if self.name is None and self.metadata is None:
            raise ValueError("Modify requires a name or metadata")
        return self


class CollectionFork(Contract):
    name: str
    metadata: dict | None = None
    expected_generation: int | None = Field(default=None, ge=0, le=2 ** 63 - 1)

    _name = field_validator("name")(validate_collection_name)

    @field_validator("metadata")
    @classmethod
    def metadata_value(cls, value):
        return canonical_metadata(value) if value is not None else None


class CollectionList(Contract):
    name: str | None = None
    limit: int = Field(default=100, ge=1, le=100)
    offset: int = Field(default=0, ge=0, le=1_000_000)

    @field_validator("name")
    @classmethod
    def name_value(cls, value):
        return validate_collection_name(value) if value is not None else None


class GenerationCheck(Contract):
    expected_generation: int | None = Field(default=None, ge=0, le=2 ** 63 - 1)


class RecordBatch(Contract):
    ids: list[str]
    documents: list[str | None] | None = None
    metadatas: list[dict] | None = None
    embeddings: list[list[float]] | None = None
    sparse_embeddings: list[dict | None] | None = None
    embedding_space_id: str | None = None
    expected_generation: int | None = Field(default=None, ge=0, le=2 ** 63 - 1)

    _ids = field_validator("ids", mode="before")(record_ids)

    @field_validator("documents")
    @classmethod
    def texts(cls, values):
        return ([valid_text(v, MAX_DOCUMENT_CHARS, allow_empty=True) if v is not None else None
                 for v in values] if values is not None else None)

    @field_validator("metadatas")
    @classmethod
    def metadata_values(cls, values):
        return [canonical_metadata(v) for v in values] if values is not None else None

    @field_validator("embeddings", mode="before")
    @classmethod
    def dense_values(cls, values):
        if values is not None and (not isinstance(values, list) or len(values) > 100):
            raise ValueError("Embeddings require a bounded list")
        return [canonical_dense(v) for v in values] if values is not None else None

    @field_validator("sparse_embeddings", mode="before")
    @classmethod
    def sparse_values(cls, values):
        if values is not None and (not isinstance(values, list) or len(values) > 100):
            raise ValueError("Sparse embeddings require a bounded list")
        return [canonical_sparse(v) for v in values] if values is not None else None

    @field_validator("embedding_space_id")
    @classmethod
    def space_id(cls, value):
        return valid_text(value, 160) if value is not None else None

    @model_validator(mode="after")
    def alignment(self):
        fields = [self.documents, self.metadatas, self.embeddings, self.sparse_embeddings]
        if all(value is None for value in fields):
            raise ValueError("A record batch requires at least one data field")
        if any(value is not None and len(value) != len(self.ids) for value in fields):
            raise ValueError("Every supplied array must align with ids")
        if (self.embeddings is None) != (self.embedding_space_id is None):
            raise ValueError("Precomputed embeddings require exactly one matching embedding_space_id")
        return self


class RecordSelection(Contract):
    ids: list[str] | None = None
    where: dict | None = None
    where_document: dict | None = None

    @field_validator("ids", mode="before")
    @classmethod
    def ids_value(cls, value):
        return record_ids(value) if value is not None else None


class GetRecords(RecordSelection):
    limit: int = Field(default=100, ge=1, le=100)
    offset: int = Field(default=0, ge=0, le=10_000)
    include: list[str] = Field(default_factory=lambda: ["document", "metadata"], max_length=6)

    @field_validator("include")
    @classmethod
    def include_values(cls, value):
        if len(value) != len(set(value)) or not set(value) <= RECORD_FIELDS | {"id", "version"}:
            raise ValueError("Unknown or duplicated record projection")
        return value


class DeleteRecords(RecordSelection):
    expected_generation: int | None = Field(default=None, ge=0, le=2 ** 63 - 1)

    @model_validator(mode="after")
    def qualified(self):
        if self.ids is None and not self.where and not self.where_document:
            raise ValueError("Delete requires explicit IDs or nonempty filters")
        return self


class GroupBy(Contract):
    field: str = Field(min_length=1, max_length=128)
    max_per_group: int = Field(default=1, ge=1, le=100)

    @field_validator("field")
    @classmethod
    def field_value(cls, value):
        return valid_text(value, 128)


class SearchQuery(RecordSelection):
    query_text: str | None = Field(default=None, min_length=1, max_length=2000)
    query_embedding: list[float] | None = None
    embedding_space_id: str | None = None
    query_sparse: dict | None = None
    mode: Literal["dense", "lexical", "sparse", "hybrid"] = "hybrid"
    limit: int = Field(default=10, ge=1, le=100)
    offset: int = Field(default=0, ge=0, le=99)
    candidate_limit: int = Field(default=100, ge=1, le=100)
    dense_weight: float = Field(default=1.0, ge=0, le=1000, allow_inf_nan=False)
    sparse_weight: float = Field(default=1.0, ge=0, le=1000, allow_inf_nan=False)
    rrf_k: int = Field(default=60, ge=1, le=1000)
    select: list[str] = Field(default_factory=lambda: ["document", "metadata", "score", "distance"], max_length=8)
    group_by: GroupBy | None = None

    @field_validator("query_text", "embedding_space_id")
    @classmethod
    def text_value(cls, value, info):
        return valid_text(value, 2000 if info.field_name == "query_text" else 160) if value is not None else None

    @field_validator("query_embedding", mode="before")
    @classmethod
    def vector_value(cls, value):
        return canonical_dense(value) if value is not None else None

    @field_validator("query_sparse", mode="before")
    @classmethod
    def sparse_value(cls, value):
        return canonical_sparse(value)

    @field_validator("dense_weight", "sparse_weight", mode="before")
    @classmethod
    def weight_type(cls, value):
        if type(value) not in (int, float):
            raise ValueError("Ranking weights must be numbers")
        return value

    @field_validator("select")
    @classmethod
    def projection(cls, value):
        if len(value) != len(set(value)) or not set(value) <= SEARCH_FIELDS:
            raise ValueError("Unknown or duplicated search projection")
        return value

    @property
    def uses_dense(self) -> bool:
        return self.mode == "dense" or (self.mode == "hybrid" and self.dense_weight > 0)

    @property
    def uses_sparse(self) -> bool:
        return self.mode == "sparse" or (self.mode == "hybrid" and self.sparse_weight > 0)

    @model_validator(mode="after")
    def inputs(self):
        if self.offset + self.limit > self.candidate_limit:
            raise ValueError("Pagination exceeds the ranked candidate window")
        if (self.query_embedding is None) != (self.embedding_space_id is None):
            raise ValueError("Precomputed query vector requires matching embedding_space_id")
        if self.mode == "dense" and ((self.query_text is None) == (self.query_embedding is None)):
            raise ValueError("Dense search requires exactly one text or vector query")
        if self.mode == "lexical" and (not self.query_text or self.query_embedding is not None or self.query_sparse is not None):
            raise ValueError("Lexical search requires only a text query")
        if self.mode == "sparse" and (self.query_sparse is None or self.query_embedding is not None or self.query_text is not None):
            raise ValueError("Sparse search requires only a sparse query")
        if self.mode == "dense" and self.query_sparse is not None:
            raise ValueError("Dense search does not accept sparse queries")
        if self.mode == "hybrid":
            if self.dense_weight == self.sparse_weight == 0:
                raise ValueError("At least one hybrid leg must have positive weight")
            if self.uses_dense and self.query_text is None and self.query_embedding is None:
                raise ValueError("Active dense leg requires a text or vector query")
            if self.uses_sparse and self.query_text is None and self.query_sparse is None:
                raise ValueError("Active sparse leg requires text or a sparse query")
        return self


class SearchBatch(Contract):
    searches: list[SearchQuery] = Field(min_length=1, max_length=16)


class CollectionAsk(RecordSelection):
    question: str = Field(min_length=2, max_length=2000)
    mode: Literal["dense", "lexical", "hybrid"] = "hybrid"
    top_k: int = Field(default=5, ge=1, le=20)
    generate: bool = True
    dense_weight: float = Field(default=1.0, ge=0, le=1000, allow_inf_nan=False)
    sparse_weight: float = Field(default=1.0, ge=0, le=1000, allow_inf_nan=False)
    rrf_k: int = Field(default=60, ge=1, le=1000)

    @field_validator("question")
    @classmethod
    def question_value(cls, value):
        return valid_text(value, 2000)

    @field_validator("dense_weight", "sparse_weight", mode="before")
    @classmethod
    def weight_type(cls, value):
        if type(value) not in (int, float):
            raise ValueError("Ranking weights must be numbers")
        return value

    def search_query(self) -> SearchQuery:
        return SearchQuery(query_text=self.question, mode=self.mode, limit=self.top_k,
                           ids=self.ids, where=self.where, where_document=self.where_document,
                           dense_weight=self.dense_weight, sparse_weight=self.sparse_weight,
                           rrf_k=self.rrf_k, select=["document", "metadata", "score"])

    @model_validator(mode="after")
    def ranking(self):
        self.search_query()
        return self
