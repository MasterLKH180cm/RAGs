"""Snapshot-bound collection search, exact sparse ranking and result projection."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import math
import re

from rags.collection_filters import compile_filters, filter_remaining_seconds
from rags.collection_models import GetRecords, SearchQuery, MAX_REQUEST_BYTES, RECORD_FIELDS, SEARCH_FIELDS
from rags.registry import CapabilityError

BM25_K1 = 1.2
BM25_B = 0.75
_MISSING = object()
_WORDS = re.compile(r"\w+")


def _invalid_data() -> CapabilityError:
    return CapabilityError("invalid_collection_data", "Stored collection data is invalid", 500)


def _invalid_search() -> CapabilityError:
    return CapabilityError("invalid_collection_search", "Collection search parameters are invalid", 422)


def _tokens(text: str | None) -> list[str]:
    return _WORDS.findall(text.casefold()) if text else []


def bm25_rank(records: list[dict], query_text: str, limit: int) -> list[tuple[str, float]]:
    """Okapi BM25 with nonnegative IDF over the complete filtered record corpus.

    Empty/vector-only documents remain zero-length members of that corpus.
    Repeated query terms are counted once. No corpus or source is truncated.
    """
    query = set(_tokens(query_text))
    if not query or not records:
        return []
    frequencies, lengths = [], []
    appearances: Counter = Counter()
    for record in records:
        counts: Counter = Counter()
        length = 0
        # Stream all corpus terms to preserve exact lengths without retaining
        # irrelevant vocabulary or materializing a token list for every record.
        for match in _WORDS.finditer((record["document"] or "").casefold()):
            length += 1
            term = match.group()
            if term in query:
                counts[term] += 1
        frequencies.append(counts)
        lengths.append(length)
        appearances.update(counts.keys())
    average = math.fsum(lengths) / len(records)
    if average == 0:
        return []
    inverse_frequency = {
        term: math.log1p((len(records) - appearances[term] + 0.5) / (appearances[term] + 0.5))
        for term in sorted(appearances)
    }
    ranked = []
    for record, counts, length in zip(records, frequencies, lengths, strict=True):
        normalization = BM25_K1 * (1.0 - BM25_B + BM25_B * length / average)
        score = math.fsum(
            inverse_frequency[term] * (counts[term] * (BM25_K1 + 1.0)) / (counts[term] + normalization)
            for term in sorted(counts)
        )
        if score > 0:
            ranked.append((record["id"], score))
    return sorted(ranked, key=lambda item: (-item[1], item[0]))[:limit]


def sparse_rank(records: list[dict], query_sparse: dict, limit: int) -> list[tuple[str, float]]:
    """Exact dot product of validated sparse vectors in the same immutable space."""
    query = dict(zip(query_sparse["indices"], query_sparse["values"], strict=True))
    ranked = []
    for record in records:
        vector = record["sparse_embedding"]
        if vector is None:
            continue
        if vector["space_id"] != query_sparse["space_id"]:
            raise _invalid_data()
        score = math.fsum(
            query.get(index, 0.0) * value
            for index, value in zip(vector["indices"], vector["values"], strict=True)
        )
        if not math.isfinite(score):
            raise _invalid_data()
        if score > 0:
            ranked.append((record["id"], score))
    return sorted(ranked, key=lambda item: (-item[1], item[0]))[:limit]


def weighted_rrf(
    rankings: list[list[str]], weights: list[float], *, k: int = 60, limit: int = 100,
) -> list[tuple[str, float]]:
    """Positive weighted reciprocal ranks, with one-based positions and stable ties."""
    if (len(rankings) != len(weights) or type(k) is not int or not 1 <= k <= 1000
            or type(limit) is not int or not 1 <= limit <= 100
            or any(type(weight) not in (int, float) or not 0 <= weight <= 1000
                   or not math.isfinite(weight) for weight in weights)
            or not any(weight > 0 for weight in weights)):
        raise _invalid_search()
    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        if weight == 0:
            continue
        for rank, identifier in enumerate(dict.fromkeys(ranking), start=1):
            scores[identifier] = scores.get(identifier, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]


def select_matching_records(
    snapshot, where: dict | None = None, where_document: dict | None = None,
    ids: list[str] | None = None, *, deadline: float,
) -> list[dict]:
    """Filter a single authorized collection snapshot before ranking or mutation."""
    compiled = compile_filters(where, where_document)
    filter_remaining_seconds(deadline)
    allowed = set(ids) if ids is not None else None
    records = sorted(snapshot.records, key=lambda record: record["id"])
    selected = []
    for record in records:
        filter_remaining_seconds(deadline)
        if allowed is not None and record["id"] not in allowed:
            continue
        if compiled.matches(record["metadata"], record["document"], deadline=deadline):
            selected.append(record)
    return selected


def _typed_group(value) -> tuple:
    if value is _MISSING:
        return ("missing",)
    if value is None:
        return ("null",)
    if type(value) is bool:
        return ("boolean", value)
    if type(value) in (int, float):
        return ("number", value)
    if isinstance(value, str):
        return ("string", value)
    if isinstance(value, list):
        return ("array", tuple(_typed_group(item) for item in value))
    raise _invalid_data()


def _project(
    snapshot, records: list[dict], fields: list[str], *,
    scores: dict[str, float] | None = None, distances: dict[str, float] | None = None,
) -> list[dict]:
    if not set(fields) <= SEARCH_FIELDS:
        raise _invalid_search()
    embeddings = {}
    if "embedding" in fields and records:
        identifiers = [record["id"] for record in records]
        embeddings = snapshot.embeddings(identifiers)
        if set(embeddings) != set(identifiers):
            raise _invalid_data()
    result = []
    for record in records:
        item = {"id": record["id"], "version": record["version"]}
        for field in fields:
            if field in {"id", "version"}:
                continue
            if field == "embedding":
                vector = embeddings[record["id"]]
                if (not isinstance(vector, list) or not 1 <= len(vector) <= 4096
                        or any(type(value) not in (int, float) or not math.isfinite(value) for value in vector)
                        or not any(value != 0 for value in vector)):
                    raise _invalid_data()
                item[field] = list(vector)
            elif field == "score":
                item[field] = (scores or {}).get(record["id"])
            elif field == "distance":
                item[field] = (distances or {}).get(record["id"])
            else:
                item[field] = deepcopy(record[field])
        result.append(item)
    return result


def _bounded_result(result: dict) -> dict:
    try:
        size = len(json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8"))
    except (ValueError, TypeError, UnicodeError) as exc:
        raise _invalid_data() from exc
    if size > MAX_REQUEST_BYTES:
        raise CapabilityError("collection_response_limit", "Collection response exceeds its encoded byte limit", 422)
    return result


def execute_get(snapshot, request: GetRecords, *, deadline: float) -> dict:
    if not set(request.include) <= RECORD_FIELDS | {"id", "version"}:
        raise _invalid_search()
    records = select_matching_records(
        snapshot, request.where, request.where_document, request.ids, deadline=deadline,
    )
    page = records[request.offset:request.offset + request.limit]
    return _bounded_result({
        "items": _project(snapshot, page, request.include), "total": len(records),
        "limit": request.limit, "offset": request.offset,
    })


def execute_search(
    snapshot, search: SearchQuery, *, query_embedding: list[float] | None, deadline: float,
) -> dict:
    """Rank an already-authorized snapshot; this module never calls a provider."""
    records = select_matching_records(
        snapshot, search.where, search.where_document, search.ids, deadline=deadline,
    )
    lookup = {record["id"]: record for record in records}
    distances = {}
    dense_ranking = []
    # An empty filtered snapshot has no vector distances to compute. In particular,
    # the service can avoid a provider call for a collection with no records.
    if search.uses_dense and records:
        if query_embedding is None:
            raise _invalid_search()
        dense = snapshot.dense(query_embedding, set(lookup), search.candidate_limit)
        if len(dense) > search.candidate_limit:
            raise _invalid_data()
        for identifier, distance in dense:
            if (identifier not in lookup or identifier in distances
                    or type(distance) not in (int, float) or not math.isfinite(distance)
                    or not -1e-6 <= distance <= 2.0 + 1e-6):
                raise _invalid_data()
            distances[identifier] = min(2.0, max(0.0, distance))
        dense_ranking = sorted(distances, key=lambda identifier: (distances[identifier], identifier))
    lexical_ranking = []
    sparse_scores = []
    if search.mode == "lexical" or search.uses_sparse:
        if search.query_sparse is not None:
            sparse_scores = sparse_rank(records, search.query_sparse, search.candidate_limit)
        else:
            sparse_scores = bm25_rank(records, search.query_text or "", search.candidate_limit)
        lexical_ranking = [identifier for identifier, _score in sparse_scores]
    if search.mode == "dense":
        ranked = [(identifier, 1.0 - distances[identifier]) for identifier in dense_ranking]
        score_kind = "cosine_similarity"
    elif search.mode in {"lexical", "sparse"}:
        ranked = sparse_scores
        score_kind = "bm25" if search.mode == "lexical" else "sparse_dot_product"
    elif search.mode == "hybrid":
        ranked = weighted_rrf(
            [dense_ranking, lexical_ranking], [search.dense_weight, search.sparse_weight],
            k=search.rrf_k, limit=search.candidate_limit,
        )
        score_kind = "weighted_rrf"
    else:
        raise _invalid_search()
    candidate_count = len(ranked)
    if search.group_by is not None:
        counts: Counter = Counter()
        diversified = []
        for identifier, score in ranked:
            group = _typed_group(lookup[identifier]["metadata"].get(search.group_by.field, _MISSING))
            if counts[group] < search.group_by.max_per_group:
                diversified.append((identifier, score))
                counts[group] += 1
        ranked = diversified
    scores = dict(ranked)
    page = ranked[search.offset:search.offset + search.limit]
    return _bounded_result({
        "items": _project(snapshot, [lookup[identifier] for identifier, _score in page],
                          search.select, scores=scores, distances=distances),
        "mode": search.mode, "score_kind": score_kind, "matched_count": len(records),
        "candidate_count": candidate_count, "has_more": search.offset + search.limit < len(ranked),
        "limit": search.limit, "offset": search.offset,
    })
