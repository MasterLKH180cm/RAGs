"""Configurable, source-preserving chunk plans with explicit provider requirements."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from itertools import chain, pairwise
import json
import math
import re
from typing import Callable, Iterable, Iterator, Literal, Mapping, Sequence

from rags.tokenization import Tokenizer, resolve_tokenizer


ALGORITHM_VERSION = "rags-chunking-v2.0"
MAX_INDEX_RECORDS = 1024
MAX_MATERIALIZED_BYTES = 8 * 1024 * 1024
STRATEGIES = ("fixed", "recursive", "semantic", "structure", "parent_child", "contextual")
_SEPARATORS = (r"\n[ \t]*\n", r"\n", r"(?<=[.!?。！？])\s*", r"[ \t]+")
_HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)\s*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_LIST = re.compile(r"^[ \t]*(?:[-+*]|\d+[.)])[ \t]+")


@dataclass(frozen=True)
class ChunkConfig:
    strategy: Literal["fixed", "recursive", "semantic", "structure", "parent_child", "contextual"] = "structure"
    max_tokens: int = 256
    overlap_tokens: int = 32
    parent_max_tokens: int = 1024
    semantic_threshold: float = 0.35
    context_max_tokens: int = 96

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError("Unsupported chunking strategy")
        for name in ("max_tokens", "parent_max_tokens", "context_max_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 32768:
                raise ValueError(f"{name} must be an integer in 1..32768")
        if (isinstance(self.overlap_tokens, bool) or not isinstance(self.overlap_tokens, int)
                or not 0 <= self.overlap_tokens < self.max_tokens):
            raise ValueError("overlap_tokens must be in 0..max_tokens-1")
        if self.strategy == "parent_child" and self.parent_max_tokens <= self.max_tokens:
            raise ValueError("parent_max_tokens must exceed max_tokens for parent_child")
        if (isinstance(self.semantic_threshold, bool)
                or not isinstance(self.semantic_threshold, (float, int))
                or not math.isfinite(self.semantic_threshold)
                or not 0 <= self.semantic_threshold <= 2):
            raise ValueError("semantic_threshold must be a cosine distance in 0..2")


@dataclass(frozen=True)
class ChunkDraft:
    local_id: str
    role: Literal["parent", "leaf"]
    parent_id: str | None
    ordinal: int
    text: str
    start_char: int
    end_char: int
    section: str
    section_path: list[str]
    token_count: int
    tokenizer_id: str
    token_unit: str
    context: str
    embedding_text: str
    metadata: dict = field(default_factory=dict)

    @property
    def level(self) -> str:
        return self.role

    @property
    def indexable(self) -> bool:
        return self.role == "leaf"


@dataclass(frozen=True)
class ChunkPlan:
    chunks: list[ChunkDraft]
    profile: dict

    @property
    def fingerprint(self) -> str:
        return index_fingerprint(self.profile)


def index_fingerprint(profile: Mapping, **identities) -> str:
    """Canonical processing identity; callers add embedding/model identities.

    Source checksums and tenant/document IDs are separate storage concerns. Never
    pass credentials: this profile is intended for persisted, inspectable metadata.
    """
    try:
        encoded = json.dumps({"profile": dict(profile), "identities": identities},
                             sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                             allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError("Index profile must contain canonical JSON values") from exc
    return hashlib.sha256(encoded).hexdigest()


fingerprint = index_fingerprint


def _recursive_units(text: str, start: int, end: int, tokenizer: Tokenizer,
                     limit: int, level: int = 0) -> Iterator[tuple[int, int]]:
    if start == end:
        return
    if tokenizer.count(text[start:end]) <= limit:
        yield start, end
        return
    for index in range(level, len(_SEPARATORS)):
        boundaries = [start]
        for match in re.finditer(_SEPARATORS[index], text[start:end]):
            offset = start + match.end()
            if start < offset < end:
                boundaries.append(offset)
        boundaries = sorted(set([*boundaries, end]))
        if len(boundaries) > 2:
            for left, right in pairwise(boundaries):
                yield from _recursive_units(text, left, right, tokenizer, limit, index + 1)
            return
    for span in tokenizer.iter_spans(text[start:end], limit):
        yield start + span.start_char, start + span.end_char


def _pack_units(text: str, units: Iterable[tuple[int, int]], tokenizer: Tokenizer,
                limit: int, overlap: int) -> Iterator[tuple[int, int]]:
    iterator = iter(units)
    first = next(iterator, None)
    if first is None:
        return
    start, end = first
    for left, right in iterator:
        if left != end:
            raise ValueError("Chunk units must cover a contiguous source range")
        if tokenizer.count(text[start:right]) <= limit:
            end = right
            continue
        yield start, end
        suffix = tokenizer.suffix(text[start:end], overlap)
        candidate = max(start + 1, end - len(suffix))
        # Keep only overlap that leaves room for the next complete unit. This
        # can legitimately be zero at a large paragraph/block boundary.
        while candidate < left and (
            tokenizer.count(text[candidate:right]) > limit
            or tokenizer.count(text[candidate:left]) > overlap
        ):
            candidate += 1
        start, end = candidate, right
        if tokenizer.count(text[start:end]) > limit:
            raise ValueError("A source unit exceeds the chunk budget")
    yield start, end


def _line_spans(text: str, start: int, end: int) -> list[tuple[int, int, str]]:
    result = []
    offset = start
    for line in text[start:end].splitlines(keepends=True):
        result.append((offset, offset + len(line), line))
        offset += len(line)
    return result


def _sections(text: str) -> list[tuple[int, int, list[str]]]:
    """Markdown heading paths; heading text remains in the original source span."""
    result = []
    lines = _line_spans(text, 0, len(text))
    stack: list[tuple[int, str]] = []
    start = 0
    fence: str | None = None
    for index, (left, _, line) in enumerate(lines):
        marker = _FENCE.match(line)
        if fence:
            if re.match(r"^ {0,3}" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}\s*$", line):
                fence = None
            continue
        if marker:
            fence = marker.group(1)
            continue
        matched = _HEADING.match(line.rstrip("\r\n"))
        heading, depth = None, 0
        if matched:
            depth = len(matched.group(1))
            heading = re.sub(r"[ \t]+#+[ \t]*$", "", matched.group(2)).strip()
        elif line.strip() and index + 1 < len(lines):
            underline = lines[index + 1][2].strip()
            if re.fullmatch(r"={3,}|-{3,}", underline):
                heading, depth = line.strip(), 1 if underline[0] == "=" else 2
        if heading is not None:
            if left > start:
                result.append((start, left, [entry[1] for entry in stack]))
            stack = [entry for entry in stack if entry[0] < depth]
            stack.append((depth, heading))
            start = left
    if start < len(text):
        result.append((start, len(text), [entry[1] for entry in stack]))
    return result


def _table_header(line: str, following: str) -> bool:
    return "|" in line and bool(re.fullmatch(r"\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*",
                                             following.rstrip("\r\n")))


def _blocks(text: str, start: int, end: int) -> list[tuple[int, int, str]]:
    lines = _line_spans(text, start, end)
    result: list[tuple[int, int, str]] = []
    index = 0
    while index < len(lines):
        left, right, line = lines[index]
        following = lines[index + 1][2] if index + 1 < len(lines) else ""
        fence = _FENCE.match(line)
        kind = "paragraph"
        stop = index + 1
        if fence:
            kind = "code"
            marker = fence.group(1)
            while stop < len(lines):
                closing = re.match(r"^ {0,3}" + re.escape(marker[0]) + "{" + str(len(marker)) + r",}\s*$",
                                   lines[stop][2])
                stop += 1
                if closing:
                    break
        elif _HEADING.match(line) or (line.strip() and re.fullmatch(r"\s*(?:={3,}|-{3,})\s*", following)):
            kind = "heading"
            if not _HEADING.match(line):
                stop += 1
        elif _table_header(line, following):
            kind = "table"
            stop += 1
            while stop < len(lines) and "|" in lines[stop][2] and lines[stop][2].strip():
                stop += 1
        elif _LIST.match(line):
            kind = "list"
            while stop < len(lines) and lines[stop][2].strip():
                value = lines[stop][2]
                if _HEADING.match(value) or _FENCE.match(value):
                    break
                stop += 1
        elif not line.strip():
            if result:
                old_left, _, old_kind = result.pop()
                result.append((old_left, right, old_kind))
                index += 1
                continue
            kind = "whitespace"
        else:
            while stop < len(lines) and lines[stop][2].strip():
                value = lines[stop][2]
                next_value = lines[stop + 1][2] if stop + 1 < len(lines) else ""
                if _HEADING.match(value) or _FENCE.match(value) or _LIST.match(value) or _table_header(value, next_value):
                    break
                stop += 1
        result.append((left, lines[stop - 1][1], kind))
        index = stop
    return result


def _structure_ranges(text: str, tokenizer: Tokenizer, limit: int, overlap: int):
    for start, end, path in _sections(text):
        blocks = _blocks(text, start, end)
        units = chain.from_iterable(_recursive_units(text, left, right, tokenizer, limit)
                                    for left, right, _ in blocks)
        for left, right in _pack_units(text, units, tokenizer, limit, overlap):
            kinds = list(dict.fromkeys(kind for a, b, kind in blocks if a < right and b > left))
            yield left, right, path, {"block_types": kinds}


def _semantic_ranges(text: str, tokenizer: Tokenizer, config: ChunkConfig,
                     embedder: Callable[[str], Sequence[float]], model_id: str):
    ends = [match.end() for match in re.finditer(r"(?<=[.!?。！？])\s*|\n[ \t]*\n", text)]
    points = sorted(set([0, *ends, len(text)]))
    units = []
    for left, right in pairwise(points):
        for unit in _recursive_units(text, left, right, tokenizer, config.max_tokens):
            if len(units) >= 2048:
                raise ValueError("Semantic chunking is limited to 2048 sentence units per document")
            units.append(unit)
    vectors = []
    dimensions = None
    for start, end in units:
        raw = embedder(text[start:end])
        if not isinstance(raw, (list, tuple)) or not raw:
            raise ValueError("Semantic encoder must return a non-empty numeric vector")
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) for v in raw):
            raise ValueError("Semantic encoder returned invalid numeric values")
        try:
            values = [float(value) for value in raw]
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError("Semantic encoder returned invalid numeric values") from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Semantic encoder returned invalid numeric values")
        if dimensions is not None and len(raw) != dimensions:
            raise ValueError("Semantic encoder dimensions changed within the document")
        dimensions = len(raw)
        norm = math.hypot(*values)
        if not math.isfinite(norm) or norm == 0:
            raise ValueError("Semantic encoder returned an unusable vector")
        vectors.append([value / norm for value in values])
    groups, current = [], []
    for index, unit in enumerate(units):
        distance = 0.0 if index == 0 else 1 - max(-1.0, min(1.0, sum(
            left * right for left, right in zip(vectors[index - 1], vectors[index], strict=True)
        )))
        if current and distance > config.semantic_threshold:
            groups.append(current)
            current = []
        current.append(unit)
    if current:
        groups.append(current)
    for group_number, group in enumerate(groups):
        for left, right in _pack_units(text, group, tokenizer, config.max_tokens, config.overlap_tokens):
            yield left, right, [], {"semantic_model": model_id,
                                    "semantic_group": group_number,
                                    "semantic_threshold": config.semantic_threshold}


def chunk_document(
    text: str, *, title: str = "", config: ChunkConfig | Mapping | None = None,
    tokenizer: Tokenizer | None = None,
    semantic_embedder: Callable[[str], Sequence[float]] | None = None,
    contextualizer: Callable[[str, str, str], str] | None = None,
    semantic_model: str | None = None, context_model: str | None = None,
    max_records: int = MAX_INDEX_RECORDS,
    max_materialized_bytes: int = MAX_MATERIALIZED_BYTES,
) -> ChunkPlan:
    for name, value in (("max_records", max_records), ("max_materialized_bytes", max_materialized_bytes)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config is None:
        config = ChunkConfig()
    if isinstance(config, Mapping):
        try:
            config = ChunkConfig(**config)
        except TypeError as exc:
            raise ValueError("Unknown chunking configuration field") from exc
    if not isinstance(config, ChunkConfig):
        raise ValueError("Expected ChunkConfig or a configuration object")
    tokenizer = tokenizer or resolve_tokenizer()
    tokenizer.count(text)  # Validate Unicode even if the source is otherwise empty.
    if not text.strip():
        raise ValueError("Cannot chunk an empty document")
    if not isinstance(title, str):
        raise ValueError("Document title must be a string")
    tokenizer.count(title)
    if config.strategy == "semantic" and (
        semantic_embedder is None or not isinstance(semantic_model, str) or not semantic_model.strip()
        or semantic_model.casefold() == "local" or semantic_model.casefold().startswith("offline-hash")
    ):
        raise ValueError("Semantic chunking requires an explicitly identified semantic encoder; hashing is unsupported")
    if config.strategy == "contextual" and (
        contextualizer is None or not isinstance(context_model, str) or not context_model.strip()
        or context_model.casefold() in {"extractive", "offline", "local"}
    ):
        raise ValueError("Contextual chunking requires a configured context generator and model identity")
    profile = {"algorithm": ALGORITHM_VERSION, "chunking": asdict(config),
               "tokenizer": tokenizer.profile,
               "semantic_model": semantic_model if config.strategy == "semantic" else None,
               "context_model": context_model if config.strategy == "contextual" else None}

    def source_ranges():
        if config.strategy == "fixed":
            for span in tokenizer.iter_spans(text, config.max_tokens, config.overlap_tokens):
                yield span.start_char, span.end_char, [], {}, "leaf", None
        elif config.strategy == "recursive":
            units = _recursive_units(text, 0, len(text), tokenizer, config.max_tokens)
            for start, end in _pack_units(text, units, tokenizer, config.max_tokens, config.overlap_tokens):
                yield start, end, [], {}, "leaf", None
        elif config.strategy == "semantic":
            for start, end, path, extra in _semantic_ranges(text, tokenizer, config, semantic_embedder, semantic_model):
                yield start, end, path, extra, "leaf", None
        elif config.strategy == "parent_child":
            # Non-overlapping parents eliminate duplicate expansion across siblings.
            source_blocks = [block for left, right, _ in _sections(text) for block in _blocks(text, left, right)]
            parents = _structure_ranges(text, tokenizer, config.parent_max_tokens, 0)
            for ordinal, (start, end, path, extra) in enumerate(parents):
                yield start, end, path, extra, "parent", None
                units = _recursive_units(text, start, end, tokenizer, config.max_tokens)
                for left, right in _pack_units(text, units, tokenizer, config.max_tokens, config.overlap_tokens):
                    kinds = list(dict.fromkeys(kind for a, b, kind in source_blocks if a < right and b > left))
                    yield left, right, path, {"block_types": kinds}, "leaf", f"parent-{ordinal:06d}"
        else:
            for start, end, path, extra in _structure_ranges(text, tokenizer, config.max_tokens, config.overlap_tokens):
                yield start, end, path, extra, "leaf", None

    # Retain only bounded offsets here. In particular, high-overlap fixed
    # windows never accumulate their potentially gigabyte-sized string copies.
    ranges = []
    materialized_bytes = 0
    for candidate in source_ranges():
        if len(ranges) >= max_records:
            raise ValueError(f"Chunk configuration produces more than {max_records} index records")
        start, end = candidate[:2]
        remaining = max_materialized_bytes - materialized_bytes
        if end - start > remaining:
            raise ValueError("Chunk materialization exceeds the byte budget")
        size = len(text[start:end].encode("utf-8"))
        if size > remaining:
            raise ValueError("Chunk materialization exceeds the byte budget")
        materialized_bytes += size
        ranges.append(candidate)

    chunks = []
    counters = {"parent": 0, "leaf": 0}

    def append(start, end, path, extra, role="leaf", parent_id=None):
        nonlocal materialized_bytes
        value = text[start:end]
        ordinal = counters[role]
        counters[role] += 1
        context = ""
        context_truncated = False
        if config.strategy == "contextual":
            generated = contextualizer(text, value, title)
            if not isinstance(generated, str) or not generated.strip():
                raise ValueError("Context generator returned empty or invalid text")
            context = generated.strip()
            if tokenizer.count(context) > config.context_max_tokens:
                raise ValueError("Context generator exceeded context_max_tokens")
            # The retained source is already charged by the offset preflight.
            # Charge context plus the separate combined string before joining.
            extra_bytes = len(value.encode("utf-8")) + 2 * len(context.encode("utf-8")) + 2
            if extra_bytes > max_materialized_bytes - materialized_bytes:
                raise ValueError("Chunk materialization exceeds the byte budget")
            materialized_bytes += extra_bytes
        embedding_text = (context + "\n\n" + value) if context else value
        count = tokenizer.count(value)
        limit = config.parent_max_tokens if role == "parent" else config.max_tokens
        if not value or count > limit:
            raise ValueError("Chunk construction violated its source/token budget")
        draft = ChunkDraft(
            local_id=f"{role}-{ordinal:06d}", role=role, parent_id=parent_id, ordinal=ordinal,
            text=value, start_char=start, end_char=end, section=path[-1] if path else "Document",
            section_path=list(path), token_count=count, tokenizer_id=tokenizer.id,
            token_unit=tokenizer.unit, context=context, embedding_text=embedding_text,
            metadata={"strategy": config.strategy, "algorithm": ALGORITHM_VERSION,
                      "source_offset_unit": "python_characters", "count_kind": tokenizer.count_kind,
                      "embedding_token_count": tokenizer.count(embedding_text),
                      "context_token_count": tokenizer.count(context), "context_truncated": context_truncated,
                      **extra},
        )
        chunks.append(draft)
        return draft

    for start, end, path, extra, role, parent_id in ranges:
        append(start, end, path, extra, role=role, parent_id=parent_id)
    leaves = [chunk for chunk in chunks if chunk.role == "leaf"]
    for previous, current in pairwise(leaves):
        overlap = text[current.start_char:min(previous.end_char, current.end_char)] if current.start_char < previous.end_char else ""
        current.metadata["overlap_with_previous"] = tokenizer.count(overlap)
    return ChunkPlan(chunks, profile)
