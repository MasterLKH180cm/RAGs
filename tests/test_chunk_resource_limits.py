"""Construction bounds reject overlapping plans before retaining chunk copies."""
import tracemalloc

import pytest

import rags.chunking as chunking
from rags.chunking import ChunkConfig, chunk_document
from rags.tokenization import resolve_tokenizer


pytestmark = pytest.mark.unit


def test_high_overlap_memory_is_bounded_before_draft_construction(monkeypatch):
    text = "x" * 34_768
    tokenizer = resolve_tokenizer()
    original = tokenizer.truncate
    calls = []

    def counted(value, budget):
        calls.append(1)
        return original(value, budget)

    monkeypatch.setattr(tokenizer, "truncate", counted)
    monkeypatch.setattr(chunking, "ChunkDraft", lambda **_: pytest.fail("Rejected offset plan cannot retain drafts"))
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="byte budget"):
            chunk_document(text, config=ChunkConfig(strategy="fixed", max_tokens=32768, overlap_tokens=32767),
                           tokenizer=tokenizer)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(calls) <= 258
    assert peak < 8 * 1024 * 1024


@pytest.mark.parametrize("strategy", ["fixed", "recursive", "structure", "parent_child", "contextual"])
def test_record_cap_preflights_every_local_strategy_before_drafts_or_generation(monkeypatch, strategy):
    tokenizer = resolve_tokenizer()
    monkeypatch.setattr(chunking, "ChunkDraft", lambda **_: pytest.fail("No draft before plan admission"))
    with pytest.raises(ValueError, match="more than 3 index records"):
        chunk_document("A long source paragraph. " * 20, tokenizer=tokenizer,
                       config=ChunkConfig(strategy=strategy, max_tokens=20, overlap_tokens=1, parent_max_tokens=80),
                       max_records=3, context_model="fixture/context-v1",
                       contextualizer=lambda *_: pytest.fail("No contextual call before plan admission"))


def test_materialization_budget_counts_utf8_bytes_and_accepts_exact_boundary():
    config = ChunkConfig(strategy="fixed", max_tokens=2, overlap_tokens=0)
    plan = chunk_document("甲乙丙丁", config=config, max_materialized_bytes=12)
    assert [chunk.text for chunk in plan.chunks] == ["甲乙", "丙丁"]
    with pytest.raises(ValueError, match="byte budget"):
        chunk_document("甲乙丙丁", config=config, max_materialized_bytes=11)


def test_context_and_combined_embedding_are_charged_before_joining():
    options = dict(config=ChunkConfig(strategy="contextual", max_tokens=10, overlap_tokens=0),
                   contextualizer=lambda *_: "xy", context_model="fixture/context-v1")
    accepted = chunk_document("abc", max_materialized_bytes=12, **options)
    assert accepted.chunks[0].embedding_text == "xy\n\nabc"
    with pytest.raises(ValueError, match="byte budget"):
        chunk_document("abc", max_materialized_bytes=11, **options)


def test_parent_and_leaf_records_share_the_count_and_byte_budgets():
    config = ChunkConfig(strategy="parent_child", max_tokens=3, parent_max_tokens=9, overlap_tokens=0)
    plan = chunk_document("abcdef", config=config, max_records=3, max_materialized_bytes=12)
    assert [chunk.role for chunk in plan.chunks] == ["parent", "leaf", "leaf"]
    with pytest.raises(ValueError, match="more than 2 index records"):
        chunk_document("abcdef", config=config, max_records=2)
    with pytest.raises(ValueError, match="byte budget"):
        chunk_document("abcdef", config=config, max_materialized_bytes=11)


@pytest.mark.parametrize("name", ["max_records", "max_materialized_bytes"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_construction_limits_require_positive_integers(name, value):
    with pytest.raises(ValueError, match="positive integer"):
        chunk_document("source", **{name: value})


def test_semantic_unit_cap_precedes_encoder_calls():
    with pytest.raises(ValueError, match="2048 sentence units"):
        chunk_document("x." * 2049, config=ChunkConfig(strategy="semantic", max_tokens=2, overlap_tokens=0),
                       semantic_model="fixture/encoder-v1",
                       semantic_embedder=lambda *_: pytest.fail("Oversized sentence plan cannot call encoder"))
