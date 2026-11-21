"""RAG-02/RAG-04 unit acceptance: six distinct strategies and index identities."""
from dataclasses import replace

import pytest

from rags.chunking import ChunkConfig, chunk_document, index_fingerprint
from rags.tokenization import resolve_tokenizer


def assert_source_plan(plan, text, tokenizer, leaf_limit, parent_limit=None):
    coverage = set()
    parents = {chunk.local_id: chunk for chunk in plan.chunks if chunk.role == "parent"}
    assert len({chunk.local_id for chunk in plan.chunks}) == len(plan.chunks)
    for chunk in plan.chunks:
        assert chunk.text == text[chunk.start_char:chunk.end_char]
        assert chunk.token_count == tokenizer.count(chunk.text)
        assert chunk.token_count <= (parent_limit if chunk.role == "parent" else leaf_limit)
        assert chunk.tokenizer_id == tokenizer.id
        assert chunk.token_unit == tokenizer.unit
        if chunk.role == "leaf":
            coverage.update(range(chunk.start_char, chunk.end_char))
        if chunk.parent_id:
            parent = parents[chunk.parent_id]
            assert parent.start_char <= chunk.start_char < chunk.end_char <= parent.end_char
        assert chunk.metadata["embedding_token_count"] == tokenizer.count(chunk.embedding_text)
    assert coverage == set(range(len(text)))


@pytest.mark.parametrize("strategy", ["fixed", "recursive", "structure", "parent_child"])
@pytest.mark.parametrize("backend", ["unicode", "tiktoken"])
def test_all_local_strategies_preserve_multilingual_source_and_limits(strategy, backend):
    tokenizer = resolve_tokenizer({"backend": backend, **({"encoding": "cl100k_base"}
                                                           if backend == "tiktoken" else {})})
    text = "# 登入政策\n\n工作階段將在三十分鐘後到期。🔑\n\n## 注意\n\ne\u0301 👩🏽‍💻 資料必須完整保留。\n" * 5
    config = ChunkConfig(strategy=strategy, max_tokens=24, overlap_tokens=3, parent_max_tokens=80)
    plan = chunk_document(text, config=config, tokenizer=tokenizer)
    assert_source_plan(plan, text, tokenizer, 24, 80)
    assert all(chunk.metadata.get("overlap_with_previous", 0) <= 3 for chunk in plan.chunks)
    if strategy == "parent_child":
        assert any(chunk.role == "parent" for chunk in plan.chunks)
        assert all(chunk.parent_id for chunk in plan.chunks if chunk.role == "leaf")


def test_recursive_prefers_complete_paragraphs_when_they_fit():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    plan = chunk_document(text, config=ChunkConfig(strategy="recursive", max_tokens=22, overlap_tokens=0))
    assert [chunk.text for chunk in plan.chunks] == [
        "First paragraph.\n\n", "Second paragraph.\n\n", "Third paragraph.",
    ]
    assert_source_plan(plan, text, resolve_tokenizer(), 22)


def test_recursive_handles_oversized_unbroken_units():
    text = "甲" * 75
    plan = chunk_document(text, config=ChunkConfig(strategy="recursive", max_tokens=16, overlap_tokens=2))
    assert len(plan.chunks) >= 5
    assert_source_plan(plan, text, resolve_tokenizer(), 16)


def test_markdown_paths_ignore_fenced_headings_and_identify_blocks():
    text = (
        "# Product\n\n## API\n\n```python\n# not a real section\nprint('ok')\n```\n\n"
        "- first\n- second\n\n| Field | Value |\n| --- | --- |\n| timeout | 30 |\n\n"
        "## Operations\n\nRestart safely."
    )
    plan = chunk_document(text, config=ChunkConfig(strategy="structure", max_tokens=90, overlap_tokens=0))
    assert_source_plan(plan, text, resolve_tokenizer(), 90)
    assert all("not a real section" not in chunk.section_path for chunk in plan.chunks)
    assert any(chunk.section_path == ["Product", "API"] for chunk in plan.chunks)
    assert plan.chunks[-1].section_path == ["Product", "Operations"]
    kinds = {kind for chunk in plan.chunks for kind in chunk.metadata["block_types"]}
    assert {"code", "list", "table", "heading"} <= kinds
    code = next(chunk for chunk in plan.chunks if "```python" in chunk.text)
    assert "print('ok')\n```" in code.text


def test_setext_heading_is_retained_in_source():
    text = "Title\n=====\nBody\n\nSubheading\n----------\nDetails"
    plan = chunk_document(text, config=ChunkConfig(max_tokens=100, overlap_tokens=0))
    assert [chunk.section_path for chunk in plan.chunks] == [["Title"], ["Title", "Subheading"]]
    assert "Title\n=====" in plan.chunks[0].text
    assert_source_plan(plan, text, resolve_tokenizer(), 100)


def test_semantic_breaks_follow_supplied_encoder_discontinuity():
    text = "Cats purr. Dogs bark. Planets orbit. Stars shine."
    observed = []

    def encode(sentence):
        observed.append(sentence)
        return [1.0, 0.0] if "Cats" in sentence or "Dogs" in sentence else [0.0, 1.0]

    plan = chunk_document(text, config=ChunkConfig(strategy="semantic", max_tokens=100, overlap_tokens=0),
                          semantic_embedder=encode, semantic_model="fixture:two-topics")
    assert len(observed) == 4
    assert [chunk.text for chunk in plan.chunks] == ["Cats purr. Dogs bark. ", "Planets orbit. Stars shine."]
    assert [chunk.metadata["semantic_group"] for chunk in plan.chunks] == [0, 1]
    assert_source_plan(plan, text, resolve_tokenizer(), 100)


@pytest.mark.parametrize("identity", [None, "local", "offline-hash-v1"])
def test_semantic_requires_explicit_non_hash_encoder(identity):
    with pytest.raises(ValueError, match="semantic encoder"):
        chunk_document("A sentence.", config=ChunkConfig(strategy="semantic"),
                       semantic_embedder=lambda _: [1.0, 0.0], semantic_model=identity)


@pytest.mark.parametrize("vector", [[], [0.0, 0.0], [float("nan"), 1], [float("inf"), 1], [True, 0]])
def test_semantic_rejects_invalid_provider_vectors(vector):
    with pytest.raises(ValueError, match="Semantic encoder"):
        chunk_document("A sentence.", config=ChunkConfig(strategy="semantic"),
                       semantic_embedder=lambda _: vector, semantic_model="fixture:invalid")


def test_semantic_rejects_dimension_changes():
    values = iter([[1.0, 0.0], [1.0]])
    with pytest.raises(ValueError, match="dimensions changed"):
        chunk_document("One sentence. Another sentence.", config=ChunkConfig(strategy="semantic"),
                       semantic_embedder=lambda _: next(values), semantic_model="fixture:dimension-change")


def test_contextual_separates_generated_context_from_original_citation_text():
    original = "# Policy\nSessions expire after 30 minutes."
    calls = []

    def context(document, chunk, title):
        calls.append((document, chunk, title))
        return "Session policy"

    plan = chunk_document(original, title="Authentication",
                          config=ChunkConfig(strategy="contextual", max_tokens=28,
                                             overlap_tokens=2, context_max_tokens=20),
                          contextualizer=context, context_model="fixture:context-v1")
    assert len(calls) == len(plan.chunks)
    assert all(doc == original and title == "Authentication" for doc, _, title in calls)
    for chunk in plan.chunks:
        assert chunk.text == original[chunk.start_char:chunk.end_char]
        assert chunk.context == "Session policy"
        assert chunk.embedding_text == chunk.context + "\n\n" + chunk.text
        assert chunk.metadata["context_truncated"] is False
        assert chunk.metadata["context_token_count"] <= 20
    assert_source_plan(plan, original, resolve_tokenizer(), 28)


def test_contextual_missing_or_invalid_generator_fails_explicitly():
    with pytest.raises(ValueError, match="context generator"):
        chunk_document("A document.", config=ChunkConfig(strategy="contextual"))
    with pytest.raises(ValueError, match="empty or invalid"):
        chunk_document("A document.", config=ChunkConfig(strategy="contextual"),
                       contextualizer=lambda *_: " ", context_model="fixture:empty")
    with pytest.raises(ValueError, match="exceeded context_max_tokens"):
        chunk_document("A document.", config=ChunkConfig(strategy="contextual", context_max_tokens=5),
                       contextualizer=lambda *_: "too much context", context_model="fixture:oversized")


def test_excessive_contextual_plan_is_rejected_before_provider_calls():
    calls = []
    with pytest.raises(ValueError, match="1024 index records"):
        chunk_document("a" * 1100, config=ChunkConfig(strategy="contextual", max_tokens=1, overlap_tokens=0),
                       contextualizer=lambda *args: calls.append(args) or "context", context_model="fixture:context")
    assert calls == []


def test_index_profile_changes_for_every_processing_identity():
    base_config = ChunkConfig(strategy="fixed", max_tokens=20, overlap_tokens=2)
    base = chunk_document("A document for indexing.", config=base_config)
    identity = index_fingerprint(base.profile, embedding_model="encoder-a", dimensions=384)
    reordered = dict(reversed(list(base.profile.items())))
    assert identity == index_fingerprint(reordered, dimensions=384, embedding_model="encoder-a")
    for config in (replace(base_config, max_tokens=21), replace(base_config, overlap_tokens=1),
                   replace(base_config, strategy="recursive")):
        changed = chunk_document("A document for indexing.", config=config)
        assert index_fingerprint(changed.profile, embedding_model="encoder-a", dimensions=384) != identity
    assert index_fingerprint(base.profile, embedding_model="encoder-b", dimensions=384) != identity
    assert index_fingerprint(base.profile, embedding_model="encoder-a", dimensions=768) != identity
    changed = chunk_document("A document for indexing.", config=base_config,
                             tokenizer=resolve_tokenizer({"backend": "tiktoken", "encoding": "cl100k_base"}))
    assert index_fingerprint(changed.profile, embedding_model="encoder-a", dimensions=384) != identity
    assert index_fingerprint(base.profile, context_model="model-a", context_prompt="v1") != index_fingerprint(
        base.profile, context_model="model-a", context_prompt="v2")
    with pytest.raises(ValueError, match="canonical"):
        index_fingerprint({"not_finite": float("nan")})


@pytest.mark.parametrize("changes", [
    {"strategy": "made-up"}, {"max_tokens": 0}, {"overlap_tokens": 256},
    {"max_tokens": True}, {"strategy": "parent_child", "parent_max_tokens": 128},
    {"semantic_threshold": float("nan")}, {"context_max_tokens": 0},
])
def test_chunk_configuration_is_validated(changes):
    with pytest.raises(ValueError):
        ChunkConfig(**changes)
