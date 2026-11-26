"""Local artifact registration and provider-bound counters without native extras."""
import hashlib
import json
import os

import pytest

from rags.tokenization import ExactEncodingTokenizer, TokenizerConfig, resolve_tokenizer
from rags.tokenizer_artifacts import parse_tokenizer_artifacts, read_tokenizer_artifact


pytestmark = pytest.mark.unit


def registration(path, body=b"registered fixture"):
    return {"backend": "huggingface", "path": str(path), "sha256": hashlib.sha256(body).hexdigest()}


def test_registry_validation_is_pure_and_copies_its_inputs(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "open", lambda *_args, **_kwargs: pytest.fail("Validation opened an artifact"))
    supplied = {"fixture-v1": registration(tmp_path / "absent.json")}
    parsed = parse_tokenizer_artifacts(json.dumps(supplied))
    assert parsed == supplied
    parsed["fixture-v1"]["path"] = "changed only in the returned copy"
    assert supplied["fixture-v1"]["path"].endswith("absent.json")
    assert parse_tokenizer_artifacts(None) == {}


@pytest.mark.parametrize("mutation", [
    "non_object", "too_many", "bad_alias", "unknown_field", "relative", "remote",
    "bad_hash", "uppercase_hash", "unknown_backend", "duplicate_json", "oversized_json",
])
def test_invalid_registries_fail_without_optional_imports(tmp_path, mutation):
    entry = registration(tmp_path / "tokenizer.json")
    registry = {"fixture": entry}
    if mutation == "non_object":
        registry = []
    elif mutation == "too_many":
        registry = {f"item{index}": entry for index in range(17)}
    elif mutation == "bad_alias":
        registry = {"../host-path": entry}
    elif mutation == "unknown_field":
        entry["trust_remote_code"] = True
    elif mutation == "relative":
        entry["path"] = "relative.json"
    elif mutation == "remote":
        entry["path"] = "https://models.invalid/tokenizer.json"
    elif mutation == "bad_hash":
        entry["sha256"] = "main"
    elif mutation == "uppercase_hash":
        entry["sha256"] = "A" * 64
    elif mutation == "unknown_backend":
        entry["backend"] = "remote"
    elif mutation == "duplicate_json":
        registry = '{"duplicate": {}, "duplicate": {}}'
    else:
        registry = " " * (128 * 1024 + 1)
    with pytest.raises(ValueError):
        parse_tokenizer_artifacts(registry)


@pytest.mark.parametrize("mutation", ["missing", "directory", "symlink", "fifo", "oversize", "wrong_hash"])
def test_artifact_read_rejects_unusable_files_without_blocking(tmp_path, mutation):
    target = tmp_path / "artifact"
    body = b"registered fixture"
    if mutation == "directory":
        target.mkdir()
    elif mutation == "symlink":
        original = tmp_path / "original"
        original.write_bytes(body)
        target.symlink_to(original)
    elif mutation == "fifo":
        os.mkfifo(target)
    elif mutation == "oversize":
        with target.open("wb") as stream:
            stream.truncate(64 * 1024 * 1024 + 1)
    elif mutation == "wrong_hash":
        target.write_bytes(b"changed artifact")
    with pytest.raises(ValueError):
        read_tokenizer_artifact(registration(target, body))


def test_verified_artifact_bytes_come_from_one_descriptor_even_when_path_is_replaced(monkeypatch, tmp_path):
    body = b"registered fixture"
    target = tmp_path / "artifact"
    target.write_bytes(body)
    original_open = os.open

    def replace_after_open(path, flags):
        descriptor = original_open(path, flags)
        target.rename(tmp_path / "original-inode")
        target.write_bytes(b"replacement must not be parsed")
        return descriptor

    monkeypatch.setattr(os, "open", replace_after_open)
    assert read_tokenizer_artifact(registration(target, body)) == body


def bound_counter(identity="a" * 64):
    return ExactEncodingTokenizer(
        tokenizer_id="embedding:fixture", count=lambda text: len(text) + 2,
        profile={"backend": "embedding", "embedding_identity": identity,
                 "input_policy": {"special_tokens": 2}},
    )


def test_embedding_selection_requires_current_immutable_provider_binding():
    tokenizer = bound_counter()
    assert resolve_tokenizer({"backend": "embedding"}, embedding_tokenizer=tokenizer) is tokenizer
    assert resolve_tokenizer({"backend": "embedding", "embedding_identity": "a" * 64},
                             embedding_tokenizer=tokenizer) is tokenizer
    for binding in (None, resolve_tokenizer(), bound_counter("not-a-digest")):
        with pytest.raises(ValueError, match="binding"):
            resolve_tokenizer({"backend": "embedding"}, embedding_tokenizer=binding)
    with pytest.raises(ValueError, match="identity changed"):
        resolve_tokenizer({"backend": "embedding", "embedding_identity": "b" * 64},
                          embedding_tokenizer=tokenizer)
    profile = tokenizer.profile
    profile["input_policy"]["special_tokens"] = 999
    assert tokenizer.profile["input_policy"] == {"special_tokens": 2}


@pytest.mark.parametrize("config", [
    {"backend": "embedding", "model": "request-selected-model"},
    {"backend": "embedding", "encoding": "cl100k_base"},
    {"backend": "embedding", "path": "/arbitrary/host/file"},
    {"backend": "embedding", "artifact_sha256": "a" * 64},
    {"backend": "unicode", "embedding_identity": "a" * 64},
    {"backend": "tiktoken", "encoding": "cl100k_base", "artifact_sha256": "a" * 64},
    {"backend": "huggingface", "model": "/host/tokenizer.json"},
    {"backend": "sentencepiece", "model": "alias", "encoding": "override"},
    {"backend": "huggingface", "model": "alias", "artifact_sha256": "main"},
])
def test_request_tokenizer_config_cannot_select_paths_or_an_unbound_model(config):
    with pytest.raises(ValueError):
        resolve_tokenizer(config)


def test_count_only_embedding_adapter_keeps_source_slices_and_counts_input_framing():
    tokenizer = bound_counter()
    text = "中文 e\u0301 👩🏽‍💻\r\n" * 10
    spans = tokenizer.spans(text, max_tokens=20, overlap_tokens=5)
    assert spans[0].start == 0 and spans[-1].end == len(text)
    previous = None
    for span in spans:
        assert span.text == text[span.start:span.end]
        assert span.token_count == len(span.text) + 2 <= 20
        if previous:
            assert previous.start < span.start <= previous.end < span.end
            assert tokenizer.count(text[span.start:previous.end]) <= 5
        previous = span
    with pytest.raises(ValueError, match="token IDs"):
        tokenizer.encode(text)
    with pytest.raises(ValueError, match="decoding"):
        tokenizer.decode([1])


@pytest.mark.parametrize("overlap", [0, 1, 2, 4])
def test_bound_counter_framing_does_not_create_empty_overlap_loops(overlap):
    from rags.chunking import ChunkConfig, chunk_document
    from rags.tokenization import ContextItem, pack_context

    tokenizer = bound_counter()
    assert tokenizer.count("") == 2
    source = "Source e\u0301中文🙂 " * 4
    plan = chunk_document(source, tokenizer=tokenizer,
                          config=ChunkConfig(strategy="fixed", max_tokens=5, overlap_tokens=overlap))
    assert plan.chunks[0].start_char == 0 and plan.chunks[-1].end_char == len(source)
    previous = None
    for chunk in plan.chunks:
        assert chunk.text == source[chunk.start_char:chunk.end_char]
        assert chunk.token_count == len(chunk.text) + 2 <= 5
        assert chunk.metadata["context_token_count"] == 0
        if previous:
            assert previous.start_char < chunk.start_char <= previous.end_char < chunk.end_char
            shared = source[chunk.start_char:previous.end_char]
            assert not shared or tokenizer.count(shared) <= overlap
            assert chunk.metadata["overlap_with_previous"] == (tokenizer.count(shared) if shared else 0)
        previous = chunk
    for budget in (3, 4, 10):
        prefix, suffix = tokenizer.truncate(source, budget), tokenizer.suffix(source, budget)
        assert source.startswith(prefix) and source.endswith(suffix)
        assert tokenizer.count(prefix) <= budget and tokenizer.count(suffix) <= budget
    # A zero source budget represents no source slice, not an encoded request.
    assert tokenizer.truncate(source, 0) == tokenizer.suffix(source, 0) == ""
    with pytest.raises(ValueError, match="too small"):
        tokenizer.spans(source, max_tokens=2)
    packed = pack_context([ContextItem("source", source)], tokenizer=tokenizer,
                          input_budget=32, fixed_prefix="Question\n", fixed_suffix="\nAnswer")
    assert packed.token_count == len(packed.text) + 2 <= 32
    assert source.startswith(packed.items[0].text)


def test_bound_counter_context_metadata_keeps_nonempty_encoder_framing():
    from rags.chunking import ChunkConfig, chunk_document

    tokenizer = bound_counter()
    plan = chunk_document("abcdefgh", tokenizer=tokenizer,
                          config=ChunkConfig(strategy="contextual", max_tokens=6,
                                             overlap_tokens=0, context_max_tokens=4),
                          context_model="fixture/context-v1", contextualizer=lambda *_: "xy")
    for chunk in plan.chunks:
        assert chunk.metadata["context_token_count"] == len(chunk.context) + 2 == 4
        assert chunk.metadata["embedding_token_count"] == len(chunk.embedding_text) + 2


def test_legacy_tokenizer_profiles_have_no_new_null_fields():
    import tiktoken
    from importlib.metadata import version

    assert resolve_tokenizer().profile == {
        "backend": "unicode", "id": "unicode-codepoints-v1", "unit": "unicode_codepoints",
        "count_kind": "fallback_units", "version": 1,
    }
    config = TokenizerConfig(backend="tiktoken", encoding="cl100k_base")
    assert resolve_tokenizer(config).profile == {
        "backend": "tiktoken", "model": None, "encoding": tiktoken.get_encoding("cl100k_base").name,
        "id": "tiktoken:cl100k_base", "unit": "model_tokens", "count_kind": "exact_encoding",
        "version": version("tiktoken"),
    }
    assert set(resolve_tokenizer({"backend": "utf8_bytes"}).profile) == {
        "backend", "id", "unit", "count_kind", "version",
    }
