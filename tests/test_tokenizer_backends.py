"""Actual local tokenizer artifacts, with no downloaded weights or credentials."""
from __future__ import annotations

import builtins
import hashlib
import io
import os

import pytest

from rags.tokenization import ContextItem, TokenizerConfig, pack_context, resolve_tokenizer

pytestmark = pytest.mark.integration

SOURCES = [
    "Hello WORLD! Café Ｈｅｌｌｏ e\u0301   repeated whitespace.\n\t" * 5,
    "繁體中文測試。沒有空格也要保留原始內容。日本語も保持します。" * 5,
    "👩🏽‍💻 e\u0301 🏳️‍🌈 👨‍👩‍👧‍👦 🇹🇼 🚀" * 5,
    "[CLS] [SEP] [UNK] <special> literal source tokens. " * 5,
    "abcde" * 400,
    " \n\t" * 120,
]


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    import sentencepiece
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors

    directory = tmp_path_factory.mktemp("tokenizer-artifacts")
    vocabulary = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "hello", "world", "cafe", "!", ".",
                  "repeated", "whitespace", "literal", "source", "tokens", "a", "##b", "##c",
                  "##d", "##e"]
    native = Tokenizer(models.WordPiece({word: index for index, word in enumerate(vocabulary)},
                                       unk_token="[UNK]"))
    native.normalizer = normalizers.BertNormalizer(lowercase=True, strip_accents=True)
    native.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    native.decoder = decoders.WordPiece(prefix="##")
    native.add_special_tokens(["[CLS]", "[SEP]", "[PAD]"])
    native.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)])
    native.enable_truncation(max_length=4)
    native.enable_padding(length=8, pad_id=3, pad_token="[PAD]")
    hf_path = directory / "tokenizer.json"
    native.save(str(hf_path))
    output = io.BytesIO()
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=iter(SOURCES * 2), model_writer=output, model_type="bpe",
        vocab_size=180, hard_vocab_limit=False, character_coverage=1.0, num_threads=1,
        shuffle_input_sentence=False, user_defined_symbols=["<special>"], minloglevel=2,
    )
    sp_path = directory / "tokenizer.model"
    sp_path.write_bytes(output.getvalue())
    return {"huggingface": hf_path, "sentencepiece": sp_path}


@pytest.fixture(params=["huggingface", "sentencepiece"])
def tokenizer(request, artifacts):
    return resolve_tokenizer({"backend": request.param, "path": str(artifacts[request.param])})


@pytest.mark.parametrize("source", SOURCES)
def test_local_artifact_spans_preserve_original_unicode(tokenizer, source):
    spans = tokenizer.spans(source, 20, 4)
    assert spans[0].start_char == 0
    assert spans[-1].end_char == len(source)
    previous = None
    covered = set()
    for span in spans:
        assert span.text == source[span.start_char:span.end_char]
        assert span.token_count == tokenizer.count(span.text) <= 20
        covered.update(range(span.start_char, span.end_char))
        if previous:
            assert previous.start_char < span.start_char <= previous.end_char
            assert span.end_char > previous.end_char
            assert tokenizer.count(source[span.start_char:previous.end_char]) <= 4
        previous = span
    assert covered == set(range(len(source)))


def test_saved_huggingface_truncation_padding_and_bos_are_disabled(artifacts):
    from tokenizers import Tokenizer

    native = Tokenizer.from_file(str(artifacts["huggingface"]))
    source = "hello world cafe repeated whitespace " * 20
    assert len(native.encode(source).ids) == 8
    selected = resolve_tokenizer({"backend": "huggingface", "path": str(artifacts["huggingface"])})
    native.no_truncation()
    native.no_padding()
    expected = native.encode(source, add_special_tokens=False).ids
    assert selected.encode(source) == expected
    assert selected.count(source) == 100
    assert selected.decode(selected.encode("Café WORLD")) == "cafe world"
    assert selected.truncate("Café WORLD", 2) == "Café WORLD"
    assert selected.profile["add_special_tokens"] is False


def test_sentencepiece_matches_native_without_automatic_bos_eos(artifacts):
    import sentencepiece

    native = sentencepiece.SentencePieceProcessor(model_proto=artifacts["sentencepiece"].read_bytes())
    selected = resolve_tokenizer({"backend": "sentencepiece", "path": str(artifacts["sentencepiece"])})
    for source in SOURCES:
        expected = native.encode(source, out_type=int, add_bos=False, add_eos=False, enable_sampling=False)
        assert selected.encode(source) == expected
        assert selected.count(source) == len(expected)
        assert selected.decode(expected) == native.decode(expected)
    assert selected.profile["add_special_tokens"] is False


def test_artifact_identity_uses_loaded_bytes_not_path(artifacts, tmp_path):
    original = artifacts["huggingface"]
    moved = tmp_path / "renamed.json"
    moved.write_bytes(original.read_bytes())
    first = resolve_tokenizer({"backend": "huggingface", "path": str(original)})
    second = resolve_tokenizer({"backend": "huggingface", "path": str(moved)})
    assert first.profile == second.profile
    assert first.profile["artifact_sha256"] == hashlib.sha256(original.read_bytes()).hexdigest()
    assert str(original) not in str(first.profile)
    assert "path" not in first.profile
    moved.write_bytes(moved.read_bytes() + b"\n")
    third = resolve_tokenizer({"backend": "huggingface", "path": str(moved)})
    assert first.id != third.id
    assert first.profile != third.profile
    # Existing instances continue to describe the immutable bytes they loaded.
    assert first.encode("hello") == second.encode("hello") == third.encode("hello")


@pytest.mark.parametrize("tokens", [[True], [1.0], ["1"], [-1], [2 ** 62], "12", b"12", None])
def test_local_decode_rejects_invalid_token_ids(tokenizer, tokens):
    with pytest.raises(ValueError, match="token sequence"):
        tokenizer.decode(tokens)


def test_local_tokenizer_input_and_budget_contract(tokenizer):
    with pytest.raises(ValueError, match="surrogate"):
        tokenizer.encode("invalid\ud800")
    with pytest.raises(ValueError, match="string"):
        tokenizer.count(10)
    for budget in (True, -1, 1.5):
        with pytest.raises(ValueError):
            tokenizer.truncate("test", budget)
        with pytest.raises(ValueError):
            tokenizer.suffix("test", budget)
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""
    assert tokenizer.truncate("hello", 0) == ""
    assert tokenizer.suffix("hello", 0) == ""


def test_normalization_does_not_rewrite_packed_evidence(tokenizer):
    source = "Café Ｈｅｌｌｏ e\u0301 繁體中文 🚀 " * 40
    packed = pack_context([ContextItem("source", source)], tokenizer=tokenizer, input_budget=30,
                          fixed_prefix="Evidence:\n", fixed_suffix="\nAnswer:")
    assert packed.items
    assert source.startswith(packed.items[0].text)
    assert packed.token_count == tokenizer.count(packed.text) <= 30
    assert packed.tokenizer_id == tokenizer.id
    assert packed.count_kind == "exact_encoding"


@pytest.mark.parametrize("config", [
    {"backend": []}, {"backend": {}}, {"backend": None},
    {"backend": "huggingface"}, {"backend": "sentencepiece", "path": ""},
    {"backend": "sentencepiece", "path": True},
    {"backend": "huggingface", "path": "https://example.test/tokenizer.json"},
    {"backend": "huggingface", "path": "model\x00.json"},
    {"backend": "huggingface", "path": "x" * 4097},
    {"backend": "huggingface", "path": "local.json", "model": "remote/model"},
    {"backend": "sentencepiece", "path": "local.model", "encoding": "utf8"},
    {"backend": "tiktoken", "encoding": "cl100k_base", "path": "local.json"},
    {"backend": "unicode", "path": "local.json"},
    {"backend": "utf8_bytes", "path": "local.json"},
])
def test_local_artifact_configuration_fails_before_loading(config):
    with pytest.raises(ValueError):
        TokenizerConfig(**config)


@pytest.mark.parametrize("backend", ["huggingface", "sentencepiece"])
def test_malformed_missing_empty_or_oversized_artifacts_are_rejected(backend, tmp_path, monkeypatch):
    import rags.tokenizer_backends as backends

    path = tmp_path / "model"
    with pytest.raises(ValueError, match="regular local file"):
        resolve_tokenizer({"backend": backend, "path": str(path)})
    path.touch()
    with pytest.raises(ValueError, match="empty"):
        resolve_tokenizer({"backend": backend, "path": str(path)})
    path.write_bytes(b"not a valid tokenizer artifact")
    with pytest.raises(ValueError, match="Invalid .* artifact"):
        resolve_tokenizer({"backend": backend, "path": str(path)})
    monkeypatch.setattr(backends, "MAX_ARTIFACT_BYTES", 8)
    with pytest.raises(ValueError, match="regular local file"):
        resolve_tokenizer({"backend": backend, "path": str(path)})


def test_local_artifact_rejects_directory_symlink_and_fifo(artifacts, tmp_path):
    paths = [tmp_path]
    link = tmp_path / "link.json"
    link.symlink_to(artifacts["huggingface"])
    paths.append(link)
    if hasattr(os, "mkfifo"):
        fifo = tmp_path / "pipe.json"
        os.mkfifo(fifo)
        paths.append(fifo)
    for path in paths:
        with pytest.raises(ValueError, match="regular local file"):
            resolve_tokenizer({"backend": "huggingface", "path": str(path)})


@pytest.mark.parametrize("backend,dependency", [("huggingface", "tokenizers"), ("sentencepiece", "sentencepiece")])
def test_optional_dependency_error_is_actionable(artifacts, monkeypatch, backend, dependency):
    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == dependency:
            raise ImportError("not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ValueError, match="uv sync --extra tokenizers"):
        resolve_tokenizer({"backend": backend, "path": str(artifacts[backend])})


def test_tokenizer_loading_uses_the_same_bytes_for_identity_and_runtime(artifacts, monkeypatch):
    from rags import tokenizer_backends

    raw = artifacts["huggingface"].read_bytes()
    reads = []

    def bounded(path, *, max_bytes):
        reads.append((path, max_bytes))
        return raw

    monkeypatch.setattr(tokenizer_backends, "read_regular_bytes", bounded)
    result = resolve_tokenizer({"backend": "huggingface", "path": "selected.json"})
    assert reads == [("selected.json", 64 * 1024 * 1024)]
    assert result.profile["artifact_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result.encode("hello") == [4]


def test_local_search_recounts_bounded_original_slices(artifacts, monkeypatch):
    from rags import tokenizer_backends

    tokenizer = resolve_tokenizer({"backend": "huggingface", "path": str(artifacts["huggingface"])})
    calls = []

    def nonmonotone(text):
        calls.append(text)
        # Simulates normalization/merges that invalidate monotone token counts.
        return 1 if len(text) in {3, 5} else len(text)

    monkeypatch.setattr(tokenizer, "count", nonmonotone)
    monkeypatch.setattr(tokenizer_backends, "MAX_WINDOW_CHARS", 8)
    source = "Café繁體中文🚀" * 1000
    prefix = tokenizer.truncate(source, 2)
    suffix = tokenizer.suffix(source, 2)
    assert source.startswith(prefix) and source.endswith(suffix)
    assert nonmonotone(prefix) <= 2 and nonmonotone(suffix) <= 2
    assert max(map(len, calls)) <= 8
    assert len(calls) <= 12


def test_local_spans_do_not_revalidate_the_entire_remaining_document(artifacts, monkeypatch):
    from rags import tokenizer_backends

    selected = resolve_tokenizer({"backend": "huggingface", "path": str(artifacts["huggingface"])})
    source = "hello world " * 7000
    assert len(source) > tokenizer_backends.MAX_WINDOW_CHARS
    longest_checked_slice = 0
    validate = tokenizer_backends._text

    def checked(text):
        nonlocal longest_checked_slice
        longest_checked_slice = max(longest_checked_slice, len(text))
        validate(text)

    monkeypatch.setattr(tokenizer_backends, "_text", checked)
    spans = selected.spans(source, 20)
    assert "".join(span.text for span in spans) == source
    assert all(span.token_count <= 20 for span in spans)
    assert longest_checked_slice <= tokenizer_backends.MAX_WINDOW_CHARS


def test_huggingface_decode_validates_actual_vocabulary_ids(tmp_path):
    from tokenizers import Tokenizer, models

    path = tmp_path / "sparse-ids.json"
    Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 2}, unk_token="[UNK]")).save(str(path))
    tokenizer = resolve_tokenizer({"backend": "huggingface", "path": str(path)})
    assert tokenizer.encode("hello") == [2]
    assert tokenizer.decode([2]) == "hello"
    with pytest.raises(ValueError, match="token sequence"):
        tokenizer.decode([1])


@pytest.mark.parametrize("config", [
    {"backend": "unicode"}, {"backend": "utf8_bytes"},
    {"backend": "tiktoken", "encoding": "cl100k_base"},
])
@pytest.mark.parametrize("token", [True, False, 1.0, "1", -1, 2 ** 100])
def test_existing_backends_reject_coerced_token_ids(config, token):
    tokenizer = resolve_tokenizer(config)
    with pytest.raises(ValueError):
        tokenizer.decode([token])
