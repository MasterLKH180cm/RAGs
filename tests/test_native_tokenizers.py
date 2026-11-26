"""Actual optional tokenizers with tiny local artifacts and no model downloads."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import io
import json
from threading import Barrier
from types import SimpleNamespace

import pytest

from rags.chunking import ChunkConfig, chunk_document
from rags.features import FeatureFlags
from rags.tokenization import ContextItem, pack_context, resolve_tokenizer
import rags.tokenization as tokenization


pytestmark = [pytest.mark.integration, pytest.mark.tokenizers]

SOURCES = [
    "CAFÉ  Alpha\r\nBeta!\tCafe\u0301. " * 15,
    "繁體中文測試。這段沒有以空格分隔，仍然必須完整保留。" * 8,
    "お誕生日おめでとうございます。日本語も保持します。" * 6,
    "👩🏽‍💻 e\u0301 🏳️‍🌈 👨‍👩‍👧‍👦 🇹🇼 🚀 " * 12,
    "[CLS] <s> [SEP] <|endoftext|> literal source " * 8,
    "abcdefghijklmnopqrstuvwxyz0123456789" * 20,
    " \n\t\r\n\x00" * 30,
]


@pytest.fixture(scope="module", params=["wordpiece", "bpe", "sentencepiece"])
def artifact(request, tmp_path_factory):
    directory = tmp_path_factory.mktemp("native-tokenizer-" + request.param)
    if request.param == "sentencepiece":
        sp = pytest.importorskip("sentencepiece", reason="Install the optional tokenizers test extra")
        writer = io.BytesIO()
        sp.SentencePieceTrainer.train(
            sentence_iterator=iter([line.replace("\x00", "") for line in SOURCES]),
            model_writer=writer, model_type="unigram", vocab_size=256, hard_vocab_limit=False,
            character_coverage=1.0, num_threads=1, shuffle_input_sentence=False,
            normalization_rule_name="nmt_nfkc", minloglevel=2,
        )
        raw = writer.getvalue()
        native = sp.SentencePieceProcessor(model_proto=raw)
        backend = "sentencepiece"
        def reference(text):
            return native.encode(text, out_type=int, enable_sampling=False,
                                 add_bos=False, add_eos=False, reverse=False)
    else:
        lib = pytest.importorskip("tokenizers", reason="Install the optional tokenizers test extra")
        from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers

        if request.param == "wordpiece":
            vocab = {word: index for index, word in enumerate(
                ["[UNK]", "[CLS]", "[SEP]", "cafe", "alpha", "beta", ".", "!", "literal", "source"]
            )}
            native = lib.Tokenizer(models.WordPiece(vocab=vocab, unk_token="[UNK]"))
            native.normalizer = normalizers.BertNormalizer(lowercase=True, strip_accents=True)
            native.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
            native.decoder = decoders.WordPiece()
        else:
            native = lib.Tokenizer(models.BPE(unk_token="[UNK]"))
            native.normalizer = normalizers.NFKC()
            native.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            native.decoder = decoders.ByteLevel()
            native.train_from_iterator(SOURCES, trainer=trainers.BpeTrainer(
                vocab_size=450, show_progress=False, special_tokens=["[UNK]", "[CLS]", "[SEP]"],
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            ))
        native.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
        )
        native.enable_padding(length=32)
        native.enable_truncation(max_length=8)
        raw = native.to_str().encode("utf-8")
        native = lib.Tokenizer.from_buffer(raw)
        native.no_padding()
        native.no_truncation()
        backend = "huggingface"
        def reference(text):
            return native.encode(text, add_special_tokens=False).ids
    path = directory / ("tokenizer.model" if backend == "sentencepiece" else "tokenizer.json")
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    return SimpleNamespace(
        backend=backend, raw=raw, path=path, digest=digest, reference=reference,
        config={"backend": backend, "model": "fixture-v1"},
        registry={"fixture-v1": {"backend": backend, "path": str(path), "sha256": digest}},
    )


def test_native_counts_match_selected_artifact_without_padding_or_truncation(artifact):
    tokenizer = resolve_tokenizer(artifact.config, artifacts=artifact.registry)
    for text in [*SOURCES, "", "CAFÉ   Alpha"]:
        expected = artifact.reference(text)
        assert tokenizer.encode(text) == expected
        assert tokenizer.count(text) == len(expected)
    assert tokenizer.count("Alpha Beta " * 100) > 32
    assert tokenizer.profile["artifact_sha256"] == artifact.digest
    assert tokenizer.profile["unit"] == "model_tokens"
    assert tokenizer.profile["count_kind"] == "exact_encoding"
    assert tokenizer.profile["special_tokens"] is False
    assert str(artifact.path) not in json.dumps(tokenizer.profile)
    assert tokenizer.decode(tokenizer.encode("Cafe\u0301   Alpha")) != "Cafe\u0301   Alpha"
    with pytest.raises(ValueError, match="token ID"):
        tokenizer.decode([1_000_000_000])


@pytest.mark.parametrize("text", SOURCES)
def test_native_spans_preserve_every_original_character_despite_normalization(artifact, text):
    tokenizer = resolve_tokenizer(artifact.config, artifacts=artifact.registry)
    spans = tokenizer.spans(text, max_tokens=20, overlap_tokens=4)
    assert spans[0].start == 0 and spans[-1].end == len(text)
    previous = None
    covered_until = 0
    for span in spans:
        assert span.text == text[span.start:span.end]
        assert span.token_count == len(artifact.reference(span.text)) <= 20
        assert span.start <= covered_until
        covered_until = max(covered_until, span.end)
        if previous:
            assert previous.start < span.start and previous.end < span.end
            assert tokenizer.count(text[span.start:previous.end]) <= 4
        previous = span
    assert covered_until == len(text)
    assert text.startswith(tokenizer.truncate(text, 8))
    assert text.endswith(tokenizer.suffix(text, 8))
    assert tokenizer.count(tokenizer.truncate(text, 8)) <= 8
    assert tokenizer.count(tokenizer.suffix(text, 8)) <= 8


def test_native_chunk_plans_and_prompt_packing_keep_exact_source_and_count_labels(artifact):
    tokenizer = resolve_tokenizer(artifact.config, artifacts=artifact.registry)
    text = SOURCES[0] + SOURCES[1]
    plan = chunk_document(text, tokenizer=tokenizer,
                          config=ChunkConfig(strategy="recursive", max_tokens=24, overlap_tokens=4))
    assert plan.profile["tokenizer"]["artifact_sha256"] == artifact.digest
    for chunk in plan.chunks:
        assert chunk.text == text[chunk.start_char:chunk.end_char]
        assert chunk.token_count == len(artifact.reference(chunk.text)) <= 24
    packed = pack_context([ContextItem("original", text)], tokenizer=tokenizer,
                          input_budget=35, fixed_prefix="Question: Alpha\n", fixed_suffix="\nAnswer:")
    assert packed.token_count == len(artifact.reference(packed.text)) <= 35
    assert packed.count_kind == "exact_encoding" and packed.unit == "model_tokens"
    assert text.startswith(packed.items[0].text)


def test_native_truncation_only_encodes_bounded_source_candidates(artifact, monkeypatch):
    tokenizer = resolve_tokenizer(artifact.config, artifacts=artifact.registry)
    original = tokenizer._encode
    examined = []

    def counted(text):
        examined.append(len(text))
        return original(text)

    monkeypatch.setattr(tokenizer, "_encode", counted)
    text = "Alpha 中文 " * 10000
    result = tokenizer.truncate(text, 10)
    assert result and text.startswith(result)
    assert len(artifact.reference(result)) <= 10
    assert examined and max(examined) <= tokenization.MAX_SOURCE_WINDOW_CHARACTERS


def test_artifact_rebinding_or_changed_file_fails_before_cached_execution(artifact, tmp_path):
    target = tmp_path / artifact.path.name
    target.write_bytes(artifact.raw)
    registry = {"fixture-v1": {**artifact.registry["fixture-v1"], "path": str(target)}}
    first = resolve_tokenizer(artifact.config, artifacts=registry)
    selected = {**artifact.config, "artifact_sha256": first.profile["artifact_sha256"]}
    rebound = {"fixture-v1": {**registry["fixture-v1"], "sha256": "b" * 64}}
    with pytest.raises(ValueError, match="identity changed"):
        resolve_tokenizer(selected, artifacts=rebound)
    target.write_bytes(b"replaced even though a native object is cached")
    with pytest.raises(ValueError, match="registered SHA256"):
        resolve_tokenizer(artifact.config, artifacts=registry)


def test_concurrent_resolution_loads_one_verified_native_object(artifact, monkeypatch):
    monkeypatch.setattr(tokenization, "_NATIVE_CACHE", OrderedDict())
    monkeypatch.setattr(tokenization, "_NATIVE_CACHE_BYTES", 0)
    read = tokenization.read_tokenizer_artifact
    load = tokenization._load_native_callbacks
    readers = Barrier(2)
    loads = []

    def synchronized_read(entry):
        raw = read(entry)
        readers.wait(timeout=5)
        return raw

    def counted_load(*args):
        loads.append(args[0])
        return load(*args)

    monkeypatch.setattr(tokenization, "read_tokenizer_artifact", synchronized_read)
    monkeypatch.setattr(tokenization, "_load_native_callbacks", counted_load)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(resolve_tokenizer, artifact.config, artifacts=artifact.registry) for _ in range(2)]
        tokenizers = [future.result(timeout=10) for future in futures]
    assert loads == [artifact.backend]
    assert tokenizers[0].encode("Alpha 中文") == tokenizers[1].encode("Alpha 中文")


def test_research_generation_bridges_receive_registered_tokenizer_artifacts(artifact):
    from rags.research.common import ProviderBridge
    from rags.research.crag_adapter import GenerationBridge

    settings = SimpleNamespace(
        features=FeatureFlags(), embedding_provider="openai", generation_tokenizer_config=artifact.config,
        tokenizer_artifacts=artifact.registry, context_budget=16000, output_budget=100, envelope_budget=64,
    )
    provider = SimpleNamespace(settings=settings, has_generation=True, embedding_model="fixture/encoder",
                               model_identity="fixture/generator", require_generation=lambda: None)
    for bridge in (ProviderBridge(provider), GenerationBridge(provider)):
        assert bridge.tokenizer.profile["artifact_sha256"] == artifact.digest
        assert bridge.tokenizer.count("CAFÉ 中文") == len(artifact.reference("CAFÉ 中文"))


def test_ingestion_reindex_preserves_artifact_pin_and_source_spans(artifact, tmp_path, monkeypatch):
    from rags.config import Settings
    from rags.providers import Providers
    from rags.retrieval import RAGService
    from rags.storage import Store

    monkeypatch.setenv("RAGS_ENV", "test")
    settings = replace(Settings.from_env(), db_path=str(tmp_path / "source.db"),
                       tokenizer_config=artifact.config, tokenizer_artifacts=artifact.registry)
    service = RAGService(Store(settings.db_path), Providers(settings))
    source = SOURCES[0]
    first = service.ingest("tenant", "Original", source, "fixture://source", document_id="document",
                           chunking={"strategy": "fixed", "max_tokens": 24, "overlap_tokens": 4})
    stored = service.store.get_document("tenant", "document")
    assert stored["index_profile"]["tokenizer"]["artifact_sha256"] == artifact.digest
    for chunk in service.store.scope("tenant"):
        assert chunk["content"] == source[chunk["start_char"]:chunk["end_char"]]
        assert chunk["token_count"] == len(artifact.reference(chunk["content"]))
    rebuilt = service.reindex("tenant", "document", first["revision"])
    assert rebuilt["revision"] != first["revision"]
    rebound = {"fixture-v1": {**artifact.registry["fixture-v1"], "sha256": "b" * 64}}
    changed = RAGService(service.store, Providers(replace(settings, tokenizer_artifacts=rebound)))
    with pytest.raises(ValueError, match="identity changed"):
        changed.reindex("tenant", "document", rebuilt["revision"])
    assert service.store.get_document("tenant", "document")["revision"] == rebuilt["revision"]


def test_huggingface_stochastic_dropout_is_rejected_before_counting(tmp_path):
    lib = pytest.importorskip("tokenizers", reason="Install the optional tokenizers test extra")
    from tokenizers import models

    native = lib.Tokenizer(models.BPE(vocab={"a": 0}, merges=[], dropout=0.2))
    raw = native.to_str().encode("utf-8")
    path = tmp_path / "stochastic.json"
    path.write_bytes(raw)
    registry = {"stochastic": {"backend": "huggingface", "path": str(path),
                               "sha256": hashlib.sha256(raw).hexdigest()}}
    with pytest.raises(ValueError, match="deterministic"):
        resolve_tokenizer({"backend": "huggingface", "model": "stochastic"}, artifacts=registry)


@pytest.mark.parametrize("bound", ["entries", "bytes"])
def test_native_lru_evicts_by_count_and_retained_artifact_bytes(tmp_path, monkeypatch, bound):
    lib = pytest.importorskip("tokenizers", reason="Install the optional tokenizers test extra")
    from tokenizers import models

    raw = lib.Tokenizer(models.WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]")).to_str().encode()
    registry = {}
    for index in range(3):
        body = raw + b" " * index
        path = tmp_path / f"tokenizer-{index}.json"
        path.write_bytes(body)
        registry[f"fixture-{index}"] = {"backend": "huggingface", "path": str(path),
                                       "sha256": hashlib.sha256(body).hexdigest()}
    monkeypatch.setattr(tokenization, "_NATIVE_CACHE", OrderedDict())
    monkeypatch.setattr(tokenization, "_NATIVE_CACHE_BYTES", 0)
    monkeypatch.setattr(tokenization, "MAX_NATIVE_TOKENIZERS", 2 if bound == "entries" else 8)
    monkeypatch.setattr(tokenization, "MAX_NATIVE_ARTIFACT_CACHE_BYTES", len(raw) * 2 + 3)
    load = tokenization._load_native_callbacks
    loads = []

    def counted_load(backend, body):
        loads.append(hashlib.sha256(body).hexdigest())
        return load(backend, body)

    monkeypatch.setattr(tokenization, "_load_native_callbacks", counted_load)
    for index in (0, 1, 0, 2, 1):
        resolve_tokenizer({"backend": "huggingface", "model": f"fixture-{index}"}, artifacts=registry)
    assert len(loads) == 4  # Recently reused 0 survives insertion of 2; 1 must be reloaded.
    assert len(tokenization._NATIVE_CACHE) <= 2
    assert tokenization._NATIVE_CACHE_BYTES <= tokenization.MAX_NATIVE_ARTIFACT_CACHE_BYTES
