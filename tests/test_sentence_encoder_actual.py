"""Actual local Sentence Transformers preprocessing; random tiny weights, no quality claim."""
from __future__ import annotations

import copy
import math
import os
from dataclasses import replace

import pytest

from rags.config import Settings
from rags.registry import CapabilityError

pytestmark = [pytest.mark.integration, pytest.mark.external, pytest.mark.encoders]


@pytest.fixture(autouse=True)
def require_sentence_runtime(monkeypatch):
    if os.getenv("RAGS_TEST_SENTENCE_TRANSFORMERS") != "1":
        pytest.skip("Set RAGS_TEST_SENTENCE_TRANSFORMERS=1 with the optional CPU Sentence Transformers runtime")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


def _encoder(directory, *, specials=False):
    import torch
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import Pooling, Transformer
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
    from transformers import GPT2Config, GPT2Model, PreTrainedTokenizerFast

    torch.manual_seed(7)
    vocabulary = {char: index for index, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    native = Tokenizer(models.BPE(vocab=vocabulary, merges=[]))
    native.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    native.decoder = decoders.ByteLevel()
    if specials:
        native.add_special_tokens(["[BOS]", "[EOS]"])
        native.post_processor = processors.TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[("[BOS]", native.token_to_id("[BOS]")), ("[EOS]", native.token_to_id("[EOS]"))])
    maximum = 12 if specials else 8
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=native, pad_token="[PAD]", unk_token="[UNK]",
                                       model_max_length=maximum)
    tokenizer.save_pretrained(directory)
    GPT2Model(GPT2Config(vocab_size=len(tokenizer), n_embd=8, n_layer=1, n_head=1,
                        n_positions=maximum)).save_pretrained(directory, safe_serialization=True)
    transformer = Transformer(str(directory), max_seq_length=maximum,
                              model_kwargs={"local_files_only": True, "use_safetensors": True})
    return SentenceTransformer(modules=[transformer, Pooling(8)], device="cpu")


@pytest.fixture
def actual_encoder(tmp_path, require_sentence_runtime):
    return _encoder(tmp_path / "tiny-transformer")


def _settings(**kwargs):
    base = Settings(db_path=":memory:", keys={}, embedding_provider="sentence_transformers",
                    model_provider="extractive", openai_key="", openai_base_url="https://unused.invalid/v1",
                    embedding_model="local-fixture", chat_model="unused", anthropic_key="", anthropic_model="unused")
    return replace(base, **kwargs)


def _provider(encoder, monkeypatch, **kwargs):
    from rags import providers

    monkeypatch.setattr(providers, "sentence_encoder", lambda *args: encoder)
    return providers.Providers(_settings(**kwargs))


def test_actual_byte_level_whitespace_overflow_rejected_before_inference(actual_encoder, monkeypatch):
    source = "a" + " " * 8
    first = actual_encoder._first_module()
    assert len(first.tokenizer(source, truncation=False)["input_ids"]) == 9
    assert len(first.preprocess([source])["input_ids"][0]) == 8
    adapter = _provider(actual_encoder, monkeypatch)
    monkeypatch.setattr(actual_encoder, "encode_document", lambda *args, **kwargs: pytest.fail("Oversize source reached inference"))
    assert adapter.embedding_token_count(source) == 9
    with pytest.raises(CapabilityError) as error:
        adapter.embedding(source)
    assert error.value.code == "context_budget_exceeded"


def test_actual_role_prefixes_and_special_tokens_match_full_native_input(tmp_path, monkeypatch):
    encoder = _encoder(tmp_path / "specials", specials=True)
    first = encoder._first_module()
    adapter = _provider(encoder, monkeypatch, embedding_query_prefix="query: ", embedding_document_prefix="doc: ")
    calls = []
    original = first.model.forward

    def capture(*args, **kwargs):
        calls.append(kwargs["input_ids"].detach().cpu().tolist()[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(first.model, "forward", capture)
    for query, prefix in ((False, "doc: "), (True, "query: ")):
        expected = first.tokenizer(prefix + "a", add_special_tokens=True, truncation=False)["input_ids"]
        assert adapter.embedding_token_count("a", query=query) == len(expected)
        vector = adapter.query_embedding("a") if query else adapter.embedding("a")
        assert len(vector) == 8 and all(math.isfinite(value) for value in vector)
        assert calls[-1] == expected
    with pytest.raises(CapabilityError) as error:
        adapter.query_embedding("abcdef")
    assert error.value.code == "context_budget_exceeded" and len(calls) == 2


def test_actual_model_processing_override_never_silently_truncates_source(actual_encoder, monkeypatch):
    first = actual_encoder._first_module()
    first.processing_kwargs = {"text": {"max_length": 3, "truncation": True}}
    before = copy.deepcopy(first.processing_kwargs)
    source = "abcde"
    expected = first.tokenizer(source, truncation=False)["input_ids"]
    assert len(expected) == 5 and len(first.preprocess([source])["input_ids"][0]) == 3
    calls = []
    original = first.model.forward

    def capture(*args, **kwargs):
        calls.append(kwargs["input_ids"].detach().cpu().tolist()[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(first.model, "forward", capture)
    adapter = _provider(actual_encoder, monkeypatch)
    try:
        adapter.embedding(source)
    except CapabilityError as error:
        assert error.code in {"requires_provider", "context_budget_exceeded"}
        assert calls == []
    else:
        assert calls == [expected]
    assert first.processing_kwargs == before
    assert first.max_seq_length == 8


def test_actual_counter_does_not_reconfigure_shared_encoder(actual_encoder):
    from rags.embedding_runtime import sentence_window

    first = actual_encoder._first_module()
    source = "a" + " " * 8
    before_processing = copy.deepcopy(first.processing_kwargs)
    before_length = first.max_seq_length
    native_before = first.preprocess([source])["input_ids"].detach().cpu().tolist()
    backend = first.tokenizer.backend_tokenizer
    before_truncation, before_padding = copy.deepcopy(backend.truncation), copy.deepcopy(backend.padding)
    window = sentence_window(actual_encoder, "q: ", "d: ")
    assert window.count("ab", query=True) == 5
    assert window.count("ab", query=False) == 5
    assert backend.truncation == before_truncation
    assert backend.padding == before_padding
    assert first.processing_kwargs == before_processing
    assert first.max_seq_length == before_length
    native_after = first.preprocess([source])["input_ids"].detach().cpu().tolist()
    assert native_before == native_after
