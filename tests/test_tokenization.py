"""RAG-01/RAG-03: source preservation and complete prompt budgets."""
import pytest

from rags.tokenization import (
    ContextItem, TokenizerConfig, pack_context, resolve_tokenizer,
)


TEXTS = [
    "Traditional English sentences. Another paragraph!\n" * 4,
    "繁體中文測試。這段沒有以空格分隔，仍然必須完整保留。" * 4,
    "お誕生日おめでとうございます。日本語も保持します。" * 3,
    "👩🏽‍💻 e\u0301 🏳️‍🌈 👨‍👩‍👧‍👦 🇹🇼 🚀" * 4,
    "<|endoftext|> literal source <|fim_prefix|> " * 4,
    "abcd" * 200,
    " \n\t" * 40,
]


@pytest.fixture(params=["unicode", "tiktoken"])
def tokenizer(request):
    if request.param == "unicode":
        return resolve_tokenizer()
    return resolve_tokenizer(TokenizerConfig(backend="tiktoken", encoding="cl100k_base"))


@pytest.mark.parametrize("text", TEXTS)
def test_source_spans_are_lossless_and_bounded(tokenizer, text):
    spans = tokenizer.spans(text, 20, 4)
    assert spans[0].start_char == 0
    assert spans[-1].end_char == len(text)
    covered = set()
    previous = None
    for span in spans:
        assert span.text == text[span.start_char:span.end_char]
        assert span.token_count == tokenizer.count(span.text) <= 20
        assert "\ufffd" not in span.text
        covered.update(range(span.start_char, span.end_char))
        if previous:
            assert previous.start_char < span.start_char <= previous.end_char
            overlap = text[span.start_char:previous.end_char]
            assert tokenizer.count(overlap) <= 4
        previous = span
    assert covered == set(range(len(text)))


def test_unicode_backend_is_explicitly_not_a_model_tokenizer():
    tokenizer = resolve_tokenizer()
    assert tokenizer.id == "unicode-codepoints-v1"
    assert tokenizer.unit == "unicode_codepoints"
    assert tokenizer.count_kind == "fallback_units"
    assert tokenizer.count("中文👩🏽‍💻") == len("中文👩🏽‍💻")
    assert tokenizer.decode(tokenizer.encode("e\u0301中文")) == "e\u0301中文"


def test_tiktoken_counts_match_actual_selected_encoding():
    import tiktoken

    tokenizer = resolve_tokenizer({"backend": "tiktoken", "model": "text-embedding-3-small"})
    encoding = tiktoken.encoding_for_model("text-embedding-3-small")
    assert tokenizer.id == "tiktoken:cl100k_base"
    for text in TEXTS:
        expected = encoding.encode(text, disallowed_special=())
        assert tokenizer.encode(text) == expected
        assert tokenizer.count(text) == len(expected)
        assert tokenizer.decode(expected) == text
    assert tokenizer.unit == "model_tokens"
    assert tokenizer.profile["encoding"] == "cl100k_base"


def test_unknown_model_does_not_guess_encoding():
    with pytest.raises(ValueError, match="Unknown tokenizer model"):
        resolve_tokenizer({"backend": "tiktoken", "model": "private-unmapped-model"})
    explicit = resolve_tokenizer({"backend": "tiktoken", "model": "private-unmapped-model",
                                  "encoding": "cl100k_base"})
    assert explicit.id == "tiktoken:cl100k_base"
    with pytest.raises(ValueError, match="conflicts"):
        resolve_tokenizer({"backend": "tiktoken", "model": "text-embedding-3-small",
                           "encoding": "o200k_base"})


def test_impossible_unicode_boundary_budget_fails_without_corruption():
    tokenizer = resolve_tokenizer({"backend": "tiktoken", "encoding": "cl100k_base"})
    assert tokenizer.count("🚀") > 1
    assert tokenizer.truncate("🚀", 1) == ""
    with pytest.raises(ValueError, match="too small"):
        tokenizer.spans("🚀", 1)


def test_bpe_overlap_is_recounted_after_boundary_adjustment():
    tokenizer = resolve_tokenizer({"backend": "tiktoken", "encoding": "cl100k_base"})
    source = " dé🚀繁體中文"
    assert tokenizer.count(" dé🚀") < tokenizer.count("dé🚀")
    spans = tokenizer.spans(source, 5, 4)
    for previous, current in zip(spans, spans[1:]):
        assert current.end_char > previous.end_char
        assert tokenizer.count(source[current.start_char:previous.end_char]) <= 4
    assert spans[-1].end_char == len(source)


@pytest.mark.parametrize("config", [
    {"backend": "unknown"}, {"backend": "tiktoken"},
    {"backend": "unicode", "model": "gpt-4"}, {"invented": True},
])
def test_invalid_tokenizer_configuration_is_rejected(config):
    with pytest.raises(ValueError):
        resolve_tokenizer(config)


def test_invalid_unicode_and_overlap_are_rejected(tokenizer):
    with pytest.raises(ValueError, match="surrogate"):
        tokenizer.count("invalid\ud800")
    with pytest.raises(ValueError, match="overlap"):
        tokenizer.spans("text", 3, 3)
    with pytest.raises(ValueError):
        tokenizer.spans("text", 0)
    with pytest.raises(ValueError):
        tokenizer.truncate("text", True)
    assert tokenizer.spans("", 3) == []
    assert tokenizer.truncate("text", 0) == ""
    assert tokenizer.suffix("text", 0) == ""


def test_prompt_packing_counts_headers_fixed_text_and_delimiters(tokenizer):
    prefix, suffix = "SYSTEM: Evidence only.\nQUESTION: 何時到期？\nEVIDENCE:\n", "\nANSWER:"
    fixed = tokenizer.count(prefix + suffix)
    packed = pack_context(
        [ContextItem("source-a", "中文與 emoji 🚀。 " * 30, "[source-a] Session policy"),
         ContextItem("source-b", "Another source that may not fit.")],
        tokenizer=tokenizer, input_budget=fixed + 35, fixed_prefix=prefix, fixed_suffix=suffix,
    )
    assert packed.text.startswith(prefix) and packed.text.endswith(suffix)
    assert packed.text == prefix + packed.context + suffix
    assert tokenizer.count(packed.text) == packed.token_count <= fixed + 35
    assert packed.included_ids == [item.id for item in packed.items]
    assert set(packed.included_ids + packed.dropped_ids) == {"source-a", "source-b"}
    assert "source-a" in packed.truncated_ids
    assert ("中文與 emoji 🚀。 " * 30).startswith(packed.items[0].text)
    assert packed.count_kind == tokenizer.count_kind


def test_fixed_prompt_is_never_silently_removed():
    with pytest.raises(ValueError, match="fixed prompt"):
        pack_context([], tokenizer=resolve_tokenizer(), input_budget=2, fixed_prefix="Question")
    empty = pack_context([], tokenizer=resolve_tokenizer(), input_budget=10, fixed_prefix="Question")
    assert empty.text == "Question"
    assert empty.token_count == 8


def test_external_counter_is_used_and_provenance_is_explicit():
    # Deterministic callback represents an externally counted request envelope.
    counted = []

    def provider_counter(text):
        counted.append(text)
        return len(text) + 7

    packed = pack_context([{"chunk_id": "a", "text": "abcdefghijklmnop"}],
                          tokenizer=resolve_tokenizer(), input_budget=20,
                          counter=provider_counter, counter_id="fixture:request-count")
    assert counted[-1] == packed.text
    assert packed.token_count == len(packed.text) + 7 <= 20
    assert packed.count_kind == "provider_count"
    assert packed.counter_id == "fixture:request-count"
    assert packed.truncated_ids == ["a"]


def test_duplicate_context_ids_and_unlabelled_counters_fail():
    tokenizer = resolve_tokenizer()
    with pytest.raises(ValueError, match="unique"):
        pack_context([ContextItem("same", "one"), ContextItem("same", "two")],
                     tokenizer=tokenizer, input_budget=100)
    with pytest.raises(ValueError, match="counter_id"):
        pack_context([], tokenizer=tokenizer, input_budget=100, counter=len)
