"""GraphRAG boundary tests run without its optional dependency environment."""
import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest

from rags.registry import CapabilityError
from rags.research.graphrag_adapter import (
    PIPELINE_STAGES,
    _cleanup,
    _messages,
    _ProviderState,
    group_sources,
    project_sources,
    search,
    validate_extraction,
    validate_pipeline,
    validate_report,
)

pytestmark = pytest.mark.unit


def lineage():
    sources = {"original-a": "Original source one.", "duplicate-a": "Original source one.",
               "original-b": "Original source two."}
    docs, groups = group_sources(sources)
    units = [{"id": "unit-" + str(index), "human_readable_id": index,
              "document_id": doc["id"], "text": doc["text"]} for index, doc in enumerate(docs)]
    selected = [{"id": str(index), "text": doc["text"]} for index, doc in enumerate(docs)]
    return selected, units, docs, groups, sources


def test_identical_content_preserves_each_explicit_original_identifier():
    selected, units, docs, groups, sources = lineage()
    assert len(docs) == 2
    assert project_sources(selected, units, docs, groups, sources, 5) == [
        {"chunk_id": "duplicate-a", "score": 1.0}, {"chunk_id": "original-a", "score": 1.0},
        {"chunk_id": "original-b", "score": 0.5},
    ]
    assert len(project_sources(selected, units, docs, groups, sources, 1)) == 1
    assert project_sources([], units, docs, groups, sources, 5) == []


@pytest.mark.parametrize("mutation", [
    "unknown_selected", "changed_selected", "duplicate_selected", "selected_number", "malformed_selected",
    "unknown_document", "changed_document", "duplicate_document", "missing_document", "malformed_document",
    "unknown_unit_document", "changed_unit", "duplicate_unit", "missing_unit", "unit_short_bool",
    "unit_short_negative", "unit_id_duplicate", "malformed_unit", "lost_original", "changed_original",
])
def test_projection_rejects_unverified_or_changed_source_lineage(mutation):
    selected, units, docs, groups, sources = deepcopy(lineage())
    if mutation == "unknown_selected":
        selected[0]["id"] = "foreign"
    elif mutation == "changed_selected":
        selected[0]["text"] = "Generated report."
    elif mutation == "duplicate_selected":
        selected.append(selected[0])
    elif mutation == "selected_number":
        selected[0]["id"] = 0
    elif mutation == "malformed_selected":
        selected[0] = None
    elif mutation == "unknown_document":
        docs[0]["id"] = "foreign"
    elif mutation == "changed_document":
        docs[0]["text"] = "Changed source"
    elif mutation == "duplicate_document":
        docs[1] = docs[0]
    elif mutation == "missing_document":
        docs.pop()
    elif mutation == "malformed_document":
        docs[0] = None
    elif mutation == "unknown_unit_document":
        units[0]["document_id"] = "foreign"
    elif mutation == "changed_unit":
        units[0]["text"] = "Changed source"
    elif mutation == "duplicate_unit":
        units[1] = units[0]
    elif mutation == "missing_unit":
        units.pop()
    elif mutation == "unit_short_bool":
        units[0]["human_readable_id"] = False
    elif mutation == "unit_short_negative":
        units[0]["human_readable_id"] = -1
    elif mutation == "unit_id_duplicate":
        units[1]["id"] = units[0]["id"]
    elif mutation == "malformed_unit":
        units[0] = None
    elif mutation == "lost_original":
        sources.pop("original-a")
    elif mutation == "changed_original":
        sources["original-a"] = "Another tenant's text"
    with pytest.raises(CapabilityError, match="GraphRAG") as error:
        project_sources(selected, units, docs, groups, sources, 5)
    assert error.value.code == "invalid_source_lineage"


GOOD_EXTRACTION = ('("entity"<|>A<|>ORGANIZATION<|>Entity A)##'
                   '("entity"<|>B<|>ORGANIZATION<|>Entity B)##'
                   '("relationship"<|>A<|>B<|>A uses B<|>8)\n<|COMPLETE|>')


def test_upstream_extraction_format_and_valid_finite_relationships():
    assert validate_extraction(GOOD_EXTRACTION) == GOOD_EXTRACTION.replace("\n<|COMPLETE|>", "##<|COMPLETE|>")


def test_extraction_returns_canonical_record_types_instead_of_unparsed_model_whitespace():
    spaced = GOOD_EXTRACTION.replace('("entity"', '( "entity" ').replace('("relationship"', '( "relationship" ')
    spaced = spaced.replace("<|>", " <|> ")
    assert validate_extraction(spaced) == validate_extraction(GOOD_EXTRACTION)


def test_html_and_control_normalization_use_the_actual_upstream_entity_identity():
    same_identity = GOOD_EXTRACTION.replace("<|>A<|>", "<|>&#65;\x01<|>", 1)
    assert validate_extraction(same_identity).endswith("##<|COMPLETE|>")


@pytest.mark.parametrize("value", [
    "not extraction", "<|COMPLETE|>", 'unparsed##<|COMPLETE|>',
    GOOD_EXTRACTION.replace("<|>8)", "<|>NaN)"),
    GOOD_EXTRACTION.replace("<|>8)", "<|>11)"),
    GOOD_EXTRACTION.replace("<|>8)", "<|>wrong)"),
    GOOD_EXTRACTION.replace("<|>A<|>B<|>", "<|>A<|>C<|>"),
    GOOD_EXTRACTION.replace("Entity A", ""),
    GOOD_EXTRACTION.replace("Entity A", "\x01"),
    GOOD_EXTRACTION.replace("<|>A<|>", '<|>""<|>', 1),
    GOOD_EXTRACTION.replace("<|>A<|>", '<|>"A"<|>', 1),
    GOOD_EXTRACTION.replace("<|>B<|>", '<|>"B"<|>', 1),
    GOOD_EXTRACTION.replace("<|>A<|>", "<|>\x01<|>", 1),
    GOOD_EXTRACTION.replace("A uses B", "\x01"),
    GOOD_EXTRACTION.replace('"entity"', '"other"'),
    '##'.join(['("entity"<|>A<|>PERSON<|>A)'] * 65) + '<|COMPLETE|>',
])
def test_malformed_extraction_cannot_be_silently_dropped(value):
    with pytest.raises(CapabilityError) as error:
        validate_extraction(value)
    assert error.value.code == "invalid_model_output"


@pytest.mark.parametrize("mutation", ["error", "partial", "wrong_order", "provider_caught", "invalid"])
def test_partial_pipeline_and_swallowed_provider_errors_are_not_success(mutation):
    results = [SimpleNamespace(workflow=name, error=None) for name in PIPELINE_STAGES]
    errors = []
    validate_pipeline(results, errors)
    if mutation == "error":
        results[-1].error = ValueError("private provider failure")
    elif mutation == "partial":
        results.pop()
    elif mutation == "wrong_order":
        results.reverse()
    elif mutation == "provider_caught":
        errors = ["CapabilityError"]
    else:
        results = None
    with pytest.raises(CapabilityError, match="did not complete"):
        validate_pipeline(results, errors)


def test_provider_failure_latch_does_not_expose_error_text_or_continue():
    state = _ProviderState(None)
    assert state.call(lambda: "answer") == "answer"
    def failure():
        raise ValueError("private-source-do-not-copy")
    with pytest.raises(ValueError):
        state.call(failure)
    assert state.errors == ["ValueError"]
    with pytest.raises(CapabilityError, match="previously failed"):
        state.call(lambda: pytest.fail("Failed upstream must not continue provider calls"))


@pytest.mark.parametrize("code,expected,status", [
    ("context_budget_exceeded", "context_budget_exceeded", 422),
    ("research_limit_exceeded", "research_limit_exceeded", 422),
    ("invalid_model_output", "invalid_upstream_output", 502),
    ("invalid_embedding", "invalid_upstream_output", 502),
    ("unknown-private-code", "invalid_upstream_output", 502),
])
def test_swallowed_provider_budget_failures_keep_the_safe_public_status(code, expected, status):
    state = _ProviderState(None)

    def failure():
        raise CapabilityError(code, "private-source-or-credential-do-not-copy", 422)

    with pytest.raises(CapabilityError):
        state.call(failure)
    for operation in (state.raise_for_failure,
                      lambda: state.call(lambda: pytest.fail("Failed provider must not continue"))):
        with pytest.raises(CapabilityError) as error:
            operation()
        assert error.value.code == expected and error.value.status_code == status
        assert "private" not in error.value.reason


@pytest.mark.parametrize("messages", [None, [], [{"role": "tool", "content": "x"}],
                                      [{"role": "user", "content": []}],
                                      [{"role": "assistant", "content": "x"}],
                                      [{"role": "user", "content": "x", "url": "https://unapproved.test"}]])
def test_graph_provider_rejects_unsupported_message_shapes(messages):
    with pytest.raises(CapabilityError):
        _messages(messages)


def test_messages_preserve_system_and_history_without_slicing_json():
    assert _messages("text") == ("", "text", [])
    history = [{"role": "assistant", "content": "complete original answer"}]
    assert _messages([{"role": "system", "content": "rules"}, *history,
                      {"role": "user", "content": "question"}]) == ("rules", "question", history)


@pytest.mark.parametrize("field,value", [("title", ""), ("summary", None), ("rating", True),
                                        ("rating", 11), ("rating", float("nan")),
                                        ("findings", []), ("findings", [{"summary": "x", "explanation": ""}]),
                                        ("findings", [{"summary": "x", "explanation": "x", "foreign": True}])])
def test_community_report_cannot_pass_with_empty_or_unbounded_content(field, value):
    report = {"title": "Community", "summary": "Summary", "rating": 1.0, "rating_explanation": "Explanation",
              "findings": [{"summary": "Finding", "explanation": "Details"}]}
    validate_report(report)
    report[field] = value
    with pytest.raises(CapabilityError):
        validate_report(report)


def test_demo_provider_is_rejected_before_optional_import_or_directory_creation(tmp_path):
    provider = SimpleNamespace(settings=SimpleNamespace(embedding_provider="local"),
                               embedding_model="hash-v1", model_identity="fixture", has_generation=True)
    with pytest.raises(CapabilityError) as error:
        asyncio.run(search("question", [{"id": "a", "text": "source"}], 1, tmp_path, provider))
    assert error.value.code == "requires_provider"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure", [False, True])
def test_cleanup_closes_both_vector_handles_before_removing_private_directory(tmp_path, failure):
    directory = tmp_path / "private-index"
    directory.mkdir()
    (directory / "source-data").write_text("private synthetic source")
    closed = []
    def close(kind):
        assert directory.is_dir()
        closed.append(kind)
        if failure and kind == "table":
            raise OSError("resource failure")
    state = _ProviderState(None)
    store = SimpleNamespace(document_collection=SimpleNamespace(_table=SimpleNamespace(close=lambda: close("table"))),
                            db_connection=SimpleNamespace(_conn=SimpleNamespace(close=lambda: close("connection"))))
    state.stores.append(store)
    if failure:
        with pytest.raises(CapabilityError, match="did not close"):
            _cleanup(state, directory)
    else:
        _cleanup(state, directory)
    assert closed == ["table", "connection"] and not directory.exists() and not state.stores
    assert store.document_collection is store.db_connection is None
