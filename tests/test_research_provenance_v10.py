"""Independent regressions for external-source identity and literal URL binding."""
import hashlib

import pytest

from rags.research_protocol import (
    UPSTREAMS,
    ResearchRequest,
    external_source_id,
    validate_external_sources,
    validate_result,
)

pytestmark = pytest.mark.unit

NAMESPACE = "provenance-review-tenant"
HOST = "approved.example.org"
TEXT = "A synthetic approved external source snapshot."


def snapshot(url="https://approved.example.org/reference?section=one%20two"):
    digest = hashlib.sha256(TEXT.encode("utf-8")).hexdigest()
    return {"id": external_source_id(NAMESPACE, url, digest), "url": url,
            "title": "Synthetic source", "text": TEXT, "content_sha256": digest,
            "fetched_at": "2026-11-23T12:00:00+00:00"}


def request(passages):
    return ResearchRequest(protocol="authorized-evidence-v2", engine="crag", namespace=NAMESPACE,
                           question="What does the approved source say?", top_k=3,
                           passages=passages, external_hosts=[HOST])


def response(source, source_count):
    return {"hits": [{"chunk_id": source["id"], "score": 0.5}],
            "grounded_answer": {"text": "The approved source supplies a snapshot.", "cited_ids": [source["id"]]},
            "external_sources": [source], "implementation": UPSTREAMS["crag"],
            "diagnostics": {"source_count": source_count, "engine_executed": True}, "usage": None}


@pytest.mark.parametrize("internal_text", [TEXT, "An entirely different authorized internal source."])
def test_external_identity_cannot_replace_an_original_internal_passage(internal_text):
    source = snapshot()
    contract = request([{"id": source["id"], "text": internal_text}])
    # Reject during acquisition intake, before the evaluator/generator sees two
    # different provenance records sharing one ID, and at the public result edge.
    with pytest.raises(ValueError):
        validate_external_sources([source], contract)
    with pytest.raises(ValueError):
        validate_result(response(source, 1), contract)


@pytest.mark.parametrize("url", [
    " https://approved.example.org/reference",
    "\thttps://approved.example.org/reference",
    "\nhttps://approved.example.org/reference",
    "\x01https://approved.example.org/reference",
    "https://approved.example.org/part\tother",
    "https://approved.example.org/part\nother",
    "https://approved.example.org/part\rother",
    "https://approved.example.org/part\x00other",
    "https://approved.example.org/part\x7fother",
    "https://approved.example.org/part\\other",
])
def test_literal_external_url_cannot_be_normalized_into_a_different_approved_url(url):
    source = snapshot(url)
    # Compute a matching ID from the malformed literal, so failure must be URL
    # validation rather than an incidental digest/identifier mismatch.
    contract = request([])
    with pytest.raises(ValueError):
        validate_result(response(source, 0), contract)


def test_valid_approved_snapshot_preserves_exact_url_text_hash_and_tenant_identity():
    source = snapshot()
    contract = request([{"id": "original-internal", "text": "An independent internal source."}])
    result = validate_result(response(source, 1), contract)
    assert result["external_sources"] == [source]
    assert result["hits"] == [{"chunk_id": source["id"], "score": 0.5}]
    assert result["grounded_answer"]["cited_ids"] == [source["id"]]
    assert source["id"] != external_source_id("another-tenant", source["url"], source["content_sha256"])
