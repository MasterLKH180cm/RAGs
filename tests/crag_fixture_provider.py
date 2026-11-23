"""Deterministic CRAG generation and source fixtures; not model quality evidence."""
import hashlib
import json
from types import SimpleNamespace

from rags.research_protocol import external_source_id

HAMLET_QUESTION = "Who wrote Hamlet?"
HAMLET_TEXT = "Hamlet is a tragedy written by William Shakespeare."
FRANCE_QUESTION = "What is the capital of France?"
FRANCE_TEXT = "The capital of France is Paris."
IRRELEVANT_TEXT = "Bananas are yellow fruit."
HAMLET_ANSWER = "William Shakespeare wrote Hamlet."
FRANCE_ANSWER = "Paris is the capital of France."


def crag_completion(system: str, user: str) -> str:
    data = json.loads(user)
    if '"query"' in system and "question" in data:
        return json.dumps({"query": data["question"]})
    evidence = data.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise AssertionError("CRAG fixture requires selected source strips")
    question = data["question"]
    expected, answer = ((HAMLET_TEXT, HAMLET_ANSWER) if question == HAMLET_QUESTION
                        else (FRANCE_TEXT, FRANCE_ANSWER) if question == FRANCE_QUESTION
                        else (None, None))
    if expected is None:
        raise AssertionError("CRAG fixture received an unknown question")
    ids = list(dict.fromkeys(item["id"] for item in evidence if item["text"] == expected))
    return json.dumps({"answer": answer if ids else "", "cited_ids": ids, "abstain": not ids})


class CRAGFixtureProvider:
    model_identity = "fixture:crag-deterministic-generation-v1"

    def __init__(self):
        self.settings = SimpleNamespace(generation_tokenizer_config={"backend": "unicode"},
                                        context_budget=16000, output_budget=1024, envelope_budget=256)
        self.calls = 0

    def require_generation(self):
        pass

    def complete(self, system, user):
        self.calls += 1
        return crag_completion(system, user)


def external_france(namespace: str) -> dict:
    url = "https://reference.example.org/france"
    digest = hashlib.sha256(FRANCE_TEXT.encode()).hexdigest()
    return {"id": external_source_id(namespace, url, digest), "url": url, "title": "Synthetic France reference",
            "text": FRANCE_TEXT, "content_sha256": digest, "fetched_at": "2026-11-23T00:00:00+00:00"}
