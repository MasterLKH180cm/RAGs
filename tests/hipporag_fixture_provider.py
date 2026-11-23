"""Synthetic HippoRAG generation fixture; embeddings use the actual BGE model."""
import json
import re

from research_fixture_provider import RealBGEFixtureProvider


def hipporag_completion(system: str, user: str) -> str:
    prompt = user.rsplit("\nCurrent request:\n", 1)[-1]
    if system.startswith("Your task is to extract named entities"):
        names = list(dict.fromkeys(re.findall(r"\b[A-Z][a-z]+[A-Z][A-Za-z]*\b", prompt)))
        return json.dumps({"named_entities": names})
    if system.startswith("Your task is to construct an RDF"):
        passage = prompt.split("Paragraph:\n```\n", 1)[1].split("\n```", 1)[0]
        facts = re.findall(r"\b([A-Z][A-Za-z]+) (develops|uses|monitors|contains|supports|operates|supplies) ([A-Z][A-Za-z]+)\b", passage)
        return json.dumps({"triples": [list(fact) for fact in facts]})
    if "[[ ## fact_before_filter ## ]]" in prompt:
        raw = prompt.split("[[ ## fact_before_filter ## ]]\n", 1)[1].split("\n\nRespond with", 1)[0]
        facts = json.loads(raw)["fact"]
        return "[[ ## fact_after_filter ## ]]\n" + json.dumps({"fact": facts[:3]}) + "\n[[ ## completed ## ]]"
    raise AssertionError("Unknown HippoRAG fixture operation")


class HippoRAGFixtureProvider(RealBGEFixtureProvider):
    model_identity = "fixture:hipporag-deterministic-generation-v1"

    def complete(self, system, user):
        self.generations += 1
        return hipporag_completion(system, user)
