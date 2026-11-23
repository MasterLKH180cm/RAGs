"""Test-only GraphRAG generation, with actual BGE embeddings inherited below."""
import ast
import json
import re

from research_fixture_provider import RealBGEFixtureProvider


def fixture_completion(system: str, user: str) -> str:
    """Generate only from supplied synthetic source/context, never answer labels."""
    if "-Real Data-" in user and "Entity_types:" in user:
        source = user.rsplit("Text: ", 1)[1].rsplit("\n######################\nOutput:", 1)[0]
        facts = re.findall(r"\b([A-Z][A-Za-z]+) (develops|uses|monitors|contains|supports|operates|supplies) ([A-Z][A-Za-z]+)\b", source)
        if not facts:
            raise AssertionError("Unknown GraphRAG fixture source")
        names = sorted({name for left, _, right in facts for name in (left, right)})
        rows = [f'("entity"<|>{name}<|>ORGANIZATION<|>{source})' for name in names]
        rows.extend(f'("relationship"<|>{left}<|>{right}<|>{source}<|>8)' for left, _, right in facts)
        return "\n##\n".join(rows) + "\n<|COMPLETE|>"
    if "Description List:" in user:
        descriptions = ast.literal_eval(user.split("Description List:", 1)[1].split("\n#######", 1)[0].strip())
        if not isinstance(descriptions, list) or any(not isinstance(item, str) for item in descriptions):
            raise AssertionError("Unknown GraphRAG description fixture")
        return " ".join(dict.fromkeys(descriptions))[:1000]
    if "# Real Data" in user and "# Report Structure" in user:
        context = user.split("# Real Data", 1)[1].split("# Output", 1)[0]
        names = sorted(set(re.findall(r"\b[A-Z][A-Z]+\b", context)))[:8]
        text = "Community connects " + ", ".join(names) + "."
        return json.dumps({"title": "Synthetic source community", "summary": text,
                           "rating": 1.0, "rating_explanation": "Synthetic test data only.",
                           "findings": [{"summary": "Source relationship", "explanation": text}]})
    raise AssertionError("Unknown GraphRAG generation request")


class GraphRAGFixtureProvider(RealBGEFixtureProvider):
    model_identity = "fixture:graphrag-deterministic-generation-v1"

    def complete(self, system, user):
        self.generations += 1
        return fixture_completion(system, user)
