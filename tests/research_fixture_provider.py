"""Explicit test-only generation fixtures plus actual trained BGE CPU vectors.

This module is never imported by a production worker. Deterministic generation
proves upstream algorithm execution and lineage, not language-model quality.
"""
import json
import os
import re
from types import SimpleNamespace

PASSAGES = [
    {"id": "water-1", "text": "OrionLab develops HarborSensor for monitoring river water pollution."},
    {"id": "water-2", "text": "HarborSensor monitors RiverDelta and detects chemical contamination in water."},
    {"id": "water-3", "text": "RiverDelta supplies CoastalTown with drinking water from a protected reservoir."},
    {"id": "water-4", "text": "CoastalTown uses HarborSensor to measure river water quality every morning."},
    {"id": "space-1", "text": "LunarTeam operates MoonRover to explore rocks on the lunar surface."},
    {"id": "space-2", "text": "MoonRover contains SolarArray for generating electricity during daylight."},
    {"id": "space-3", "text": "SolarArray supports ResearchStation with renewable solar power in space."},
    {"id": "space-4", "text": "ResearchStation uses MoonRover for collecting geological samples on the moon."},
    {"id": "health-1", "text": "CedarClinic operates HeartMonitor for observing patient cardiac rhythms."},
    {"id": "health-2", "text": "HeartMonitor supports CareTeam in detecting abnormal heart rates."},
    {"id": "health-3", "text": "CareTeam uses HealthArchive for keeping medical appointment records."},
    {"id": "health-4", "text": "HealthArchive supports CedarClinic with patient scheduling and treatment history."},
    {"id": "library-1", "text": "MapleLibrary operates BookCatalog for lending books to local readers."},
    {"id": "library-2", "text": "BookCatalog contains PoetryCollection with classic poems and literary essays."},
    {"id": "library-3", "text": "PoetryCollection supports ReadingClub with materials for weekly discussions."},
    {"id": "library-4", "text": "ReadingClub uses MapleLibrary for public lectures about literature."},
]
QUESTION = "How does OrionLab use HarborSensor to monitor river water pollution?"


def fixture_completion(system: str, user: str) -> str:
    """Match only fixture facts inside upstream prompts; never inspect labels."""
    if "expert keyword extractor" in user:
        query = user.split("---Real Data---\nUser Query: ", 1)[1].split("\n\n---Output---", 1)[0]
        names = re.findall(r"\b[A-Z][a-z]+[A-Z][A-Za-z]*\b", query)
        return json.dumps({"high_level_keywords": ["river water pollution"] if "water" in query else [query],
                           "low_level_keywords": names})
    if user.startswith("---Task---\nExtract entities and relationships"):
        source = user.split("---Input Text---\n```\n", 1)[1].rsplit("\n```\n\n---Output---", 1)[0]
        facts = re.findall(r"\b([A-Z][A-Za-z]+) (develops|uses|monitors|contains|supports|operates|supplies) ([A-Z][A-Za-z]+)\b", source)
        entities = sorted({name for subject, _, obj in facts for name in (subject, obj)})
        if not facts:
            raise AssertionError("The fixture generator received an unknown source shape")
        rows = [f"entity<|#|>{name}<|#|>Organization<|#|>{source}" for name in entities]
        rows += [f"relation<|#|>{subject}<|#|>{obj}<|#|>{verb}<|#|>{source}"
                 for subject, verb, obj in facts]
        return "\n".join(rows + ["<|COMPLETE|>"])
    # RAPTOR cluster summaries and LightRAG entity-description summaries use
    # deterministic extractive text. They are explicitly not a trained LLM.
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")
    result = " ".join(user.split())
    tokens = encoding.encode(result, disallowed_special=())
    return encoding.decode(tokens[:96])


class RealBGEFixtureProvider:
    has_generation = True
    embedding_model = "fastembed:BAAI/bge-small-en-v1.5"
    model_identity = "fixture:deterministic-generation-v1"

    def __init__(self):
        from rags.research.common import disable_encoder_telemetry, fastembed_token_window
        disable_encoder_telemetry()
        from fastembed import TextEmbedding
        self.settings = SimpleNamespace(
            embedding_provider="fastembed", generation_tokenizer_config={"backend": "tiktoken", "encoding": "cl100k_base"},
            context_budget=16000, output_budget=1024, envelope_budget=256,
        )
        self.encoder = TextEmbedding(
            "BAAI/bge-small-en-v1.5", cache_dir=os.environ.get("RAGS_MODEL_CACHE"), threads=2,
        )
        self.embedding_token_count, self.embedding_token_limit = fastembed_token_window(self.encoder)
        self.queries = 0
        self.documents = 0
        self.generations = 0

    def embedding(self, text):
        self.documents += 1
        return next(self.encoder.embed([text])).tolist()

    def query_embedding(self, text):
        self.queries += 1
        return next(self.encoder.query_embed(text)).tolist()

    def complete(self, system, user):
        self.generations += 1
        return fixture_completion(system, user)


def record_actual_report(engine: str, result: dict, provider: RealBGEFixtureProvider) -> None:
    """Optional, reproducible test artifact; contains synthetic data metadata only."""
    destination = os.getenv("RAGS_RESEARCH_REPORT_PATH")
    if not destination:
        return
    import hashlib
    import importlib.metadata
    import platform
    from datetime import UTC, datetime
    from pathlib import Path
    root = Path(provider.encoder.model._model_dir)
    artifacts = {}
    for name in [provider.encoder.model.model_description.model_file, "tokenizer.json", "tokenizer_config.json"]:
        path = root / name
        artifacts[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    libraries = ["fastembed", "onnxruntime", "tokenizers", "numpy", "tiktoken"]
    libraries += ["lightrag-hku", "networkx", "nano-vectordb"] if engine == "lightrag" else ["umap-learn", "scikit-learn", "torch"]
    report = {
        "schema": "actual-upstream-verification-v1", "engine": engine,
        "verified_at": datetime.now(UTC).isoformat(), "python": platform.python_version(),
        "verification": "actual_upstream_algorithm_real_bge_embeddings_deterministic_generation_fixture",
        "not_verified": ["trained_generation_quality", "paper_benchmark_reproduction", "paid_provider_quality"],
        "runtime_versions": {name: importlib.metadata.version(name) for name in libraries},
        "embedding_artifacts_sha256": artifacts, "result": result,
    }
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
