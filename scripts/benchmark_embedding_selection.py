"""Reproducible actual-encoder latency smoke; no retrieval-quality promotion."""
from __future__ import annotations

import argparse
from dataclasses import replace
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import statistics
from time import perf_counter

from rags.config import Settings
from rags.model_catalog import embedding_catalog, get_embedding_preset
from rags.providers import Providers
from rags.registry import CapabilityError


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction + .5))]


def measure(identifier: str, iterations: int) -> dict:
    item = get_embedding_preset(identifier)
    env = item["env"]
    settings = Settings(
        db_path=":memory:", keys={"benchmark-fixture": "benchmark"}, embedding_provider=item["provider"],
        model_provider="extractive", openai_key="", openai_base_url="https://unused.invalid/v1",
        embedding_model=item["model"], chat_model="", anthropic_key="", anthropic_model="",
        fastembed_model=item["model"], embedding_revision=item.get("revision") or "",
        embedding_query_prefix=env.get("RAGS_EMBEDDING_QUERY_PREFIX", ""),
        embedding_document_prefix=env.get("RAGS_EMBEDDING_DOCUMENT_PREFIX", ""),
        embedding_normalize=True if env.get("RAGS_EMBEDDING_NORMALIZE") == "true" else None,
    )
    provider = Providers(settings)
    document = "The Moon orbits the Earth. 月亮繞著地球運行。 Exact source evidence remains unchanged."
    query = "What orbits Earth? 月亮繞著什麼運行？"
    started = perf_counter()
    vector = provider.embedding(document)
    first_ms = (perf_counter() - started) * 1000
    provider.query_embedding(query)  # Query warm-up is excluded from warm samples.
    samples = {"document": [], "query": []}
    for _ in range(iterations):
        for role, function, text in (("document", provider.embedding, document), ("query", provider.query_embedding, query)):
            started = perf_counter()
            function(text)
            samples[role].append((perf_counter() - started) * 1000)
    oversize = " context " * (provider.embedding_token_limit + 20)
    try:
        provider.embedding(oversize)
    except CapabilityError as exc:
        rejected = exc.code == "context_budget_exceeded"
    else:
        rejected = False
    if not rejected:
        raise RuntimeError("Complete-input overflow was not rejected")
    source = provider.embedding_tokenizer()
    spans = source.spans(document * 6, 32, 4)
    if not spans or any((document * 6)[span.start:span.end] != span.text for span in spans):
        raise RuntimeError("Source span smoke failed")
    # A generation counter change must not alter the stored embedding space.
    if Providers(replace(settings, generation_tokenizer_config={"backend": "utf8_bytes"})).embedding_model != provider.embedding_model:
        raise RuntimeError("Generation counter incorrectly changes vector identity")
    return {"id": identifier, "provider": item["provider"], "model": item["model"],
            "revision": item.get("revision"), "trained_model": True, "dimensions": len(vector),
            "embedding_identity": provider.embedding_model, "input_policy": provider.embedding_window_policy,
            "input_limit": provider.embedding_token_limit,
            "document_input_tokens": provider.embedding_token_count(document),
            "query_input_tokens": provider.embedding_token_count(query, query=True),
            "load_and_first_document_ms": round(first_ms, 3),
            "cache_state_before_load": "not_inspected_includes_download_if_required",
            "warm_samples_per_role": iterations,
            "warm": {role: {"median_ms": round(statistics.median(values), 3),
                             "p95_ms": round(percentile(values, .95), 3),
                             "min_ms": round(min(values), 3), "max_ms": round(max(values), 3)}
                     for role, values in samples.items()},
            "overflow_rejected": rejected, "original_source_spans": len(spans),
            "quality_claim": "none_smoke_only_no_admission_promotion"}


def main() -> None:
    choices = [item["id"] for item in embedding_catalog() if item["provider"] in {"fastembed", "sentence_transformers"}]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=choices, default=choices)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not 3 <= args.iterations <= 100:
        parser.error("iterations must be from 3 to 100")
    report = {"schema_version": 1, "workload": "actual_pretrained_embedding_latency_smoke",
              "runtime": {name: version(name) for name in ("rags-platform", "sentence-transformers", "transformers",
                                                          "torch", "tokenizers", "fastembed", "onnxruntime")},
              "machine": {"architecture": platform.machine(), "python": platform.python_version(),
                          "logical_cpus": os.cpu_count(),
                          "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS"), "MKL_NUM_THREADS": os.getenv("MKL_NUM_THREADS")},
              "limitations": ["Small sequential workload, not concurrency/production capacity or retrieval quality.",
                              "First-use timing includes checkpoint loading and any missing downloads.",
                              "Warm timings include adapter preflight; model RAM and process RSS are not measured."],
              "models": [measure(identifier, args.iterations) for identifier in dict.fromkeys(args.models)]}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.report), "models": len(report["models"])}))


if __name__ == "__main__":
    main()
