"""Authenticated public selection metadata without loading model resources."""
from importlib.metadata import PackageNotFoundError, version

from rags.model_catalog import embedding_catalog, tokenizer_catalog

_DISTRIBUTIONS = {"fastembed": "fastembed", "sentence_transformers": "sentence-transformers",
                  "huggingface": "tokenizers", "sentencepiece": "sentencepiece", "tiktoken": "tiktoken"}


def dependency_state(runtime: str) -> dict:
    distribution = _DISTRIBUTIONS.get(runtime)
    if distribution is None:
        return {"required": None, "installed": True, "version": None}
    try:
        installed = version(distribution)
    except PackageNotFoundError:
        installed = None
    return {"required": distribution, "installed": installed is not None, "version": installed}


def tokenizer_selection(config: dict, settings) -> dict:
    backend = config.get("backend", "unicode")
    native = backend in {"huggingface", "sentencepiece"}
    alias = config.get("model") if native else None
    registered = settings.tokenizer_artifacts.get(alias) if alias else None
    return {"backend": backend, "id": config.get("model") or config.get("encoding") or backend,
            "unit": "unicode_codepoints" if backend == "unicode" else "utf8_bytes"
                    if backend == "utf8_bytes" else "model_tokens",
            "count_kind": "fallback_units" if backend == "unicode" else
                          "conservative_bytes" if backend == "utf8_bytes" else "exact_encoding",
            "artifact_registered": bool(registered) if native else None,
            "artifact_sha256": (config.get("artifact_sha256") or registered.get("sha256")) if registered else None,
            "enabled": settings.features.enabled("tokenizer." + backend),
            "dependency": dependency_state(backend), "runtime_validation": "not_run"}


def configured_models(settings) -> dict:
    provider = settings.embedding_provider
    model = ("offline-hash-v1" if provider == "local" else settings.fastembed_model
             if provider == "fastembed" else settings.embedding_model)
    policy = ("offline_demo" if provider == "local" else "server_reject"
              if provider == "ollama" and settings.embedding_tokenizer_config is None else
              "resolve_alias_at_runtime" if provider == "router" and settings.embedding_tokenizer_config is None else
              "requires_counter" if (provider == "openai" and model not in {
                  "text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"})
              and settings.embedding_tokenizer_config is None else "exact")
    return {"embedding": {"provider": provider, "model": model, "revision": settings.embedding_revision or None,
                          "dimensions": settings.embedding_dimensions, "input_policy": policy,
                          "readiness": {"enabled": settings.features.enabled("embedding." + provider),
                                        "dependency": dependency_state(provider),
                                        "runtime_validation": "not_run", "weights_checked": False}},
            "source_tokenizer": tokenizer_selection(settings.tokenizer_config, settings),
            "generation_tokenizer": tokenizer_selection(settings.generation_tokenizer_config, settings)}


def model_discovery(settings) -> dict:
    return {"embeddings": [{**item, "enabled": settings.features.enabled("embedding." + item["provider"]),
                            "dependency": dependency_state(item["provider"]), "runtime_validation": "not_run"}
                           for item in embedding_catalog()],
            "tokenizers": tokenizer_catalog(), "registered_artifacts": tokenizer_discovery(settings)["registered_artifacts"],
            "configured": configured_models(settings),
            "approved_profiles": [{"id": alias, **profile.public(),
                                   "enabled": settings.features.enabled("embedding." + profile.provider),
                                   "dependency": dependency_state(profile.provider), "runtime_validation": "not_run"}
                                  for alias, profile in sorted(settings.embedding_profiles.items())]}


def tokenizer_discovery(settings) -> dict:
    return {"tokenizers": tokenizer_catalog(),
            "registered_artifacts": [{"id": alias, "backend": entry["backend"], "sha256": entry["sha256"],
                                      "enabled": settings.features.enabled("tokenizer." + entry["backend"]),
                                      "dependency": dependency_state(entry["backend"]), "runtime_validation": "not_run"}
                                     for alias, entry in sorted(settings.tokenizer_artifacts.items())],
            "source": tokenizer_selection(settings.tokenizer_config, settings),
            "generation": tokenizer_selection(settings.generation_tokenizer_config, settings)}
