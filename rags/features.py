"""Operator feature permissions, independent of provider/backend availability."""
from dataclasses import dataclass
import json
import os
from pathlib import Path

from rags.chunking import STRATEGIES
from rags.registry import CapabilityError, MODEL_MODES, MODES, UPSTREAM_IDS


MANIFEST = {
    "ingestion": "Create or update source documents",
    "documents": "Read source documents and revision history",
    "uploads": "Ingest supported uploaded files",
    "reindex": "Rebuild existing document or image indexes",
    "deletion": "Delete source documents or image assets",
    "query": "Search the knowledge service",
    "api_docs": "Expose OpenAPI and interactive API documentation",
    "mcp": "Expose the shipped HTTP-backed MCP tools",
    "model_generation": "Call a configured generation model",
    "federation": "Query configured tenant-authorized remote knowledge services",
    "research": "Call configured external research engines",
    "image_captioning": "Generate and ingest image descriptions",
    "multimodal": "Store and retrieve original image assets",
    "original_images": "Read original image bytes through the asset API",
    "admission_scoring": "Use an operator-selected evidence scoring profile",
    **{f"embedding.{name}": f"Use the {name} embedding provider" for name in ("local", "openai", "fastembed")},
    **{f"index.{name}": f"Build the {name} index" for name in ("hierarchy", "graph")},
    **{f"tokenizer.{name}": f"Use the {name} text counter" for name in ("unicode", "tiktoken")},
    **{f"chunking.{name}": f"Use the {name} chunker" for name in STRATEGIES},
    **{f"strategy.{name}": f"Execute the {name} retrieval strategy" for name in MODES},
}

_HYBRID_MODES = {"fusion", "contextual", "parent_child", "corrective", "agentic", "self_reflective",
                 "structured", "temporal", "caption", "federated"}


def strategy_features(mode: str) -> tuple[str, ...]:
    """Include reusable retrieval dependencies, before any model/network call."""
    if mode not in MODES:
        raise ValueError("Unknown retrieval strategy")
    names = {"query", "strategy." + mode}
    if mode in _HYBRID_MODES:
        names.add("strategy.hybrid")
    if mode == "hybrid" or "strategy.hybrid" in names:
        names.update(("strategy.sparse", "strategy.dense"))
    if mode in {"naive", "hyde", "hierarchical"}:
        names.add("strategy.dense")
    if mode in MODEL_MODES:
        names.add("model_generation")
    if mode == "federated":
        names.add("federation")
    if mode in set(UPSTREAM_IDS) - {"colbert", "multivector", "multimodal_native"}:
        names.update(("research", "model_generation"))
    if mode in {"raptor", "lightrag"}:
        names.add("tokenizer.tiktoken")
    if mode == "multimodal_native":
        names.add("multimodal")
    return tuple(sorted(names))


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Feature configuration contains a duplicate key")
        result[key] = value
    return result


def _parse(raw: str) -> dict[str, bool]:
    if len(raw.encode("utf-8")) > 65536:
        raise ValueError("Feature configuration exceeds 64 KiB")
    try:
        value = json.loads(raw, object_pairs_hook=_object)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Feature configuration must be a JSON boolean object") from exc
    if not isinstance(value, dict) or any(key not in MANIFEST or type(flag) is not bool
                                         for key, flag in value.items()):
        raise ValueError("Feature configuration requires known names and boolean values")
    return value


@dataclass(frozen=True)
class FeatureFlags:
    overrides: tuple[tuple[str, bool], ...] = ()

    def __post_init__(self):
        if not isinstance(self.overrides, tuple) or any(
            not isinstance(item, tuple) or len(item) != 2 or not isinstance(item[0], str) or item[0] not in MANIFEST
            or type(item[1]) is not bool for item in self.overrides
        ) or len(dict(self.overrides)) != len(self.overrides):
            raise ValueError("Feature overrides require distinct known boolean entries")

    @classmethod
    def from_env(cls) -> "FeatureFlags":
        flags = {}
        filename = os.getenv("RAGS_FEATURES_FILE")
        if filename:
            try:
                with Path(filename).open(encoding="utf-8") as stream:
                    raw = stream.read(65537)
            except (OSError, UnicodeError) as exc:
                raise ValueError("Unable to read RAGS_FEATURES_FILE") from exc
            flags.update(_parse(raw))
        flags.update(_parse(os.getenv("RAGS_FEATURES_JSON") or "{}"))
        return cls(tuple(sorted(flags.items())))

    def enabled(self, name: str) -> bool:
        if name not in MANIFEST:
            raise ValueError("Unknown feature name")
        return dict(self.overrides).get(name, True)

    def require(self, *names: str) -> None:
        for name in names:
            if not self.enabled(name):
                raise CapabilityError("feature_disabled", "Feature is disabled: " + name, 403)

    def as_dict(self) -> dict[str, bool]:
        values = dict(self.overrides)
        return {name: values.get(name, True) for name in MANIFEST}

    def disabled_for(self, mode: str) -> list[str]:
        return [name for name in strategy_features(mode) if not self.enabled(name)]
