"""Download explicit checkpoint files without importing any inference runtime.

Use this separate preparation phase before running actual encoder validation in
a network namespace. No ONNX Runtime, Torch or FastEmbed module is imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from rags.model_catalog import embedding_catalog, get_embedding_preset

_FASTEMBED = {
    "bge-small-en": ("Qdrant/bge-small-en-v1.5-onnx-Q", "model_optimized.onnx"),
    "bge-small-zh": ("Qdrant/bge-small-zh-v1.5", "model_optimized.onnx"),
    "fastembed-minilm": ("Qdrant/all-MiniLM-L6-v2-onnx", "model.onnx"),
}
_TOKENIZER_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                    "preprocessor_config.json", "vocab.txt", "vocab.json", "merges.txt",
                    "sentencepiece.bpe.model", "spiece.model"]


def prepare(identifier: str, cache: Path) -> dict:
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import hf_hub_download, snapshot_download
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
    item = get_embedding_preset(identifier)
    if identifier in _FASTEMBED:
        repo, filename = _FASTEMBED[identifier]
        patterns, revision = [*_TOKENIZER_FILES, filename], "main"
    else:
        repo, revision = item["model"], item["revision"]
        patterns = [*_TOKENIZER_FILES, "modules.json", "sentence_bert_config.json", "1_Pooling/config.json",
                    "2_Normalize/config.json", "model.safetensors", "model.safetensors.index.json", "model-*.safetensors"]
        # Record immutable negative metadata for the offline anti-adapter check.
        try:
            hf_hub_download(repo, "adapter_config.json", revision=revision, cache_dir=cache, token=False)
        except EntryNotFoundError as exc:
            if isinstance(exc, LocalEntryNotFoundError) or exc.response is None or exc.response.status_code != 404:
                raise
        else:
            raise ValueError("This prepared recipe does not accept adapter checkpoints")
    snapshot = Path(snapshot_download(repo, revision=revision, cache_dir=cache, allow_patterns=patterns,
                                      token=False, max_workers=2))
    files = []
    for path in sorted(snapshot.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            files.append({"name": path.relative_to(snapshot).as_posix(), "bytes": path.stat().st_size,
                          "sha256": digest.hexdigest()})
    return {"id": identifier, "repository": repo, "revision": snapshot.name, "files": files}


def main() -> None:
    identifiers = [item["id"] for item in embedding_catalog() if item["provider"] in {"fastembed", "sentence_transformers"}]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=identifiers, default=identifiers)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "phase": "download_only_no_inference_runtime", "models": []}
    for identifier in dict.fromkeys(args.models):
        report["models"].append(prepare(identifier, args.cache))
        print(json.dumps({"prepared": identifier}), flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
