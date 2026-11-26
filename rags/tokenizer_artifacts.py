"""Operator-only local tokenizer registrations and verified artifact reads.

Registry validation performs no filesystem access or optional native imports.
Tenant requests select aliases; paths are never accepted as tokenizer selectors.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import stat


MAX_TOKENIZER_ARTIFACTS = 16
MAX_TOKENIZER_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_TOKENIZER_REGISTRY_BYTES = 128 * 1024
TOKENIZER_ALIAS = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
ARTIFACT_BACKENDS = frozenset({"huggingface", "sentencepiece"})


def _distinct_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Tokenizer registry contains duplicate fields")
        result[key] = value
    return result


def parse_tokenizer_artifacts(value) -> dict[str, dict[str, str]]:
    """Return a validated copy of a bounded operator registry, without loading it."""
    if value is None:
        value = {}
    if isinstance(value, str):
        try:
            if len(value.encode("utf-8")) > MAX_TOKENIZER_REGISTRY_BYTES:
                raise ValueError("Tokenizer registry exceeds its byte budget")
            value = json.loads(value, object_pairs_hook=_distinct_object)
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise ValueError("Tokenizer registry must be a JSON object") from exc
    if not isinstance(value, Mapping) or len(value) > MAX_TOKENIZER_ARTIFACTS:
        raise ValueError("Tokenizer registry requires at most 16 named artifacts")
    result = {}
    for alias, entry in value.items():
        if not isinstance(alias, str) or not TOKENIZER_ALIAS.fullmatch(alias):
            raise ValueError("Tokenizer artifact aliases must be bounded simple names")
        if not isinstance(entry, Mapping) or set(entry) != {"backend", "path", "sha256"}:
            raise ValueError("Tokenizer artifacts require backend, absolute path and sha256")
        backend, path, digest = entry["backend"], entry["path"], entry["sha256"]
        if not isinstance(backend, str) or backend not in ARTIFACT_BACKENDS:
            raise ValueError("Tokenizer artifacts support huggingface or sentencepiece")
        if (not isinstance(path, str) or not path or len(path) > 4096
                or "\x00" in path or not Path(path).is_absolute()):
            raise ValueError("Tokenizer artifact requires a bounded absolute local path")
        try:
            path.encode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise ValueError("Tokenizer artifact path contains invalid Unicode") from exc
        if not isinstance(digest, str) or not SHA256.fullmatch(digest):
            raise ValueError("Tokenizer artifact requires a lowercase SHA256 digest")
        result[alias] = {"backend": backend, "path": path, "sha256": digest}
    return result


def read_tokenizer_artifact(entry: Mapping[str, str]) -> bytes:
    """Read, bound and hash the same regular-file descriptor passed to the parser."""
    path = Path(entry["path"])
    try:
        if path.is_symlink():
            raise ValueError("Tokenizer artifact must not be a symbolic link")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                             | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("Tokenizer artifact must be a regular file")
            if not 0 < info.st_size <= MAX_TOKENIZER_ARTIFACT_BYTES:
                raise ValueError("Tokenizer artifact exceeds the 64 MiB size contract")
            raw = stream.read(MAX_TOKENIZER_ARTIFACT_BYTES + 1)
    except OSError as exc:
        raise ValueError("Registered tokenizer artifact is unavailable") from exc
    if len(raw) != info.st_size or len(raw) > MAX_TOKENIZER_ARTIFACT_BYTES:
        raise ValueError("Tokenizer artifact changed or exceeds its size contract")
    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError("Tokenizer artifact differs from its registered SHA256")
    return raw
