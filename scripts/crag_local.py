#!/usr/bin/env python3
"""Provision the exact public evaluator artifacts linked by the CRAG authors."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import tempfile
from time import monotonic
from urllib.parse import urljoin, urlsplit

import httpx

from rags.research.crag_adapter import ARTIFACTS, verify_artifacts


FILE_IDS = {
    "added_tokens.json": "1fiYYBunQGs3rQ14y28mCg5Ap_KKBCCDy",
    "config.json": "1J2DCva5L6Mmx0JhzVWYiZfLg7YhRoY84",
    "model.safetensors": "19p2bAYL2QxvUyloKv6udF0R69u1yie-H",
    "special_tokens_map.json": "13LAXfli1JFO-Nan918RwYZnOpEB3nr4E",
    "spiece.model": "132NL1aJIjdIUCCab6uk3R2EZuayTyy8F",
    "tokenizer_config.json": "11GRPEmkDQ4h9EKFK5kb8OW0CEhXByyEd",
}


def approved_url(current, location):
    result = urljoin(current, location)
    parsed = urlsplit(result)
    if (parsed.scheme != "https" or parsed.hostname not in {"drive.usercontent.google.com", "drive.google.com"}
            or parsed.port not in (None, 443) or parsed.username is not None
            or parsed.password is not None or parsed.fragment):
        raise ValueError("Evaluator artifacts require the pinned public Google Drive download hosts")
    return result


async def download_one(directory, name, *, deadline, client):
    expected_size, expected_digest = ARTIFACTS[name]
    target = directory / name
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_file() or target.stat().st_size != expected_size:
            raise ValueError("Existing evaluator artifact is not the pinned regular file")
        with target.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != expected_digest:
                raise ValueError("Existing evaluator artifact has a different hash")
        return
    url = "https://drive.usercontent.google.com/download?id=" + FILE_IDS[name] + "&export=download&confirm=t"
    descriptor, filename = tempfile.mkstemp(prefix=".crag-part-", dir=directory)
    try:
        async with asyncio.timeout(max(0, deadline - monotonic())):
            with os.fdopen(descriptor, "wb") as handle:
                for _ in range(5):
                    async with client.stream("GET", url, headers={"Accept-Encoding": "identity"}) as response:
                        if response.status_code in {301, 302, 303, 307, 308}:
                            url = approved_url(url, response.headers["location"])
                            continue
                        response.raise_for_status()
                        if response.headers.get("content-encoding", "identity").lower() != "identity":
                            raise ValueError("Compressed evaluator downloads are unsupported")
                        count, digest = 0, hashlib.sha256()
                        async for chunk in response.aiter_raw():
                            count += len(chunk)
                            if count > expected_size:
                                raise ValueError("Evaluator download exceeds the pinned artifact size")
                            handle.write(chunk)
                            digest.update(chunk)
                            await asyncio.sleep(0)
                        if count != expected_size or digest.hexdigest() != expected_digest:
                            raise ValueError("Evaluator download differs from the pinned artifact")
                        handle.flush()
                        os.fsync(handle.fileno())
                        break
                else:
                    raise ValueError("Too many evaluator download redirects")
        # Atomic no-clobber publication, including a target created concurrently.
        os.link(filename, target)
    finally:
        Path(filename).unlink(missing_ok=True)


async def download(directory, timeout):
    if directory.is_symlink():
        raise ValueError("Evaluator directory cannot be a symlink")
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    deadline = monotonic() + timeout
    async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
        for name in FILE_IDS:
            await download_one(directory, name, deadline=deadline, client=client)
    verify_artifacts(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("download", "check"))
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=600,
                        help="Shared deadline for missing-artifact network acquisition; local hash verification is separate")
    args = parser.parse_args()
    if not 1 <= args.timeout_seconds <= 3600:
        parser.error("--timeout-seconds must be 1..3600")
    directory = args.directory.absolute()
    if args.operation == "download":
        asyncio.run(download(directory, args.timeout_seconds))
    else:
        verify_artifacts(directory)
    print(json.dumps({"status": "verified", "engine": "crag", "artifacts": {
        name: {"bytes": size, "sha256": digest} for name, (size, digest) in ARTIFACTS.items()
    }, "inference_executed": False}, indent=2))


if __name__ == "__main__":
    main()
