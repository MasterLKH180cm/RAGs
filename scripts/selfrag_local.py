"""Reproduce the pinned Self-RAG Q4_K_M CPU artifact from official weights.

Preparation is an explicit operator action. Research requests never download,
convert, or import model code. Install requirements/research-selfrag-convert.txt
in an isolated Python 3.12 environment and pass its Python with --python.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rags.research.selfrag_runtime import (
    CHECKPOINT_REPOSITORY,
    CHECKPOINT_REVISION,
    LIBRARY_HASHES,
    MODEL_BYTES,
    MODEL_SHA256,
    RUNTIME_REVISION,
    TOKEN_IDS,
    file_digest,
)
from scripts.local_model import Artifact, stream_model, write_json

MODEL_FILENAME = "selfrag_llama2_7b.Q4_K_M.gguf"
F16_FILENAME = "selfrag_llama2_7b.f16.gguf"
F16_BYTES = 13478368000
F16_SHA256 = "e7ad486ae418d03fda1e0258863ad66ec3c19266b4773c8e206cc1750c2ce583"
SOURCE_ARCHIVE_BYTES = 39307663
SOURCE_ARCHIVE_SHA256 = "c3c576cc3411ebbc284bcdb19d79fe23baee6c7d6aa49e297d370c5b06599805"
SOURCE_FILES = {
    "added_tokens.json": (433, "dcb083dc232eb79aa0eac5e2a5f2013dcfda0980561ca313ce191ad69fbbf954"),
    "config.json": (635, "eeaf19f35c1b9aee755d6903f5a628b27c310824efc273b991e02dcd67c7ae35"),
    "generation_config.json": (188, "7b81a582ca675c32477c0d4fc8272b3265c4f314fe24ec79a16b20e33740471d"),
    "pytorch_model-00001-of-00002.bin": (9976751194, "5c726ea09de32bff5ad0053f98bfd68847e18830d54b4c3d08b67c270380bbd1"),
    "pytorch_model-00002-of-00002.bin": (3500441859, "0883291fa5ac8b05ff5afb22cf08921be94c34988424a5e9ae3931663ad8ed93"),
    "pytorch_model.bin.index.json": (23950, "0ca9b8f5a8f500eafd4bdcbd52898a75834f2f5d596630cdca4da3c30034b625"),
    "special_tokens_map.json": (470, "dc76ec5e7d56ed1a0b80b582c30714d117b9b3ba15f3f5bc8bb8333c0312a48f"),
    "tokenizer.model": (499723, "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347"),
    "tokenizer_config.json": (855, "e5ad35cc8dd767a57020a885f2e2687c65912203566625f188a46af7eab789be"),
}
CONVERSION_VERSIONS = {"torch": "2.15.0+cpu", "transformers": "4.57.6", "numpy": "2.5.5",
                       "sentencepiece": "0.2.1", "safetensors": "0.8.0"}


def verified(path: Path, size: int, digest: str) -> bool:
    return bool(not path.is_symlink() and path.is_file() and path.stat().st_size == size
                and file_digest(path) == digest)


def conversion_environment() -> dict[str, str]:
    allowed = {"PATH", "LANG", "LC_ALL", "TZ", "SYSTEMROOT", "LD_LIBRARY_PATH"}
    return {**{key: value for key, value in os.environ.items() if key in allowed},
            "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
            "PYTHONNOUSERSITE": "1"}


def verify_conversion_python(python: str) -> None:
    code = "import importlib.metadata,json; print(json.dumps({name:importlib.metadata.version(name) for name in " + repr(list(CONVERSION_VERSIONS)) + "}))"
    run = subprocess.run([python, "-c", code], capture_output=True, text=True,
                         timeout=30, check=True, env=conversion_environment())
    if json.loads(run.stdout) != CONVERSION_VERSIONS:
        raise ValueError("Conversion dependencies differ from the pinned manifest")


def verify_runtime(directory: Path) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Pinned llama.cpp directory is unavailable")
    for filename, digest in LIBRARY_HASHES.items():
        path = directory / filename
        if path.is_symlink() or not path.is_file() or file_digest(path) != digest:
            raise ValueError("Pinned llama.cpp native library differs")
    quantizer = directory / "llama-quantize"
    if not quantizer.is_file():
        raise ValueError("Pinned llama.cpp quantizer is unavailable")
    # The release's executable loads its companion implementation dynamically.
    for filename, expected in {
        "llama-quantize": "e3db84214d6985a1661ee9507846375d3d78b148619b86c43602c65a474fa123",
        "libllama-quantize-impl.so": "5a37c340dce5f79fe7040f78cee216d169e902ef7ac39c34fbee9c20feefde7c",
        "libllama-common.so.1.0.0": "5c8a54d50233f8c3e440d6d35a7de45a9d656bc062ff93a7adc444b09b9e219f",
    }.items():
        path = directory / filename
        if path.is_symlink() or file_digest(path) != expected:
            raise ValueError("Pinned llama.cpp quantizer differs")


async def download_checkpoint(cache: Path, filename: str, size: int, digest: str):
    target = cache / filename
    if verified(target, size, digest):
        return
    if target.exists() or target.is_symlink():
        raise ValueError("Existing Self-RAG checkpoint file differs; choose a clean cache")
    descriptor, temporary = tempfile.mkstemp(prefix=".selfrag-download-", dir=cache)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            await stream_model(stream, Artifact(CHECKPOINT_REPOSITORY, CHECKPOINT_REVISION,
                                               filename, size, digest, "original_bfloat16"), timeout_seconds=1800)
        # Publish without replacing a file created while the download ran.
        os.link(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


async def download_converter(cache: Path) -> Path:
    target = cache / "llama-source.tar.gz"
    if verified(target, SOURCE_ARCHIVE_BYTES, SOURCE_ARCHIVE_SHA256):
        return target
    if target.exists() or target.is_symlink():
        raise ValueError("Existing pinned converter archive differs")
    descriptor, temporary = tempfile.mkstemp(prefix=".selfrag-source-", dir=cache)
    digest, size = hashlib.sha256(), 0
    try:
        with os.fdopen(descriptor, "wb") as stream:
            async with (asyncio.timeout(300), httpx.AsyncClient(timeout=30, follow_redirects=False) as client,
                        client.stream("GET", f"https://codeload.github.com/ggml-org/llama.cpp/tar.gz/{RUNTIME_REVISION}",
                                      headers={"Accept-Encoding": "identity"}) as response):
                if response.status_code != 200 or response.headers.get("Content-Encoding", "identity") != "identity":
                    raise ValueError("Pinned converter download failed")
                async for data in response.aiter_raw():
                    size += len(data)
                    if size > SOURCE_ARCHIVE_BYTES:
                        raise ValueError("Pinned converter archive exceeds its recorded size")
                    digest.update(data)
                    stream.write(data)
            if size != SOURCE_ARCHIVE_BYTES or digest.hexdigest() != SOURCE_ARCHIVE_SHA256:
                raise ValueError("Pinned converter archive digest differs")
        os.link(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def extract_converter(archive: Path, directory: Path) -> Path:
    """Only regular source files under the exact pinned root are extracted."""
    root_name = "llama.cpp-" + RUNTIME_REVISION
    with tarfile.open(archive) as source:
        for member in source:
            parts = Path(member.name).parts
            if (not parts or parts[0] != root_name or any(part in {".", ".."} for part in parts)
                    or Path(member.name).is_absolute()):
                raise ValueError("Unexpected converter archive path")
            if not (member.name.endswith(".py") or "/gguf-py/" in member.name):
                continue
            if member.isdir():
                continue
            if not member.isfile() or member.size > 10_000_000:
                raise ValueError("Unsupported converter archive member")
            target = directory / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as incoming, target.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
    return directory / root_name


def normalize_selfrag_eos(python: str, converter: Path, model: Path) -> None:
    # The pinned converter mistakes vocab_size=32016 for CodeLlama and adds
    # EOT=32010. In Self-RAG that ID is [Utility:4], NOT an end token. Preserve
    # every tensor and token ID while aligning EOT with official EOS=2.
    code = (
        "import sys; sys.path.insert(0,sys.argv[1]); from gguf import GGUFReader; "
        "r=GGUFReader(sys.argv[2],'r+'); f=r.get_field('tokenizer.ggml.eot_token_id'); "
        "assert f is not None and int(f.parts[f.data[0]][0])==32010; f.parts[f.data[0]][0]=2"
    )
    subprocess.run([python, "-c", code, str(converter / "gguf-py"), str(model)],
                   timeout=60, check=True, env=conversion_environment())


def manifest(cache: Path) -> dict:
    return {"schema": "selfrag-cpu-artifacts-v1", "checkpoint_repository": CHECKPOINT_REPOSITORY,
            "checkpoint_revision": CHECKPOINT_REVISION, "quantization": "Q4_K_M",
            "model": {"filename": MODEL_FILENAME, "bytes": MODEL_BYTES, "sha256": MODEL_SHA256,
                      "verified": verified(cache / MODEL_FILENAME, MODEL_BYTES, MODEL_SHA256)},
            "source_files": {name: {"bytes": size, "sha256": digest} for name, (size, digest) in SOURCE_FILES.items()},
            "conversion_runtime": CONVERSION_VERSIONS, "runtime_revision": RUNTIME_REVISION,
            "runtime_library_hashes": LIBRARY_HASHES, "reflection_token_ids": TOKEN_IDS,
            "conversion_correction": "CodeLlama EOT metadata 32010 normalized to official Self-RAG EOS 2; tensors unchanged",
            "model_inference_executed": False}


@contextmanager
def preparation_lock(cache: Path):
    """Allow one preparer per cache; the kernel releases ownership on exit/crash.

    Keep the lock file in place: unlinking it could give concurrent processes
    different lock inodes. Its presence is harmless after a crash and requires
    no manual lock removal. This recipe uses the pinned Linux CPU runtime.
    """
    import fcntl

    descriptor = os.open(cache / ".selfrag-prepare.lock",
                         os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("Self-RAG preparation lock must be a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Self-RAG preparation is already running in this cache") from exc
        yield
    finally:
        os.close(descriptor)


def prepare(cache: Path, directory: Path, python: str) -> dict:
    if cache.is_symlink():
        raise ValueError("Self-RAG cache must not be a symbolic link")
    cache.mkdir(parents=True, exist_ok=True)
    with preparation_lock(cache):
        return _prepare_locked(cache, directory, python)


def _prepare_locked(cache: Path, directory: Path, python: str) -> dict:
    verify_runtime(directory)
    if verified(cache / MODEL_FILENAME, MODEL_BYTES, MODEL_SHA256):
        result = manifest(cache)
        write_json(cache / "selfrag-manifest.json", result)
        return result
    if (cache / MODEL_FILENAME).exists() or (cache / MODEL_FILENAME).is_symlink():
        raise ValueError("Existing Self-RAG model differs; choose a clean cache")
    verify_conversion_python(python)
    required = sum(size for name, (size, _) in SOURCE_FILES.items() if not (cache / name).exists())
    required += (0 if (cache / F16_FILENAME).exists() else F16_BYTES) + MODEL_BYTES + 1024 ** 3
    if shutil.disk_usage(cache).free < required:
        raise ValueError("Insufficient disk for pinned source weights, F16 conversion and Q4_K_M artifact")

    async def downloads():
        for name, (size, digest) in SOURCE_FILES.items():
            await download_checkpoint(cache, name, size, digest)
        return await download_converter(cache)

    archive = asyncio.run(downloads())
    f16 = cache / F16_FILENAME
    if not verified(f16, F16_BYTES, F16_SHA256):
        if f16.exists() or f16.is_symlink():
            raise ValueError("Existing Self-RAG F16 conversion differs")
        with tempfile.TemporaryDirectory(prefix="selfrag-converter-", dir=cache) as temporary:
            converter = extract_converter(archive, Path(temporary))
            partial = cache / (F16_FILENAME + ".partial")
            if partial.exists() or partial.is_symlink():
                raise ValueError("Incomplete conversion already exists; choose a clean cache")
            try:
                subprocess.run([python, str(converter / "convert_hf_to_gguf.py"), str(cache),
                                "--outfile", str(partial), "--outtype", "f16", "--model-name", CHECKPOINT_REPOSITORY],
                               timeout=1800, check=True, env=conversion_environment())
                normalize_selfrag_eos(python, converter, partial)
                if not verified(partial, F16_BYTES, F16_SHA256):
                    raise ValueError("Converted Self-RAG F16 identity differs")
                os.link(partial, f16)
            finally:
                partial.unlink(missing_ok=True)
    partial = cache / (MODEL_FILENAME + ".partial")
    if partial.exists() or partial.is_symlink():
        raise ValueError("Incomplete quantization already exists; choose a clean cache")
    try:
        subprocess.run([str(directory / "llama-quantize"), str(f16), str(partial), "Q4_K_M", "4"],
                       timeout=1800, check=True, env=conversion_environment())
        if not verified(partial, MODEL_BYTES, MODEL_SHA256):
            raise ValueError("Quantized Self-RAG identity differs")
        os.link(partial, cache / MODEL_FILENAME)
    finally:
        partial.unlink(missing_ok=True)
    result = manifest(cache)
    write_json(cache / "selfrag-manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "prepare"))
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--llama-dir", type=Path)
    parser.add_argument("--python", default=sys.executable, help="Pinned isolated conversion Python")
    args = parser.parse_args()
    if args.action == "prepare" and args.llama_dir is None:
        parser.error("prepare requires --llama-dir")
    try:
        result = prepare(args.cache_dir.absolute(), args.llama_dir.absolute(), args.python) if args.action == "prepare" else manifest(args.cache_dir)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, subprocess.SubprocessError, TimeoutError, httpx.HTTPError) as error:
        print(json.dumps({"status": "blocked", "error_type": type(error).__name__,
                          "model_inference_executed": False}), file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
