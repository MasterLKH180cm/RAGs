"""Bounded HTTP for direct generation, embedding and image-caption providers.

The public adapter remains synchronous. Its owned async request supplies a wall
deadline across connection, headers and body reads, including slow-drip peers.
Operator credentials and response bodies are never included in public errors.
"""
import asyncio
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
import json as json_module
import math
from time import monotonic

import httpx

from rags.registry import CapabilityError

MAX_REQUEST_BYTES = 1024 * 1024
MAX_IMAGE_REQUEST_BYTES = 3_000_000  # Existing 2 MB uploads plus base64/JSON framing.
MAX_RESPONSE_BYTES = 512 * 1024


def _encode(payload: dict, limit: int) -> bytes:
    body = bytearray()
    encoder = json_module.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    try:
        for part in encoder.iterencode(payload):
            # Avoid allocating a second UTF-8 copy of an already oversized string.
            if len(part) > limit - len(body):
                raise CapabilityError("invalid_model_input", "Provider request exceeds the byte limit", 422)
            encoded = part.encode("utf-8")
            if len(encoded) > limit - len(body):
                raise CapabilityError("invalid_model_input", "Provider request exceeds the byte limit", 422)
            body.extend(encoded)
    except CapabilityError:
        raise
    except (ValueError, TypeError, RecursionError) as exc:
        raise CapabilityError("invalid_model_input", "Provider request is not valid JSON", 422) from exc
    return bytes(body)


async def _request(url: str, headers: Mapping[str, str], encoded: bytes,
                   deadline: float, client_factory: Callable[..., httpx.AsyncClient]) -> httpx.Response:
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise TimeoutError
    request_headers = httpx.Headers(headers)
    request_headers["Accept-Encoding"] = "identity"
    request_headers["Content-Type"] = "application/json"
    request_headers["Content-Length"] = str(len(encoded))
    async with asyncio.timeout(remaining):
        async with client_factory(timeout=remaining, follow_redirects=False) as client:
            async with client.stream("POST", url, headers=request_headers, content=encoded) as response:
                response.raise_for_status()
                if response.headers.get("content-encoding", "identity").strip().lower() != "identity":
                    raise CapabilityError("provider_failed", "Compressed provider responses are not accepted", 502)
                length = response.headers.get("content-length")
                if length is not None and (not length.isascii() or not length.isdecimal()
                                           or len(length) > 20 or int(length) > MAX_RESPONSE_BYTES):
                    raise CapabilityError("provider_failed", "Provider response has an invalid byte length", 502)
                body = bytearray()
                # Raw iteration prevents automatic decompression; fixed chunks
                # keep the buffering boundary independent of server chunk sizes.
                async for part in response.aiter_raw(chunk_size=16 * 1024):
                    if monotonic() >= deadline:
                        raise TimeoutError
                    if len(part) > MAX_RESPONSE_BYTES - len(body):
                        raise CapabilityError("provider_failed", "Provider response exceeds the byte limit", 502)
                    body.extend(part)
                if monotonic() >= deadline:
                    raise TimeoutError
                return httpx.Response(response.status_code, headers=response.headers,
                                      content=bytes(body), request=response.request)


def post(url: str, *, headers: Mapping[str, str], json: dict, timeout: float,
         max_request_bytes: int = MAX_REQUEST_BYTES,
         client_factory: Callable[..., httpx.AsyncClient] = httpx.AsyncClient) -> httpx.Response:
    """Post bounded JSON with an owned client; a factory is an explicit adapter seam.

    Async application paths already calling a synchronous provider use a short-
    lived worker event loop. It is joined before return, including timeout/error,
    so no request continues after the caller observes a completed operation.
    """
    if type(timeout) not in (int, float) or not 0 < timeout <= 360 or not math.isfinite(timeout):
        raise ValueError("Provider timeout must be finite and between 0 and 360 seconds")
    if type(max_request_bytes) is not int or not 1 <= max_request_bytes <= MAX_IMAGE_REQUEST_BYTES:
        raise ValueError("Provider request byte limit is invalid")
    deadline = monotonic() + timeout
    encoded = _encode(json, max_request_bytes)

    def run() -> httpx.Response:
        return asyncio.run(_request(url, headers, encoded, deadline, client_factory))

    try:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return run()
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="rags-provider") as executor:
            return executor.submit(run).result()
    except (httpx.HTTPError, TimeoutError):
        raise CapabilityError("provider_failed", "Configured provider request failed", 502) from None
