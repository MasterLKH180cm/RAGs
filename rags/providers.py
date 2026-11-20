"""Model and embedding adapters: deterministic offline demo and opt-in external models."""
from collections import Counter
import hashlib
import math
import re

import httpx

from rags.config import Settings


def local_embedding(text: str, dims: int = 384) -> list[float]:
    """Hashing features for an OFFLINE DEMO only; not a trained semantic embedding."""
    t = text.casefold()
    features = re.findall(r"[\w]+", t)
    features.extend(t[i:i + 3] for i in range(max(0, len(t) - 2)) if not t[i:i + 3].isspace())
    vector = [0.0] * dims
    for feature, count in Counter(features).items():
        hashed = hashlib.sha256(feature.encode("utf-8")).digest()
        idx = int.from_bytes(hashed[:4], "big") % dims
        sign = 1 if hashed[4] & 1 else -1
        vector[idx] += sign * (1 + math.log(count))
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


class Providers:
    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def embedding_model(self) -> str:
        return (
            "offline-hash-v1"
            if self.settings.embedding_provider == "local"
            else "openai:" + self.settings.embedding_model
        )

    def embedding(self, text: str) -> list[float]:
        if self.settings.embedding_provider == "local":
            return local_embedding(text)
        if not self.settings.openai_key:
            raise ValueError("OPENAI_API_KEY required for OpenAI embeddings")
        response = httpx.post(
            self.settings.openai_base_url + "/embeddings",
            headers={"Authorization": "Bearer " + self.settings.openai_key},
            json={"model": self.settings.embedding_model, "input": text},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    def complete(self, system: str, user: str) -> str | None:
        if self.settings.model_provider == "extractive":
            return None
        if self.settings.model_provider == "openai":
            if not self.settings.openai_key:
                raise ValueError("OPENAI_API_KEY required for OpenAI generation")
            response = httpx.post(
                self.settings.openai_base_url + "/chat/completions",
                headers={"Authorization": "Bearer " + self.settings.openai_key},
                json={
                    "model": self.settings.chat_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0,
                },
                timeout=90,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        if not self.settings.anthropic_key:
            raise ValueError("ANTHROPIC_API_KEY required for Anthropic generation")
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.settings.anthropic_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": self.settings.anthropic_model,
                "system": system,
                "messages": [{"role": "user", "content": user}],
                "max_tokens": 1024,
            },
            timeout=90,
        )
        response.raise_for_status()
        return "".join(
            b["text"] for b in response.json()["content"] if b.get("type") == "text"
        )
