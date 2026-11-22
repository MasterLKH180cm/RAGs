"""Optional stdio MCP integration for Codex and Claude Code.

Install with: uv sync --locked --extra mcp (include other selected extras)
Launch with: uv run --no-sync python -m rags.mcp_server
"""
import os

import httpx

from rags.features import FeatureFlags, strategy_features
from rags.registry import CapabilityError

FEATURES = FeatureFlags.from_env()
if __name__ == "__main__":
    FEATURES.require("mcp")

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:
    raise SystemExit("Install MCP support: uv sync --locked --extra mcp (include other selected extras)") from exc

mcp = FastMCP("rags")
BASE_URL = os.getenv("RAGS_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.getenv("RAGS_API_KEY", "local-development-key")
HEADERS = {"Authorization": "Bearer " + API_KEY, "X-RAGS-Interface": "mcp"}


def capabilities(client: httpx.Client, *requirements: str) -> dict:
    FEATURES.require("mcp", *requirements)
    response = client.get(BASE_URL + "/capabilities", headers=HEADERS)
    response.raise_for_status()
    payload = response.json()
    flags = payload.get("features") if isinstance(payload, dict) else None
    if not isinstance(flags, dict):
        raise CapabilityError("invalid_capabilities", "RAG service did not report feature permissions", 502)
    for name in ("mcp", *requirements):
        if flags.get(name) is not True:
            raise CapabilityError("feature_disabled", "Feature is disabled: " + name, 403)
    return payload


def configure_tools() -> None:
    """Advertise only tools permitted by both the bridge and the HTTP service."""
    FEATURES.require("mcp")
    with httpx.Client(timeout=20) as client:
        remote = capabilities(client)["features"]
    for tool, feature in (("search_knowledge", "query"), ("list_knowledge_documents", "documents")):
        if not FEATURES.enabled(feature) or remote.get(feature) is not True:
            mcp.remove_tool(tool)


@mcp.tool()
def search_knowledge(question: str, mode: str = "hybrid", top_k: int = 5, generate: bool = True) -> dict:
    """Retrieve tenant-authorized documents and generate an evidence-based response."""
    requirements = strategy_features(mode)
    FEATURES.require("mcp", *requirements)
    with httpx.Client(timeout=90) as client:
        remote = capabilities(client, *requirements)
        if generate and remote.get("generation", {}).get("provider") != "extractive":
            FEATURES.require("model_generation")
            if remote["features"].get("model_generation") is not True:
                raise CapabilityError("feature_disabled", "Feature is disabled: model_generation", 403)
        response = client.post(
            BASE_URL + "/query",
            headers=HEADERS,
            json={"question": question, "mode": mode, "top_k": top_k, "generate": generate},
        )
        response.raise_for_status()
        return response.json()


@mcp.tool()
def list_knowledge_documents() -> list[dict]:
    """List the documents available to this configured tenant."""
    FEATURES.require("mcp", "documents")
    with httpx.Client(timeout=20) as client:
        capabilities(client, "documents")
        response = client.get(
            BASE_URL + "/documents",
            headers=HEADERS,
        )
        response.raise_for_status()
        return response.json()


if __name__ == "__main__":
    configure_tools()
    mcp.run(transport="stdio")
