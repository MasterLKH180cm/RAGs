"""Optional stdio MCP integration for Codex and Claude Code.

Install with: pip install -e '.[mcp]'
Launch with: python -m rags.mcp_server
"""
import os

import httpx

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:
    raise SystemExit("Install MCP support: pip install -e '.[mcp]'") from exc

mcp = FastMCP("rags")
BASE_URL = os.getenv("RAGS_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.getenv("RAGS_API_KEY", "local-development-key")


@mcp.tool()
def search_knowledge(question: str, mode: str = "hybrid", top_k: int = 5) -> dict:
    """Retrieve tenant-authorized documents and generate an evidence-based response."""
    with httpx.Client(timeout=90) as client:
        response = client.post(
            BASE_URL + "/query",
            headers={"Authorization": "Bearer " + API_KEY},
            json={"question": question, "mode": mode, "top_k": top_k},
        )
        response.raise_for_status()
        return response.json()


@mcp.tool()
def list_knowledge_documents() -> list[dict]:
    """List the documents available to this configured tenant."""
    with httpx.Client(timeout=20) as client:
        response = client.get(
            BASE_URL + "/documents",
            headers={"Authorization": "Bearer " + API_KEY},
        )
        response.raise_for_status()
        return response.json()


if __name__ == "__main__":
    mcp.run(transport="stdio")
