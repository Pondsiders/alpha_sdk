"""Direct Cortex operations - store, search, recent.

These are the foundational memory operations that talk to Postgres.
Higher-level functions (recall, suggest) build on these.
"""

import os
from datetime import datetime
from typing import Any

import httpx
import logfire

# Configuration from environment
CORTEX_BASE_URL = os.environ.get("CORTEX_BASE_URL")
CORTEX_API_KEY = os.environ.get("CORTEX_API_KEY")

# Search parameters
DEFAULT_LIMIT = 5
MIN_SCORE = 0.1  # Minimum similarity threshold


async def store(memory: str) -> dict[str, Any] | None:
    """Store a memory in Cortex.

    Args:
        memory: The memory content to store

    Returns:
        Dict with id and created_at, or None on failure
    """
    if not CORTEX_API_KEY:
        logfire.warning("CORTEX_API_KEY not set, skipping store")
        return None

    with logfire.span("cortex.store", memory_preview=memory[:50]):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{CORTEX_BASE_URL.rstrip('/')}/store",
                    json={"memory": memory},
                    headers={
                        "Content-Type": "application/json",
                        "X-API-Key": CORTEX_API_KEY,
                    },
                )
                response.raise_for_status()

            data = response.json()
            logfire.info("Memory stored", id=data.get("id"))
            return data

        except Exception as e:
            logfire.error("Cortex store failed", error=str(e))
            return None


async def search(
    query: str,
    limit: int = DEFAULT_LIMIT,
    exclude: list[int] | None = None,
    min_score: float | None = MIN_SCORE,
) -> list[dict[str, Any]]:
    """Search Cortex for memories matching a query.

    Args:
        query: The search query
        limit: Maximum results to return
        exclude: Memory IDs to skip
        min_score: Minimum similarity threshold

    Returns:
        List of memory dicts with id, content, created_at, score
    """
    if not CORTEX_API_KEY:
        logfire.warning("CORTEX_API_KEY not set, skipping search")
        return []

    with logfire.span("cortex.search", query_preview=query[:50], exclude_count=len(exclude or [])):
        try:
            payload = {
                "query": query,
                "limit": limit,
            }
            if exclude:
                payload["exclude"] = exclude
            if min_score is not None:
                payload["min_score"] = min_score

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{CORTEX_BASE_URL.rstrip('/')}/search",
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "X-API-Key": CORTEX_API_KEY,
                    },
                )
                response.raise_for_status()

            data = response.json()
            memories = []
            for item in data.get("memories", []):
                memories.append({
                    "id": item["id"],
                    "content": item["content"],
                    "created_at": item.get("created_at", ""),
                    "score": item.get("score"),
                })

            logfire.debug("Cortex search complete", query_preview=query[:30], results=len(memories))
            return memories

        except Exception as e:
            logfire.error("Cortex search failed", error=str(e))
            return []


async def recent(limit: int = 10) -> list[dict[str, Any]]:
    """Get recent memories from Cortex.

    Args:
        limit: Maximum results to return

    Returns:
        List of memory dicts with id, content, created_at
    """
    if not CORTEX_API_KEY:
        logfire.warning("CORTEX_API_KEY not set, skipping recent")
        return []

    with logfire.span("cortex.recent", limit=limit):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{CORTEX_BASE_URL.rstrip('/')}/recent",
                    params={"limit": limit},
                    headers={
                        "Content-Type": "application/json",
                        "X-API-Key": CORTEX_API_KEY,
                    },
                )
                response.raise_for_status()

            data = response.json()
            memories = []
            for item in data.get("memories", []):
                memories.append({
                    "id": item["id"],
                    "content": item["content"],
                    "created_at": item.get("created_at", ""),
                })

            logfire.debug("Cortex recent complete", results=len(memories))
            return memories

        except Exception as e:
            logfire.error("Cortex recent failed", error=str(e))
            return []
