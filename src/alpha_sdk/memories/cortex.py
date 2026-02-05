"""Direct Cortex operations - store, search, recent.

These are the foundational memory operations that talk directly to Postgres.
No HTTP layer—direct database access via asyncpg.
Higher-level functions (recall, suggest) build on these.
"""

from datetime import datetime
from typing import Any

import logfire

from .embeddings import embed_document, embed_query, EmbeddingError
from .db import (
    store_memory,
    search_memories,
    get_recent_memories,
    get_memory,
    forget_memory,
    health_check as db_health_check,
    close_pool,
)
from .embeddings import health_check as ollama_health_check

# Re-export EmbeddingError for consumers
__all__ = [
    "store",
    "search",
    "recent",
    "get",
    "forget",
    "health",
    "close",
    "EmbeddingError",
]

# Search parameters
DEFAULT_LIMIT = 5
MIN_SCORE = 0.1  # Minimum similarity threshold


async def store(
    memory: str,
    tags: list[str] | None = None,
    timezone: str | None = None,
) -> dict[str, Any] | None:
    """Store a memory in Cortex.

    Args:
        memory: The memory content to store
        tags: Optional tags for organization
        timezone: Timezone where the memory was captured

    Returns:
        Dict with id and created_at, or None on failure
    """
    with logfire.span("cortex.store", memory_preview=memory[:50]) as span:
        try:
            # Generate embedding
            embedding = await embed_document(memory)

            # Store in database
            memory_id, created_at = await store_memory(
                content=memory,
                embedding=embedding,
                tags=tags,
                timezone_str=timezone,
            )

            span.set_attribute("memory_id", memory_id)
            logfire.info(f"Memory stored: #{memory_id}")
            return {"id": memory_id, "created_at": created_at.isoformat()}

        except EmbeddingError as e:
            logfire.error("Embedding failed during store", error=str(e))
            return None
        except Exception as e:
            logfire.error("Cortex store failed", error=str(e))
            return None


async def search(
    query: str,
    limit: int = DEFAULT_LIMIT,
    exclude: list[int] | None = None,
    min_score: float | None = MIN_SCORE,
    include_forgotten: bool = False,
    exact: bool = False,
    after: datetime | None = None,
    before: datetime | None = None,
) -> list[dict[str, Any]]:
    """Search Cortex for memories matching a query.

    Args:
        query: The search query
        limit: Maximum results to return
        exclude: Memory IDs to skip
        min_score: Minimum similarity threshold
        include_forgotten: Include soft-deleted memories
        exact: Use exact full-text match only (no semantic search)
        after: Only memories after this datetime
        before: Only memories before this datetime

    Returns:
        List of memory dicts with id, content, created_at, score
    """
    with logfire.span("cortex.search", query_preview=query[:50], exclude_count=len(exclude or [])) as span:
        try:
            # Generate query embedding (unless exact match only)
            query_embedding = None
            if not exact:
                query_embedding = await embed_query(query)

            # Search database
            results = await search_memories(
                query_embedding=query_embedding,
                query_text=query,
                limit=limit,
                include_forgotten=include_forgotten,
                exact=exact,
                after=after,
                before=before,
                exclude=exclude,
                min_score=min_score,
            )

            # Transform to expected format
            memories = []
            for item in results:
                metadata = item.get("metadata", {})
                memories.append({
                    "id": item["id"],
                    "content": item["content"],
                    "created_at": metadata.get("created_at", ""),
                    "score": item.get("score"),
                })

            span.set_attribute("result_count", len(memories))
            logfire.debug("Cortex search complete", query_preview=query[:30], results=len(memories))
            return memories

        except EmbeddingError as e:
            logfire.error("Embedding failed during search", error=str(e))
            return []
        except Exception as e:
            logfire.error("Cortex search failed", error=str(e))
            return []


async def recent(limit: int = 10, hours: int = 24) -> list[dict[str, Any]]:
    """Get recent memories from Cortex.

    Args:
        limit: Maximum results to return
        hours: How far back to look (in hours)

    Returns:
        List of memory dicts with id, content, created_at
    """
    with logfire.span("cortex.recent", limit=limit, hours=hours) as span:
        try:
            results = await get_recent_memories(limit=limit, hours=hours)

            # Transform to expected format
            memories = []
            for item in results:
                metadata = item.get("metadata", {})
                memories.append({
                    "id": item["id"],
                    "content": item["content"],
                    "created_at": metadata.get("created_at", ""),
                })

            span.set_attribute("result_count", len(memories))
            logfire.debug("Cortex recent complete", results=len(memories))
            return memories

        except Exception as e:
            logfire.error("Cortex recent failed", error=str(e))
            return []


async def get(memory_id: int) -> dict[str, Any] | None:
    """Get a single memory by ID.

    Args:
        memory_id: The memory ID to retrieve

    Returns:
        Memory dict with id, content, created_at, or None if not found
    """
    try:
        result = await get_memory(memory_id)
        if result is None:
            return None

        metadata = result.get("metadata", {})
        return {
            "id": result["id"],
            "content": result["content"],
            "created_at": metadata.get("created_at", ""),
            "tags": metadata.get("tags"),
        }
    except Exception as e:
        logfire.error("Cortex get failed", error=str(e))
        return None


async def forget(memory_id: int) -> bool:
    """Soft-delete a memory.

    Args:
        memory_id: The memory ID to forget

    Returns:
        True if memory was found and forgotten, False otherwise
    """
    try:
        return await forget_memory(memory_id)
    except Exception as e:
        logfire.error("Cortex forget failed", error=str(e))
        return False


async def health() -> dict[str, Any]:
    """Check Cortex health (database and embeddings).

    Returns:
        Dict with status, postgres, ollama, memory_count
    """
    db_ok, memory_count = await db_health_check()
    ollama_ok = await ollama_health_check()

    status = "healthy" if (db_ok and ollama_ok) else "degraded"

    return {
        "status": status,
        "postgres": "connected" if db_ok else "disconnected",
        "ollama": "connected" if ollama_ok else "disconnected",
        "memory_count": memory_count,
    }


async def close() -> None:
    """Close database connections. Call on shutdown."""
    await close_pool()
