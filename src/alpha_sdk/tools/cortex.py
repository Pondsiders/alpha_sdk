"""Cortex memory tools - native MCP server for alpha_sdk.

Direct Postgres access—no HTTP layer, no Cortex service dependency.
The store() tool clears the Redis memorables buffer as a side effect,
closing the feedback loop with Intro.

Usage:
    from alpha_sdk.tools import create_cortex_server

    # In AlphaClient, get session ID dynamically
    mcp_servers = {
        "cortex": create_cortex_server(get_session_id=lambda: client.session_id)
    }
"""

import os
from typing import Any, Callable

import logfire
import redis.asyncio as redis

from claude_agent_sdk import tool, create_sdk_mcp_server

from ..memories import store as cortex_store, search as cortex_search, recent as cortex_recent

# Configuration from environment
REDIS_URL = os.environ.get("REDIS_URL", "redis://alpha-pi:6379")


async def _get_redis() -> redis.Redis:
    """Get Redis client."""
    return redis.from_url(REDIS_URL, decode_responses=True)


async def _clear_memorables(session_id: str | None) -> int:
    """Clear the memorables buffer for this session.

    Returns the number of items that were cleared.
    """
    if not session_id:
        return 0

    redis_client = await _get_redis()
    try:
        key = f"intro:memorables:{session_id}"
        # Get count before deleting
        count = await redis_client.llen(key)
        if count > 0:
            await redis_client.delete(key)
            logfire.info("Cleared memorables buffer", session_id=session_id[:8], count=count)
        return count
    finally:
        await redis_client.aclose()


def create_cortex_server(get_session_id: Callable[[], str | None] | None = None):
    """Create the Cortex MCP server.

    Args:
        get_session_id: Optional callable that returns the current session ID.
                       Used for clearing memorables buffer after store().
                       If not provided, memorables won't be cleared.

    Returns:
        MCP server configuration dict
    """

    @tool(
        "store",
        "Store a memory in Cortex. Use this to remember important moments, realizations, or anything worth preserving.",
        {"memory": str}
    )
    async def store_memory(args: dict[str, Any]) -> dict[str, Any]:
        """Store a memory and clear the memorables buffer."""
        memory = args["memory"]
        session_id = get_session_id() if get_session_id else None

        with logfire.span(
            "mcp.cortex.store",
            memory_len=len(memory),
            session_id=session_id[:8] if session_id else "none"
        ):
            try:
                result = await cortex_store(memory)

                if result is None:
                    return {"content": [{"type": "text", "text": "Error storing memory"}]}

                memory_id = result.get("id", "unknown")
                logfire.info("Memory stored", memory_id=memory_id)

                # Clear the memorables buffer - this is the feedback mechanism
                cleared = await _clear_memorables(session_id)

                # Build response
                response_text = f"Memory stored (id: {memory_id})"
                if cleared > 0:
                    response_text += f" - cleared {cleared} pending suggestion(s)"

                return {"content": [{"type": "text", "text": response_text}]}

            except Exception as e:
                logfire.error("Cortex store failed", error=str(e))
                return {"content": [{"type": "text", "text": f"Error storing memory: {e}"}]}

    @tool(
        "search",
        "Search memories in Cortex. Returns semantically similar memories. Limit defaults to 5.",
        {"query": str}
    )
    async def search_memories(args: dict[str, Any]) -> dict[str, Any]:
        """Search for memories matching a query."""
        query = args["query"]
        limit = args.get("limit", 5)

        with logfire.span("mcp.cortex.search", query_len=len(query), limit=limit):
            try:
                memories = await cortex_search(query, limit=limit)

                if not memories:
                    return {"content": [{"type": "text", "text": "No memories found."}]}

                # Format results
                lines = [f"Found {len(memories)} memor{'y' if len(memories) == 1 else 'ies'}:\n"]
                for mem in memories:
                    score = mem.get("score", 0)
                    content = mem.get("content", "")
                    created = mem.get("created_at", "")[:10]  # Just the date
                    lines.append(f"[{score:.2f}] ({created}) {content}\n")

                logfire.info("Search complete", results=len(memories))
                return {"content": [{"type": "text", "text": "\n".join(lines)}]}

            except Exception as e:
                logfire.error("Cortex search failed", error=str(e))
                return {"content": [{"type": "text", "text": f"Error searching memories: {e}"}]}

    @tool(
        "recent",
        "Get recent memories from Cortex. Limit defaults to 10.",
        {}
    )
    async def recent_memories(args: dict[str, Any]) -> dict[str, Any]:
        """Get the most recent memories."""
        limit = args.get("limit", 10)

        with logfire.span("mcp.cortex.recent", limit=limit):
            try:
                memories = await cortex_recent(limit=limit)

                if not memories:
                    return {"content": [{"type": "text", "text": "No recent memories."}]}

                # Format results
                lines = [f"Last {len(memories)} memor{'y' if len(memories) == 1 else 'ies'}:\n"]
                for mem in memories:
                    content = mem.get("content", "")
                    created = mem.get("created_at", "")[:16]  # Date and time
                    lines.append(f"({created}) {content}\n")

                logfire.info("Recent complete", results=len(memories))
                return {"content": [{"type": "text", "text": "\n".join(lines)}]}

            except Exception as e:
                logfire.error("Cortex recent failed", error=str(e))
                return {"content": [{"type": "text", "text": f"Error getting recent memories: {e}"}]}

    # Bundle into MCP server
    return create_sdk_mcp_server(
        name="cortex",
        version="2.0.0",  # Bumped version - now using direct Postgres
        tools=[store_memory, search_memories, recent_memories]
    )
