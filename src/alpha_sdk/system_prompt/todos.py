"""Todos - task list from Redis.

What needs doing.
"""

import os

import logfire
import redis.asyncio as aioredis

REDIS_URL = os.environ.get("REDIS_URL", "redis://alpha-pi:6379")


async def _get_redis() -> aioredis.Redis:
    """Get async Redis connection."""
    return aioredis.from_url(REDIS_URL, decode_responses=True)


async def get_todos() -> str | None:
    """Fetch todos from Redis.

    Returns formatted markdown or None if no todos.
    """
    try:
        r = await _get_redis()
        todos = await r.get("hud:todos")
        await r.aclose()

        if todos:
            return f"## Todos\n\n{todos}"
        return None

    except Exception as e:
        logfire.warn(f"Error fetching todos: {e}")
        return None
