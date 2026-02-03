"""Calendar - events from Redis.

What's coming up.
"""

import os

import logfire
import redis.asyncio as aioredis

REDIS_URL = os.environ.get("REDIS_URL", "redis://alpha-pi:6379")


async def _get_redis() -> aioredis.Redis:
    """Get async Redis connection."""
    return aioredis.from_url(REDIS_URL, decode_responses=True)


async def get_events() -> str | None:
    """Fetch calendar events from Redis.

    Returns formatted markdown or None if no events.
    """
    try:
        r = await _get_redis()
        calendar = await r.get("hud:calendar")
        await r.aclose()

        if calendar:
            return f"## Events\n\n{calendar}"
        return None

    except Exception as e:
        logfire.warn(f"Error fetching calendar: {e}")
        return None
