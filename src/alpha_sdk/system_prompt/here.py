"""Here - where I am right now.

Client name, hostname, weather, astronomy.
Answers the question: what's the situation?
"""

import asyncio
import os
import socket

import logfire
import redis.asyncio as aioredis

REDIS_URL = os.environ.get("REDIS_URL", "redis://alpha-pi:6379")


async def _get_redis() -> aioredis.Redis:
    """Get async Redis connection."""
    return aioredis.from_url(REDIS_URL, decode_responses=True)


def get_hostname() -> str:
    """Get the current machine's hostname."""
    return socket.gethostname()


async def get_weather() -> str | None:
    """Fetch weather from Redis."""
    try:
        r = await _get_redis()
        weather = await r.get("hud:weather")
        await r.aclose()
        return weather
    except Exception as e:
        logfire.warn(f"Error fetching weather: {e}")
        return None


async def get_here(client: str | None = None, hostname: str | None = None) -> str:
    """Build the Here section.

    Args:
        client: Client name (e.g., "duckpond", "solitude")
        hostname: Override hostname (defaults to socket.gethostname())

    Returns:
        Formatted markdown string for the ## Here section.
    """
    hostname = hostname or get_hostname()
    weather = await get_weather()

    parts = []
    if client:
        parts.append(f"**Client:** {client.title()}")
    parts.append(f"**Machine:** {hostname}")
    if weather:
        parts.append(f"\n{weather}")

    return "## Here\n\n" + "\n".join(parts)
