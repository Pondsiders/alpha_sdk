"""Weave - the transformation that turns Claude into Alpha.

Takes an API request, replaces the system prompt with Alpha's
assembled prompt, and returns the modified request.
"""

import logging
from typing import Any

from .system_prompt import assemble

logger = logging.getLogger(__name__)


async def weave(
    body: dict[str, Any],
    client: str | None = None,
    hostname: str | None = None,
) -> dict[str, Any]:
    """Transform a Claude API request into an Alpha API request.

    Args:
        body: The original request body
        client: Client name (e.g., "duckpond")
        hostname: Machine hostname

    Returns:
        Modified request body with Alpha's system prompt injected.
    """
    # Build the complete system prompt
    system_blocks = await assemble(client=client, hostname=hostname)

    # Get existing system from request
    existing_system = body.get("system")

    if existing_system is None:
        # No system prompt - just add ours
        body["system"] = system_blocks

    elif isinstance(existing_system, list) and len(existing_system) >= 1:
        # SDK sends: [0]=billing header, [1]=SDK boilerplate, [2]=safety envelope
        # Keep the billing header (element 0), replace everything else
        billing_header = existing_system[0]
        body["system"] = [billing_header] + system_blocks

    elif isinstance(existing_system, str):
        # Old-style string system prompt - replace entirely
        body["system"] = system_blocks

    else:
        logger.warning(f"Unexpected system format: {type(existing_system)}, replacing")
        body["system"] = system_blocks

    logger.info(f"Wove Alpha into request ({len(system_blocks)} blocks)")
    return body
