"""Weave - the transformation that turns Claude into Alpha.

Takes an API request, replaces the system prompt with Alpha's
assembled prompt, and returns the modified request.
"""

from typing import Any

import logfire

from .system_prompt import assemble


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
    with logfire.span(
        "weave",
        client=client or "unknown",
        hostname=hostname or "unknown",
        model=body.get("model", "unknown"),
    ) as span:
        # Build the complete system prompt
        system_blocks = await assemble(client=client, hostname=hostname)
        span.set_attribute("system_blocks", len(system_blocks))

        # Calculate total system prompt size
        total_chars = sum(
            len(block.get("text", "")) if isinstance(block, dict) else 0
            for block in system_blocks
        )
        span.set_attribute("system_chars", total_chars)

        # Get existing system from request
        existing_system = body.get("system")

        if existing_system is None:
            # No system prompt - just add ours
            body["system"] = system_blocks
            span.set_attribute("merge_mode", "replace_empty")

        elif isinstance(existing_system, list) and len(existing_system) >= 1:
            # SDK sends: [0]=billing header, [1]=SDK boilerplate, [2]=safety envelope
            # Keep the billing header (element 0), replace everything else
            billing_header = existing_system[0]
            body["system"] = [billing_header] + system_blocks
            span.set_attribute("merge_mode", "keep_billing_header")
            span.set_attribute("original_blocks", len(existing_system))

        elif isinstance(existing_system, str):
            # Old-style string system prompt - replace entirely
            body["system"] = system_blocks
            span.set_attribute("merge_mode", "replace_string")

        else:
            logfire.warn(f"Unexpected system format: {type(existing_system)}, replacing")
            body["system"] = system_blocks
            span.set_attribute("merge_mode", "replace_unexpected")

        logfire.info(f"Wove Alpha into request ({len(system_blocks)} blocks)")
        return body
