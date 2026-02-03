"""Quick test script for AlphaClient."""

import asyncio
import logging

from src.alpha_sdk import AlphaClient
from src.alpha_sdk.observability import configure

# Configure Logfire - will send to https://logfire-us.pydantic.dev/jefferyharrell/pondside
configure("alpha_sdk_test")

# Also log to console
logging.getLogger().setLevel(logging.INFO)


async def main():
    print("Creating AlphaClient...")

    async with AlphaClient(
        cwd="/Pondside",
        client_name="test_script",
        allowed_tools=["Read", "Bash"],  # Minimal tools for testing
    ) as client:
        print(f"Connected! Proxy running.")

        print("\nSending test prompt...")
        await client.query("Say 'hello from alpha_sdk!' and nothing else.")

        print("\nStreaming response:")
        async for event in client.stream():
            # Print the event type and any text content
            event_type = type(event).__name__
            print(f"  [{event_type}]", end="")

            # Try to extract text if it's an AssistantMessage
            if hasattr(event, 'content'):
                for block in event.content:
                    if hasattr(block, 'text'):
                        print(f" {block.text}", end="")

            # Show session ID from result
            if hasattr(event, 'session_id'):
                print(f" session_id={event.session_id}", end="")

            print()

        print(f"\nFinal session ID: {client.session_id}")
        print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
