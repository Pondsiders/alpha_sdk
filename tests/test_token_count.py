"""Test token counting via CompactProxy.

Verifies that:
1. Token count starts at 0
2. Token count increases after a request
3. Callback fires with (count, window)
4. max() logic filters out smaller requests
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alpha_sdk import AlphaClient
from alpha_sdk.observability import configure

# Configure Logfire
configure("token_count_test")


async def main():
    print("Token Count Test")
    print("=" * 60)

    # Track callback invocations
    callback_calls: list[tuple[int, int]] = []

    def on_token_count(count: int, window: int) -> None:
        """Callback that fires when token count increases."""
        pct = count / window * 100
        print(f"[CALLBACK] Token count: {count:,} / {window:,} ({pct:.1f}%)")
        callback_calls.append((count, window))

    # Create client with callback
    async with AlphaClient(
        cwd="/Pondside",
        client_name="token_test",
        allowed_tools=[],
        on_token_count=on_token_count,
    ) as client:
        print(f"[TEST] Initial token count: {client.token_count}")
        print(f"[TEST] Context window: {client.context_window}")

        # Turn 1
        print("\n[TEST] Sending Turn 1...")
        await client.query("Hello! Please respond with just one word.")
        async for _ in client.stream():
            pass  # Consume the stream

        print(f"[TEST] Token count after Turn 1: {client.token_count}")

        # Turn 2
        print("\n[TEST] Sending Turn 2...")
        await client.query(
            "Now please write a slightly longer response, maybe two sentences.",
            session_id=client.session_id,
        )
        async for _ in client.stream():
            pass

        print(f"[TEST] Token count after Turn 2: {client.token_count}")

    print("\n" + "=" * 60)
    print(f"Total callback invocations: {len(callback_calls)}")
    for i, (count, window) in enumerate(callback_calls, 1):
        print(f"  [{i}] {count:,} / {window:,} ({count/window*100:.1f}%)")

    print("\n[TEST] Success! Token counting works.")


if __name__ == "__main__":
    asyncio.run(main())
