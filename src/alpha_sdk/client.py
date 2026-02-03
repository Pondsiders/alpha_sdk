"""AlphaClient - the main interface to Alpha.

Wraps Claude Agent SDK with:
- Automatic proxy setup for request transformation
- Memory recall before prompts
- Memorables extraction after turns
- Session management
"""

import asyncio
import logging
import os
from typing import Any, AsyncGenerator

from .proxy import AlphaProxy
from .weave import weave
from .memories.recall import recall
from .memories.suggest import suggest

logger = logging.getLogger(__name__)


class AlphaClient:
    """Context manager that wraps Claude Agent SDK with Alpha transformation.

    Usage:
        async with AlphaClient(session_id="abc123") as client:
            await client.query("Hello!")
            async for event in client.stream():
                print(event)

    The client automatically:
    - Starts a proxy to intercept and transform requests
    - Recalls relevant memories before the prompt
    - Extracts memorables after each turn
    - Manages session state
    """

    def __init__(
        self,
        session_id: str | None = None,
        fork_from: str | None = None,
        client_name: str = "alpha_sdk",
        hostname: str | None = None,
        allowed_tools: list[str] | None = None,
        mcp_servers: dict | None = None,
        archive: bool = True,
        cwd: str = "/Pondside",
    ):
        """Initialize the Alpha client.

        Args:
            session_id: Session ID to resume, or None for new session
            fork_from: Session ID to fork from (creates new session with context)
            client_name: Name of the client (for logging, HUD)
            hostname: Machine hostname (auto-detected if not provided)
            allowed_tools: List of allowed tool names
            mcp_servers: Dict of MCP server configurations
            archive: Whether to archive turns to Postgres
            cwd: Working directory for the agent
        """
        self.session_id = session_id
        self.fork_from = fork_from
        self.client_name = client_name
        self.hostname = hostname
        self.allowed_tools = allowed_tools
        self.mcp_servers = mcp_servers or {}
        self.archive = archive
        self.cwd = cwd

        # Internal state
        self._proxy: AlphaProxy | None = None
        self._sdk_client = None
        self._last_user_content: str = ""
        self._last_assistant_content: str = ""

    async def __aenter__(self) -> "AlphaClient":
        """Start the proxy and initialize the SDK client."""
        # Start the proxy
        self._proxy = AlphaProxy(
            weaver=weave,
            client=self.client_name,
            hostname=self.hostname,
        )
        port = self._proxy.start()

        # Set environment for SDK
        os.environ["ANTHROPIC_BASE_URL"] = self._proxy.base_url

        logger.info(f"AlphaClient ready (proxy on port {port})")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Stop the proxy and clean up."""
        # Fire-and-forget: extract memorables from this turn
        if self._last_user_content and self._last_assistant_content and self.session_id:
            asyncio.create_task(
                suggest(
                    self._last_user_content,
                    self._last_assistant_content,
                    self.session_id,
                )
            )

        # Stop proxy
        if self._proxy:
            self._proxy.stop()
            self._proxy = None

        # Restore environment
        if "ANTHROPIC_BASE_URL" in os.environ:
            del os.environ["ANTHROPIC_BASE_URL"]

        logger.info("AlphaClient closed")

    async def query(self, prompt: str | AsyncGenerator) -> None:
        """Send a query to the agent.

        Args:
            prompt: Either a string (single message) or an async generator
                   that yields message dicts for streaming input.
        """
        # For now, we just store the prompt for memory purposes
        # The actual SDK interaction will be added when we integrate
        # with the full Claude Agent SDK
        if isinstance(prompt, str):
            self._last_user_content = prompt

            # Recall memories based on the prompt
            if self.session_id:
                memories = await recall(prompt, self.session_id)
                if memories:
                    logger.info(f"Recalled {len(memories)} memories")

        logger.info("Query submitted (SDK integration pending)")

    async def stream(self) -> AsyncGenerator[Any, None]:
        """Stream responses from the agent.

        Yields:
            Event objects from the SDK (StreamEvent, ToolCall, etc.)
        """
        # Placeholder - will yield SDK events when integrated
        logger.info("Streaming (SDK integration pending)")
        yield {"type": "placeholder", "message": "SDK integration pending"}

    def get_session_id(self) -> str | None:
        """Get the current session ID."""
        return self.session_id
