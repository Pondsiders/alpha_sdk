"""AlphaClient - the main interface to Alpha.

Wraps Claude Agent SDK with:
- Automatic proxy setup for request transformation
- Long-lived client with session switching
- Memory recall before prompts
- Memorables extraction after turns
- Session discovery and management
"""

import asyncio
import logging
import os
from typing import Any, AsyncGenerator, AsyncIterable

import logfire

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    TextBlock,
)
from claude_agent_sdk.types import StreamEvent

from .proxy import AlphaProxy
from .weave import weave
from .memories.recall import recall
from .memories.suggest import suggest
from .sessions import list_sessions, get_session_path, get_sessions_dir, SessionInfo

logger = logging.getLogger(__name__)


class AlphaClient:
    """Long-lived client that wraps Claude Agent SDK with Alpha transformation.

    The SDK has ~4 second startup cost, so we keep one client alive and reuse
    it across conversations. Session switching is handled internally.

    Usage (long-lived):
        client = AlphaClient(cwd="/Pondside")
        await client.connect()

        # Multiple conversations
        await client.query(prompt, session_id="abc123")
        async for event in client.stream():
            yield event

        await client.query(other_prompt, session_id="xyz789")
        async for event in client.stream():
            yield event

        await client.disconnect()

    Usage (context manager for one-shot):
        async with AlphaClient(cwd="/Pondside") as client:
            await client.query(prompt, session_id=session_id)
            async for event in client.stream():
                yield event
    """

    def __init__(
        self,
        cwd: str = "/Pondside",
        client_name: str = "alpha_sdk",
        hostname: str | None = None,
        allowed_tools: list[str] | None = None,
        mcp_servers: dict | None = None,
        archive: bool = True,
        include_partial_messages: bool = True,
    ):
        """Initialize the Alpha client.

        Args:
            cwd: Working directory for the agent
            client_name: Name of the client (for logging, HUD)
            hostname: Machine hostname (auto-detected if not provided)
            allowed_tools: List of allowed tool names
            mcp_servers: Dict of MCP server configurations
            archive: Whether to archive turns to Postgres
            include_partial_messages: Stream partial messages for real-time updates
        """
        self.cwd = cwd
        self.client_name = client_name
        self.hostname = hostname
        self.allowed_tools = allowed_tools
        self.mcp_servers = mcp_servers or {}
        self.archive = archive
        self.include_partial_messages = include_partial_messages

        # Internal state
        self._proxy: AlphaProxy | None = None
        self._sdk_client: ClaudeSDKClient | None = None
        self._current_session_id: str | None = None
        self._last_user_content: str = ""
        self._last_assistant_content: str = ""

    # -------------------------------------------------------------------------
    # Session Discovery (static methods)
    # -------------------------------------------------------------------------

    @staticmethod
    def list_sessions(cwd: str = "/Pondside", limit: int = 50) -> list[SessionInfo]:
        """List available sessions for resumption.

        Args:
            cwd: Working directory
            limit: Maximum sessions to return

        Returns:
            List of SessionInfo objects, most recent first
        """
        return list_sessions(cwd, limit)

    @staticmethod
    def get_session_path(session_id: str, cwd: str = "/Pondside") -> str:
        """Get the filesystem path for a session.

        Args:
            session_id: The session UUID
            cwd: Working directory

        Returns:
            Path to the session's JSONL file
        """
        return str(get_session_path(session_id, cwd))

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def connect(self, session_id: str | None = None) -> None:
        """Connect to Claude, optionally resuming a session.

        Args:
            session_id: Session to resume, or None for new session
        """
        # Start the proxy
        self._proxy = AlphaProxy(
            weaver=weave,
            client=self.client_name,
            hostname=self.hostname,
        )
        port = self._proxy.start()

        # Set environment for SDK
        os.environ["ANTHROPIC_BASE_URL"] = self._proxy.base_url

        # Create SDK client
        await self._create_sdk_client(session_id)

        logger.info(f"AlphaClient connected (proxy on port {port})")

    async def disconnect(self) -> None:
        """Disconnect and clean up resources."""
        # Fire-and-forget: extract memorables from last turn
        if self._last_user_content and self._last_assistant_content:
            asyncio.create_task(
                suggest(
                    self._last_user_content,
                    self._last_assistant_content,
                    self._current_session_id or "unknown",
                )
            )

        # Disconnect SDK client
        if self._sdk_client:
            await self._sdk_client.disconnect()
            self._sdk_client = None

        # Stop proxy
        if self._proxy:
            self._proxy.stop()
            self._proxy = None

        # Restore environment
        if "ANTHROPIC_BASE_URL" in os.environ:
            del os.environ["ANTHROPIC_BASE_URL"]

        self._current_session_id = None
        logger.info("AlphaClient disconnected")

    async def __aenter__(self) -> "AlphaClient":
        """Context manager entry - connects the client."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - disconnects the client."""
        await self.disconnect()

    # -------------------------------------------------------------------------
    # Conversation
    # -------------------------------------------------------------------------

    async def query(
        self,
        prompt: str | AsyncIterable[dict[str, Any]],
        session_id: str | None = None,
        fork_from: str | None = None,
    ) -> None:
        """Send a query to the agent.

        Args:
            prompt: The user's message (string or async generator for streaming input)
            session_id: Session to resume, or None for new session
            fork_from: Session to fork from (creates new session with context)
        """
        with logfire.span(
            "alpha.query",
            session_id=session_id or "new",
            fork_from=fork_from,
            client_name=self.client_name,
        ) as span:
            # Handle session switching
            await self._ensure_session(session_id, fork_from)

            # Store prompt for memory extraction
            if isinstance(prompt, str):
                self._last_user_content = prompt
                span.set_attribute("prompt_length", len(prompt))
                span.set_attribute("prompt_preview", prompt[:200])

                # Recall memories based on the prompt
                if self._current_session_id:
                    memories = await recall(prompt, self._current_session_id)
                    if memories:
                        span.set_attribute("memories_recalled", len(memories))
                        logger.info(f"Recalled {len(memories)} memories")

            # Send to SDK
            if self._sdk_client:
                await self._sdk_client.query(prompt)
            else:
                raise RuntimeError("Client not connected. Call connect() first.")

    async def stream(self) -> AsyncGenerator[Any, None]:
        """Stream responses from the agent.

        Yields:
            Message objects from the SDK (StreamEvent, AssistantMessage, ResultMessage, etc.)
        """
        if not self._sdk_client:
            raise RuntimeError("Client not connected. Call connect() first.")

        with logfire.span("alpha.stream", session_id=self._current_session_id or "unknown") as span:
            assistant_text_parts: list[str] = []
            message_count = 0

            async for message in self._sdk_client.receive_response():
                message_count += 1

                # Accumulate assistant text for memory extraction
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            assistant_text_parts.append(block.text)

                # Capture session ID and stats from result
                if isinstance(message, ResultMessage):
                    self._current_session_id = message.session_id
                    span.set_attribute("final_session_id", message.session_id)
                    span.set_attribute("duration_ms", message.duration_ms)
                    span.set_attribute("num_turns", message.num_turns)
                    if message.total_cost_usd:
                        span.set_attribute("cost_usd", message.total_cost_usd)
                    if message.usage:
                        span.set_attribute("usage", str(message.usage))

                yield message

            # Store accumulated text for memorables extraction
            self._last_assistant_content = "".join(assistant_text_parts)
            span.set_attribute("message_count", message_count)
            span.set_attribute("response_length", len(self._last_assistant_content))

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def session_id(self) -> str | None:
        """Get the current session ID."""
        return self._current_session_id

    @property
    def connected(self) -> bool:
        """Check if the client is connected."""
        return self._sdk_client is not None

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    async def _ensure_session(
        self,
        session_id: str | None,
        fork_from: str | None = None,
    ) -> None:
        """Ensure we have the right SDK client for the requested session.

        If session_id matches current, reuse. Otherwise, recreate.
        """
        needs_new_client = False

        if fork_from:
            # Forking always creates a new session
            needs_new_client = True
        elif session_id is None:
            # New session requested
            if self._current_session_id is not None:
                needs_new_client = True
        elif session_id != self._current_session_id:
            # Different session requested
            needs_new_client = True

        if needs_new_client:
            await self._create_sdk_client(session_id, fork_from)

    async def _create_sdk_client(
        self,
        session_id: str | None = None,
        fork_from: str | None = None,
    ) -> None:
        """Create or recreate the SDK client.

        Args:
            session_id: Session to resume
            fork_from: Session to fork from
        """
        # Disconnect existing client if any
        if self._sdk_client:
            await self._sdk_client.disconnect()

        # Build options
        options = ClaudeAgentOptions(
            cwd=self.cwd,
            allowed_tools=self.allowed_tools or [],
            mcp_servers=self.mcp_servers,
            include_partial_messages=self.include_partial_messages,
            resume=session_id,
            fork_session=fork_from is not None,
        )

        # If forking, we need to set resume to the fork source
        if fork_from:
            options.resume = fork_from

        # Create and connect
        self._sdk_client = ClaudeSDKClient(options)
        await self._sdk_client.connect()

        self._current_session_id = session_id
        logger.info(
            f"SDK client created (session={session_id or 'new'}, fork={fork_from})"
        )
