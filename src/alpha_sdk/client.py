"""AlphaClient - the main interface to Alpha.

Proxyless architecture:
- System prompt is assembled once at connect() and passed to SDK
- Orientation (capsules, letter, here, context, etc.) goes in first user message
- Memories and memorables go in user content, not system prompt
- No proxy, no canary, no interception
"""

import asyncio
import json
import os
from typing import Any, AsyncGenerator, AsyncIterable, Literal

import logfire
import redis.asyncio as aioredis

# Permission modes supported by Claude Agent SDK
PermissionMode = Literal[
    "default",           # Standard permission behavior
    "acceptEdits",       # Auto-accept file edits
    "plan",              # Planning mode - no execution
    "bypassPermissions"  # Bypass all permission checks (use with caution)
]

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    UserMessage,
)
from claude_agent_sdk.types import (
    StreamEvent,
    HookMatcher,
    PreCompactHookInput,
    HookContext,
    ToolUseBlock,
    ToolResultBlock,
)

from .memories.recall import recall
from .memories.suggest import suggest
from .sessions import list_sessions, get_session_path, get_sessions_dir, SessionInfo
from .system_prompt import assemble
from .system_prompt.soul import get_soul

# Redis URL for memorables
REDIS_URL = os.environ.get("REDIS_URL", "redis://alpha-pi:6379")


def _message_to_dict(message: Any) -> dict:
    """Convert an SDK message to a dict for logging.

    Handles the various dataclass types from claude_agent_sdk.
    """
    from dataclasses import asdict, is_dataclass

    if is_dataclass(message) and not isinstance(message, type):
        try:
            return asdict(message)
        except Exception:
            # Some fields might not be serializable
            return {"type": type(message).__name__, "repr": repr(message)[:500]}
    elif hasattr(message, "__dict__"):
        return {"type": type(message).__name__, **message.__dict__}
    else:
        return {"type": type(message).__name__, "repr": repr(message)[:500]}


def _format_memory(memory: dict) -> str:
    """Format a memory for inclusion in user content.

    Creates human-readable memory text with relative timestamps.
    """
    import pendulum

    mem_id = memory.get("id", "?")
    created_at = memory.get("created_at", "")
    content = memory.get("content", "").strip()
    score = memory.get("score")

    # Simple relative time formatting
    relative_time = created_at  # fallback
    try:
        dt = pendulum.parse(created_at)
        now = pendulum.now(dt.timezone or "America/Los_Angeles")
        diff = now.diff(dt)
        if diff.in_days() == 0:
            relative_time = f"today at {dt.format('h:mm A')}"
        elif diff.in_days() == 1:
            relative_time = f"yesterday at {dt.format('h:mm A')}"
        elif diff.in_days() < 7:
            relative_time = f"{diff.in_days()} days ago"
        elif diff.in_days() < 30:
            weeks = diff.in_days() // 7
            relative_time = f"{weeks} week{'s' if weeks > 1 else ''} ago"
        else:
            relative_time = dt.format("ddd MMM D YYYY")
    except Exception:
        pass

    # Include score if present (helps with debugging/transparency)
    score_str = f", score {score:.2f}" if score else ""
    return f"## Memory #{mem_id} ({relative_time}{score_str})\n{content}"


async def _get_pending_memorables(session_id: str) -> list[str] | None:
    """Get pending memorables from Redis and clear them.

    Returns:
        List of memorable strings, or None if none pending
    """
    if not REDIS_URL:
        return None

    try:
        redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
        key = f"intro:memorables:{session_id}"

        # Get all memorables
        memorables = await redis_client.lrange(key, 0, -1)

        # Clear them (we've consumed them)
        if memorables:
            await redis_client.delete(key)

        await redis_client.aclose()
        return memorables if memorables else None

    except Exception as e:
        logfire.warn(f"Failed to get memorables: {e}")
        return None


class AlphaClient:
    """Long-lived client that wraps Claude Agent SDK.

    Proxyless architecture:
    - System prompt = just the soul (small, truly static)
    - Orientation = injected in first user message of session
    - Memories = per-turn, in user content
    - Memorables = per-turn nudge, in user content

    Usage:
        async with AlphaClient(cwd="/Pondside") as client:
            await client.query("Hello!", session_id=None)  # New session
            async for event in client.stream():
                print(event)

            await client.query("Continue...", session_id=client.session_id)
            async for event in client.stream():
                print(event)
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
        permission_mode: PermissionMode = "default",
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
            permission_mode: How to handle tool permission requests
        """
        self.cwd = cwd
        self.client_name = client_name
        self.hostname = hostname
        self.allowed_tools = allowed_tools
        self.mcp_servers = mcp_servers or {}
        self.archive = archive
        self.include_partial_messages = include_partial_messages
        self.permission_mode = permission_mode

        # Internal state
        self._sdk_client: ClaudeSDKClient | None = None
        self._current_session_id: str | None = None
        self._system_prompt: str | None = None  # Just the soul, assembled once
        self._orientation_blocks: list[dict] | None = None  # Cached for re-injection

        # Turn state
        self._last_user_content: str = ""  # Just the user's text (for memorables)
        self._last_content_blocks: list[dict] = []  # Full content array (for observability)
        self._last_assistant_content: str = ""
        self._turn_span: logfire.LogfireSpan | None = None
        self._suggest_task: asyncio.Task | None = None

        # Compaction flag - set by PreCompact hook, cleared after re-orientation
        self._needs_reorientation: bool = False

    # -------------------------------------------------------------------------
    # Session Discovery (static methods)
    # -------------------------------------------------------------------------

    @staticmethod
    def list_sessions(cwd: str = "/Pondside", limit: int = 50) -> list[SessionInfo]:
        """List available sessions for resumption."""
        return list_sessions(cwd, limit)

    @staticmethod
    def get_session_path(session_id: str, cwd: str = "/Pondside") -> str:
        """Get the filesystem path for a session."""
        return str(get_session_path(session_id, cwd))

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def connect(self, session_id: str | None = None) -> None:
        """Connect to Claude.

        Assembles the system prompt (soul only) and creates the SDK client.
        Orientation will be injected on the first query.

        Args:
            session_id: Session to resume, or None for new session
        """
        with logfire.span("alpha.connect") as span:
            # Build the system prompt - just the soul
            self._system_prompt = f"# Alpha\n\n{get_soul()}"
            span.set_attribute("system_prompt_length", len(self._system_prompt))

            # Pre-build orientation blocks (will be injected on first turn)
            self._orientation_blocks = await self._build_orientation()
            span.set_attribute("orientation_blocks", len(self._orientation_blocks))

            # Create SDK client with system prompt
            await self._create_sdk_client(session_id)

            logfire.info(f"AlphaClient connected (soul: {len(self._system_prompt)} chars)")

    async def disconnect(self) -> None:
        """Disconnect and clean up resources."""
        # Wait for any pending suggest task
        if self._suggest_task is not None:
            try:
                await self._suggest_task
            except Exception as e:
                logfire.debug(f"Suggest task error on disconnect: {e}")
            self._suggest_task = None

        # Disconnect SDK client
        if self._sdk_client:
            await self._sdk_client.disconnect()
            self._sdk_client = None

        self._current_session_id = None
        logfire.debug("AlphaClient disconnected")

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
        prompt: str | list[dict[str, Any]],
        session_id: str | None = None,
    ) -> None:
        """Send a query to the agent.

        Args:
            prompt: The user's message - string or content blocks
            session_id: Session to resume, or None for new session
        """
        # Start the root turn span
        self._turn_span = logfire.span(
            "alpha.turn",
            session_id=session_id or "new",
            client_name=self.client_name,
        )
        self._turn_span.__enter__()

        with logfire.span("alpha.query") as span:
            # Handle session switching
            await self._ensure_session(session_id)

            if not self._sdk_client:
                raise RuntimeError("Client not connected. Call connect() first.")

            # Extract text for memory operations
            if isinstance(prompt, str):
                prompt_text = prompt
            else:
                text_parts = [b.get("text", "") for b in prompt if b.get("type") == "text"]
                prompt_text = " ".join(text_parts)

            self._last_user_content = prompt_text
            span.set_attribute("prompt_preview", prompt_text[:200])

            # Build content blocks
            content_blocks: list[dict[str, Any]] = []

            # Check if we need orientation (new session or post-compact)
            needs_orientation = (session_id is None) or self._needs_reorientation

            if needs_orientation:
                # Re-build orientation in case it's stale (post-compact)
                if self._needs_reorientation:
                    self._orientation_blocks = await self._build_orientation()

                # Add orientation blocks
                if self._orientation_blocks:
                    content_blocks.extend(self._orientation_blocks)
                    span.set_attribute("orientation_injected", True)
                    span.set_attribute("orientation_blocks", len(self._orientation_blocks))

                self._needs_reorientation = False

            # Check for memorables from previous turn (the nudge)
            if self._current_session_id:
                memorables = await _get_pending_memorables(self._current_session_id)
                if memorables:
                    nudge = "## Intro speaks\n\n"
                    nudge += "Alpha, consider storing these from the previous turn:\n"
                    nudge += "\n".join(f"- {m}" for m in memorables)
                    content_blocks.append({"type": "text", "text": nudge})
                    span.set_attribute("memorables_nudged", len(memorables))

            # Recall memories for this prompt
            memories = await recall(prompt_text, self._current_session_id or "new")
            if memories:
                for mem in memories:
                    content_blocks.append({
                        "type": "text",
                        "text": _format_memory(mem)
                    })
                span.set_attribute("memories_recalled", len(memories))

            # Add the user's actual prompt
            if isinstance(prompt, str):
                content_blocks.append({"type": "text", "text": prompt})
            else:
                content_blocks.extend(prompt)

            span.set_attribute("content_blocks", len(content_blocks))

            # Store for observability (full content, not just user text)
            self._last_content_blocks = content_blocks

            # Send via transport bypass (SDK query() only takes strings)
            message = {
                "type": "user",
                "message": {"role": "user", "content": content_blocks},
                "session_id": self._current_session_id or "new",
            }
            await self._sdk_client._transport.write(json.dumps(message) + "\n")
            logfire.debug(f"Sent message with {len(content_blocks)} content blocks")

    async def stream(self) -> AsyncGenerator[Any, None]:
        """Stream responses from the agent.

        Progressive observability: gen_ai.* attributes update after each message,
        so if the turn hangs you can see everything up to that point in Logfire.

        Yields:
            Message objects from the SDK
        """
        if not self._sdk_client:
            raise RuntimeError("Client not connected. Call connect() first.")

        try:
            with logfire.span("alpha.stream") as span:
                assistant_text_parts: list[str] = []
                message_count = 0

                # Progressive accumulation for gen_ai.* attributes
                # Input: user message + tool results
                # Output: assistant text + tool calls
                input_messages: list[dict] = []
                output_messages: list[dict] = []

                # Initialize with user message (our injected content)
                user_parts = []
                for block in self._last_content_blocks:
                    if block.get("type") == "text":
                        user_parts.append({
                            "type": "text",
                            "content": block.get("text", "")
                        })
                input_messages.append({"role": "user", "parts": user_parts})

                # Set initial gen_ai attributes (system prompt + user message)
                if self._turn_span:
                    if self._system_prompt:
                        self._turn_span.set_attribute(
                            "gen_ai.system_instructions",
                            json.dumps([{"type": "text", "content": self._system_prompt}])
                        )
                    self._turn_span.set_attribute("gen_ai.input.messages", json.dumps(input_messages))
                    self._turn_span.set_attribute("gen_ai.output.messages", json.dumps(output_messages))
                    self._turn_span.set_attribute("gen_ai.operation.name", "chat")
                    self._turn_span.set_attribute("gen_ai.system", "anthropic")

                async for message in self._sdk_client.receive_response():
                    message_count += 1

                    # Log non-streaming messages for debugging
                    # StreamEvent is too noisy (one per SSE delta), skip it
                    if not isinstance(message, StreamEvent):
                        logfire.debug(
                            f"sdk.message.{type(message).__name__}",
                            message=_message_to_dict(message),
                        )

                    # Handle assistant messages (text + tool calls)
                    if isinstance(message, AssistantMessage):
                        assistant_parts = []
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                assistant_text_parts.append(block.text)
                                assistant_parts.append({
                                    "type": "text",
                                    "content": block.text
                                })
                            elif isinstance(block, ToolUseBlock):
                                assistant_parts.append({
                                    "type": "tool_call",
                                    "id": block.id,
                                    "name": block.name,
                                    "input": block.input
                                })
                        if assistant_parts:
                            output_messages.append({"role": "assistant", "parts": assistant_parts})
                            # Update span progressively
                            if self._turn_span:
                                self._turn_span.set_attribute("gen_ai.output.messages", json.dumps(output_messages))

                    # Handle user messages (tool results)
                    elif isinstance(message, UserMessage):
                        if isinstance(message.content, list):
                            tool_result_parts = []
                            for block in message.content:
                                if isinstance(block, ToolResultBlock):
                                    # Format tool result content
                                    result_content = block.content
                                    if isinstance(result_content, list):
                                        result_content = json.dumps(result_content)
                                    elif result_content is None:
                                        result_content = ""
                                    tool_result_parts.append({
                                        "type": "tool_result",
                                        "tool_use_id": block.tool_use_id,
                                        "content": str(result_content)[:500],  # Truncate for sanity
                                        "is_error": block.is_error or False
                                    })
                            if tool_result_parts:
                                input_messages.append({"role": "user", "parts": tool_result_parts})
                                # Update span progressively
                                if self._turn_span:
                                    self._turn_span.set_attribute("gen_ai.input.messages", json.dumps(input_messages))

                    # Capture session ID and stats from result
                    if isinstance(message, ResultMessage):
                        self._current_session_id = message.session_id
                        span.set_attribute("session_id", message.session_id)
                        span.set_attribute("duration_ms", message.duration_ms)
                        span.set_attribute("num_turns", message.num_turns)
                        if message.total_cost_usd:
                            span.set_attribute("cost_usd", message.total_cost_usd)
                        if message.usage:
                            span.set_attribute("usage", str(message.usage))
                            if self._turn_span:
                                input_tokens = message.usage.get("input_tokens", 0)
                                output_tokens = message.usage.get("output_tokens", 0)
                                if input_tokens:
                                    self._turn_span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
                                if output_tokens:
                                    self._turn_span.set_attribute("gen_ai.usage.output_tokens", output_tokens)

                        if self._turn_span:
                            self._turn_span.set_attribute("session_id", message.session_id)
                            self._turn_span.set_attribute("duration_ms", message.duration_ms)
                            if message.total_cost_usd:
                                self._turn_span.set_attribute("cost_usd", message.total_cost_usd)

                    yield message

                # Store accumulated text for memorables extraction
                self._last_assistant_content = "".join(assistant_text_parts)
                span.set_attribute("message_count", message_count)
                span.set_attribute("response_length", len(self._last_assistant_content))

                if self._turn_span:
                    self._turn_span.set_attribute("response_length", len(self._last_assistant_content))

                # Launch suggest as background task
                if self._last_user_content and self._last_assistant_content:
                    self._suggest_task = asyncio.create_task(
                        suggest(
                            self._last_user_content,
                            self._last_assistant_content,
                            self._current_session_id or "unknown",
                        )
                    )
        finally:
            # End the root turn span
            if self._turn_span:
                self._turn_span.__exit__(None, None, None)
                self._turn_span = None

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

    async def _build_orientation(self) -> list[dict[str, Any]]:
        """Build orientation blocks for session start.

        This includes everything except the soul (which is in system prompt):
        - Capsules (yesterday, last night)
        - Letter from last night
        - Today so far
        - Here (client, machine, weather)
        - ALPHA.md context files
        - Events
        - Todos
        """
        with logfire.span("build_orientation") as span:
            # Use the existing assemble() but we'll extract just the non-soul parts
            all_blocks = await assemble(
                client=self.client_name,
                hostname=self.hostname,
            )

            # Skip the first block (which is the soul)
            # The soul starts with "# Alpha\n\n"
            orientation_blocks = []
            for block in all_blocks:
                text = block.get("text", "")
                if not text.startswith("# Alpha\n\n"):
                    orientation_blocks.append(block)

            span.set_attribute("orientation_blocks", len(orientation_blocks))
            return orientation_blocks

    async def _on_pre_compact(
        self,
        input: PreCompactHookInput,
        tool_use_id: str | None,
        context: HookContext,
    ) -> dict[str, Any]:
        """Hook called before compaction - flag that we need to re-orient."""
        logfire.info("Compaction triggered, will re-orient on next turn")
        self._needs_reorientation = True
        return {"continue_": True}

    async def _ensure_session(self, session_id: str | None) -> None:
        """Ensure we have the right SDK client for the requested session."""
        needs_new_client = False

        if session_id is None:
            # New session requested
            if self._current_session_id is not None:
                needs_new_client = True
        elif session_id != self._current_session_id:
            # Different session requested
            needs_new_client = True

        if needs_new_client:
            await self._create_sdk_client(session_id)

    async def _create_sdk_client(self, session_id: str | None = None) -> None:
        """Create or recreate the SDK client."""
        # Disconnect existing client if any
        if self._sdk_client:
            await self._sdk_client.disconnect()

        # Build hooks config for PreCompact
        hooks = {
            "PreCompact": [
                HookMatcher(
                    matcher=None,  # Match all
                    hooks=[self._on_pre_compact],
                )
            ]
        }

        # Build options with our system prompt
        options = ClaudeAgentOptions(
            cwd=self.cwd,
            system_prompt=self._system_prompt,  # Just the soul!
            allowed_tools=self.allowed_tools or [],
            mcp_servers=self.mcp_servers,
            include_partial_messages=self.include_partial_messages,
            resume=session_id,
            permission_mode=self.permission_mode,
            hooks=hooks,
        )

        # Create and connect
        self._sdk_client = ClaudeSDKClient(options)
        await self._sdk_client.connect()

        self._current_session_id = session_id
        logfire.debug(f"SDK client created (session={session_id or 'new'})")
