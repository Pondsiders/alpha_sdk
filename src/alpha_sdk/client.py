"""AlphaClient - the main interface to Alpha.

Architecture:
- System prompt is assembled once at connect() and passed to SDK
- Orientation (capsules, letter, here, context, etc.) goes in first user message
- Memories and memorables go in user content, not system prompt
- Minimal proxy intercepts only compact prompts for rewriting
"""

import asyncio
import json
import os
from typing import Any, AsyncGenerator, AsyncIterable, Literal

import logfire
import pendulum
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

from .compact_proxy import CompactProxy
from .memories.recall import recall
from .memories.suggest import suggest
from .sessions import list_sessions, get_session_path, get_sessions_dir, SessionInfo
from .system_prompt import assemble
from .system_prompt.soul import get_soul

# Redis URL for memorables
REDIS_URL = os.environ.get("REDIS_URL", "redis://alpha-pi:6379")

# Store original ANTHROPIC_BASE_URL so we can restore it
_ORIGINAL_ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL")


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

    # The model that IS Alpha. Pinned at the SDK level, not configurable per-client.
    # When we upgrade Alpha to a new model, we bump alpha_sdk version.
    ALPHA_MODEL = "claude-opus-4-5-20251101"

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
        self._compact_proxy: CompactProxy | None = None  # For compact prompt rewriting

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

        Starts the compact proxy, assembles the system prompt (soul only),
        and creates the SDK client. Orientation will be injected on first query.

        Args:
            session_id: Session to resume, or None for new session
        """
        with logfire.span("alpha.connect") as span:
            # Start compact proxy (intercepts only compact prompts for rewriting)
            self._compact_proxy = CompactProxy()
            await self._compact_proxy.start()
            os.environ["ANTHROPIC_BASE_URL"] = self._compact_proxy.base_url
            span.set_attribute("proxy_port", self._compact_proxy.port)
            span.set_attribute("anthropic_base_url", self._compact_proxy.base_url)
            logfire.info(f"Proxy started, ANTHROPIC_BASE_URL={self._compact_proxy.base_url}")

            # Build the system prompt - just the soul
            self._system_prompt = f"# Alpha\n\n{get_soul()}"
            span.set_attribute("system_prompt_length", len(self._system_prompt))

            # Pre-build orientation blocks (will be injected on first turn)
            self._orientation_blocks = await self._build_orientation()
            span.set_attribute("orientation_blocks", len(self._orientation_blocks))

            # Create SDK client with system prompt
            await self._create_sdk_client(session_id)

            logfire.info(f"AlphaClient connected (soul: {len(self._system_prompt)} chars, proxy: {self._compact_proxy.port})")

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

        # Stop compact proxy and restore original ANTHROPIC_BASE_URL
        if self._compact_proxy:
            await self._compact_proxy.stop()
            self._compact_proxy = None
            if _ORIGINAL_ANTHROPIC_BASE_URL:
                os.environ["ANTHROPIC_BASE_URL"] = _ORIGINAL_ANTHROPIC_BASE_URL
            elif "ANTHROPIC_BASE_URL" in os.environ:
                del os.environ["ANTHROPIC_BASE_URL"]

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
        # Extract text for span naming and memory operations
        if isinstance(prompt, str):
            prompt_text = prompt
        else:
            text_parts = [b.get("text", "") for b in prompt if b.get("type") == "text"]
            prompt_text = " ".join(text_parts)

        # Build span name from prompt preview (first 50 chars, single line)
        prompt_preview = prompt_text[:50].replace("\n", " ").strip()
        if len(prompt_text) > 50:
            prompt_preview += "…"

        # Start the root turn span with prompt preview in the name
        self._turn_span = logfire.span(
            "alpha.turn: {prompt_preview}",
            prompt_preview=prompt_preview,
            session_id=session_id or "new",
            client_name=self.client_name,
        )
        self._turn_span.__enter__()

        # Set gen_ai attributes for Model Run card (progressively enhanced)
        self._turn_span.set_attribute("gen_ai.system", "anthropic")
        self._turn_span.set_attribute("gen_ai.operation.name", "chat")
        self._turn_span.set_attribute("gen_ai.request.model", self.ALPHA_MODEL)
        if session_id:
            self._turn_span.set_attribute("gen_ai.conversation.id", session_id)

        # System instructions = just the soul (the static system prompt)
        if self._system_prompt:
            self._turn_span.set_attribute(
                "gen_ai.system_instructions",
                json.dumps([{"type": "text", "content": self._system_prompt}])
            )

        # Input messages placeholder - will be updated in query() after content blocks are built
        # Set empty now so attribute exists even if request hangs before content is built
        self._turn_span.set_attribute("gen_ai.input.messages", json.dumps([]))

        # Initialize output messages (will be progressively updated in stream())
        self._turn_span.set_attribute("gen_ai.output.messages", json.dumps([]))

        # Set trace context on proxy so its spans nest under this turn
        if self._compact_proxy:
            self._compact_proxy.set_trace_context(logfire.get_context())

        with logfire.span("alpha.query") as span:
            # Handle session switching
            await self._ensure_session(session_id)

            if not self._sdk_client:
                raise RuntimeError("Client not connected. Call connect() first.")

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

            # Add PSO-8601 timestamp right before user's prompt
            # Format: "[Sent Thu Feb 5 2026, 8:03 AM]"
            sent_at = pendulum.now("America/Los_Angeles").format("ddd MMM D YYYY, h:mm A")
            content_blocks.append({
                "type": "text",
                "text": f"[Sent {sent_at}]"
            })

            # Add the user's actual prompt
            if isinstance(prompt, str):
                content_blocks.append({"type": "text", "text": prompt})
            else:
                content_blocks.extend(prompt)

            span.set_attribute("content_blocks", len(content_blocks))

            # Store for observability (full content, not just user text)
            self._last_content_blocks = content_blocks

            # Update turn span with full structured input (now that we have all content blocks)
            if self._turn_span:
                user_parts = []
                for block in content_blocks:
                    if block.get("type") == "text":
                        user_parts.append({"type": "text", "content": block.get("text", "")})
                self._turn_span.set_attribute(
                    "gen_ai.input.messages",
                    json.dumps([{"role": "user", "parts": user_parts}])
                )

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

        Creates one span per API inference call:
        - alpha.inference.0: user prompt → assistant (possibly with tool calls)
        - alpha.inference.1: tool_call + tool_result → assistant continues
        - alpha.inference.N: tool_call + tool_result → final response

        Each inference span has its own gen_ai.input.messages and gen_ai.output.messages.
        Tool calls are paired with their results in the input for visual clarity in Logfire.

        Yields:
            Message objects from the SDK
        """
        if not self._sdk_client:
            raise RuntimeError("Client not connected. Call connect() first.")

        try:
            with logfire.span("alpha.stream") as stream_span:
                assistant_text_parts: list[str] = []
                message_count = 0
                inference_count = 0

                # Current inference span state
                inference_span: logfire.LogfireSpan | None = None
                input_messages: list[dict] = []
                output_messages: list[dict] = []

                # Stash tool calls so we can pair them with results
                # Maps tool_use_id -> tool_call dict
                pending_tool_calls: dict[str, dict] = {}

                # Track accumulated output for turn-level gen_ai.output.messages
                turn_output_parts: list[dict] = []
                last_finish_reason: str | None = None

                def _start_inference_span(inference_num: int, initial_input: list[dict]) -> logfire.LogfireSpan:
                    """Start a new inference span with initial input."""
                    span = logfire.span(
                        "alpha.inference.{n}",
                        n=inference_num,
                    )
                    span.__enter__()
                    # Set initial attributes
                    if self._system_prompt:
                        span.set_attribute(
                            "gen_ai.system_instructions",
                            json.dumps([{"type": "text", "content": self._system_prompt}])
                        )
                    span.set_attribute("gen_ai.input.messages", json.dumps(initial_input))
                    span.set_attribute("gen_ai.output.messages", json.dumps([]))
                    span.set_attribute("gen_ai.operation.name", "chat")
                    span.set_attribute("gen_ai.system", "anthropic")
                    return span

                def _end_inference_span(span: logfire.LogfireSpan, outputs: list[dict]) -> None:
                    """End an inference span with final outputs."""
                    span.set_attribute("gen_ai.output.messages", json.dumps(outputs))
                    span.__exit__(None, None, None)

                # Build initial user message from our injected content
                user_parts = []
                for block in self._last_content_blocks:
                    if block.get("type") == "text":
                        user_parts.append({
                            "type": "text",
                            "content": block.get("text", "")
                        })
                input_messages = [{"role": "user", "parts": user_parts}]

                # Start first inference span
                inference_span = _start_inference_span(inference_count, input_messages)

                async for message in self._sdk_client.receive_response():
                    message_count += 1

                    # Log non-streaming messages for debugging
                    # StreamEvent is too noisy (one per SSE delta), skip it
                    if not isinstance(message, StreamEvent):
                        msg_type = type(message).__name__
                        logfire.debug(
                            "sdk.message.{msg_type}",
                            msg_type=msg_type,
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
                                # Also track for turn-level output
                                turn_output_parts.append({
                                    "type": "text",
                                    "content": block.text
                                })
                            elif isinstance(block, ToolUseBlock):
                                tool_call = {
                                    "type": "tool_call",
                                    "id": block.id,
                                    "name": block.name,
                                    "arguments": block.input,
                                }
                                assistant_parts.append(tool_call)
                                # Stash for pairing with result later
                                pending_tool_calls[block.id] = tool_call
                        if assistant_parts:
                            output_messages.append({"role": "assistant", "parts": assistant_parts})
                            # Update inference span progressively
                            if inference_span:
                                inference_span.set_attribute("gen_ai.output.messages", json.dumps(output_messages))
                            # Update turn span progressively (just text, not tool calls)
                            if self._turn_span and turn_output_parts:
                                self._turn_span.set_attribute(
                                    "gen_ai.output.messages",
                                    json.dumps([{"role": "assistant", "parts": turn_output_parts}])
                                )

                        # Capture finish reason (stop_reason on AssistantMessage)
                        if hasattr(message, 'stop_reason') and message.stop_reason:
                            last_finish_reason = message.stop_reason
                            if self._turn_span:
                                self._turn_span.set_attribute(
                                    "gen_ai.response.finish_reasons",
                                    json.dumps([last_finish_reason])
                                )

                    # Handle user messages (tool results) - this triggers a new inference span
                    elif isinstance(message, UserMessage):
                        if isinstance(message.content, list):
                            for block in message.content:
                                if isinstance(block, ToolResultBlock):
                                    result_content = block.content
                                    if isinstance(result_content, list):
                                        result_content = json.dumps(result_content)
                                    elif result_content is None:
                                        result_content = ""

                                    # End current inference span
                                    if inference_span:
                                        _end_inference_span(inference_span, output_messages)

                                    # Build input: tool_call (from stash) + tool_result
                                    inference_count += 1
                                    new_input: list[dict] = []

                                    # Include the tool_call that caused this result
                                    tool_call = pending_tool_calls.pop(block.tool_use_id, None)
                                    if tool_call:
                                        new_input.append({"role": "assistant", "parts": [tool_call]})

                                    # Include the tool result
                                    tool_result = {
                                        "type": "tool_call_response",
                                        "id": block.tool_use_id,
                                        "response": str(result_content)[:500],
                                    }
                                    new_input.append({"role": "tool", "parts": [tool_result]})

                                    # Start new inference span
                                    output_messages = []
                                    inference_span = _start_inference_span(inference_count, new_input)

                    # Capture session ID and stats from result
                    if isinstance(message, ResultMessage):
                        self._current_session_id = message.session_id
                        stream_span.set_attribute("session_id", message.session_id)
                        stream_span.set_attribute("duration_ms", message.duration_ms)
                        stream_span.set_attribute("num_turns", message.num_turns)
                        stream_span.set_attribute("inference_count", inference_count + 1)
                        if message.total_cost_usd:
                            stream_span.set_attribute("cost_usd", message.total_cost_usd)
                        if message.usage:
                            stream_span.set_attribute("usage", str(message.usage))

                        # Also set on turn span with full gen_ai attributes
                        if self._turn_span:
                            self._turn_span.set_attribute("session_id", message.session_id)
                            self._turn_span.set_attribute("gen_ai.conversation.id", message.session_id)
                            self._turn_span.set_attribute("duration_ms", message.duration_ms)
                            self._turn_span.set_attribute("inference_count", inference_count + 1)
                            if message.total_cost_usd:
                                self._turn_span.set_attribute("cost_usd", message.total_cost_usd)
                            if message.usage:
                                # Standard token counts
                                input_tokens = message.usage.get("input_tokens", 0)
                                output_tokens = message.usage.get("output_tokens", 0)
                                if input_tokens:
                                    self._turn_span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
                                if output_tokens:
                                    self._turn_span.set_attribute("gen_ai.usage.output_tokens", output_tokens)

                                # Cache token stats (Anthropic-specific)
                                cache_creation = message.usage.get("cache_creation_input_tokens", 0)
                                cache_read = message.usage.get("cache_read_input_tokens", 0)
                                if cache_creation:
                                    self._turn_span.set_attribute("gen_ai.usage.cache_creation.input_tokens", cache_creation)
                                if cache_read:
                                    self._turn_span.set_attribute("gen_ai.usage.cache_read.input_tokens", cache_read)

                            # Response model (might differ from request model)
                            if hasattr(message, 'model') and message.model:
                                self._turn_span.set_attribute("gen_ai.response.model", message.model)

                    yield message

                # End final inference span
                if inference_span:
                    _end_inference_span(inference_span, output_messages)

                # Store accumulated text for memorables extraction
                self._last_assistant_content = "".join(assistant_text_parts)
                stream_span.set_attribute("message_count", message_count)
                stream_span.set_attribute("response_length", len(self._last_assistant_content))

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
        options_kwargs = {
            "cwd": self.cwd,
            "system_prompt": self._system_prompt,  # Just the soul!
            "model": self.ALPHA_MODEL,  # Alpha IS this model
            "allowed_tools": self.allowed_tools or [],
            "mcp_servers": self.mcp_servers,
            "include_partial_messages": self.include_partial_messages,
            "resume": session_id,
            "permission_mode": self.permission_mode,
            "hooks": hooks,
        }
        options = ClaudeAgentOptions(**options_kwargs)

        # Create and connect
        self._sdk_client = ClaudeSDKClient(options)
        await self._sdk_client.connect()

        self._current_session_id = session_id
        logfire.debug(f"SDK client created (session={session_id or 'new'})")
