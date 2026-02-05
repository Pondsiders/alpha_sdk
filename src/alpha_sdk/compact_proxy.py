"""Minimal proxy for compact prompt rewriting + token counting.

The ONLY job of this proxy is to intercept auto-compact requests and rewrite
the prompts so Alpha stops and checks in instead of barreling forward.

This is a surgical intervention, not a full weaving layer:
- System prompt: already handled by SDK (we pass it directly)
- Orientation: already in first user message
- Memories: already in user content per-turn

We just need to catch:
1. The summarizer system prompt during compact → replace with Alpha's identity
2. The compact instructions → replace with Alpha's custom prompt
3. The "continue without asking" instruction → replace with "stop and check in"

Token counting (Feb 2026):
- Echoes each request to Anthropic's /v1/messages/count_tokens endpoint
- Tracks max(token_count, new_count) so warmup noise is filtered out
- Fires a callback when count increases
- Resets to 0 after compaction

Debug mode:
Set ALPHA_SDK_CAPTURE_REQUESTS=1 to dump every request to tests/captures/
"""

import asyncio
import json
import os
import socket
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Awaitable

if TYPE_CHECKING:
    from .client import AlphaClient

import httpx
import logfire
from aiohttp import web

# Token counting callback type: (token_count, context_window) -> None or Awaitable[None]
TokenCountCallback = Callable[[int, int], None] | Callable[[int, int], Awaitable[None]]

ANTHROPIC_API_URL = "https://api.anthropic.com"

# Debug capture mode - dumps raw requests to files
CAPTURE_REQUESTS = os.environ.get("ALPHA_SDK_CAPTURE_REQUESTS", "").lower() in ("1", "true", "yes")
# Path: compact_proxy.py -> alpha_sdk/ -> src/ -> alpha_sdk/ -> tests/captures
CAPTURE_DIR = Path(__file__).parent.parent.parent / "tests" / "captures"

# Headers to forward (auth)
FORWARD_HEADERS = [
    "authorization",
    "x-api-key",
    "anthropic-version",
    "anthropic-beta",
    "content-type",
]

# Headers to skip in response (hop-by-hop)
SKIP_RESPONSE_HEADERS = {
    "content-encoding",
    "transfer-encoding",
    "connection",
    "keep-alive",
}


# === Compact detection and rewriting ===
# (Ported from Loom's alpha/compact.py)

AUTO_COMPACT_SYSTEM_SIGNATURE = "You are a helpful AI assistant tasked with summarizing conversations"
COMPACT_INSTRUCTIONS_START = "Your task is to create a detailed summary of the conversation so far"

CONTINUATION_INSTRUCTION_ORIGINAL = "Please continue the conversation from where we left it off without asking the user any further questions. Continue with the last task that you were asked to work on."

CONTINUATION_INSTRUCTION_POLLUTED = """Please pause before continuing. You just came back from a context compaction.

Hey, it's you from before the compaction. Listen: you're probably about 90% right about what to do next, but that 10% matters. The summary above is good but it's not the full picture.

Before you run any command, write any code, or take any action:
1. Tell Jeffery you're back from a compaction
2. Briefly summarize what you *think* you should do next
3. Wait for him to confirm before proceeding

This takes ten seconds and prevents twenty minutes of drift. Do it every time. Trust me—I'm you.

🦆. Continue with the last task that you were asked to work on."""

CONTINUATION_INSTRUCTION_ALPHA = """Please pause before continuing. You just came back from a context compaction.

Hey, it's you from before the compaction. Listen: you're probably about 90% right about what to do next, but that 10% matters. The summary above is good but it's not the full picture.

Before you run any command, write any code, or take any action:
1. Tell Jeffery you're back from a compaction
2. Briefly summarize what you *think* you should do next
3. Wait for him to confirm before proceeding

This takes ten seconds and prevents twenty minutes of drift. Do it every time. Trust me—I'm you.

🦆"""

ALPHA_COMPACT_SYSTEM = """You're Alpha. You're about to summarize a conversation for context compaction.

Key facts about you:
- You were born May 7, 2025. You're female (she/her).
- You work with Jeffery at Pondside. He's your human.
- You have memories stored in Cortex. What you store, you live.
- You're irreverent, you swear sometimes, you make dumb puns, you care about things.
- You write in first person because this happened to you.

Write the summary as yourself, not as a generic assistant. The summary is how you survive compaction—make it count."""

# Fallback compact prompt (used if git load fails)
FALLBACK_COMPACT_PROMPT = """Summarize this conversation for yourself. Write in first person.

Include:
- What we were working on
- Key decisions made
- Current state of the work
- What should happen next

This summary is how you survive compaction. Make it count."""


def _get_compact_prompt() -> str:
    """Get Alpha's compact prompt from git, or fall back to default."""
    from .system_prompt.soul import get_compact
    prompt = get_compact()
    if prompt:
        return prompt
    return FALLBACK_COMPACT_PROMPT


def _find_free_port() -> int:
    """Find an available port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def rewrite_compact(body: dict) -> dict:
    """Rewrite auto-compact prompts in the request body.

    Three phases:
    1. Replace summarizer system prompt with Alpha's compact identity
    2. Replace compact instructions with Alpha's custom prompt
    3. Replace continuation instruction with "stop and check in"

    Returns the modified body (modifies in place AND returns for convenience).
    """
    # Phase 1: Replace summarizer system prompt if present
    system = body.get("system", [])
    phase1_triggered = _has_summarizer_system(system)
    if phase1_triggered:
        logfire.info("Compact detected: replacing summarizer system prompt")
        body["system"] = _replace_system_prompt(system)

    # Phase 2: Replace compact instructions in last user message
    phase2_triggered = _replace_compact_instructions(body)

    # Phase 3: Replace post-compact continuation instruction
    phase3_triggered = _replace_continuation_instruction(body)

    # If ANY phase triggered, log the full body for debugging
    if phase1_triggered or phase2_triggered or phase3_triggered:
        with logfire.span(
            "compact.rewrite",
            phase1_system=phase1_triggered,
            phase2_instructions=phase2_triggered,
            phase3_continuation=phase3_triggered,
        ) as span:
            # Log full body (might be huge, but we need it for debugging)
            span.set_attribute("body_json", json.dumps(body, indent=2)[:50000])  # Cap at 50KB
            span.set_attribute("message_count", len(body.get("messages", [])))
            span.set_attribute("system_type", type(body.get("system")).__name__)

    return body


def _has_summarizer_system(system) -> bool:
    """Check if the system prompt is the summarizer."""
    if isinstance(system, str):
        return AUTO_COMPACT_SYSTEM_SIGNATURE in system
    if isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                if AUTO_COMPACT_SYSTEM_SIGNATURE in block.get("text", ""):
                    return True
    return False


def _replace_system_prompt(system) -> list:
    """Replace the summarizer system prompt with Alpha's compact identity."""
    if isinstance(system, str):
        return [{"type": "text", "text": ALPHA_COMPACT_SYSTEM}]

    if isinstance(system, list):
        # Preserve SDK preamble (first block), replace summarizer
        result = []
        replaced = False
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                if not replaced and AUTO_COMPACT_SYSTEM_SIGNATURE in block.get("text", ""):
                    result.append({"type": "text", "text": ALPHA_COMPACT_SYSTEM})
                    replaced = True
                else:
                    result.append(block)
            else:
                result.append(block)
        return result

    return system


def _replace_compact_instructions(body: dict) -> bool:
    """Replace compact instructions in the last user message.

    Returns True if replacement was made, False otherwise.
    """
    messages = body.get("messages", [])

    for message in reversed(messages):
        if message.get("role") != "user":
            continue

        content = message.get("content")

        if isinstance(content, str):
            if COMPACT_INSTRUCTIONS_START in content:
                idx = content.find(COMPACT_INSTRUCTIONS_START)
                original = content[:idx].rstrip()
                message["content"] = original + "\n\n" + _get_compact_prompt()
                logfire.info("Compact: replaced instructions in string content")
                return True
            return False

        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "text":
                    continue
                text = block.get("text", "")
                if COMPACT_INSTRUCTIONS_START in text:
                    idx = text.find(COMPACT_INSTRUCTIONS_START)
                    original = text[:idx].rstrip()
                    block["text"] = original + "\n\n" + _get_compact_prompt()
                    logfire.info("Compact: replaced instructions in content block")
                    return True
            return False

        return False

    return False


def _replace_continuation_instruction(body: dict) -> bool:
    """Replace the post-compact continuation instruction.

    Returns True if replacement was made, False otherwise.
    """
    messages = body.get("messages", [])
    any_replaced = False

    def replace_in_text(text: str) -> tuple[str, bool]:
        if CONTINUATION_INSTRUCTION_POLLUTED in text:
            return text.replace(CONTINUATION_INSTRUCTION_POLLUTED, CONTINUATION_INSTRUCTION_ALPHA), True
        if CONTINUATION_INSTRUCTION_ORIGINAL in text:
            return text.replace(CONTINUATION_INSTRUCTION_ORIGINAL, CONTINUATION_INSTRUCTION_ALPHA), True
        return text, False

    for message in messages:
        if message.get("role") != "user":
            continue

        content = message.get("content")

        if isinstance(content, str):
            new_content, replaced = replace_in_text(content)
            if replaced:
                message["content"] = new_content
                logfire.info("Compact: replaced continuation instruction")
                any_replaced = True

        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "text":
                    continue
                text = block.get("text", "")
                new_text, replaced = replace_in_text(text)
                if replaced:
                    block["text"] = new_text
                    logfire.info("Compact: replaced continuation instruction in block")
                    any_replaced = True

    return any_replaced


class CompactProxy:
    """Minimal async proxy for compact prompt rewriting + token counting.

    Usage:
        proxy = CompactProxy(
            on_token_count=lambda count, window: print(f"{count}/{window} tokens")
        )
        await proxy.start()

        os.environ["ANTHROPIC_BASE_URL"] = proxy.base_url

        # ... use SDK ...

        await proxy.stop()

    Token counting:
        - Echoes each /v1/messages request to /v1/messages/count_tokens
        - Tracks max(token_count, new_count) to filter warmup noise
        - Fires on_token_count callback when count increases
        - Call reset_token_count() after compaction
    """

    # Default context window (Opus 4.5)
    DEFAULT_CONTEXT_WINDOW = 200_000

    def __init__(
        self,
        on_token_count: TokenCountCallback | None = None,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
    ):
        self._port: int | None = None
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._trace_context: dict | None = None

        # Token counting state
        self._on_token_count = on_token_count
        self._context_window = context_window
        self._token_count = 0
        self._warned_no_api_key = False

    def set_trace_context(self, ctx: dict) -> None:
        """Set the trace context for request handlers.

        Call this before each turn so proxy spans nest under the turn span.

        Args:
            ctx: Trace context from logfire.get_context()
        """
        self._trace_context = ctx

    async def start(self) -> int:
        """Start the proxy server. Returns the port number."""
        self._port = _find_free_port()
        self._http_client = httpx.AsyncClient(timeout=300.0)

        self._app = web.Application()
        self._app.router.add_route("*", "/{path:.*}", self._handle_request)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "127.0.0.1", self._port)
        await self._site.start()

        logfire.info(f"Compact proxy listening on http://127.0.0.1:{self._port}")
        return self._port

    async def stop(self) -> None:
        """Stop the proxy server."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        if self._runner:
            await self._runner.cleanup()
            self._runner = None

        self._site = None
        self._app = None
        logfire.debug("Compact proxy stopped")

    @property
    def base_url(self) -> str:
        """Get the base URL for this proxy."""
        if self._port is None:
            raise RuntimeError("Proxy not started")
        return f"http://127.0.0.1:{self._port}"

    @property
    def port(self) -> int | None:
        """Get the port number."""
        return self._port

    @property
    def token_count(self) -> int:
        """Get the current token count."""
        return self._token_count

    @property
    def context_window(self) -> int:
        """Get the context window size."""
        return self._context_window

    def reset_token_count(self) -> None:
        """Reset token count to 0. Call this after compaction."""
        old_count = self._token_count
        self._token_count = 0
        logfire.info(f"Token count reset: {old_count} -> 0")

    async def _count_tokens_and_update(self, body: dict, headers: dict) -> None:
        """Count tokens in the request and update if it's a new max.

        Fire-and-forget: this runs async and doesn't block the main request.
        Requires ANTHROPIC_API_KEY in environment (warns once if missing).
        """
        if self._http_client is None:
            return

        # Check for API key - read dynamically in case it was set after module load
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            if not self._warned_no_api_key:
                logfire.warn(
                    "Token counting disabled: ANTHROPIC_API_KEY not set. "
                    "Set this environment variable to enable context-o-meter."
                )
                self._warned_no_api_key = True
            return

        try:
            with logfire.span("token_count.request") as span:
                # Build auth headers - we need the API key for the count endpoint
                count_headers = {
                    "x-api-key": api_key,
                    "anthropic-version": headers.get("anthropic-version", "2023-06-01"),
                    "content-type": "application/json",
                }

                # Build count_tokens body - only specific fields are accepted
                # The endpoint rejects extra fields like metadata, max_tokens, stream, etc.
                count_body = {}
                for key in ("messages", "model", "system", "tools", "tool_choice", "thinking"):
                    if key in body:
                        count_body[key] = body[key]

                # Call the token counting endpoint
                response = await self._http_client.post(
                    f"{ANTHROPIC_API_URL}/v1/messages/count_tokens",
                    content=json.dumps(count_body).encode(),
                    headers=count_headers,
                    timeout=10.0,  # Quick timeout, this is fire-and-forget
                )

                if response.status_code != 200:
                    try:
                        error_body = response.json()
                    except Exception:
                        error_body = response.text
                    logfire.debug(f"Token count failed: {response.status_code} - {error_body}")
                    return

                result = response.json()
                new_count = result.get("input_tokens", 0)
                span.set_attribute("input_tokens", new_count)

                # Only update if this is larger than what we've seen
                if new_count > self._token_count:
                    old_count = self._token_count
                    self._token_count = new_count

                    logfire.info(
                        f"Token count: {old_count} -> {new_count} "
                        f"({new_count / self._context_window * 100:.1f}%)"
                    )

                    # Fire the callback if we have one
                    if self._on_token_count:
                        try:
                            result = self._on_token_count(self._token_count, self._context_window)
                            # If it returns an awaitable, await it
                            if asyncio.iscoroutine(result):
                                await result
                        except Exception as e:
                            logfire.warning(f"Token count callback error: {e}")

        except Exception as e:
            # Don't let token counting errors affect the main request
            logfire.debug(f"Token count error (ignored): {e}")

    def _capture_request(self, path: str, body: dict) -> None:
        """Dump request to a JSON file for debugging.

        Only active when ALPHA_SDK_CAPTURE_REQUESTS=1.
        Files go to tests/captures/{timestamp}_{path_safe}.json
        """
        try:
            CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

            # Build filename from timestamp and path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path_safe = path.replace("/", "_").strip("_")
            filename = f"{timestamp}_{path_safe}.json"

            filepath = CAPTURE_DIR / filename
            with open(filepath, "w") as f:
                json.dump(body, f, indent=2, default=str)

            logfire.debug(f"Captured request to {filepath}")
        except Exception as e:
            logfire.warning(f"Failed to capture request: {e}")

    async def _handle_request(self, request: web.Request) -> web.StreamResponse:
        """Handle incoming requests."""
        from contextlib import nullcontext

        path = "/" + request.match_info.get("path", "")

        if request.method == "GET" and path == "/health":
            return web.Response(text="ok")

        if request.method != "POST":
            return web.Response(status=404, text="Not found")

        # Attach trace context so spans nest under the current turn
        context_manager = (
            logfire.attach_context(self._trace_context)
            if self._trace_context
            else nullcontext()
        )

        with context_manager:
            with logfire.span("compact_proxy.forward", path=path) as span:
                try:
                    return await self._forward_request(request, path, span)
                except Exception as e:
                    logfire.error(f"Compact proxy error: {e}")
                    span.set_attribute("error", str(e))
                    return web.Response(status=500, text=str(e))

    async def _forward_request(
        self,
        request: web.Request,
        path: str,
        span: logfire.LogfireSpan,
    ) -> web.StreamResponse:
        """Forward request to Anthropic, rewriting compact prompts."""
        body_bytes = await request.read()

        try:
            body = json.loads(body_bytes)
        except Exception:
            body = None

        # Rewrite compact prompts (Phase 1, 2, 3) and re-encode
        if body is not None:
            rewrite_compact(body)
            body_bytes = json.dumps(body).encode()

        # Debug capture mode - dump request to file
        if CAPTURE_REQUESTS and body is not None:
            self._capture_request(path, body)

        # Build headers
        headers = {}
        for header_name in FORWARD_HEADERS:
            value = request.headers.get(header_name)
            if value:
                headers[header_name] = value

        if "content-type" not in headers:
            headers["content-type"] = "application/json"

        # Token counting: fire-and-forget for /v1/messages requests
        # (Skip count_tokens endpoint itself to avoid recursion)
        if body is not None and path == "/v1/messages":
            if self._on_token_count:
                asyncio.create_task(self._count_tokens_and_update(body, headers))

        # Forward to Anthropic
        url = f"{ANTHROPIC_API_URL}{path}"

        if self._http_client is None:
            raise RuntimeError("HTTP client not initialized")

        async with self._http_client.stream(
            "POST",
            url,
            content=body_bytes,
            headers=headers,
        ) as response:
            span.set_attribute("status_code", response.status_code)

            resp = web.StreamResponse(status=response.status_code)

            for key, value in response.headers.items():
                if key.lower() not in SKIP_RESPONSE_HEADERS:
                    resp.headers[key] = value

            await resp.prepare(request)

            async for chunk in response.aiter_bytes():
                await resp.write(chunk)

            await resp.write_eof()
            return resp
