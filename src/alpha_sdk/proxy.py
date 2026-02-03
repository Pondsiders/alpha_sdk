"""Async proxy server for request interception.

Runs an aiohttp server in the same event loop as the SDK client:
1. Receives requests from Claude Agent SDK (via Claude Code subprocess)
2. Transforms them via weave()
3. Forwards to Anthropic
4. Streams responses back

Because it's async in the same event loop, all spans share trace context.
One turn → one span → everything nested inside.
"""

import asyncio
import socket
from contextlib import nullcontext
from typing import Any, Callable, Awaitable

import httpx
import logfire
from aiohttp import web

ANTHROPIC_API_URL = "https://api.anthropic.com"

# Headers to forward from incoming request (SDK → Anthropic)
# These handle authentication - DO NOT add our own API keys!
FORWARD_HEADERS = [
    "authorization",      # OAuth Bearer token (Claude Max)
    "x-api-key",          # API key auth (fallback)
    "anthropic-version",
    "anthropic-beta",
    "content-type",
]

# Headers to skip when forwarding response (hop-by-hop)
SKIP_RESPONSE_HEADERS = {
    "content-encoding",
    "transfer-encoding",
    "connection",
    "keep-alive",
}


def _find_free_port() -> int:
    """Find an available port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class AlphaProxy:
    """Async proxy server for intercepting SDK requests.

    Runs in the same event loop as the caller, enabling shared trace context.

    Usage:
        proxy = AlphaProxy(weaver=weave, client="duckpond")
        await proxy.start()

        os.environ["ANTHROPIC_BASE_URL"] = proxy.base_url

        # Before each turn, set the trace context so proxy spans nest properly
        proxy.set_trace_context(logfire.get_context())

        # ... use SDK ...

        await proxy.stop()
    """

    def __init__(
        self,
        weaver: Callable[[dict, str | None, str | None], Awaitable[dict]],
        client: str | None = None,
        hostname: str | None = None,
    ):
        """Initialize the proxy.

        Args:
            weaver: Async function to transform request bodies
            client: Client name (for logging, passed to weaver)
            hostname: Machine hostname (passed to weaver)
        """
        self.weaver = weaver
        self.client = client
        self.hostname = hostname

        self._port: int | None = None
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._trace_context: dict | None = None

    async def start(self) -> int:
        """Start the proxy server.

        Returns:
            The port number the server is listening on.
        """
        self._port = _find_free_port()

        # Create long-lived httpx client for forwarding requests
        self._http_client = httpx.AsyncClient(timeout=300.0)

        # Build aiohttp app
        self._app = web.Application()
        self._app.router.add_route("*", "/{path:.*}", self._handle_request)

        # Start server
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "127.0.0.1", self._port)
        await self._site.start()

        logfire.info(f"Alpha proxy started on port {self._port}")
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

        logfire.info("Alpha proxy stopped")

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

    def set_trace_context(self, ctx: dict) -> None:
        """Set the trace context for request handlers.

        Call this before each turn so proxy spans nest under the turn span.

        Args:
            ctx: Trace context from logfire.get_context()
        """
        self._trace_context = ctx

    async def _handle_request(self, request: web.Request) -> web.StreamResponse:
        """Handle incoming requests from the SDK."""
        path = "/" + request.match_info.get("path", "")

        # Health check
        if request.method == "GET" and path == "/health":
            return web.Response(text="ok")

        # Only handle POST to messages endpoints
        if request.method != "POST":
            return web.Response(status=404, text="Not found")

        # Attach trace context so spans nest under the current turn
        context_manager = (
            logfire.attach_context(self._trace_context)
            if self._trace_context
            else nullcontext()
        )

        with context_manager:
            with logfire.span(
                "proxy.forward",
                path=path,
                method=request.method,
            ) as span:
                try:
                    return await self._forward_request(request, path, span)
                except Exception as e:
                    logfire.error(f"Proxy error: {e}")
                    span.set_attribute("error", str(e))
                    return web.Response(status=500, text=str(e))

    async def _forward_request(
        self,
        request: web.Request,
        path: str,
        span: logfire.LogfireSpan,
    ) -> web.StreamResponse:
        """Transform and forward a request to Anthropic."""
        # Read request body
        body_bytes = await request.read()

        try:
            body = await request.json()
        except Exception:
            # Not JSON - forward as-is (shouldn't happen for messages API)
            body = None

        # Transform via weave (if it's JSON and we have a weaver)
        if body is not None and self.weaver:
            body = await self.weaver(body, self.client, self.hostname)
            span.set_attribute("transformed", True)

        # Build headers - forward auth headers from SDK
        headers = {}
        for header_name in FORWARD_HEADERS:
            value = request.headers.get(header_name)
            if value:
                headers[header_name] = value

        # Ensure content-type
        if "content-type" not in headers:
            headers["content-type"] = "application/json"

        # Forward to Anthropic
        url = f"{ANTHROPIC_API_URL}{path}"

        if self._http_client is None:
            raise RuntimeError("HTTP client not initialized")

        # Use streaming to forward response in real-time
        async with self._http_client.stream(
            "POST",
            url,
            json=body if body is not None else None,
            content=body_bytes if body is None else None,
            headers=headers,
        ) as response:
            span.set_attribute("status_code", response.status_code)

            # Create streaming response
            resp = web.StreamResponse(status=response.status_code)

            # Forward response headers
            for key, value in response.headers.items():
                if key.lower() not in SKIP_RESPONSE_HEADERS:
                    resp.headers[key] = value

            await resp.prepare(request)

            # Stream response body
            async for chunk in response.aiter_bytes():
                await resp.write(chunk)

            await resp.write_eof()
            return resp
