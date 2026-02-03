"""Minimal proxy server for request interception.

Runs a lightweight HTTP server that:
1. Receives requests from Claude Agent SDK
2. Transforms them via weave()
3. Forwards to Anthropic
4. Streams responses back

This allows us to modify requests after the SDK builds them
but before they reach Anthropic.
"""

import asyncio
import json
import logging
import os
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from typing import Callable, Any
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com"


def _find_free_port() -> int:
    """Find an available port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class ProxyHandler(BaseHTTPRequestHandler):
    """HTTP handler that intercepts and transforms requests."""

    weaver: Callable[[dict], Any] | None = None
    client_name: str | None = None
    hostname: str | None = None
    api_key: str | None = None

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

    def do_POST(self):
        """Handle POST requests (the messages endpoint)."""
        try:
            # Read request body
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length)
            body = json.loads(body_bytes)

            # Transform via weave
            if self.weaver:
                # Run async weave in sync context
                loop = asyncio.new_event_loop()
                try:
                    body = loop.run_until_complete(
                        self.weaver(body, self.client_name, self.hostname)
                    )
                finally:
                    loop.close()

            # Build headers for Anthropic
            headers = {
                "Content-Type": "application/json",
                "anthropic-version": self.headers.get("anthropic-version", "2023-06-01"),
                "x-api-key": self.api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
            }

            # Copy authorization header if present (for OAuth)
            if self.headers.get("Authorization"):
                headers["Authorization"] = self.headers["Authorization"]

            # Forward to Anthropic
            # Use sync httpx for simplicity in the handler
            with httpx.Client(timeout=300.0) as client:
                response = client.post(
                    f"{ANTHROPIC_API_URL}{self.path}",
                    json=body,
                    headers=headers,
                    # Don't follow redirects, stream response
                )

            # Send response back
            self.send_response(response.status_code)
            for key, value in response.headers.items():
                if key.lower() not in ("content-encoding", "transfer-encoding", "connection"):
                    self.send_header(key, value)
            self.end_headers()
            self.wfile.write(response.content)

        except Exception as e:
            logger.error(f"Proxy error: {e}")
            self.send_error(500, str(e))

    def do_GET(self):
        """Handle GET requests (health check, etc.)."""
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_error(404)


class AlphaProxy:
    """Minimal proxy server for intercepting SDK requests."""

    def __init__(
        self,
        weaver: Callable[[dict, str | None, str | None], Any],
        client: str | None = None,
        hostname: str | None = None,
    ):
        self.weaver = weaver
        self.client = client
        self.hostname = hostname
        self.server: HTTPServer | None = None
        self.thread: Thread | None = None
        self.port: int | None = None

    def start(self) -> int:
        """Start the proxy server.

        Returns:
            The port number the server is listening on.
        """
        self.port = _find_free_port()

        # Create handler class with our settings
        handler = type(
            "ConfiguredProxyHandler",
            (ProxyHandler,),
            {
                "weaver": staticmethod(self.weaver),
                "client_name": self.client,
                "hostname": self.hostname,
                "api_key": os.environ.get("ANTHROPIC_API_KEY"),
            },
        )

        self.server = HTTPServer(("127.0.0.1", self.port), handler)

        # Run in background thread
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

        logger.info(f"Alpha proxy started on port {self.port}")
        return self.port

    def stop(self):
        """Stop the proxy server."""
        if self.server:
            self.server.shutdown()
            self.server = None
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
        logger.info("Alpha proxy stopped")

    @property
    def base_url(self) -> str:
        """Get the base URL for this proxy."""
        if self.port is None:
            raise RuntimeError("Proxy not started")
        return f"http://127.0.0.1:{self.port}"
