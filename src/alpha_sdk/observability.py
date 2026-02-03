"""Observability setup - Logfire configuration.

Centralizes logging and tracing configuration so all
alpha_sdk consumers get consistent observability.
"""

import logging
import os

import logfire


def configure(service_name: str = "alpha_sdk") -> None:
    """Configure Logfire for observability.

    Args:
        service_name: Name to identify this service in traces.
    """
    logfire.configure(
        service_name=service_name,
        distributed_tracing=True,
        scrubbing=False,  # Too aggressive, redacts normal words
        send_to_logfire="if-token-present",
    )

    # Route Python logging through Logfire
    logging.basicConfig(
        handlers=[logfire.LogfireLoggingHandler()],
        level=logging.DEBUG,
    )

    # Instrument httpx for trace propagation
    logfire.instrument_httpx()


def get_logger(name: str) -> logging.Logger:
    """Get a logger configured for Logfire.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)
