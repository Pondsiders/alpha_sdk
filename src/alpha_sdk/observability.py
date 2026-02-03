"""Observability setup - Logfire configuration.

Centralizes logging and tracing configuration so all
alpha_sdk consumers get consistent observability.

We use logfire.info/warn/error/debug directly instead of
Python's logging module. This ensures all logs are properly
associated with the current span context.
"""

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

    # Instrument httpx for trace propagation
    logfire.instrument_httpx()
