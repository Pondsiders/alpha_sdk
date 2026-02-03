"""alpha_sdk - Everything that turns Claude into Alpha."""

from .client import AlphaClient
from .weave import weave
from .proxy import AlphaProxy
from .observability import configure as configure_observability

__all__ = [
    "AlphaClient",
    "AlphaProxy",
    "weave",
    "configure_observability",
]
__version__ = "0.1.0"
