"""Memory operations - Cortex integration and smart recall.

Cortex is now fully absorbed into alpha_sdk:
- Direct Postgres access via asyncpg (no HTTP layer)
- Direct Ollama embeddings (nomic-embed-text with proper task prefixes)
- store/search/recent are the foundational operations
- recall/suggest are higher-level functions that use them
"""

from .cortex import store, search, recent, get, forget, health, close, EmbeddingError
from .recall import recall
from .suggest import suggest

__all__ = [
    # Core Cortex operations
    "store",
    "search",
    "recent",
    "get",
    "forget",
    "health",
    "close",
    "EmbeddingError",
    # Higher-level functions
    "recall",
    "suggest",
]
