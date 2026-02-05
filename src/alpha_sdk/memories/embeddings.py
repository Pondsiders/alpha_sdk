"""Ollama embeddings client for Cortex.

Uses nomic-embed-text with proper task prefixes:
- search_document: for storing memories
- search_query: for searching memories
"""

import os

import httpx
import logfire

# Configuration from environment
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://alpha-pi:11434")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""
    pass


async def embed_document(content: str) -> list[float]:
    """Generate embedding for a document (for storage).

    Uses the 'search_document:' task prefix as required by nomic-embed-text.
    """
    return await _embed(f"search_document: {content}")


async def embed_query(query: str) -> list[float]:
    """Generate embedding for a query (for search).

    Uses the 'search_query:' task prefix as required by nomic-embed-text.
    """
    return await _embed(f"search_query: {query}")


async def _embed(prompt: str, timeout: float = 5.0) -> list[float]:
    """Call Ollama API to generate embedding."""
    with logfire.span(
        "cortex.embed",
        model=OLLAMA_EMBED_MODEL,
        prompt_preview=prompt[:50],
    ) as span:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{OLLAMA_URL.rstrip('/')}/api/embeddings",
                    json={
                        "model": OLLAMA_EMBED_MODEL,
                        "prompt": prompt,
                        "keep_alive": -1,  # Keep model loaded indefinitely
                    },
                )
                response.raise_for_status()
                data = response.json()
                embedding = data["embedding"]
                span.set_attribute("embedding_dim", len(embedding))
                return embedding
        except httpx.TimeoutException:
            logfire.warning(f"Ollama timeout after {timeout}s")
            raise EmbeddingError("Embedding service timed out")
        except httpx.HTTPStatusError as e:
            logfire.warning(f"Ollama HTTP error: {e.response.status_code}")
            raise EmbeddingError(f"Embedding service error: {e.response.status_code}")
        except httpx.ConnectError:
            logfire.warning(f"Ollama unreachable at {OLLAMA_URL}")
            raise EmbeddingError("Embedding service unreachable")
        except Exception as e:
            logfire.error(f"Ollama unexpected error: {e}")
            raise EmbeddingError(f"Embedding failed: {e}")


async def health_check() -> bool:
    """Check if Ollama is reachable."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{OLLAMA_URL.rstrip('/')}/api/tags")
            return response.status_code == 200
    except Exception:
        return False
