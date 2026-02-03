"""Associative recall - what sounds familiar from this prompt?

Given a user prompt, searches Cortex using two parallel strategies:
1. Direct embedding search (fast, catches overall semantic similarity)
2. OLMo query extraction (slower, catches distinctive terms in long prompts)

Results are merged and deduped. Filters via session-scoped seen-cache.

The dual approach solves the "Mrs. Hughesbot problem": when a distinctive
term is buried in a long meta-prompt, direct embedding averages it out.
OLMo can isolate it as a separate query.
"""

import asyncio
import json
import os
from typing import Any

import httpx
import logfire
import redis.asyncio as aioredis

from .cortex import search as cortex_search

# Configuration from environment
REDIS_URL = os.environ.get("REDIS_URL")
OLLAMA_URL = os.environ.get("OLLAMA_URL")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL")

# Search parameters
DIRECT_LIMIT = 1   # Just top 1 for "wtf is Jeffery talking about generally"
QUERY_LIMIT = 1    # Top 1 per extracted query
MIN_SCORE = 0.1    # Minimum similarity threshold

# Query extraction prompt
QUERY_EXTRACTION_PROMPT = """Jeffery just said:

"{message}"

---

Alpha is searching her memories. A separate system already handles the MAIN topic of this message. Your job is to catch the PERIPHERAL details that might get lost:

- Names mentioned in passing (people, pets, projects)
- Brief references to past events or inside jokes
- Asides that take only 10-20 words of a longer message
- Distinctive terms that aren't the central subject

IGNORE the main thrust of what Jeffery is talking about. Focus on the edges.

Give me 0-3 short search queries (2-5 words each) for these peripheral mentions: {{"queries": ["query one", "query two"]}}

If there are no peripheral details worth searching (just one focused topic), return {{"queries": []}}

Return only the JSON object, nothing else."""


async def _get_redis() -> aioredis.Redis:
    """Get Redis client."""
    return aioredis.from_url(REDIS_URL, decode_responses=True)


async def _get_seen_ids(redis_client: aioredis.Redis, session_id: str) -> list[int]:
    """Get the list of memory IDs already seen this session."""
    key = f"memories:seen:{session_id}"
    members = await redis_client.smembers(key)
    return [int(m) for m in members]


async def _mark_seen(redis_client: aioredis.Redis, session_id: str, memory_ids: list[int]) -> None:
    """Mark memory IDs as seen for this session."""
    if not memory_ids:
        return
    key = f"memories:seen:{session_id}"
    await redis_client.sadd(key, *[str(m) for m in memory_ids])
    await redis_client.expire(key, 60 * 60 * 24)  # 24h TTL


async def _extract_queries(message: str) -> list[str]:
    """Extract search queries from a user message using OLMo.

    Returns 0-3 short queries, or empty list if message doesn't warrant search.
    """
    if not OLLAMA_URL or not OLLAMA_MODEL:
        logfire.debug("OLLAMA not configured, skipping query extraction")
        return []

    prompt = QUERY_EXTRACTION_PROMPT.format(message=message[:2000])

    with logfire.span(
        "recall.extract_queries",
        **{
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": "ollama",
            "gen_ai.request.model": OLLAMA_MODEL,
        }
    ) as span:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/chat",
                    json={
                        "model": OLLAMA_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "format": "json",
                        "options": {"num_ctx": 4096},
                    },
                )
                response.raise_for_status()

            result = response.json()
            output = result.get("message", {}).get("content", "")

            span.set_attribute("gen_ai.usage.input_tokens", result.get("prompt_eval_count", 0))
            span.set_attribute("gen_ai.usage.output_tokens", result.get("eval_count", 0))
            span.set_attribute("gen_ai.response.model", OLLAMA_MODEL)

            parsed = json.loads(output)
            queries = parsed.get("queries", [])

            if isinstance(queries, list):
                valid = [q for q in queries if isinstance(q, str) and q.strip()]
                logfire.info("Extracted queries", count=len(valid), queries=valid)
                return valid

            return []

        except json.JSONDecodeError as e:
            logfire.warning("Failed to parse OLMo response as JSON", error=str(e))
            return []
        except Exception as e:
            logfire.error("Query extraction failed", error=str(e))
            return []


async def _search_extracted_queries(
    queries: list[str],
    exclude: list[int],
) -> list[dict[str, Any]]:
    """Search Cortex for each extracted query, taking top 1 per query."""
    if not queries:
        return []

    async def search_one(query: str) -> dict[str, Any] | None:
        results = await cortex_search(
            query=query,
            limit=QUERY_LIMIT,
            exclude=exclude,
            min_score=MIN_SCORE,
        )
        return results[0] if results else None

    with logfire.span("recall.search_extracted", query_count=len(queries)) as span:
        tasks = [search_one(q) for q in queries]
        results = await asyncio.gather(*tasks)

        # Instrumentation
        query_results = {
            q: (r["id"] if r else None)
            for q, r in zip(queries, results)
        }
        span.set_attribute("query_results", str(query_results))

        # Filter None and dedupe
        memories = []
        seen_in_batch = set(exclude)
        for i, mem in enumerate(results):
            if mem and mem["id"] not in seen_in_batch:
                memories.append(mem)
                seen_in_batch.add(mem["id"])
                logfire.debug(f"Query '{queries[i]}' -> memory #{mem['id']}")
            elif mem:
                logfire.debug(f"Query '{queries[i]}' -> memory #{mem['id']} (deduped)")
            else:
                logfire.debug(f"Query '{queries[i]}' -> no result above threshold")

        return memories


async def recall(prompt: str, session_id: str) -> list[dict[str, Any]]:
    """
    Associative recall: what sounds familiar from this prompt?

    Uses two parallel strategies:
    1. Direct embedding search (fast, semantic similarity)
    2. OLMo query extraction + search (slower, catches distinctive terms)

    Results are merged and deduped. Filters via Redis seen-cache.

    Args:
        prompt: The user's message
        session_id: Current session ID (for seen-cache scoping)

    Returns:
        List of memory dicts with keys: id, content, created_at, score
    """
    with logfire.span("recall", session_id=session_id[:8] if session_id else "none") as span:
        redis_client = await _get_redis()
        try:
            seen_ids = await _get_seen_ids(redis_client, session_id)
            logfire.debug("Seen IDs loaded", count=len(seen_ids))

            # Run direct search and query extraction in parallel
            direct_task = cortex_search(
                query=prompt,
                limit=DIRECT_LIMIT,
                exclude=seen_ids if seen_ids else None,
                min_score=MIN_SCORE,
            )
            extract_task = _extract_queries(prompt)

            direct_memories, extracted_queries = await asyncio.gather(direct_task, extract_task)

            span.set_attribute("extracted_queries", extracted_queries)
            span.set_attribute("direct_memory_ids", [m["id"] for m in direct_memories])

            # Build exclude list for extracted searches
            exclude_for_extracted = set(seen_ids)
            for mem in direct_memories:
                exclude_for_extracted.add(mem["id"])

            # Search extracted queries
            extracted_memories = await _search_extracted_queries(
                extracted_queries,
                list(exclude_for_extracted),
            )

            span.set_attribute("extracted_memory_ids", [m["id"] for m in extracted_memories])

            # Merge: extracted first, then direct
            all_memories = extracted_memories + direct_memories
            span.set_attribute("total_memories", len(all_memories))

            if not all_memories:
                logfire.info("No memories above threshold")
                return []

            # Mark as seen
            new_ids = [m["id"] for m in all_memories]
            await _mark_seen(redis_client, session_id, new_ids)

            logfire.info(
                "Recall complete",
                extracted=len(extracted_memories),
                direct=len(direct_memories),
                total=len(all_memories),
            )

            return all_memories

        finally:
            await redis_client.aclose()
