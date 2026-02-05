"""Postgres database operations for Cortex.

Direct asyncpg access to the cortex database with pgvector.
"""

import json
import os
from datetime import datetime, timezone, timedelta
from typing import Any

import asyncpg
import logfire

# Configuration from environment
DATABASE_URL = os.environ.get("DATABASE_URL")

# Module-level connection pool (lazy initialized)
_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    """Get or create the connection pool."""
    global _pool
    if _pool is None:
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL environment variable not set")
        _pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=2,
            max_size=10,
        )
        logfire.info("Cortex database pool created")
    return _pool


async def close_pool() -> None:
    """Close the connection pool."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logfire.info("Cortex database pool closed")


async def store_memory(
    content: str,
    embedding: list[float],
    tags: list[str] | None = None,
    timezone_str: str | None = None,
) -> tuple[int, datetime]:
    """Store a new memory. Returns (id, created_at)."""
    pool = await get_pool()
    created_at = datetime.now(timezone.utc)
    metadata = {
        "created_at": created_at.isoformat(),
        "captured_tz": timezone_str,
    }
    if tags:
        metadata["tags"] = tags

    with logfire.span("cortex.db.store", content_preview=content[:50]) as span:
        async with pool.acquire() as conn:
            memory_id = await conn.fetchval(
                """
                INSERT INTO cortex.memories (content, embedding, metadata)
                VALUES ($1, $2, $3)
                RETURNING id
                """,
                content,
                json.dumps(embedding),  # pgvector accepts JSON array
                json.dumps(metadata),
            )
            span.set_attribute("memory_id", memory_id)
            return memory_id, created_at


async def search_memories(
    query_embedding: list[float] | None,
    query_text: str,
    limit: int = 10,
    include_forgotten: bool = False,
    exact: bool = False,
    after: datetime | None = None,
    before: datetime | None = None,
    exclude: list[int] | None = None,
    min_score: float | None = None,
) -> list[dict[str, Any]]:
    """
    Search memories with hybrid (full-text + semantic) scoring.

    If exact=True, only uses full-text search (no embedding needed).
    exclude: list of memory IDs to skip (e.g., already seen this session)
    min_score: minimum similarity threshold (0-1), results below this are filtered
    """
    pool = await get_pool()

    with logfire.span(
        "cortex.db.search",
        query_preview=query_text[:50],
        limit=limit,
        exact=exact,
        exclude_count=len(exclude or []),
    ) as span:
        async with pool.acquire() as conn:
            # Build the WHERE clause
            conditions = []
            params = []
            param_idx = 1

            if not include_forgotten:
                conditions.append("NOT forgotten")

            if exclude:
                conditions.append(f"id != ALL(${param_idx}::int[])")
                params.append(exclude)
                param_idx += 1

            if after:
                conditions.append(f"(metadata->>'created_at')::timestamptz >= ${param_idx}")
                params.append(after)
                param_idx += 1

            if before:
                conditions.append(f"(metadata->>'created_at')::timestamptz < ${param_idx}")
                params.append(before)
                param_idx += 1

            where_clause = " AND ".join(conditions) if conditions else "TRUE"

            if exact:
                # Full-text search only
                query = f"""
                    SELECT
                        id,
                        content,
                        metadata,
                        ts_rank(content_tsv, plainto_tsquery('english', ${param_idx})) as score
                    FROM cortex.memories
                    WHERE {where_clause}
                      AND content_tsv @@ plainto_tsquery('english', ${param_idx})
                    ORDER BY score DESC
                    LIMIT ${param_idx + 1}
                """
                params.extend([query_text, limit])
            else:
                # Three-way search: exact match + full-text + semantic
                embedding_json = json.dumps(query_embedding)

                # Build WHERE clause for min_score threshold
                min_score_clause = ""
                if min_score is not None:
                    min_score_clause = f"AND GREATEST(exact_score, 0.5 * LEAST(fts_score, 1.0) + 0.5 * sem_score) >= ${param_idx + 3}"

                query = f"""
                    WITH scored AS (
                        SELECT
                            id,
                            content,
                            metadata,
                            -- Exact match: 1.0 if query appears with word boundaries
                            CASE WHEN content ~* ('\\m' || ${param_idx} || '\\M')
                                 THEN 1.0 ELSE 0.0 END as exact_score,
                            -- Full-text ranking (ts_rank returns 0-1ish)
                            COALESCE(
                                ts_rank(content_tsv, plainto_tsquery('english', ${param_idx})),
                                0
                            ) as fts_score,
                            -- Cosine similarity is already 0-1 for normalized vectors
                            1 - (embedding <=> ${param_idx + 1}::vector) as sem_score
                        FROM cortex.memories
                        WHERE {where_clause}
                          AND embedding IS NOT NULL
                    )
                    SELECT
                        id,
                        content,
                        metadata,
                        GREATEST(exact_score, 0.5 * LEAST(fts_score, 1.0) + 0.5 * sem_score) as score
                    FROM scored
                    WHERE 1=1
                    {min_score_clause}
                    ORDER BY score DESC
                    LIMIT ${param_idx + 2}
                """
                params.extend([query_text, embedding_json, limit])
                if min_score is not None:
                    params.append(min_score)

            rows = await conn.fetch(query, *params)
            span.set_attribute("result_count", len(rows))

            return [
                {
                    "id": row["id"],
                    "content": row["content"],
                    "metadata": json.loads(row["metadata"]),
                    "score": float(row["score"]),
                }
                for row in rows
            ]


async def get_recent_memories(
    limit: int = 10,
    hours: int = 24,
) -> list[dict[str, Any]]:
    """Get recent memories within the specified time window."""
    pool = await get_pool()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    with logfire.span("cortex.db.recent", limit=limit, hours=hours) as span:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, content, metadata
                FROM cortex.memories
                WHERE NOT forgotten
                  AND (metadata->>'created_at')::timestamptz >= $1
                ORDER BY (metadata->>'created_at')::timestamptz DESC
                LIMIT $2
                """,
                cutoff,
                limit,
            )
            span.set_attribute("result_count", len(rows))

            return [
                {
                    "id": row["id"],
                    "content": row["content"],
                    "metadata": json.loads(row["metadata"]),
                }
                for row in rows
            ]


async def get_memory(memory_id: int) -> dict[str, Any] | None:
    """Get a single memory by ID. Returns None if not found."""
    pool = await get_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, content, metadata
            FROM cortex.memories
            WHERE id = $1 AND NOT forgotten
            """,
            memory_id,
        )
        if row is None:
            return None
        return {
            "id": row["id"],
            "content": row["content"],
            "metadata": json.loads(row["metadata"]),
        }


async def forget_memory(memory_id: int) -> bool:
    """Soft-delete a memory. Returns True if found and updated."""
    pool = await get_pool()

    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE cortex.memories
            SET forgotten = TRUE
            WHERE id = $1 AND NOT forgotten
            """,
            memory_id,
        )
        return result == "UPDATE 1"


async def health_check() -> tuple[bool, int | None]:
    """Check database health, return (healthy, memory_count)."""
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM cortex.memories WHERE NOT forgotten"
            )
            return True, count
    except Exception as e:
        logfire.warning(f"Cortex database health check failed: {e}")
        return False, None
