"""Capsules - daily summaries from Postgres.

The past section: what happened yesterday and last night.
"""

import os

import logfire
import pendulum

DATABASE_URL = os.environ.get("DATABASE_URL", "")


def _format_summary(row: tuple) -> str:
    """Format a capsule summary row into a markdown section."""
    period_start, period_end, summary = row

    start = pendulum.instance(period_start).in_timezone("America/Los_Angeles")
    end = pendulum.instance(period_end).in_timezone("America/Los_Angeles")

    is_night = start.hour >= 22 or start.hour < 6

    if is_night:
        header = f"## {start.format('dddd')} night, {start.format('MMMM')} {start.day}-{end.day}, {end.year}"
    else:
        header = f"## {start.format('dddd, MMMM D, YYYY')}"

    return f"{header}\n\n{summary}"


async def get_capsules() -> tuple[str | None, str | None]:
    """Get the two most recent capsule summaries.

    Returns (older_summary, newer_summary) as formatted strings.
    """
    if not DATABASE_URL:
        logfire.debug("No DATABASE_URL, skipping capsules")
        return None, None

    try:
        import psycopg

        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT period_start, period_end, summary
                    FROM cortex.summaries
                    ORDER BY period_start DESC
                    LIMIT 2
                """)
                rows = cur.fetchall()

        if not rows:
            return None, None

        summaries = [_format_summary(row) for row in rows]

        if len(summaries) >= 2:
            return summaries[1], summaries[0]  # (older, newer)
        elif len(summaries) == 1:
            return None, summaries[0]
        else:
            return None, None

    except Exception as e:
        logfire.warn(f"Error fetching capsules: {e}")
        return None, None
