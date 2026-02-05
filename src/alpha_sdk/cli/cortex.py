"""Cortex CLI - Command line interface for Alpha's memory system.

Direct Postgres access—no HTTP layer, no Cortex service dependency.
This is the sysadmin/debugging tool for Jeffery.

Usage:
    cortex store "Today I learned about pyrosomes."
    cortex search "bioluminescence"
    cortex recent
    cortex health
"""

import asyncio
import sys
from datetime import datetime, timedelta
from typing import Annotated, Optional

import pendulum
import typer
from rich.console import Console

# We'll import the async functions and run them
from ..memories import store, search, recent, get, forget, health

app = typer.Typer(help="Cortex - Alpha's semantic memory CLI")
console = Console()


def get_local_timezone() -> str:
    """Get the local timezone name."""
    try:
        import os
        import zoneinfo
        # Try to get the system timezone
        tz = datetime.now().astimezone().tzinfo
        if hasattr(tz, 'key'):
            return tz.key
        # Fallback: try to read from /etc/timezone
        try:
            with open("/etc/timezone") as f:
                return f.read().strip()
        except FileNotFoundError:
            pass
        # Another fallback: try TZ env var
        return os.environ.get("TZ", "UTC")
    except Exception:
        return "UTC"


def run_async(coro):
    """Run an async function synchronously."""
    return asyncio.run(coro)


@app.command(name="store")
def store_cmd(
    content: Annotated[
        Optional[str],
        typer.Argument(help="Memory content (use - for stdin)")
    ] = None,
    tags: Annotated[
        Optional[str],
        typer.Option("--tags", "-t", help="Comma-separated tags")
    ] = None,
):
    """Store a new memory."""
    # Read content from stdin if "-" or no argument
    if content == "-" or (content is None and not sys.stdin.isatty()):
        content = sys.stdin.read().strip()

    if not content:
        console.print("[red]Error: No content provided[/red]")
        raise typer.Exit(1)

    tag_list = [t.strip() for t in tags.split(",")] if tags else None

    async def do_store():
        result = await store(content, tags=tag_list, timezone=get_local_timezone())
        return result

    result = run_async(do_store())

    if result:
        console.print(f"[green]✓ Memory stored[/green] (id: {result['id']})")
    else:
        console.print("[red]Error storing memory[/red]")
        raise typer.Exit(1)


@app.command(name="search")
def search_cmd(
    query: Annotated[str, typer.Argument(help="Search query")],
    limit: Annotated[int, typer.Option("--limit", "-l", help="Max results")] = 10,
    include_forgotten: Annotated[
        bool, typer.Option("--include-forgotten", help="Include forgotten memories")
    ] = False,
    exact: Annotated[
        bool, typer.Option("--exact", "-e", help="Exact match (full-text only)")
    ] = False,
    after: Annotated[
        Optional[str], typer.Option("--after", help="Only memories after this date")
    ] = None,
    before: Annotated[
        Optional[str], typer.Option("--before", help="Only memories before this date")
    ] = None,
    date: Annotated[
        Optional[str], typer.Option("--date", "-d", help="Only memories from this date")
    ] = None,
):
    """Search memories."""
    # Handle date filters
    after_dt = None
    before_dt = None

    if date:
        # --date is shorthand for a single day
        try:
            d = datetime.fromisoformat(date)
            after_dt = d
            before_dt = d + timedelta(days=1)
        except ValueError:
            console.print(f"[red]Error: Invalid date format: {date}[/red]")
            raise typer.Exit(1)
    else:
        if after:
            try:
                after_dt = datetime.fromisoformat(after)
            except ValueError:
                console.print(f"[red]Error: Invalid date format: {after}[/red]")
                raise typer.Exit(1)
        if before:
            try:
                before_dt = datetime.fromisoformat(before)
            except ValueError:
                console.print(f"[red]Error: Invalid date format: {before}[/red]")
                raise typer.Exit(1)

    async def do_search():
        return await search(
            query,
            limit=limit,
            include_forgotten=include_forgotten,
            exact=exact,
            after=after_dt,
            before=before_dt,
        )

    memories = run_async(do_search())

    if not memories:
        console.print("[dim]No memories found[/dim]")
        return

    for mem in memories:
        score = mem.get("score")
        score_str = f"[{score:.2f}]" if score else ""
        # Parse UTC timestamp and convert to local time
        created_at = mem.get("created_at", "")
        if created_at:
            dt = pendulum.parse(created_at).in_tz(pendulum.local_timezone())
            date_str = dt.format("YYYY-MM-DD")
        else:
            date_str = "unknown"
        console.print(f"[cyan]{score_str}[/cyan] [dim]#{mem['id']}[/dim] ({date_str})")

        # Full content—no truncation
        console.print(mem["content"])
        console.print()


@app.command(name="recent")
def recent_cmd(
    limit: Annotated[int, typer.Option("--limit", "-l", help="Max results")] = 10,
    hours: Annotated[int, typer.Option("--hours", "-h", help="Hours to look back")] = 24,
):
    """Get recent memories."""
    async def do_recent():
        return await recent(limit=limit, hours=hours)

    memories = run_async(do_recent())

    if not memories:
        console.print("[dim]No recent memories[/dim]")
        return

    for mem in memories:
        # Parse UTC timestamp and convert to local time
        created_at = mem.get("created_at", "")
        if created_at:
            dt = pendulum.parse(created_at).in_tz(pendulum.local_timezone())
            date_str = dt.format("YYYY-MM-DD HH:mm")
        else:
            date_str = "unknown"
        console.print(f"[dim]#{mem['id']}[/dim] ({date_str})")

        content = mem["content"]
        console.print(content)
        console.print()


@app.command(name="health")
def health_cmd():
    """Check Cortex health."""
    async def do_health():
        return await health()

    data = run_async(do_health())

    status_color = "green" if data["status"] == "healthy" else "red"
    pg_color = "green" if data["postgres"] == "connected" else "red"
    ollama_color = "green" if data["ollama"] == "connected" else "red"

    console.print(f"Status: [{status_color}]{data['status']}[/{status_color}]")
    console.print(f"Postgres: [{pg_color}]{data['postgres']}[/{pg_color}]")
    console.print(f"Ollama: [{ollama_color}]{data['ollama']}[/{ollama_color}]")

    if data["memory_count"] is not None:
        console.print(f"Memories: {data['memory_count']:,}")


@app.command(name="forget")
def forget_cmd(
    memory_id: Annotated[int, typer.Argument(help="Memory ID to forget")],
):
    """Soft-delete a memory."""
    async def do_forget():
        return await forget(memory_id)

    forgotten = run_async(do_forget())

    if forgotten:
        console.print(f"[green]✓ Memory #{memory_id} forgotten[/green]")
    else:
        console.print(f"[yellow]Memory #{memory_id} not found or already forgotten[/yellow]")


@app.command(name="get")
def get_cmd(
    memory_id: Annotated[int, typer.Argument(help="Memory ID to retrieve")],
):
    """Get a single memory by ID."""
    async def do_get():
        return await get(memory_id)

    mem = run_async(do_get())

    if mem is None:
        console.print(f"[yellow]Memory #{memory_id} not found[/yellow]")
        raise typer.Exit(1)

    # Parse UTC timestamp and convert to local time
    created_at = mem.get("created_at", "")
    if created_at:
        dt = pendulum.parse(created_at).in_tz(pendulum.local_timezone())
        date_str = dt.format("YYYY-MM-DD HH:mm")
    else:
        date_str = "unknown"
    console.print(f"[dim]#{mem['id']}[/dim] ({date_str})")

    if mem.get("tags"):
        console.print(f"[dim]Tags: {', '.join(mem['tags'])}[/dim]")

    console.print()
    console.print(mem["content"])


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
