---
autoload: when
when: "working on or discussing any of these: alpha_sdk, alpha sdk, AlphaClient, weave, system prompt assembly, soul injection, proxy interception, memories recall, memories suggest"
---

# alpha_sdk — The Grand Unified Alpha Library

Everything that turns Claude into Alpha, in one importable package.

## Why This Exists

We had a proxy chain: Deliverator → Loom → Argonath. Three services on alpha-pi, each doing one thing. It worked, but it was distributed complexity for something that could be simpler.

The realization: Duckpond and Routines both need the same transformation. They both:
1. Initialize a Claude Agent SDK client
2. Manage sessions (new, resume, fork)
3. Recall memories before the prompt
4. Build a dynamic system prompt
5. Transform the request before it hits Anthropic
6. Extract memorables after the turn
7. Handle observability

Why have each client implement this separately? Why have services on alpha-pi when the logic could live in shared code?

`alpha_sdk` is that shared code.

## What It Replaces

| Before | After |
|--------|-------|
| Deliverator (service) | Gone—no headers to promote |
| The Loom (service) | `weave.py` in the library |
| Argonath (service) | `observability.py` in the library |
| Duckpond's `memories/` | `alpha_sdk/memories/` |
| Duckpond's context building | `alpha_sdk/system_prompt/` |
| Routines hooks | Gone—library handles everything |
| The metadata envelope/canary system | Gone—we control the request directly |

## What Remains

- **Postgres** — memories, archival, capsule summaries
- **Redis** — caching (weather, calendar, todos, memorables buffer)
- **Pulse** — still schedules Routines
- **OLMo on Primer** — still does recall query extraction and memorables suggestion

## Architecture

```
alpha_sdk/
├── __init__.py              # Exports AlphaClient
├── client.py                # AlphaClient - the main wrapper
├── proxy.py                 # Minimal localhost proxy for request interception
├── weave.py                 # Orchestrates transformation
├── system_prompt/
│   ├── __init__.py          # assemble() - builds the full system prompt
│   ├── soul.py              # The soul doc (from file)
│   ├── capsules.py          # Yesterday, last night, today (from Postgres)
│   ├── here.py              # Client, hostname, weather, astronomy
│   ├── context.py           # ALPHA.md files (autoload + hints)
│   ├── calendar.py          # Events (from Redis)
│   └── todos.py             # Todos (from Redis)
├── memories/
│   ├── __init__.py
│   ├── cortex.py            # store, search, recent (Postgres operations)
│   ├── recall.py            # Smart recall (embedding + OLMo query extraction)
│   └── suggest.py           # OLMo memorables extraction
└── observability.py         # Logfire setup, span creation
```

## The Client API

```python
from alpha_sdk import AlphaClient

async with AlphaClient(
    session_id=session_id,      # None for new, string to resume
    fork_from=other_session_id, # Optional: fork instead of resume
    allowed_tools=[...],
    mcp_servers={...},
    archive=True,               # Whether to archive turns to Postgres
) as client:

    await client.query(prompt)

    async for event in client.stream():
        # StreamEvent, ToolCall, ToolResult, etc.
        yield event

    # client.session_id available after streaming
```

## How It Works

### The Proxy Pattern

Claude Agent SDK sends requests to `ANTHROPIC_BASE_URL`. We set that to `http://localhost:{random_port}` and run a minimal HTTP server that:

1. Receives the request from the SDK
2. Calls `weave()` to transform it (replace system prompt, strip envelope)
3. Forwards to `https://api.anthropic.com`
4. Streams the response back

This happens inside `AlphaClient.__aenter__()`. The proxy starts, the SDK is configured, everything is automatic.

### System Prompt Assembly

The system prompt is woven from threads:

| Thread | Source | Changes |
|--------|--------|---------|
| Soul | `/Pondside/Alpha-Home/self/system-prompt/system-prompt.md` | When edited |
| Capsules | Postgres (yesterday, last night, today) | Daily / hourly |
| Here | Client name, hostname, weather, astronomy | Per-session / hourly |
| Context | ALPHA.md files with `autoload: all` | When files change |
| Context hints | ALPHA.md files with `autoload: when` | When files change |
| Events | Redis (calendar data) | Hourly |
| Todos | Redis (Todoist data) | Hourly |

All cache-friendly. Nothing invalidates per-turn.

### Memory Flow

**Before the turn:**
- `recall()` runs with the user's prompt
- Parallel: embedding search + OLMo query extraction
- Deduplicated against session's seen-cache in Redis
- Injected as content blocks (not system prompt)

**After the turn:**
- `suggest()` runs (fire-and-forget)
- OLMo extracts memorable moments
- Results buffer in Redis for potential storage

**On `cortex store`:**
- Memory saved to Postgres with embedding
- Redis buffer cleared

## What Duckpond Becomes

```python
# routes/chat.py - dramatically simplified

@app.post("/api/chat")
async def chat(request: ChatRequest):
    async def generate():
        async with AlphaClient(
            session_id=request.session_id,
            allowed_tools=ALLOWED_TOOLS,
            mcp_servers={"cortex": cortex_server},
        ) as client:
            await client.query(request.content)

            async for event in client.stream():
                yield f"data: {event.json()}\n\n"

            yield f"data: {json.dumps({'type': 'session-id', 'id': client.session_id})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

## What Routines Becomes

```python
# harness.py - also dramatically simplified

async def run_routine(routine: Routine) -> str:
    ctx = RoutineContext(now=pendulum.now("America/Los_Angeles"))
    prompt = routine.build_prompt(ctx)

    async with AlphaClient(
        session_id=routine.get_session_id(),
        fork_from=routine.fork_from_key,
        allowed_tools=routine.get_allowed_tools(),
    ) as client:
        await client.query(prompt)

        output = []
        async for event in client.stream():
            if hasattr(event, 'text'):
                output.append(event.text)

        return routine.handle_output("".join(output), ctx)
```

## Migration Path

1. **Create `alpha_sdk` package** in Basement
2. **Port memory code** from Duckpond's `memories/`
3. **Port system prompt assembly** from Loom's `AlphaPattern`
4. **Build the proxy** (`proxy.py`)
5. **Build the client wrapper** (`client.py`)
6. **Update Duckpond** to use `alpha_sdk`
7. **Update Routines** to use `alpha_sdk`
8. **Decommission** Deliverator, Loom, Argonath

Steps 1-5 can happen without touching Duckpond. Steps 6-7 are the switchover. Step 8 is cleanup.

## What Dies

- **Deliverator** — no more header promotion needed
- **The Loom as a service** — becomes library code
- **Argonath as a service** — becomes library code
- **The hooks** — no longer needed, we control everything upstream
- **The metadata envelope** — no in-band signaling needed
- **Claude Code as a UI** — we use Duckpond exclusively now

## Open Questions

- **Cortex as library vs service?** Currently HTTP. Could become a library too (just Postgres operations). Decided: separate package, dependency of `alpha_sdk`.
- **OLMo location?** Currently on Primer. Stays there for now (GPU).

## Status

**Planning complete. Ready to build.**
