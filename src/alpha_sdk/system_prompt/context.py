"""Dynamic context loading from ALPHA.md files.

Walks /Pondside looking for ALPHA.md files with YAML frontmatter.
The 'autoload' key controls what gets injected:

- autoload: all   -> Full content becomes a system block
- autoload: when  -> Just a hint: "Read({path}) when {when}"
- autoload: no    -> Ignored entirely
"""

from pathlib import Path

import frontmatter
import logfire

CONTEXT_ROOT = Path("/Pondside")
CONTEXT_FILE_NAME = "ALPHA.md"


def find_context_files(root: Path = CONTEXT_ROOT) -> list[Path]:
    """Walk directory tree finding ALPHA.md files."""
    if not root.exists():
        logfire.warn(f"Context root does not exist: {root}")
        return []

    context_files = []
    for path in root.rglob(CONTEXT_FILE_NAME):
        if path.is_file():
            context_files.append(path)

    return sorted(context_files)


def load_context() -> tuple[list[dict], list[str]]:
    """Load ALPHA.md files and return content blocks and hints.

    Returns:
        (all_blocks, when_hints) where:
        - all_blocks: list of {"path": str, "content": str} for autoload=all
        - when_hints: list of hint strings for autoload=when
    """
    all_blocks = []
    when_hints = []

    for path in find_context_files():
        try:
            post = frontmatter.load(path)

            autoload = str(post.metadata.get("autoload", "no")).lower()
            when = post.metadata.get("when", "")

            rel_path = path.relative_to(CONTEXT_ROOT)

            if autoload == "all":
                all_blocks.append({
                    "path": str(rel_path),
                    "content": post.content.strip(),
                })
                logfire.debug(f"Loaded full context from {rel_path}")

            elif autoload == "when" and when:
                when_hints.append(f"`Read({rel_path})` — **Topics:** {when}")
                logfire.debug(f"Added context hint for {rel_path}")

        except Exception as e:
            logfire.warn(f"Failed to load context file {path}: {e}")

    if all_blocks or when_hints:
        logfire.info(f"Loaded {len(all_blocks)} context(s), {len(when_hints)} hint(s)")

    return all_blocks, when_hints
