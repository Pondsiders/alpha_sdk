"""Soul - the eternal prompt that makes Alpha who she is.

Loads from local git repository, caches for the session.
Version-controlled or bust.
"""

import os
import subprocess
from pathlib import Path

import logfire

# Configuration
SOUL_REPO_PATH = Path(os.environ.get(
    "ALPHA_SOUL_REPO",
    "/Pondside/Alpha-Home/self/system-prompt"
))
SOUL_FILE = "system-prompt.md"
COMPACT_FILE = "compact-prompt.md"

# Cached state
_soul_prompt: str | None = None
_compact_prompt: str | None = None


def _read_from_git(filename: str, ref: str = "HEAD") -> str | None:
    """Read a file from the git repository at a specific ref."""
    try:
        result = subprocess.run(
            ["git", "show", f"{ref}:{filename}"],
            cwd=SOUL_REPO_PATH,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            logfire.warn(f"git show {ref}:{filename} failed: {result.stderr}")
            return None

        commit = subprocess.run(
            ["git", "rev-parse", "--short", ref],
            cwd=SOUL_REPO_PATH,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()

        logfire.info(f"Loaded {filename} from git (commit={commit}, {len(result.stdout)} chars)")
        return result.stdout

    except Exception as e:
        logfire.error(f"Failed to read {filename} from git: {e}")
        return None


def init() -> None:
    """Initialize the soul at startup. Call once."""
    global _soul_prompt, _compact_prompt

    logfire.info("Initializing Alpha soul...")

    _soul_prompt = _read_from_git(SOUL_FILE)
    if _soul_prompt is None:
        raise RuntimeError(
            f"FATAL: Could not load Alpha soul doc from {SOUL_REPO_PATH}/{SOUL_FILE}"
        )

    _compact_prompt = _read_from_git(COMPACT_FILE)
    if _compact_prompt is None:
        logfire.warn("Compact prompt not loaded, will use fallback")


def get_soul() -> str:
    """Get the cached soul doc. Initializes if needed."""
    global _soul_prompt
    if _soul_prompt is None:
        init()
    return _soul_prompt


def get_compact() -> str | None:
    """Get the compact prompt, or None if not loaded."""
    return _compact_prompt
