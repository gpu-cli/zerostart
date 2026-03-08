"""Snapshot identity computation for Tier 1 process snapshots.

Computes the intent_hash that uniquely identifies a snapshot based on
what the user asked to run (not the installed environment).
"""

from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path


def compute_intent_hash(
    entrypoint: str,
    argv: list[str],
    python_version: str | None = None,
    target_platform: str | None = None,
) -> str:
    """Compute a deterministic intent hash from user inputs.

    The intent hash identifies WHAT the user asked to run. It is computed
    from immutable inputs before launch and used as the snapshot directory name.

    Args:
        entrypoint: Path to the script/module being run.
        argv: Full argument vector (e.g. ["python", "serve.py", "--port", "8000"]).
        python_version: Python version string (default: current interpreter).
        target_platform: Platform string (default: current platform).

    Returns:
        Hex digest string (first 16 chars of SHA-256).
    """
    if python_version is None:
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    if target_platform is None:
        target_platform = f"{platform.machine()}-{platform.system().lower()}"

    h = hashlib.sha256()

    # Hash the entrypoint content (not just the path) so edits invalidate
    entrypoint_path = Path(entrypoint)
    if entrypoint_path.is_file():
        h.update(b"entrypoint_content:")
        h.update(entrypoint_path.read_bytes())
    else:
        # If file doesn't exist yet (e.g. module mode), hash the path
        h.update(b"entrypoint_path:")
        h.update(entrypoint.encode())

    h.update(b"\x00argv:")
    for arg in argv:
        h.update(arg.encode())
        h.update(b"\x00")

    h.update(b"python:")
    h.update(python_version.encode())

    h.update(b"\x00platform:")
    h.update(target_platform.encode())

    h.update(b"\x00mode:process")

    return h.hexdigest()[:16]
