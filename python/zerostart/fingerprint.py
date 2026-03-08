"""Environment fingerprint for snapshot validation.

Computes a hash of the installed Python environment so that snapshots
are only restored when the environment matches. If packages change,
the fingerprint changes, and a cold start is triggered instead.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger("zerostart")


def compute_env_fingerprint(site_packages: Path | None = None) -> str:
    """Compute a fingerprint of the current Python environment.

    Strategy (in order of preference):
    1. If site_packages has a wheel manifest from fast-wheel daemon, hash it
    2. Fallback: hash the normalized output of `uv pip freeze` or `pip freeze`

    Args:
        site_packages: Path to site-packages directory (optional, for manifest lookup).

    Returns:
        Hex digest string (first 16 chars of SHA-256).
    """
    # Strategy 1: Check for fast-wheel manifest
    if site_packages is not None:
        manifest_path = site_packages / ".zerostart-manifest.json"
        if manifest_path.is_file():
            h = hashlib.sha256()
            h.update(manifest_path.read_bytes())
            fingerprint = h.hexdigest()[:16]
            log.debug("Env fingerprint from manifest: %s", fingerprint)
            return fingerprint

    # Strategy 2: pip freeze
    freeze_output = _get_freeze_output()
    if freeze_output:
        h = hashlib.sha256()
        h.update(freeze_output.encode())
        fingerprint = h.hexdigest()[:16]
        log.debug("Env fingerprint from pip freeze: %s", fingerprint)
        return fingerprint

    # Strategy 3: If nothing works, hash the site-packages listing
    if site_packages is not None and site_packages.is_dir():
        return _hash_directory_listing(site_packages)

    # No environment info available
    log.warning("Cannot compute env fingerprint — no freeze output or site-packages")
    return "unknown"


def _get_freeze_output() -> str | None:
    """Get normalized pip freeze output, preferring uv."""
    for cmd in [["uv", "pip", "freeze"], ["pip", "freeze"]]:
        if shutil.which(cmd[0]) is None:
            continue
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                return _normalize_freeze(result.stdout)
        except (subprocess.TimeoutExpired, OSError):
            continue
    return None


def _normalize_freeze(output: str) -> str:
    """Normalize pip freeze output for stable hashing."""
    lines = []
    for line in output.strip().splitlines():
        line = line.strip().lower()
        if line and not line.startswith("#") and not line.startswith("-"):
            lines.append(line)
    lines.sort()
    return "\n".join(lines)


def _hash_directory_listing(site_packages: Path) -> str:
    """Fallback: hash the sorted list of installed packages."""
    h = hashlib.sha256()
    entries = sorted(p.name for p in site_packages.iterdir() if not p.name.startswith("."))
    for entry in entries:
        h.update(entry.encode())
        h.update(b"\n")
    fingerprint = h.hexdigest()[:16]
    log.debug("Env fingerprint from directory listing: %s", fingerprint)
    return fingerprint
