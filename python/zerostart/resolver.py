"""Artifact resolution and manifest generation for fast-wheel daemon.

Resolves requirements via `uv pip compile --format pylock.toml` which gives
us exact wheel URLs, sizes, and hashes in a single call — no separate PyPI
lookups needed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform as platform_mod
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("zerostart.resolver")



def _detect_arch() -> str:
    """Detect the machine architecture.

    Returns the raw machine arch (e.g. 'x86_64', 'arm64', 'aarch64').
    On macOS this is 'arm64'; on Linux it's 'aarch64'.
    """
    machine = platform_mod.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "x86_64"
    # Don't normalize arm64→aarch64: macOS wheels use 'arm64', Linux uses 'aarch64'
    return machine


@dataclass
class WheelArtifact:
    """A resolved wheel artifact with exact URL, hash, and metadata."""
    distribution: str
    version: str
    url: str
    hash: str | None = None
    size: int = 0
    import_roots: list[str] = field(default_factory=list)



@dataclass
class ArtifactPlan:
    """The complete resolved artifact set for a requirements spec."""
    artifacts: list[WheelArtifact]
    python_version: str
    platform: str

    @property
    def fast_wheels(self) -> list[WheelArtifact]:
        """All wheels — handled by the daemon."""
        return self.artifacts

    @property
    def import_to_distribution(self) -> dict[str, str]:
        """Map import names to distribution names for the lazy hook."""
        mapping: dict[str, str] = {}
        for a in self.artifacts:
            for root in a.import_roots:
                mapping[root] = a.distribution
            # Fallback: distribution name itself
            if not a.import_roots:
                mapping[a.distribution] = a.distribution
        return mapping

    @property
    def cache_key_payload(self) -> dict:
        """Deterministic payload for cache key generation."""
        return {
            "python_version": self.python_version,
            "platform": self.platform,
            "artifacts": [
                {"distribution": a.distribution, "version": a.version, "url": a.url}
                for a in sorted(self.artifacts, key=lambda a: a.distribution)
            ],
        }

    @property
    def cache_key(self) -> str:
        """SHA-256 hash of the cache key payload."""
        payload = json.dumps(self.cache_key_payload, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()


def _detect_python_version() -> str:
    """Detect the running Python's major.minor version."""
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _detect_platform() -> str:
    """Detect the current platform for wheel selection."""
    if sys.platform == "linux":
        return "linux"
    if sys.platform == "darwin":
        return "macos"
    return sys.platform


def resolve_requirements(
    requirements: list[str],
    python_version: str | None = None,
    platform: str | None = None,
) -> ArtifactPlan:
    """Resolve requirements to exact artifacts via uv pip compile --format pylock.toml.

    Single call to uv gives us pinned versions, wheel URLs, sizes, and hashes.
    """
    if python_version is None:
        python_version = _detect_python_version()
    if platform is None:
        platform = _detect_platform()

    if not requirements:
        return ArtifactPlan(artifacts=[], python_version=python_version, platform=platform)

    artifacts = _uv_resolve_pylock(requirements, python_version, platform)

    return ArtifactPlan(
        artifacts=artifacts,
        python_version=python_version,
        platform=platform,
    )


def _uv_platform_tag(platform: str) -> str:
    """Build the --python-platform value for uv pip compile."""
    arch = _detect_arch()
    # uv expects 'aarch64' not 'arm64' in platform tags
    if arch == "arm64":
        arch = "aarch64"
    if platform == "macos":
        return f"{arch}-apple-darwin"
    if platform == "linux":
        return f"{arch}-unknown-linux-gnu"
    # Fallback: pass through (uv will validate)
    return platform


def _parse_pylock_toml(content: str) -> list[WheelArtifact]:
    """Parse pylock.toml output from uv pip compile.

    Uses simple string parsing — no TOML library dependency needed.
    The format is structured and predictable from uv.
    """
    artifacts = []
    current_name: str | None = None
    current_version: str | None = None
    current_url: str | None = None
    current_size: int = 0
    current_hash: str | None = None
    in_wheels = False

    for line in content.splitlines():
        stripped = line.strip()

        if stripped == "[[packages]]":
            # Save previous package if it had a wheel
            if current_name and current_version and current_url:
                artifacts.append(WheelArtifact(
                    distribution=current_name,
                    version=current_version,
                    url=current_url,
                    hash=current_hash,
                    size=current_size,
                    import_roots=_guess_import_roots(current_name),
                ))
            current_name = None
            current_version = None
            current_url = None
            current_size = 0
            current_hash = None
            in_wheels = False
            continue

        if stripped.startswith("name = "):
            current_name = _extract_quoted(stripped)
        elif stripped.startswith("version = ") and not in_wheels:
            current_version = _extract_quoted(stripped)
        elif stripped.startswith("wheels = ["):
            in_wheels = True
            # Inline wheel entry: wheels = [{ url = "...", size = N, ... }]
            if "{" in stripped:
                url, size, sha = _parse_wheel_entry(stripped)
                if url and not current_url:
                    current_url = url
                    current_size = size
                    current_hash = sha
        elif in_wheels and stripped.startswith("{"):
            # Multi-line wheels array entry
            url, size, sha = _parse_wheel_entry(stripped)
            if url and not current_url:
                current_url = url
                current_size = size
                current_hash = sha
        elif stripped == "]":
            in_wheels = False

    # Don't forget the last package
    if current_name and current_version and current_url:
        artifacts.append(WheelArtifact(
            distribution=current_name,
            version=current_version,
            url=current_url,
            hash=current_hash,
            size=current_size,
            import_roots=_guess_import_roots(current_name),
        ))

    return artifacts


def _extract_quoted(line: str) -> str:
    """Extract a quoted string value from a TOML line like 'key = "value"'."""
    match = re.search(r'"([^"]*)"', line)
    return match.group(1) if match else ""


def _parse_wheel_entry(line: str) -> tuple[str | None, int, str | None]:
    """Parse a wheel entry from pylock.toml.

    Returns (url, size, sha256_hash).
    """
    url_match = re.search(r'url\s*=\s*"([^"]+)"', line)
    size_match = re.search(r'size\s*=\s*(\d+)', line)
    hash_match = re.search(r'sha256\s*=\s*"([^"]+)"', line)

    url = url_match.group(1) if url_match else None
    size = int(size_match.group(1)) if size_match else 0
    sha = hash_match.group(1) if hash_match else None

    return url, size, sha


def _uv_resolve_pylock(
    requirements: list[str],
    python_version: str,
    platform: str,
) -> list[WheelArtifact]:
    """Resolve requirements via uv pip compile --format pylock.toml.

    Returns fully resolved artifacts with URLs, sizes, and hashes in a single call.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for req in requirements:
            f.write(req + "\n")
        req_file = f.name

    try:
        result = subprocess.run(
            [
                "uv", "pip", "compile", req_file,
                "--format", "pylock.toml",
                "--python-version", python_version,
                "--python-platform", _uv_platform_tag(platform),
                "--no-header",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            log.error("uv pip compile failed: %s", result.stderr[:500])
            return []

        return _parse_pylock_toml(result.stdout)
    finally:
        Path(req_file).unlink(missing_ok=True)


def _guess_import_roots(distribution: str) -> list[str]:
    """Guess import root names from distribution name.

    Common mismatches: PyYAML→yaml, Pillow→PIL, scikit-learn→sklearn
    """
    known_mappings = {
        "pyyaml": ["yaml"],
        "pillow": ["PIL"],
        "scikit-learn": ["sklearn"],
        "python-dateutil": ["dateutil"],
        "beautifulsoup4": ["bs4"],
        "attrs": ["attr", "attrs"],
    }

    dist_lower = distribution.lower()
    if dist_lower in known_mappings:
        return known_mappings[dist_lower]

    # Default: distribution name with hyphens → underscores
    return [distribution.replace("-", "_").lower()]


def generate_manifest(
    plan: ArtifactPlan,
    site_packages: Path,
    output_dir: Path,
) -> Path:
    """Write manifest.json for the fast-wheel daemon.

    Only includes fast-wheel-classified wheels (not tiny/uv wheels).
    """
    manifest = {
        "site_packages": str(site_packages),
        "wheels": [
            {
                "url": w.url,
                "distribution": w.distribution,
                "import_roots": w.import_roots,
                "size": w.size,
                "hash": w.hash,
            }
            for w in plan.fast_wheels
        ],
    }

    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path
