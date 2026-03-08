"""Artifact resolution and manifest generation for fast-wheel daemon.

Resolves requirements via `uv pip compile` to get exact wheel URLs,
classifies them by size, and generates the JSON manifest that the
Rust daemon reads.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform as platform_mod
import re
import subprocess
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("zerostart.resolver")

# Wheels smaller than this go through uv (tiny/metadata-sensitive)
UV_THRESHOLD = 1 * 1024 * 1024  # 1MB


def _detect_arch() -> str:
    """Detect the machine architecture for wheel selection.

    Returns arch string as used in wheel filenames (e.g. 'x86_64', 'aarch64').
    """
    machine = platform_mod.machine().lower()
    # Normalize common aliases
    if machine in ("x86_64", "amd64"):
        return "x86_64"
    if machine in ("aarch64", "arm64"):
        return "aarch64"
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

    @property
    def is_small(self) -> bool:
        return self.size < UV_THRESHOLD


@dataclass
class ArtifactPlan:
    """The complete resolved artifact set for a requirements spec."""
    artifacts: list[WheelArtifact]
    python_version: str
    platform: str

    @property
    def uv_wheels(self) -> list[WheelArtifact]:
        """Tiny wheels that uv should handle."""
        return [a for a in self.artifacts if a.is_small]

    @property
    def fast_wheels(self) -> list[WheelArtifact]:
        """Medium/large wheels for fast-wheel daemon."""
        return [a for a in self.artifacts if not a.is_small]

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
    import sys
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _detect_platform() -> str:
    """Detect the current platform for wheel selection.

    Returns 'linux' or 'macos' — the values uv pip compile accepts.
    """
    import sys
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
    """Resolve requirements to exact artifacts via uv pip compile + PyPI JSON.

    Phase 1: uses uv pip compile for resolution, then PyPI JSON API
    for wheel URLs. This works for public PyPI packages. Custom indexes
    and direct URLs need Option A/B from the planning doc.
    """
    if python_version is None:
        python_version = _detect_python_version()
    if platform is None:
        platform = _detect_platform()

    if not requirements:
        return ArtifactPlan(artifacts=[], python_version=python_version, platform=platform)

    # Step 1: Resolve with uv pip compile
    resolved = _uv_resolve(requirements, python_version, platform)
    if not resolved:
        return ArtifactPlan(artifacts=[], python_version=python_version, platform=platform)

    # Step 2: Look up wheel URLs and sizes from PyPI JSON (parallel)
    artifacts = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {
            pool.submit(_lookup_pypi_wheel, dist, version, python_version, platform): (dist, version)
            for dist, version in resolved
        }
        for future in as_completed(futures):
            dist, version = futures[future]
            artifact = future.result()
            if artifact:
                artifacts.append(artifact)
            else:
                log.warning("could not find wheel URL for %s==%s", dist, version)

    return ArtifactPlan(
        artifacts=artifacts,
        python_version=python_version,
        platform=platform,
    )


def _uv_platform_tag(platform: str) -> str:
    """Build the --python-platform value for uv pip compile."""
    arch = _detect_arch()
    if platform == "macos":
        return f"{arch}-apple-darwin"
    if platform == "linux":
        return f"{arch}-unknown-linux-gnu"
    # Fallback: pass through (uv will validate)
    return platform


def _uv_resolve(
    requirements: list[str],
    python_version: str,
    platform: str,
) -> list[tuple[str, str]]:
    """Run uv pip compile to resolve requirements to pinned versions."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for req in requirements:
            f.write(req + "\n")
        req_file = f.name

    try:
        result = subprocess.run(
            [
                "uv", "pip", "compile", req_file,
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

        resolved = []
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Remove inline comments (e.g., "# via torch")
            line = line.split("#")[0].strip()
            # Parse "package==version"
            match = re.match(r"^([a-zA-Z0-9_.-]+)==(.+)$", line)
            if match:
                resolved.append((match.group(1), match.group(2)))

        return resolved
    finally:
        Path(req_file).unlink(missing_ok=True)


def _lookup_pypi_wheel(
    distribution: str,
    version: str,
    python_version: str,
    platform: str,
) -> WheelArtifact | None:
    """Look up wheel URL from PyPI JSON API.

    This is the Phase 1 / public-PyPI-only fallback. Production
    should use uv's resolved artifact output directly.
    """
    try:
        url = f"https://pypi.org/pypi/{distribution}/{version}/json"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        log.warning("PyPI lookup failed for %s==%s: %s", distribution, version, e)
        return None

    # Find the best wheel
    py_tag = f"cp{python_version.replace('.', '')}"
    best_url = None
    best_size = 0

    for file_info in data.get("urls", []):
        filename = file_info.get("filename", "")
        if not filename.endswith(".whl"):
            continue

        file_url = file_info["url"]
        file_size = file_info.get("size", 0)

        # Prefer platform-specific wheel matching current arch
        arch = _detect_arch()
        plat_tag = "macosx" if platform == "macos" else platform
        if plat_tag in filename and py_tag in filename and arch in filename:
            best_url = file_url
            best_size = file_size
            break
        elif "none-any" in filename:
            if not best_url:
                best_url = file_url
                best_size = file_size

    if not best_url:
        # Fallback to first .whl
        for file_info in data.get("urls", []):
            if file_info.get("filename", "").endswith(".whl"):
                best_url = file_info["url"]
                best_size = file_info.get("size", 0)
                break

    if not best_url:
        return None

    # Extract import roots from top_level.txt if available
    import_roots = _guess_import_roots(distribution, data)

    return WheelArtifact(
        distribution=distribution,
        version=version,
        url=best_url,
        size=best_size,
        import_roots=import_roots,
    )


def _guess_import_roots(distribution: str, pypi_data: dict) -> list[str]:
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
