"""Discover and invoke entry points from installed packages.

Supports two kinds of entry points:
1. console_scripts (Python): e.g. comfyui = comfyui.main:main
   → Invoked in-process so the lazy import hook provides progressive loading.
2. .data/scripts binaries (Rust/Go/C): e.g. ruff ships a compiled binary
   → Invoked via subprocess (no progressive loading, but still fast install).
"""

from __future__ import annotations

import importlib
import logging
import os
import re
import subprocess
import sys
from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("zerostart")


class EntryPointError(Exception):
    """Base for entry point discovery errors."""


class NoEntryPointError(EntryPointError):
    """Package has no console_script entry points."""


class AmbiguousEntryPointError(EntryPointError):
    """Package has multiple entry points and none match the package name."""


@dataclass
class EntryPoint:
    """A Python console_script entry point (module:attr)."""
    name: str    # e.g. "comfyui"
    module: str  # e.g. "comfyui.main"
    attr: str    # e.g. "main"


@dataclass
class ScriptEntryPoint:
    """A binary script entry point (from .data/scripts/)."""
    name: str    # e.g. "ruff"
    path: Path   # absolute path to the binary


def _normalize(name: str) -> str:
    """Normalize package name for comparison (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse_entry_point_spec(spec: str) -> tuple[str, str]:
    """Parse 'module.path:function' into (module, attr)."""
    module, _, attr = spec.partition(":")
    return module.strip(), attr.strip()


def _dist_info_package_name(dirname: str) -> str:
    """Extract package name from dist-info directory name.

    Format: {name}-{version}.dist-info
    e.g. 'httpie-3.2.4.dist-info' -> 'httpie'
         'my_package-1.0.dist-info' -> 'my-package' (after normalize)
    """
    stem = dirname.removesuffix(".dist-info")
    # Version always starts with a digit. Split at first '-digit' boundary.
    match = re.match(r'^(.+?)-\d', stem)
    if match:
        return match.group(1)
    return stem


def _find_dist_info(normalized: str, site_packages: Path) -> Path | None:
    """Find the dist-info directory for a package."""
    for dist_info in site_packages.glob("*.dist-info"):
        dir_name = _normalize(_dist_info_package_name(dist_info.name))
        if dir_name == normalized:
            return dist_info
    return None


def discover_entry_point(
    package_name: str,
    site_packages: Path,
) -> EntryPoint | ScriptEntryPoint:
    """Find the entry point for a package.

    Checks console_scripts first (Python entry points), then falls back
    to .data/scripts/ binaries (compiled tools like ruff).

    Raises NoEntryPointError if nothing found.
    Raises AmbiguousEntryPointError if multiple console_scripts and none match.
    """
    normalized = _normalize(package_name)

    # Strategy 1: console_scripts from dist-info
    console_scripts = _scan_console_scripts(normalized, site_packages)

    # Strategy 2: importlib.metadata fallback
    if not console_scripts:
        console_scripts = _query_importlib_metadata(package_name)

    if console_scripts:
        if len(console_scripts) == 1:
            return console_scripts[0]

        # Multiple: pick the one matching the package name
        for ep in console_scripts:
            if _normalize(ep.name) == normalized:
                return ep

        names = ", ".join(ep.name for ep in console_scripts)
        raise AmbiguousEntryPointError(
            f"Package '{package_name}' has multiple entry points: {names}. "
            f"None match '{package_name}'."
        )

    # Strategy 3: .data/scripts/ binary (e.g. ruff, uv)
    script_ep = _scan_data_scripts(normalized, site_packages)
    if script_ep:
        return script_ep

    raise NoEntryPointError(
        f"Package '{package_name}' has no console_script entry points. "
        "It may be a library, not an application."
    )


def _scan_console_scripts(
    normalized: str,
    site_packages: Path,
) -> list[EntryPoint]:
    """Parse entry_points.txt from the package's dist-info directory."""
    dist_info = _find_dist_info(normalized, site_packages)
    if not dist_info:
        return []

    ep_file = dist_info / "entry_points.txt"
    if not ep_file.exists():
        return []

    return _parse_entry_points_txt(ep_file.read_text())


def _scan_data_scripts(
    normalized: str,
    site_packages: Path,
) -> ScriptEntryPoint | None:
    """Find a binary in the package's .data/scripts/ directory.

    Some packages (ruff, uv) ship compiled binaries instead of Python
    console_scripts. These land in {dist}-{ver}.data/scripts/.
    """
    for data_dir in site_packages.glob("*.data"):
        dir_name = _normalize(_dist_info_package_name(data_dir.name.removesuffix(".data") + ".dist-info"))
        if dir_name != normalized:
            continue

        scripts_dir = data_dir / "scripts"
        if not scripts_dir.is_dir():
            continue

        # Find executables in the scripts dir
        candidates = []
        for f in scripts_dir.iterdir():
            if f.is_file() and os.access(f, os.X_OK):
                candidates.append(f)

        if not candidates:
            continue

        # Prefer the one matching the package name
        for c in candidates:
            if _normalize(c.name) == normalized:
                return ScriptEntryPoint(name=c.name, path=c)

        # Otherwise take the first one
        return ScriptEntryPoint(name=candidates[0].name, path=candidates[0])

    return None


def _parse_entry_points_txt(content: str) -> list[EntryPoint]:
    """Parse INI-style entry_points.txt, return console_script entries."""
    parser = ConfigParser()
    parser.read_string(content)

    if not parser.has_section("console_scripts"):
        return []

    results = []
    for name, spec in parser.items("console_scripts"):
        module, attr = _parse_entry_point_spec(spec)
        results.append(EntryPoint(name=name, module=module, attr=attr))
    return results


def _query_importlib_metadata(package_name: str) -> list[EntryPoint]:
    """Use importlib.metadata to find entry points (fallback)."""
    try:
        from importlib.metadata import PackageNotFoundError, distribution
        dist = distribution(package_name)
    except (PackageNotFoundError, Exception):
        return []

    results = []
    for ep in dist.entry_points:
        if ep.group == "console_scripts":
            module, attr = _parse_entry_point_spec(ep.value)
            results.append(EntryPoint(name=ep.name, module=module, attr=attr))
    return results


def invoke_entry_point(
    ep: EntryPoint | ScriptEntryPoint,
    args: list[str],
) -> None:
    """Run an entry point — in-process for Python, subprocess for binaries."""
    if isinstance(ep, ScriptEntryPoint):
        log.info("Running binary: %s", ep.path)
        result = subprocess.run([str(ep.path)] + args)
        if result.returncode != 0:
            sys.exit(result.returncode)
        return

    sys.argv = [ep.name] + args

    log.info("Running entry point: %s (%s:%s)", ep.name, ep.module, ep.attr)

    module = importlib.import_module(ep.module)
    func = getattr(module, ep.attr)
    func()
