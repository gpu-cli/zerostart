"""Orchestrator: resolve → start DaemonHandle → lazy import hook → run.

Supports two modes:
- Script mode: `zerostart serve.py` — run a Python script with progressive loading
- Package mode: `zerostart comfyui` — install package, discover entry point, run it

Both modes use the same pipeline: resolve deps, cache venv, stream large
wheels via the Rust daemon, install small wheels via uv, and gate imports
with the lazy import hook for progressive loading.

Usage:
    zerostart serve.py                          # script with PEP 723 deps
    zerostart -r requirements.txt serve.py      # script with requirements file
    zerostart -p torch -p transformers serve.py  # script with explicit deps
    zerostart comfyui                           # package mode (like uvx)
    zerostart -p torch comfyui                   # package mode with extra deps
    zerostart vllm serve meta-llama/Llama-3-8B  # package mode with args
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field as dc_field
from pathlib import Path

from zerostart.cache import CachedEnv, EnvironmentCache
from zerostart.entrypoints import (
    EntryPointError,
    discover_entry_point,
    invoke_entry_point,
)
from zerostart.lazy_imports import install_hook, remove_hook
from zerostart.resolver import ArtifactPlan, resolve_requirements

log = logging.getLogger("zerostart")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_uv() -> str | None:
    return shutil.which("uv")


def _is_script(target: str) -> bool:
    """Determine if target is a Python script (vs a package name)."""
    if target.endswith(".py"):
        return True
    # A file that exists on disk is treated as a script
    if Path(target).is_file():
        return True
    return False


def parse_requirements(path: str) -> list[str]:
    """Parse a requirements.txt file, returning package specifiers."""
    reqs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            reqs.append(line)
    return reqs


def parse_inline_metadata(script: str) -> list[str]:
    """Parse PEP 723 inline script metadata from a Python script.

    Looks for a block like:
        # /// script
        # dependencies = ["torch", "transformers>=4.0"]
        # ///

    Returns list of dependency specifiers, or empty list if none found.
    """
    try:
        with open(script) as f:
            content = f.read()
    except (FileNotFoundError, PermissionError):
        return []

    # Match the /// script ... /// block (PEP 723 format)
    pattern = r'(?m)^# /// script\s*\n((?:#[^\n]*\n)*)# ///'
    match = re.search(pattern, content)
    if not match:
        return []

    # Strip leading "# " from each line to get TOML content
    block = match.group(1)
    toml_lines = []
    for line in block.splitlines():
        # Remove leading "# " or "#"
        stripped = re.sub(r'^#\s?', '', line)
        toml_lines.append(stripped)
    toml_content = '\n'.join(toml_lines)

    # Parse the dependencies array from TOML-like content
    # Full TOML parsing is overkill — just extract the dependencies list
    deps_match = re.search(
        r'dependencies\s*=\s*\[(.*?)\]',
        toml_content,
        re.DOTALL,
    )
    if not deps_match:
        return []

    # Extract quoted strings from the array
    raw = deps_match.group(1)
    matches = re.findall(r'"([^"]+)"|\'([^\']+)\'', raw)
    # re.findall with alternation returns tuples — take whichever group matched
    return [double or single for double, single in matches]


# ---------------------------------------------------------------------------
# Installation helpers
# ---------------------------------------------------------------------------

def _install_uv_wheels(
    plan: ArtifactPlan,
    site_packages: Path,
    venv_dir: Path,
) -> None:
    """Install small wheels via uv pip install (runs in background thread)."""
    uv = _find_uv()
    if not uv:
        log.warning("uv not found — skipping small wheel install")
        return

    if not plan.uv_wheels:
        return

    specs = [f"{w.distribution}=={w.version}" for w in plan.uv_wheels]
    python = str(venv_dir / "bin" / "python")

    log.info("[uv] Installing %d small packages...", len(specs))
    t0 = time.monotonic()

    result = subprocess.run(
        [uv, "pip", "install", "--python", python] + specs,
        capture_output=True,
        text=True,
    )

    elapsed = time.monotonic() - t0
    if result.returncode == 0:
        log.info("[uv] %d packages installed (%.1fs)", len(specs), elapsed)
    else:
        log.error("[uv] Install failed (%.1fs): %s", elapsed, result.stderr[:500])


def _start_daemon(plan: ArtifactPlan, site_packages: Path) -> object | None:
    """Start the Rust DaemonHandle for large wheels. Returns handle or None."""
    if not plan.fast_wheels:
        return None

    try:
        from zs_fast_wheel import DaemonHandle
    except ImportError:
        log.warning(
            "zs_fast_wheel not available — falling back to uv for all %d wheels",
            len(plan.fast_wheels),
        )
        return None

    wheels = [
        {
            "url": w.url,
            "distribution": w.distribution,
            "size": w.size,
            "import_roots": w.import_roots,
            "hash": w.hash,
        }
        for w in plan.fast_wheels
    ]

    daemon = DaemonHandle()
    daemon.start(
        wheels=wheels,
        site_packages=str(site_packages),
    )

    log.info("[daemon] Started for %d large packages", len(plan.fast_wheels))
    return daemon


# ---------------------------------------------------------------------------
# Prepared environment
# ---------------------------------------------------------------------------

@dataclass
class PreparedEnv:
    """Result of prepare_env(): everything needed to run a script or entry point."""
    site_packages: Path
    env: CachedEnv
    cache: EnvironmentCache
    plan: ArtifactPlan
    requirements: list[str] = dc_field(default_factory=list)
    daemon: object | None = None
    hook: object | None = None
    uv_thread: threading.Thread | None = None
    is_cached: bool = False


def prepare_env(
    requirements: list[str],
    cache_dir: str | None = None,
) -> PreparedEnv:
    """Resolve requirements, set up cache/daemon/uv/hook.

    Shared pipeline for both script mode and package mode.
    Returns PreparedEnv ready for script exec or entry point invocation.
    """
    cache = EnvironmentCache(Path(cache_dir) if cache_dir else None)

    # Fast warm path: check input hash before resolving
    cached = cache.lookup_by_input(requirements)
    if cached:
        log.info("Cache hit — skipping resolution")
        sys.path.insert(0, str(cached.site_packages))
        return PreparedEnv(
            site_packages=cached.site_packages,
            env=cached,
            cache=cache,
            plan=ArtifactPlan(artifacts=[], python_version="3.11", platform="linux"),
            requirements=requirements,
            is_cached=True,
        )

    # Cold path: resolve requirements
    log.info("Resolving %d requirements...", len(requirements))
    plan = resolve_requirements(requirements)

    if not plan.artifacts:
        log.warning("No artifacts resolved")
        env = cache.create_env(plan)
        return PreparedEnv(
            site_packages=env.site_packages,
            env=env,
            cache=cache,
            plan=plan,
            requirements=requirements,
            is_cached=False,
        )

    log.info(
        "Resolved: %d packages (%d small via uv, %d large via daemon)",
        len(plan.artifacts),
        len(plan.uv_wheels),
        len(plan.fast_wheels),
    )

    # Check plan-keyed cache (handles case where input changed but resolved same)
    cached = cache.lookup(plan)
    if cached and cached.is_complete:
        log.info("Cache hit — using cached environment")
        cache.save_input_mapping(requirements, plan)
        sys.path.insert(0, str(cached.site_packages))
        return PreparedEnv(
            site_packages=cached.site_packages,
            env=cached,
            cache=cache,
            plan=plan,
            requirements=requirements,
            is_cached=True,
        )

    # Cold path: create env and install progressively
    env = cached if cached else cache.create_env(plan)
    site_packages = env.site_packages
    sys.path.insert(0, str(site_packages))

    # Start uv for small wheels in background
    uv_thread = None
    if plan.uv_wheels:
        uv_thread = threading.Thread(
            target=_install_uv_wheels,
            args=(plan, site_packages, env.env_dir),
            daemon=True,
        )
        uv_thread.start()

    # Start Rust daemon for large wheels
    daemon = _start_daemon(plan, site_packages)

    # If daemon unavailable but we have large wheels, install them via uv too
    if not daemon and plan.fast_wheels:
        _install_large_via_uv(plan, site_packages, env.env_dir)

    # Install lazy import hook (only when daemon is running)
    hook = install_hook(
        daemon=daemon,
        import_map=plan.import_to_distribution,
    ) if daemon else None

    # If no daemon (all via uv), wait for uv to finish
    if not daemon and uv_thread:
        log.info("Waiting for uv to finish (no daemon)...")
        uv_thread.join(timeout=120)
        uv_thread = None

    return PreparedEnv(
        site_packages=site_packages,
        env=env,
        cache=cache,
        plan=plan,
        requirements=requirements,
        daemon=daemon,
        hook=hook,
        uv_thread=uv_thread,
        is_cached=False,
    )


def _install_large_via_uv(
    plan: ArtifactPlan,
    site_packages: Path,
    venv_dir: Path,
) -> None:
    """Fallback: install large wheels via uv when daemon is unavailable."""
    uv = _find_uv()
    if not uv:
        log.error("Neither zs_fast_wheel nor uv available — cannot install large wheels")
        return

    specs = [f"{w.distribution}=={w.version}" for w in plan.fast_wheels]
    python = str(venv_dir / "bin" / "python")

    log.info("[uv-fallback] Installing %d large packages...", len(specs))
    t0 = time.monotonic()

    result = subprocess.run(
        [uv, "pip", "install", "--python", python] + specs,
        capture_output=True,
        text=True,
    )

    elapsed = time.monotonic() - t0
    if result.returncode == 0:
        log.info("[uv-fallback] %d packages installed (%.1fs)", len(specs), elapsed)
    else:
        log.error("[uv-fallback] Install failed (%.1fs): %s", elapsed, result.stderr[:500])


def cleanup_env(prepared: PreparedEnv, t0: float) -> None:
    """Shared cleanup: remove hook, wait for threads/daemon, mark complete."""
    total = time.monotonic() - t0
    log.info("Finished in %.1fs", total)

    report = remove_hook() if prepared.hook else None

    if prepared.uv_thread:
        prepared.uv_thread.join(timeout=30)

    if prepared.daemon:
        try:
            prepared.daemon.wait_all(timeout_secs=60.0)
        except Exception:
            pass
        prepared.daemon.shutdown()

    prepared.cache.mark_complete(prepared.env)

    # Save input → env mapping for fast warm path on next run
    if prepared.requirements and prepared.plan.artifacts:
        prepared.cache.save_input_mapping(prepared.requirements, prepared.plan)

    if report:
        log.info("Import wait times:")
        for pkg, wait in sorted(report.items(), key=lambda x: -x[1]):
            log.info("  %s: %.2fs", pkg, wait)


# ---------------------------------------------------------------------------
# Script mode
# ---------------------------------------------------------------------------

def run(
    script: str,
    requirements: list[str] | None = None,
    requirements_file: str | None = None,
    cache_dir: str | None = None,
) -> None:
    """Run a Python script with lazy imports and progressive installation."""
    if requirements is None:
        requirements = []
    if requirements_file:
        requirements.extend(parse_requirements(requirements_file))
    elif not requirements:
        inline = parse_inline_metadata(script)
        if inline:
            log.info("Found PEP 723 inline dependencies: %s", inline)
            requirements.extend(inline)
        elif Path("requirements.txt").exists():
            requirements = parse_requirements("requirements.txt")

    if not requirements:
        log.warning("No requirements found — running script directly")
        exec(compile(open(script).read(), script, "exec"), {"__name__": "__main__"})
        return

    prepared = prepare_env(requirements, cache_dir=cache_dir)

    if prepared.is_cached:
        exec(compile(open(script).read(), script, "exec"), {"__name__": "__main__"})
        return

    log.info("Running %s (packages installing in background)...", script)
    t0 = time.monotonic()

    try:
        script_globals = {"__name__": "__main__", "__file__": script}
        exec(compile(open(script).read(), script, "exec"), script_globals)
    finally:
        cleanup_env(prepared, t0)


# ---------------------------------------------------------------------------
# Package mode (like uvx)
# ---------------------------------------------------------------------------

def run_package(
    package: str,
    args: list[str] | None = None,
    extra_packages: list[str] | None = None,
    cache_dir: str | None = None,
) -> None:
    """Install a package and run its console_script entry point.

    Like uvx but with streaming extraction and progressive loading.
    The entry point runs in-process so the lazy import hook works.
    """
    if args is None:
        args = []

    # Build requirements: target package + any extras
    requirements = [package]
    if extra_packages:
        requirements.extend(extra_packages)

    prepared = prepare_env(requirements, cache_dir=cache_dir)

    if not prepared.is_cached:
        # Wait for the target package's metadata to be on disk before
        # we can discover its entry point.
        pkg_normalized = re.sub(r"[-_.]+", "-", package.split("[")[0]).lower()
        in_daemon = any(
            re.sub(r"[-_.]+", "-", w.distribution).lower() == pkg_normalized
            for w in prepared.plan.fast_wheels
        )

        if in_daemon and prepared.daemon:
            log.info("Waiting for %s metadata (daemon)...", package)
            prepared.daemon.wait_done(package, timeout_secs=120.0)

        if not in_daemon and prepared.uv_thread:
            # Target package is in the uv batch — must wait for uv to finish
            log.info("Waiting for %s metadata (uv)...", package)
            prepared.uv_thread.join(timeout=120)
            prepared.uv_thread = None

    # Discover the console_script entry point
    try:
        ep = discover_entry_point(package, prepared.site_packages)
    except EntryPointError as e:
        log.error("%s", e)
        sys.exit(1)

    log.info("Running %s (packages installing in background)...", ep.name)
    t0 = time.monotonic()

    try:
        invoke_entry_point(ep, args)
    finally:
        cleanup_env(prepared, t0)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="zerostart",
        description="Run Python scripts or packages with progressive loading",
        usage="zerostart [-p PKG ...] [-r FILE] [--cache-dir DIR] [-v] <target> [args ...]",
    )
    parser.add_argument(
        "target",
        help="Python script (.py file) or package name to run",
    )
    parser.add_argument(
        "target_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to script or entry point",
    )
    parser.add_argument("-r", "--requirements", help="Requirements file")
    parser.add_argument("-p", "--packages", action="append", help="Additional packages to install (repeatable: -p torch -p numpy)")
    parser.add_argument("--cache-dir", help="Cache directory (default: .zerostart)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s.%(msecs)03d %(name)-10s %(message)s",
        datefmt="%H:%M:%S",
    )

    if _is_script(args.target):
        # Script mode: pass args through via sys.argv
        sys.argv = [args.target] + args.target_args
        run(
            script=args.target,
            requirements=args.packages,
            requirements_file=args.requirements,
            cache_dir=args.cache_dir,
        )
    else:
        # Package mode: target is a package name, args go to its entry point
        run_package(
            package=args.target,
            args=args.target_args,
            extra_packages=args.packages,
            cache_dir=args.cache_dir,
        )


if __name__ == "__main__":
    main()
