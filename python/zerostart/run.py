"""Orchestrator: resolve → start DaemonHandle → lazy import hook → run user script.

This is the core of `zerostart <script>`. It:
1. Resolves requirements via uv pip compile + PyPI JSON
2. Classifies wheels: small → uv, large → fast-wheel daemon
3. Creates/reuses a cached venv
4. Starts DaemonHandle (Rust, in-process) for large wheels
5. Optionally runs uv for small wheels in parallel
6. Installs the lazy import hook
7. Runs the user's script — imports resolve progressively as packages land

Usage:
    zerostart serve.py
    zerostart -r requirements.txt serve.py
    zerostart -p torch transformers serve.py
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
from pathlib import Path

from zerostart.cache import EnvironmentCache
from zerostart.lazy_imports import install_hook, remove_hook
from zerostart.resolver import ArtifactPlan, resolve_requirements

log = logging.getLogger("zerostart")


def _find_uv() -> str | None:
    return shutil.which("uv")


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
        log.error("zs_fast_wheel not available — large wheels won't be installed progressively")
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


def run(
    script: str,
    requirements: list[str] | None = None,
    requirements_file: str | None = None,
    cache_dir: str | None = None,
) -> None:
    """Run a Python script with lazy imports and progressive installation."""
    # Parse requirements from (in priority order):
    # 1. Explicit -p packages or -r file
    # 2. PEP 723 inline script metadata (# /// script)
    # 3. requirements.txt in current directory
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

    # Resolve requirements
    log.info("Resolving %d requirements...", len(requirements))
    plan = resolve_requirements(requirements)

    if not plan.artifacts:
        log.warning("No artifacts resolved — running script directly")
        exec(compile(open(script).read(), script, "exec"), {"__name__": "__main__"})
        return

    log.info(
        "Resolved: %d packages (%d small via uv, %d large via daemon)",
        len(plan.artifacts),
        len(plan.uv_wheels),
        len(plan.fast_wheels),
    )

    # Check cache
    cache = EnvironmentCache(Path(cache_dir) if cache_dir else None)
    cached = cache.lookup(plan)

    if cached and cached.is_complete:
        # Warm path: reuse cached env
        log.info("Cache hit — using cached environment")
        sys.path.insert(0, str(cached.site_packages))
        exec(compile(open(script).read(), script, "exec"), {"__name__": "__main__"})
        return

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

    # Install lazy import hook
    hook = install_hook(
        daemon=daemon,
        import_map=plan.import_to_distribution,
    ) if daemon else None

    # Run user script
    log.info("Running %s (packages installing in background)...", script)
    t0 = time.monotonic()

    try:
        script_globals = {"__name__": "__main__", "__file__": script}
        exec(compile(open(script).read(), script, "exec"), script_globals)
    finally:
        total = time.monotonic() - t0
        log.info("Script finished in %.1fs", total)

        # Cleanup
        report = remove_hook() if hook else None

        if uv_thread:
            uv_thread.join(timeout=30)

        if daemon:
            try:
                daemon.wait_all(timeout_secs=60.0)
            except Exception:
                pass
            daemon.shutdown()

        # Mark env complete if everything succeeded
        cache.mark_complete(env)

        if report:
            log.info("Import wait times:")
            for pkg, wait in sorted(report.items(), key=lambda x: -x[1]):
                log.info("  %s: %.2fs", pkg, wait)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="zerostart",
        description="Run a Python script with progressive package loading",
        usage="zerostart [-r FILE] [-p PKG ...] [--cache-dir DIR] [-v] <script> [args ...]",
    )
    parser.add_argument("script", help="Python script to run")
    parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments passed to the script")
    parser.add_argument("-r", "--requirements", help="Requirements file (default: requirements.txt)")
    parser.add_argument("-p", "--packages", nargs="+", help="Additional packages to install")
    parser.add_argument("--cache-dir", help="Cache directory (default: .zerostart)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s.%(msecs)03d %(name)-10s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Pass script args through via sys.argv so the script sees them
    sys.argv = [args.script] + args.script_args

    run(
        script=args.script,
        requirements=args.packages,
        requirements_file=args.requirements,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
