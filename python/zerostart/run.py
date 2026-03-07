"""Orchestrator: start background install + lazy import hook + run user script.

This is the core of `zerostart run python serve.py`. It:
1. Parses requirements (requirements.txt or pyproject.toml)
2. Creates/reuses a venv
3. Starts background installers (uv for small packages, fast-wheel for large)
4. Installs the lazy import hook
5. Runs the user's script — imports resolve progressively as packages land

Usage:
    python -m zerostart.run serve.py
    python -m zerostart.run -r requirements.txt serve.py
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from zerostart.lazy_imports import install_hook, remove_hook

log = logging.getLogger("zerostart")

# Threshold: wheels larger than this go to fast-wheel (if available)
LARGE_WHEEL_THRESHOLD = 50 * 1024 * 1024  # 50MB


def find_uv() -> str | None:
    for candidate in ["uv", os.path.expanduser("~/.cargo/bin/uv")]:
        if shutil.which(candidate):
            return candidate
    return None


def find_fast_wheel() -> str | None:
    for candidate in ["zs-fast-wheel", "./bin/zs-fast-wheel-linux-x86_64"]:
        if shutil.which(candidate) or Path(candidate).is_file():
            return candidate
    return None


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


def ensure_venv(venv_dir: Path) -> Path:
    """Create a venv if it doesn't exist. Returns path to site-packages."""
    if not (venv_dir / "bin" / "python").exists():
        uv = find_uv()
        if uv:
            subprocess.run([uv, "venv", str(venv_dir)], check=True, capture_output=True)
        else:
            subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True, capture_output=True)

    # Find site-packages
    candidates = list(venv_dir.glob("lib/python*/site-packages"))
    if not candidates:
        raise RuntimeError(f"No site-packages found in {venv_dir}")
    return candidates[0]


def background_install(
    requirements: list[str],
    venv_dir: Path,
    status_dir: Path,
    demand_path: Path,
) -> None:
    """Install packages in background. Monitors demand file for prioritization."""
    uv = find_uv()
    python = str(venv_dir / "bin" / "python")

    if not uv:
        log.error("uv not found — cannot install packages")
        (status_dir / "installing").unlink(missing_ok=True)
        (status_dir / "__done__").touch()
        return

    # Install all at once — uv handles parallelism internally
    log.info("[installer] Installing %d packages with uv...", len(requirements))
    t0 = time.monotonic()

    result = subprocess.run(
        [uv, "pip", "install", "--python", python] + requirements,
        capture_output=True,
        text=True,
    )

    elapsed = time.monotonic() - t0
    if result.returncode == 0:
        log.info("[installer] All packages installed (%.1fs)", elapsed)
    else:
        log.error("[installer] Install failed (%.1fs): %s", elapsed, result.stderr[:500])

    # Signal done
    (status_dir / "installing").unlink(missing_ok=True)
    (status_dir / "__done__").touch()


def run(
    script: str,
    requirements: list[str] | None = None,
    requirements_file: str | None = None,
    venv_dir: str | None = None,
) -> None:
    """Run a Python script with lazy imports and background installation."""
    # Parse requirements
    if requirements is None:
        requirements = []
    if requirements_file:
        requirements.extend(parse_requirements(requirements_file))
    elif not requirements and Path("requirements.txt").exists():
        requirements = parse_requirements("requirements.txt")

    if not requirements:
        log.warning("No requirements found — running script directly")
        exec(compile(open(script).read(), script, "exec"), {"__name__": "__main__"})
        return

    # Setup directories
    venv = Path(venv_dir) if venv_dir else Path(".zerostart") / "venv"
    status_dir = Path(tempfile.mkdtemp(prefix="zs-status-"))

    try:
        # Create venv and find site-packages
        site_packages = ensure_venv(venv)
        sys.path.insert(0, str(site_packages))

        # Signal: installer starting
        (status_dir / "installing").touch()

        # Start background installer
        installer = threading.Thread(
            target=background_install,
            args=(requirements, venv, status_dir, status_dir / "demand"),
            daemon=True,
        )
        installer.start()

        # Install lazy import hook
        hook = install_hook(status_dir=status_dir, poll_interval=0.02)

        # Run user script
        log.info("Running %s (packages installing in background)...", script)
        t0 = time.monotonic()

        script_globals = {"__name__": "__main__", "__file__": script}
        exec(compile(open(script).read(), script, "exec"), script_globals)

        total = time.monotonic() - t0
        log.info("Script finished in %.1fs", total)

        # Report
        report = remove_hook()
        if report:
            log.info("Import wait times:")
            for pkg, wait in sorted(report.items(), key=lambda x: -x[1]):
                log.info("  %s: %.2fs", pkg, wait)

        installer.join(timeout=30)

    finally:
        # Cleanup status dir
        shutil.rmtree(status_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Python script with lazy imports")
    parser.add_argument("script", help="Python script to run")
    parser.add_argument("-r", "--requirements", help="Requirements file (default: requirements.txt)")
    parser.add_argument("-p", "--packages", nargs="+", help="Additional packages to install")
    parser.add_argument("--venv", help="Venv directory (default: .zerostart/venv)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s.%(msecs)03d %(name)-10s %(message)s",
        datefmt="%H:%M:%S",
    )

    run(
        script=args.script,
        requirements=args.packages,
        requirements_file=args.requirements,
        venv_dir=args.venv,
    )


if __name__ == "__main__":
    main()
