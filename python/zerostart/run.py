"""Orchestrator: uv for caching + daemon for streaming extraction + progressive loading.

Uses uv's cache for warm starts (instant hardlinks) and our daemon for cold starts
(streaming download+extraction with progressive loading).

Supports two modes:
- Script mode: `zerostart serve.py` — run a Python script with progressive loading
- Package mode: `zerostart comfyui` — install package, discover entry point, run it

Flow:
  WARM (uv cache populated):
    uv venv → uv pip install (resolved 1ms, installed 5ms via hardlinks) → exec
  COLD (no cache):
    uv venv → uv pip install small wheels + daemon streams large wheels → progressive loading
    After daemon finishes: uv pip install saved .whls → populates uv cache for next time

Usage:
    zerostart serve.py                          # script with PEP 723 deps
    zerostart -r requirements.txt serve.py      # script with requirements file
    zerostart -p torch -p transformers serve.py # script with explicit deps
    zerostart comfyui                           # package mode (like uvx)
    zerostart -p torch comfyui                  # package mode with extra deps
    zerostart vllm serve meta-llama/Llama-3-8B  # package mode with args
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from zerostart.entrypoints import (
    EntryPointError,
    discover_entry_point,
    invoke_entry_point,
)
from zerostart.lazy_imports import install_hook, remove_hook
from zerostart.resolver import ArtifactPlan, resolve_requirements

log = logging.getLogger("zerostart")

# Environment cache directory — stores our venvs keyed by requirements
ENV_CACHE_DIR = Path(os.environ.get("ZEROSTART_CACHE", os.path.expanduser("~/.cache/zerostart")))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_uv() -> str:
    uv = shutil.which("uv")
    if not uv:
        log.error("uv is required but not found on PATH")
        sys.exit(1)
    return uv


def _activate_accelerate(opts: dict[str, str | None]) -> None:
    """Enable model loading acceleration after environment is ready."""
    from zerostart.accelerate import accelerate
    accelerate(cache_dir=opts.get("cache_dir"))


def _activate_env(site_packages: Path) -> None:
    """Activate the zerostart environment by adding it to sys.path and isolating from system packages.

    System dist-packages (e.g. /usr/local/lib/python3.11/dist-packages) can contain
    packages incompatible with the zerostart env (e.g. torchvision built for an older torch).
    We remove system dist-packages from sys.path to prevent these from leaking through.
    """
    sys.path.insert(0, str(site_packages))
    # Remove system dist-packages to prevent incompatible system packages from leaking.
    # Keep the zerostart site-packages and stdlib paths.
    zs_sp = str(site_packages)
    sys.path[:] = [
        p for p in sys.path
        if p == zs_sp or "dist-packages" not in p
    ]


def _is_script(target: str) -> bool:
    """Determine if target is a Python script (vs a package name)."""
    if target.endswith(".py"):
        return True
    if Path(target).is_file():
        return True
    return False


def _env_key(requirements: list[str]) -> str:
    """Hash requirements to get a stable environment key."""
    payload = json.dumps(sorted(requirements), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _find_site_packages(venv: Path) -> Path | None:
    candidates = list(venv.glob("lib/python*/site-packages"))
    return candidates[0] if candidates else None


class DaemonProcess:
    """Wrapper around a zs-fast-wheel subprocess that implements the daemon API.

    The binary runs to completion — it downloads all wheels and extracts them
    to site-packages, then exits. This wrapper provides the same API as the
    PyO3 DaemonHandle so the lazy import hook works transparently.
    """

    def __init__(self, proc: subprocess.Popen, manifest_path: str):  # type: ignore[type-arg]
        self._proc = proc
        self._manifest_path = manifest_path
        self._done = False

    def is_done(self, distribution: str) -> bool:
        if self._done:
            return True
        if self._proc.poll() is not None:
            self._done = True
            return True
        return False

    def signal_demand(self, distribution: str) -> None:
        # Binary doesn't support demand signaling — it processes all wheels
        pass

    def wait_done(self, distribution: str, timeout_secs: float = 120.0) -> None:
        if self._done:
            return
        try:
            self._proc.wait(timeout=timeout_secs)
        except subprocess.TimeoutExpired:
            log.warning("daemon timed out after %.0fs waiting for %s", timeout_secs, distribution)
        self._done = True

    def wait_all(self, timeout_secs: float = 60.0) -> None:
        if self._done:
            return
        try:
            self._proc.wait(timeout=timeout_secs)
        except subprocess.TimeoutExpired:
            log.warning("daemon timed out after %.0fs", timeout_secs)
            self._proc.kill()
        self._done = True

    def shutdown(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        # Clean up manifest
        try:
            Path(self._manifest_path).unlink(missing_ok=True)
        except Exception:
            pass


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
    """Parse PEP 723 inline script metadata from a Python script."""
    try:
        with open(script) as f:
            content = f.read()
    except (FileNotFoundError, PermissionError):
        return []

    pattern = r'(?m)^# /// script\s*\n((?:#[^\n]*\n)*)# ///'
    match = re.search(pattern, content)
    if not match:
        return []

    block = match.group(1)
    toml_lines = []
    for line in block.splitlines():
        stripped = re.sub(r'^#\s?', '', line)
        toml_lines.append(stripped)
    toml_content = '\n'.join(toml_lines)

    deps_match = re.search(
        r'dependencies\s*=\s*\[(.*?)\]',
        toml_content,
        re.DOTALL,
    )
    if not deps_match:
        return []

    raw = deps_match.group(1)
    matches = re.findall(r'"([^"]+)"|\'([^\']+)\'', raw)
    return [double or single for double, single in matches]


# ---------------------------------------------------------------------------
# Environment management (backed by uv)
# ---------------------------------------------------------------------------

def _get_or_create_venv(requirements: list[str]) -> Path:
    """Get or create a venv for these requirements.

    Uses uv venv + uv pip install. On warm path (uv cache populated),
    this is instant — uv resolves in <2ms and hardlinks from its cache.
    """
    key = _env_key(requirements)
    venv = ENV_CACHE_DIR / "envs" / key
    complete_marker = venv / ".complete"

    if complete_marker.exists():
        return venv

    uv = _find_uv()

    # Create venv if needed
    if not venv.exists():
        log.info("Creating environment...")
        subprocess.run(
            [uv, "venv", str(venv), "--python", f"{sys.version_info.major}.{sys.version_info.minor}"],
            capture_output=True,
            check=True,
        )

    return venv


def _uv_install(venv: Path, specs: list[str]) -> subprocess.CompletedProcess[str]:
    """Run uv pip install into a venv. Leverages uv's global cache."""
    uv = _find_uv()
    python = str(venv / "bin" / "python")
    return subprocess.run(
        [uv, "pip", "install", "--python", python] + specs,
        capture_output=True,
        text=True,
    )


def _uv_install_background(venv: Path, whl_paths: list[Path]) -> None:
    """Install local .whl files via uv to populate its cache for next run."""
    if not whl_paths:
        return
    uv = _find_uv()
    python = str(venv / "bin" / "python")
    specs = [str(p) for p in whl_paths if p.exists()]
    if not specs:
        return

    log.info("[uv-cache] Registering %d wheels with uv cache...", len(specs))
    result = subprocess.run(
        [uv, "pip", "install", "--python", python, "--reinstall"] + specs,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        log.info("[uv-cache] Done — next run will be instant")
    else:
        log.warning("[uv-cache] Failed: %s", result.stderr[:200])


# ---------------------------------------------------------------------------
# Installation: uv for small wheels, daemon for large wheels
# ---------------------------------------------------------------------------

def _find_daemon_binary() -> str | None:
    """Find the zs-fast-wheel binary.

    Search order:
    1. ZEROSTART_DAEMON env var
    2. zs-fast-wheel on PATH
    3. bin/zs-fast-wheel-linux-x86_64 (cross-compiled, relative to project root)
    """
    if env_bin := os.environ.get("ZEROSTART_DAEMON"):
        if Path(env_bin).is_file():
            return env_bin

    if path_bin := shutil.which("zs-fast-wheel"):
        return path_bin

    # Check common locations for cross-compiled binary
    import platform as _plat
    arch = _plat.machine()
    system = _plat.system().lower()
    suffix = f"{system}-{arch}" if system != "darwin" else f"darwin-{arch}"
    for candidate in [
        Path(f"bin/zs-fast-wheel-{suffix}"),
        Path(f"bin/zs-fast-wheel-linux-x86_64"),
        Path(__file__).parent.parent.parent / f"bin/zs-fast-wheel-{suffix}",
        Path(__file__).parent.parent.parent / "bin/zs-fast-wheel-linux-x86_64",
    ]:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)

    return None


def _start_daemon(plan: ArtifactPlan, site_packages: Path, wheel_cache_dir: Path) -> tuple[object | None, list[Path]]:
    """Start the Rust daemon for large wheels.

    Tries PyO3 DaemonHandle first, then falls back to the zs-fast-wheel binary
    as a subprocess. Fails hard if neither is available — the daemon is required
    for streaming large wheels.

    Returns (daemon_handle_or_process, list_of_saved_whl_paths).
    """
    if not plan.fast_wheels:
        return None, []

    # Try PyO3 module first
    try:
        from zs_fast_wheel import DaemonHandle

        wheel_cache_dir.mkdir(parents=True, exist_ok=True)

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

        log.info("[daemon] Started via PyO3 for %d large packages", len(plan.fast_wheels))

        whl_paths = [
            wheel_cache_dir / w.url.rsplit('/', 1)[-1]
            for w in plan.fast_wheels
        ]

        return daemon, whl_paths
    except ImportError:
        pass

    # Fall back to binary subprocess
    daemon_bin = _find_daemon_binary()
    if not daemon_bin:
        log.error(
            "zs-fast-wheel daemon not available (no PyO3 module and no binary found). "
            "Install via: cargo zigbuild --target x86_64-unknown-linux-musl --release -p zs-fast-wheel"
        )
        sys.exit(1)

    import tempfile

    # Write manifest for the binary
    manifest = {
        "site_packages": str(site_packages),
        "wheels": [
            {
                "url": w.url,
                "distribution": w.distribution,
                "version": w.version,
                "size": w.size,
                "import_roots": w.import_roots,
                "hash": w.hash,
            }
            for w in plan.fast_wheels
        ],
    }

    manifest_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="zs-manifest-", delete=False
    )
    json.dump(manifest, manifest_file)
    manifest_file.close()

    log.info("[daemon] Starting binary %s for %d large packages", daemon_bin, len(plan.fast_wheels))

    proc = subprocess.Popen(
        [daemon_bin, "daemon", "--manifest", manifest_file.name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    daemon = DaemonProcess(proc, manifest_file.name)

    whl_paths = [
        wheel_cache_dir / w.url.rsplit('/', 1)[-1]
        for w in plan.fast_wheels
    ]

    return daemon, whl_paths


# ---------------------------------------------------------------------------
# Core: prepare environment
# ---------------------------------------------------------------------------

def prepare_env(requirements: list[str]) -> tuple[Path, Path, ArtifactPlan | None, object | None, list[Path], threading.Thread | None]:
    """Prepare environment for running.

    Returns (venv, site_packages, plan, daemon, whl_paths, uv_thread).

    WARM path: uv pip install is instant from cache. No daemon.
    COLD path: uv handles small wheels, daemon streams large wheels.
    """
    uv = _find_uv()
    venv = _get_or_create_venv(requirements)
    site_packages = _find_site_packages(venv)
    complete_marker = venv / ".complete"

    if not site_packages:
        log.error("Could not find site-packages in %s", venv)
        sys.exit(1)

    # Try uv-only fast path first
    if complete_marker.exists():
        log.info("Cache hit — environment ready")
        _activate_env(site_packages)
        return venv, site_packages, None, None, [], None

    # Cold path: resolve to figure out which wheels need daemon
    log.info("Resolving %d requirements...", len(requirements))
    plan = resolve_requirements(requirements)

    if not plan.artifacts:
        log.warning("No artifacts resolved")
        complete_marker.touch()
        _activate_env(site_packages)
        return venv, site_packages, plan, None, [], None

    log.info(
        "Resolved: %d packages (%d small via uv, %d large via daemon)",
        len(plan.artifacts),
        len(plan.uv_wheels),
        len(plan.fast_wheels),
    )

    _activate_env(site_packages)

    # Install only SMALL wheels via uv (fast, metadata-sensitive)
    # Large wheels go through the daemon for streaming extraction
    uv_thread = None
    small_specs = [f"{w.distribution}=={w.version}" for w in plan.uv_wheels]
    if small_specs:
        uv_thread = threading.Thread(
            target=_run_uv_install,
            args=(venv, small_specs),
            daemon=True,
        )
        uv_thread.start()

    # Start daemon for large wheels (streams in parallel with uv)
    wheel_cache = ENV_CACHE_DIR / "wheels"
    daemon, whl_paths = _start_daemon(plan, site_packages, wheel_cache)

    # Install lazy import hook (only for PyO3 daemon — subprocess daemon
    # doesn't support per-package tracking, so we wait for it to finish below)
    is_subprocess_daemon = isinstance(daemon, DaemonProcess) if daemon else False
    if daemon and not is_subprocess_daemon:
        daemon_dists = {w.distribution for w in plan.fast_wheels}
        daemon_import_map = {
            k: v for k, v in plan.import_to_distribution.items()
            if v in daemon_dists
        }
        install_hook(
            daemon=daemon,
            import_map=daemon_import_map,
        )

    # Always wait for uv to finish — small packages aren't covered by
    # the lazy import hook, so they must be on disk before the script starts
    if uv_thread:
        log.info("Waiting for uv to finish...")
        uv_thread.join(timeout=120)
        uv_thread = None

    # Subprocess daemon: wait for ALL wheels to finish before running script.
    # Unlike PyO3 daemon, subprocess can't track per-package progress.
    if is_subprocess_daemon:
        log.info("Waiting for daemon to finish extracting %d packages...", len(plan.fast_wheels))
        daemon.wait_all(timeout_secs=300.0)
        # Check daemon exit status
        if hasattr(daemon, '_proc') and daemon._proc.returncode != 0:
            stderr = daemon._proc.stderr.read().decode() if daemon._proc.stderr else ""
            log.error("Daemon failed (exit %d): %s", daemon._proc.returncode, stderr[:500])
            sys.exit(1)
        log.info("Daemon finished — all packages ready")
        daemon = None  # No cleanup needed later

    if not daemon:
        complete_marker.touch()

    return venv, site_packages, plan, daemon, whl_paths, uv_thread


def _run_uv_install(venv: Path, specs: list[str]) -> None:
    """Run uv pip install (for use in background thread)."""
    t0 = time.monotonic()
    result = _uv_install(venv, specs)
    elapsed = time.monotonic() - t0
    if result.returncode == 0:
        log.info("[uv] %d packages installed (%.1fs)", len(specs), elapsed)
    else:
        log.error("[uv] Install failed (%.1fs): %s", elapsed, result.stderr[:500])


def cleanup(
    venv: Path,
    plan: ArtifactPlan | None,
    daemon: object | None,
    uv_thread: threading.Thread | None,
    whl_paths: list[Path],
    t0: float,
) -> None:
    """Cleanup: wait for daemon/uv, mark complete, populate uv cache."""
    total = time.monotonic() - t0
    log.info("Finished in %.1fs", total)

    report = remove_hook()

    if uv_thread:
        uv_thread.join(timeout=30)

    if daemon:
        try:
            daemon.wait_all(timeout_secs=60.0)
        except Exception:
            pass
        daemon.shutdown()

    # Populate uv's cache by installing ALL resolved specs.
    # This runs after the app finishes so it doesn't slow down cold start.
    # Next run will be instant — uv hardlinks from its cache.
    if plan and plan.artifacts:
        all_specs = [f"{w.distribution}=={w.version}" for w in plan.artifacts]
        log.info("[uv-cache] Populating uv cache with %d packages...", len(all_specs))
        _run_uv_install(venv, all_specs)

    # Mark environment as complete
    (venv / ".complete").touch()

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
    accelerate_opts: dict[str, str | None] | None = None,
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
        if accelerate_opts is not None:
            _activate_accelerate(accelerate_opts)
        exec(compile(open(script).read(), script, "exec"), {"__name__": "__main__"})
        return

    venv, site_packages, plan, daemon, whl_paths, uv_thread = prepare_env(requirements)

    # Enable accelerate AFTER env is ready so we import the correct torch
    if accelerate_opts is not None:
        _activate_accelerate(accelerate_opts)

    if not daemon:
        # Warm path or all-uv — just run
        exec(compile(open(script).read(), script, "exec"), {"__name__": "__main__"})
        return

    log.info("Running %s (packages installing in background)...", script)
    t0 = time.monotonic()

    try:
        script_globals = {"__name__": "__main__", "__file__": script}
        exec(compile(open(script).read(), script, "exec"), script_globals)
    finally:
        cleanup(venv, plan, daemon, uv_thread, whl_paths, t0)


# ---------------------------------------------------------------------------
# Package mode (like uvx)
# ---------------------------------------------------------------------------

def run_package(
    package: str,
    args: list[str] | None = None,
    extra_packages: list[str] | None = None,
    accelerate_opts: dict[str, str | None] | None = None,
) -> None:
    """Install a package and run its console_script entry point."""
    if args is None:
        args = []

    requirements = [package]
    if extra_packages:
        requirements.extend(extra_packages)

    venv, site_packages, plan, daemon, whl_paths, uv_thread = prepare_env(requirements)

    # Enable accelerate AFTER env is ready so we import the correct torch
    if accelerate_opts is not None:
        _activate_accelerate(accelerate_opts)

    if plan and not (venv / ".complete").exists():
        # Wait for the target package's metadata to be on disk
        pkg_normalized = re.sub(r"[-_.]+", "-", package.split("[")[0]).lower()

        if daemon:
            # Find the exact distribution name the daemon is tracking
            daemon_dist = next(
                (w.distribution for w in plan.fast_wheels
                 if re.sub(r"[-_.]+", "-", w.distribution).lower() == pkg_normalized),
                None,
            )
            if daemon_dist:
                log.info("Waiting for %s metadata (daemon)...", daemon_dist)
                daemon.wait_done(daemon_dist, timeout_secs=120.0)
            elif uv_thread:
                log.info("Waiting for %s metadata (uv)...", package)
                uv_thread.join(timeout=120)
                uv_thread = None
        elif uv_thread:
            log.info("Waiting for %s metadata (uv)...", package)
            uv_thread.join(timeout=120)
            uv_thread = None

    # Discover entry point
    try:
        ep = discover_entry_point(package, site_packages)
    except EntryPointError as e:
        log.error("%s", e)
        sys.exit(1)

    if not daemon:
        # Warm path — just run
        invoke_entry_point(ep, args)
        return

    log.info("Running %s (packages installing in background)...", ep.name)
    t0 = time.monotonic()

    try:
        invoke_entry_point(ep, args)
    finally:
        cleanup(venv, plan, daemon, uv_thread, whl_paths, t0)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="zerostart",
        description="Run Python scripts or packages with progressive loading",
        usage="zerostart [-p PKG ...] [-r FILE] [-v] <target> [args ...]",
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
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--accelerate", action="store_true", help="Enable transparent model loading acceleration (4x faster from_pretrained)")
    parser.add_argument("--model-cache-dir", help="Model cache directory for --accelerate")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s.%(msecs)03d %(name)-10s %(message)s",
        datefmt="%H:%M:%S",
    )

    # NOTE: accelerate() must be called AFTER prepare_env() sets up the
    # environment, otherwise it imports system torch (which may be older)
    # and caches it in sys.modules before the zerostart env is on sys.path.
    accelerate_opts = None
    if args.accelerate:
        accelerate_opts = {"cache_dir": args.model_cache_dir}

    if _is_script(args.target):
        sys.argv = [args.target] + args.target_args
        run(
            script=args.target,
            requirements=args.packages,
            requirements_file=args.requirements,
            accelerate_opts=accelerate_opts,
        )
    else:
        run_package(
            package=args.target,
            args=args.target_args,
            extra_packages=args.packages,
            accelerate_opts=accelerate_opts,
        )


if __name__ == "__main__":
    main()
