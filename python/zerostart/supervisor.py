"""Tier 1 Supervisor: cold-start vs restore state machine.

Orchestrates CRIU process snapshots on top of the existing run.py.
On first run: launches app, waits for readiness, dumps snapshot.
On subsequent runs: restores from snapshot if environment matches.

Usage:
    python -m zerostart.supervisor --ready=signal python serve.py
    python -m zerostart.supervisor --ready=url:http://localhost:8000/health python serve.py
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from zerostart.fingerprint import compute_env_fingerprint
from zerostart.ready import make_run_dir, wait_for_ready
from zerostart.snapshot import compute_intent_hash

log = logging.getLogger("zerostart")

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "zerostart" / "snapshots"
DEFAULT_READY_TIMEOUT = 300.0  # 5 minutes
DEFAULT_SETTLE_DELAY = 0.25  # seconds after ready before dump


@dataclass
class SupervisorConfig:
    """Configuration for the Tier 1 supervisor."""

    ready_mode: str = "signal"
    ready_timeout: float = DEFAULT_READY_TIMEOUT
    settle_delay: float = DEFAULT_SETTLE_DELAY
    cache_dir: Path = DEFAULT_CACHE_DIR
    zs_snapshot_bin: str | None = None  # Auto-detect if None


def _find_zs_snapshot() -> str | None:
    """Find the zs-snapshot binary."""
    # Check explicit path first
    explicit = os.environ.get("ZS_SNAPSHOT_BIN")
    if explicit and Path(explicit).is_file():
        return explicit

    # Check sibling to this package
    pkg_dir = Path(__file__).parent.parent.parent
    for candidate in [
        pkg_dir / "bin" / "zs-snapshot-linux-x86_64",
        pkg_dir / "target" / "release" / "zs-snapshot",
        pkg_dir / "target" / "debug" / "zs-snapshot",
        pkg_dir / "crates" / "target" / "release" / "zs-snapshot",
        pkg_dir / "crates" / "target" / "debug" / "zs-snapshot",
    ]:
        if candidate.is_file():
            return str(candidate)

    # Check PATH
    return shutil.which("zs-snapshot")


def _run_zs_snapshot(
    bin_path: str,
    args: list[str],
    timeout: float = 60.0,
) -> subprocess.CompletedProcess[str]:
    """Run zs-snapshot CLI and return result."""
    cmd = [bin_path] + args
    log.debug("Running: %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _load_snapshot_metadata(
    bin_path: str,
    cache_dir: Path,
    intent_hash: str,
) -> dict[str, object] | None:
    """Load snapshot metadata via zs-snapshot CLI. Returns None if not found."""
    metadata_path = cache_dir / intent_hash / "metadata.json"
    if not metadata_path.is_file():
        return None
    try:
        return json.loads(metadata_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Failed to read snapshot metadata: %s", exc)
        return None


def supervise(
    argv: list[str],
    config: SupervisorConfig | None = None,
    site_packages: Path | None = None,
) -> int:
    """Run the Tier 1 supervisor state machine.

    Args:
        argv: Command to run (e.g. ["python", "serve.py"]).
        config: Supervisor configuration. Uses defaults if None.
        site_packages: Path to site-packages for fingerprinting.

    Returns:
        Exit code from the supervised process.
    """
    if config is None:
        config = SupervisorConfig()

    # Find zs-snapshot binary
    bin_path = config.zs_snapshot_bin or _find_zs_snapshot()
    if not bin_path:
        log.warning("zs-snapshot not found — running without snapshot support")
        return _run_without_snapshots(argv)

    # Compute intent hash
    entrypoint = argv[1] if len(argv) > 1 else argv[0]
    intent_hash = compute_intent_hash(entrypoint, argv)
    log.info("Intent hash: %s", intent_hash)

    # Compute environment fingerprint
    env_fingerprint = compute_env_fingerprint(site_packages)
    log.info("Env fingerprint: %s", env_fingerprint)

    # Check for existing snapshot
    metadata = _load_snapshot_metadata(bin_path, config.cache_dir, intent_hash)

    if metadata and metadata.get("env_fingerprint") == env_fingerprint:
        # Restore path
        log.info("Snapshot hit — attempting restore")
        exit_code = _restore_path(bin_path, config, intent_hash, argv)
        if exit_code is not None:
            return exit_code
        log.warning("Restore failed — falling back to cold start")

    # Cold start path
    log.info("Cold start — launching process")
    return _cold_start_path(bin_path, config, intent_hash, env_fingerprint, argv, entrypoint)


def _restore_path(
    bin_path: str,
    config: SupervisorConfig,
    intent_hash: str,
    argv: list[str],
) -> int | None:
    """Attempt to restore from snapshot. Returns exit code or None on failure."""
    run_dir = make_run_dir()
    pidfile = run_dir / "restored.pid"

    try:
        result = _run_zs_snapshot(
            bin_path,
            [
                "restore",
                "--intent-hash", intent_hash,
                "--pidfile", str(pidfile),
                "--cache-dir", str(config.cache_dir),
            ],
            timeout=config.ready_timeout,
        )

        if result.returncode != 0:
            log.warning("Restore failed: %s", result.stderr.strip())
            return None

        # Read restored PID
        if not pidfile.is_file():
            log.warning("Restore succeeded but pidfile not found")
            return None

        restored_pid = int(pidfile.read_text().strip())
        log.info("Restored process PID: %d", restored_pid)

        # Send SIGUSR1 to trigger on_restore callbacks
        try:
            os.kill(restored_pid, signal.SIGUSR1)
            log.info("Sent SIGUSR1 to restored process")
        except OSError as exc:
            log.warning("Failed to send SIGUSR1: %s", exc)

        # Supervise restored process
        return _supervise_pid(restored_pid)

    except subprocess.TimeoutExpired:
        log.warning("Restore timed out")
        return None
    except (OSError, ValueError) as exc:
        log.warning("Restore error: %s", exc)
        return None


def _cold_start_path(
    bin_path: str,
    config: SupervisorConfig,
    intent_hash: str,
    env_fingerprint: str,
    argv: list[str],
    entrypoint: str,
) -> int:
    """Cold start: launch process, wait for ready, dump snapshot."""
    run_dir = make_run_dir()

    # Set environment for managed mode
    env = os.environ.copy()
    env["ZEROSTART_MANAGED"] = "1"
    env["ZEROSTART_RUN_DIR"] = str(run_dir)
    env["ZEROSTART_READY_PATH"] = str(run_dir / "ready")

    # Launch child process
    child = subprocess.Popen(argv, env=env)
    log.info("Launched child PID %d", child.pid)

    # Set up signal forwarding
    child_pid = child.pid

    def forward_signal(signum: int, frame: object) -> None:
        try:
            os.kill(child_pid, signum)
        except OSError:
            pass

    signal.signal(signal.SIGTERM, forward_signal)
    signal.signal(signal.SIGINT, forward_signal)

    # Wait for readiness
    try:
        ready_event = wait_for_ready(config.ready_mode, run_dir, config.ready_timeout)
        log.info("Process ready: %s", ready_event.details)
    except TimeoutError:
        log.warning("Ready timeout — skipping snapshot, continuing to serve")
        return child.wait()

    # Optional settle delay
    if config.settle_delay > 0:
        time.sleep(config.settle_delay)

    # Attempt snapshot dump (non-fatal on failure)
    _attempt_dump(bin_path, config, intent_hash, env_fingerprint, child.pid, entrypoint, argv)

    # Supervise child
    return child.wait()


def _attempt_dump(
    bin_path: str,
    config: SupervisorConfig,
    intent_hash: str,
    env_fingerprint: str,
    pid: int,
    entrypoint: str,
    argv: list[str],
) -> None:
    """Attempt to dump snapshot. Failures are logged but non-fatal."""
    # Write metadata to a temp file for the CLI
    import tempfile

    metadata = {
        "intent_hash": intent_hash,
        "env_fingerprint": env_fingerprint,
        "entrypoint": entrypoint,
        "argv": argv,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": f"{os.uname().machine}-{os.uname().sysname.lower()}",
    }

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(metadata, f)
            metadata_file = f.name

        log.info("Dumping snapshot for PID %d...", pid)
        t0 = time.monotonic()

        result = _run_zs_snapshot(
            bin_path,
            [
                "dump",
                "--intent-hash", intent_hash,
                "--pid", str(pid),
                "--metadata", metadata_file,
                "--cache-dir", str(config.cache_dir),
            ],
            timeout=120.0,
        )

        elapsed = time.monotonic() - t0

        if result.returncode == 0:
            log.info("Snapshot dump succeeded (%.1fs)", elapsed)
        else:
            log.warning("Snapshot dump failed (%.1fs): %s", elapsed, result.stderr.strip()[:500])

    except subprocess.TimeoutExpired:
        log.warning("Snapshot dump timed out")
    except OSError as exc:
        log.warning("Snapshot dump error: %s", exc)
    finally:
        try:
            os.unlink(metadata_file)
        except OSError:
            pass


def _supervise_pid(pid: int) -> int:
    """Supervise a detached process until it exits."""
    # Forward signals
    def forward_signal(signum: int, frame: object) -> None:
        try:
            os.kill(pid, signum)
        except OSError:
            pass

    signal.signal(signal.SIGTERM, forward_signal)
    signal.signal(signal.SIGINT, forward_signal)

    # Poll /proc/<pid> for liveness
    proc_path = Path(f"/proc/{pid}")
    poll_interval = 0.5

    while True:
        if not proc_path.exists():
            # Process exited — try to get status
            try:
                _, status = os.waitpid(pid, os.WNOHANG)
                if os.WIFEXITED(status):
                    return os.WEXITSTATUS(status)
                return 1
            except ChildProcessError:
                return 0  # Already reaped

        time.sleep(poll_interval)


def _run_without_snapshots(argv: list[str]) -> int:
    """Run the command directly without snapshot support."""
    child = subprocess.Popen(argv)

    def forward_signal(signum: int, frame: object) -> None:
        try:
            os.kill(child.pid, signum)
        except OSError:
            pass

    signal.signal(signal.SIGTERM, forward_signal)
    signal.signal(signal.SIGINT, forward_signal)
    return child.wait()
