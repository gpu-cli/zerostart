"""Ready detection for Tier 1 process snapshots.

Supports three explicit readiness modes:
- signal: app calls zerostart.ready() which touches a file
- file:<path>: wait for an arbitrary file to appear
- url:<url>: poll an HTTP endpoint until 2xx
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("zerostart")

RUN_BASE = Path("/tmp/zerostart")


@dataclass
class ReadyEvent:
    """Emitted when the managed process signals readiness."""

    mode: str
    details: str
    timestamp: float


def make_run_dir() -> Path:
    """Create a unique per-run directory under /tmp/zerostart/<run_id>/.

    Returns the run directory path. The directory contains:
    - ready: touched by the app via zerostart.ready()
    - restored.pid: written by CRIU restore
    """
    run_id = uuid.uuid4().hex[:12]
    run_dir = RUN_BASE / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def wait_for_ready(
    mode: str,
    run_dir: Path,
    timeout_s: float,
) -> ReadyEvent:
    """Wait for the managed process to become ready.

    Args:
        mode: One of "signal", "file:<path>", or "url:<url>".
        run_dir: Per-run directory created by make_run_dir().
        timeout_s: Maximum seconds to wait before raising TimeoutError.

    Returns:
        ReadyEvent with details about how readiness was detected.

    Raises:
        TimeoutError: If readiness is not detected within timeout_s.
        ValueError: If mode is not recognized.
    """
    if mode == "signal":
        return _wait_signal(run_dir / "ready", timeout_s)
    elif mode.startswith("file:"):
        file_path = Path(mode[5:])
        return _wait_file(file_path, timeout_s)
    elif mode.startswith("url:"):
        url = mode[4:]
        return _wait_url(url, timeout_s)
    else:
        raise ValueError(f"Unknown ready mode: {mode!r}. Expected 'signal', 'file:<path>', or 'url:<url>'.")


def _wait_signal(ready_path: Path, timeout_s: float) -> ReadyEvent:
    """Wait for the ready file to appear (touched by zerostart.ready())."""
    log.info("Waiting for ready signal at %s (timeout %.0fs)", ready_path, timeout_s)
    deadline = time.monotonic() + timeout_s
    poll_interval = 0.05  # 50ms

    while time.monotonic() < deadline:
        if ready_path.exists():
            log.info("Ready signal received: %s", ready_path)
            return ReadyEvent(
                mode="signal",
                details=str(ready_path),
                timestamp=time.time(),
            )
        time.sleep(poll_interval)

    raise TimeoutError(f"Ready signal not received at {ready_path} within {timeout_s}s")


def _wait_file(file_path: Path, timeout_s: float) -> ReadyEvent:
    """Wait for an explicit file to appear."""
    log.info("Waiting for file %s (timeout %.0fs)", file_path, timeout_s)
    deadline = time.monotonic() + timeout_s
    poll_interval = 0.1

    while time.monotonic() < deadline:
        if file_path.exists():
            log.info("Ready file detected: %s", file_path)
            return ReadyEvent(
                mode="file",
                details=str(file_path),
                timestamp=time.time(),
            )
        time.sleep(poll_interval)

    raise TimeoutError(f"File {file_path} not created within {timeout_s}s")


def _wait_url(url: str, timeout_s: float) -> ReadyEvent:
    """Poll a URL until it returns a 2xx response."""
    import urllib.request
    import urllib.error

    log.info("Polling %s for readiness (timeout %.0fs)", url, timeout_s)
    deadline = time.monotonic() + timeout_s
    poll_interval = 0.25
    request_timeout = 2.0

    last_error = ""
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=request_timeout) as resp:
                if 200 <= resp.status < 300:
                    log.info("Ready URL responded %d: %s", resp.status, url)
                    return ReadyEvent(
                        mode="url",
                        details=f"{url} -> {resp.status}",
                        timestamp=time.time(),
                    )
                last_error = f"HTTP {resp.status}"
        except (urllib.error.URLError, OSError) as exc:
            last_error = str(exc)
        time.sleep(poll_interval)

    raise TimeoutError(f"URL {url} did not return 2xx within {timeout_s}s (last: {last_error})")
