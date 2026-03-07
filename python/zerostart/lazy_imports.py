"""Lazy import hook with demand signaling and progressive loading.

The app starts immediately. When it hits `import torch`, the hook:
1. Checks if torch is importable right now → yes? normal import.
2. No? Writes "torch" to a demand file so the installer can prioritize it.
3. Blocks, polling until torch appears on sys.path.
4. As torch's __init__.py imports its own submodules (torch._C, torch.nn),
   each resolves progressively as files land on disk.

Bidirectional protocol with the background installer:

    Hook → Installer (demand signal):
        Appends import names to <status_dir>/demand (one per line).
        Installer watches this file and bumps requested packages to
        the front of the install queue.

    Installer → Hook (progress):
        <status_dir>/installing     — exists while installer is running
        <status_dir>/__done__       — created when installer finishes

    No per-package markers needed. The hook just retries find_spec()
    against the real filesystem. When a wheel is extracted, its modules
    become findable automatically.

Usage:
    from zerostart.lazy_imports import install_hook, remove_hook

    hook = install_hook(status_dir="/tmp/zs-status")

    import torch        # blocks ~3s, signals demand for "torch"
    import torch.nn     # instant — torch is already on disk
    import requests     # instant — small, installed before torch

    report = remove_hook()
    # {"torch": 3.2}
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import logging
import os
import sys
import threading
import time
from pathlib import Path

log = logging.getLogger("zerostart")


class LazyImportHook(importlib.abc.MetaPathFinder):
    """Blocks imports until packages appear on disk. Signals demand to installer.

    Never returns a loader — just gates the import, then lets normal
    machinery do the actual loading once files are on disk.
    """

    def __init__(
        self,
        status_dir: str | Path,
        timeout: float = 300.0,
        speculative_timeout: float = 5.0,
        poll_interval: float = 0.05,
        expected_packages: set[str] | None = None,
    ) -> None:
        self.status_dir = Path(status_dir)
        self.timeout = timeout
        self.speculative_timeout = speculative_timeout
        self.poll_interval = poll_interval
        # If provided, only wait full timeout for these top-level names.
        # Others get the shorter speculative_timeout (for try/except imports).
        self._expected: set[str] | None = (
            {p.lower().replace("-", "_") for p in expected_packages}
            if expected_packages else None
        )

        # Track resolved modules — only cache at fullname level while
        # installer is running (submodules may arrive later). Once installer
        # is done, we cache at top-level to avoid repeated checks.
        self._resolved: set[str] = set()
        self._waiting: set[str] = set()  # currently blocked, for logging
        self._wait_times: dict[str, float] = {}
        self._demand_lock = threading.Lock()
        self._demand_path = self.status_dir / "demand"

    # ------------------------------------------------------------------
    # Demand signaling — tell the installer what we need NOW
    # ------------------------------------------------------------------

    def _signal_demand(self, module_name: str) -> None:
        """Append a module name to the demand file.

        The installer watches this file and reprioritizes accordingly.
        Thread-safe — multiple imports can signal concurrently.
        """
        with self._demand_lock:
            try:
                self.status_dir.mkdir(parents=True, exist_ok=True)
                with open(self._demand_path, "a") as f:
                    f.write(module_name + "\n")
                    f.flush()
                    os.fsync(f.fileno())
            except OSError:
                pass  # best effort — installer may not be watching

    # ------------------------------------------------------------------
    # Installer state
    # ------------------------------------------------------------------

    def _installer_running(self) -> bool:
        return (self.status_dir / "installing").exists()

    # ------------------------------------------------------------------
    # Import probing — can Python find this module right now?
    # ------------------------------------------------------------------

    def _can_import(self, fullname: str) -> bool:
        """Ask Python's import machinery if this module exists, bypassing ourselves."""
        try:
            idx = sys.meta_path.index(self)
        except ValueError:
            return False
        sys.meta_path.pop(idx)
        try:
            spec = importlib.util.find_spec(fullname)
            return spec is not None
        except (ModuleNotFoundError, ValueError, ImportError):
            return False
        finally:
            sys.meta_path.insert(idx, self)

    # ------------------------------------------------------------------
    # The main hook — called on every import statement
    # ------------------------------------------------------------------

    def _mark_resolved(self, fullname: str) -> None:
        """Mark a module as resolved.

        While installer is running, only cache exact fullname (submodules
        may still be landing). Once installer is done, mark the top-level
        package as fully resolved to skip all future checks.
        """
        self._resolved.add(fullname)
        if not self._installer_running():
            self._resolved.add(fullname.split(".")[0] + ".*")

    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: object = None,
    ) -> None:
        # Fast path: already resolved this exact module, or entire package
        # is resolved (installer done, marked with "pkg.*" sentinel)
        top = fullname.split(".")[0]
        if fullname in self._resolved or top + ".*" in self._resolved:
            return None

        # Already importable? Done.
        if self._can_import(fullname):
            self._mark_resolved(fullname)
            return None

        # Not importable. Is the installer even running?
        if not self._installer_running():
            self._mark_resolved(fullname)
            return None  # genuine ImportError

        # --- Installer is running, module not yet available ---
        # Signal demand so installer can prioritize this package
        self._signal_demand(top)

        # Use short timeout for speculative imports (try/except in libraries)
        # unless this is a package we know is being installed.
        is_expected = (
            self._expected is None  # no list = wait for everything
            or top.lower().replace("-", "_") in self._expected
        )
        max_wait = self.timeout if is_expected else self.speculative_timeout

        log.info("import %s — waiting for install... (max %.0fs)",
                 fullname, max_wait)
        self._waiting.add(fullname)
        start = time.monotonic()

        while True:
            importlib.invalidate_caches()

            if self._can_import(fullname):
                wait = time.monotonic() - start
                # Record wait against top-level package, keep longest
                if top not in self._wait_times or wait > self._wait_times[top]:
                    self._wait_times[top] = wait
                self._waiting.discard(fullname)
                self._mark_resolved(fullname)
                log.info("import %s — ready (%.1fs)", fullname, wait)
                return None

            if not self._installer_running():
                wait = time.monotonic() - start
                self._waiting.discard(fullname)
                self._mark_resolved(fullname)
                log.warning(
                    "import %s — installer finished, module not found (%.1fs)",
                    fullname,
                    wait,
                )
                return None

            elapsed = time.monotonic() - start
            if elapsed > max_wait:
                self._waiting.discard(fullname)
                self._mark_resolved(fullname)
                if is_expected:
                    log.error("import %s — timed out (%.0fs)", fullname, elapsed)
                else:
                    log.debug("import %s — speculative timeout (%.1fs)", fullname, elapsed)
                return None

            time.sleep(self.poll_interval)

    def report(self) -> dict[str, float]:
        """Return {package: seconds_waited} for imports that blocked."""
        return dict(self._wait_times)


# ---------------------------------------------------------------------------
# Module API
# ---------------------------------------------------------------------------

_active_hook: LazyImportHook | None = None


def install_hook(
    status_dir: str | Path,
    timeout: float = 300.0,
    speculative_timeout: float = 5.0,
    poll_interval: float = 0.05,
    expected_packages: set[str] | None = None,
) -> LazyImportHook:
    """Install the lazy import hook into sys.meta_path.

    Args:
        status_dir: Directory with 'installing' sentinel file.
        timeout: Max seconds to wait for expected packages.
        speculative_timeout: Max seconds to wait for unknown packages
            (e.g. try/except imports in libraries). Default 5s.
        poll_interval: Seconds between import retry checks.
        expected_packages: Set of package names we know are being installed.
            If None, wait full timeout for everything.
    """
    global _active_hook  # noqa: PLW0603
    if _active_hook is not None:
        remove_hook()

    hook = LazyImportHook(
        status_dir=status_dir,
        timeout=timeout,
        speculative_timeout=speculative_timeout,
        poll_interval=poll_interval,
        expected_packages=expected_packages,
    )
    sys.meta_path.insert(0, hook)
    _active_hook = hook
    return hook


def remove_hook() -> dict[str, float] | None:
    """Remove the lazy import hook. Returns {package: wait_seconds}."""
    global _active_hook  # noqa: PLW0603
    if _active_hook is None:
        return None

    report = _active_hook.report()
    try:
        sys.meta_path.remove(_active_hook)
    except ValueError:
        pass
    _active_hook = None

    if report:
        total = sum(report.values())
        log.info("zerostart: total import wait %.1fs", total)
        for pkg, wait in sorted(report.items(), key=lambda x: -x[1]):
            log.info("  %s: %.2fs", pkg, wait)

    return report
