"""Lazy import hook with in-memory demand signaling via DaemonHandle.

The app starts immediately. When it hits `import torch`, the hook:
1. Checks if torch is importable right now → yes? normal import.
2. No? Calls daemon.signal_demand("torch") so the daemon prioritizes it.
3. Calls daemon.wait_done("torch") — blocks until it's extracted.
4. Retries the import.

No file-based IPC — the DaemonHandle is an in-memory Rust object.

Usage:
    from zs_fast_wheel import DaemonHandle
    from zerostart.lazy_imports import install_hook, remove_hook

    daemon = DaemonHandle()
    daemon.start(wheels=[...], site_packages="...")

    hook = install_hook(daemon=daemon)

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
import sys
import time

log = logging.getLogger("zerostart")


class LazyImportHook(importlib.abc.MetaPathFinder):
    """Blocks imports until packages appear on disk. Signals demand to daemon.

    Never returns a loader — just gates the import, then lets normal
    machinery do the actual loading once files are on disk.
    """

    def __init__(
        self,
        daemon: object,
        import_map: dict[str, str] | None = None,
        timeout: float = 300.0,
        speculative_timeout: float = 5.0,
    ) -> None:
        """
        Args:
            daemon: DaemonHandle instance (from zs_fast_wheel PyO3 module).
            import_map: Maps import names → distribution names
                        (e.g. {"yaml": "PyYAML", "PIL": "Pillow"}).
                        If None, assumes import name == distribution name.
            timeout: Max seconds to wait for known packages.
            speculative_timeout: Max seconds for unknown packages (try/except imports).
        """
        self._daemon = daemon
        self._import_map = import_map or {}
        self.timeout = timeout
        self.speculative_timeout = speculative_timeout

        self._resolved: set[str] = set()
        self._wait_times: dict[str, float] = {}

    def _dist_for_import(self, top_level: str) -> str:
        """Map an import name to a distribution name."""
        return self._import_map.get(top_level, top_level)

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

    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: object = None,
    ) -> None:
        top = fullname.split(".")[0]

        # Fast path: already resolved
        if fullname in self._resolved or top + ".*" in self._resolved:
            return None

        # Already importable
        if self._can_import(fullname):
            self._resolved.add(fullname)
            return None

        # Check if daemon knows about this distribution
        dist = self._dist_for_import(top)
        try:
            done = self._daemon.is_done(dist)
        except Exception:
            # Daemon not started or errored — pass through
            self._resolved.add(fullname)
            return None

        if done:
            # Daemon says done but Python can't find it — genuinely missing
            self._resolved.add(top + ".*")
            return None

        # Signal demand and wait
        self._daemon.signal_demand(dist)

        # Use short timeout for speculative imports
        is_known = dist.lower().replace("-", "_") in {
            d.lower().replace("-", "_") for d in self._import_map.values()
        } or top.lower() in {k.lower() for k in self._import_map}

        max_wait = self.timeout if is_known or not self._import_map else self.speculative_timeout

        log.info("import %s — waiting for %s... (max %.0fs)", fullname, dist, max_wait)
        start = time.monotonic()

        try:
            self._daemon.wait_done(dist, timeout_secs=max_wait)
        except Exception as e:
            elapsed = time.monotonic() - start
            log.warning("import %s — wait failed after %.1fs: %s", fullname, elapsed, e)
            self._resolved.add(fullname)
            return None

        # Invalidate caches so Python sees the new files
        importlib.invalidate_caches()

        elapsed = time.monotonic() - start
        if elapsed > 0.01:
            if top not in self._wait_times or elapsed > self._wait_times[top]:
                self._wait_times[top] = elapsed
            log.info("import %s — ready (%.1fs)", fullname, elapsed)

        self._resolved.add(fullname)
        return None

    def report(self) -> dict[str, float]:
        """Return {package: seconds_waited} for imports that blocked."""
        return dict(self._wait_times)


# ---------------------------------------------------------------------------
# Module API
# ---------------------------------------------------------------------------

_active_hook: LazyImportHook | None = None


def install_hook(
    daemon: object,
    import_map: dict[str, str] | None = None,
    timeout: float = 300.0,
    speculative_timeout: float = 5.0,
) -> LazyImportHook:
    """Install the lazy import hook.

    Args:
        daemon: DaemonHandle instance.
        import_map: Import name → distribution name mapping.
        timeout: Max seconds to wait for known packages.
        speculative_timeout: Max seconds for speculative imports.
    """
    global _active_hook  # noqa: PLW0603
    if _active_hook is not None:
        remove_hook()

    hook = LazyImportHook(
        daemon=daemon,
        import_map=import_map,
        timeout=timeout,
        speculative_timeout=speculative_timeout,
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
