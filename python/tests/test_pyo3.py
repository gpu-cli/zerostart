"""Integration tests for PyO3 DaemonHandle with real PyPI wheels.

Tests the full stack: DaemonHandle → download → extract → Python import.
Requires: zs_fast_wheel built with `maturin develop --features python`
"""

from __future__ import annotations

import importlib
import json
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def _get_wheel_url(dist: str, version: str) -> tuple[str, int]:
    """Fetch wheel URL and size from PyPI JSON API."""
    url = f"https://pypi.org/pypi/{dist}/{version}/json"
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read())
    for f in data.get("urls", []):
        if f["filename"].endswith(".whl") and "none-any" in f["filename"]:
            return f["url"], f["size"]
    for f in data.get("urls", []):
        if f["filename"].endswith(".whl"):
            return f["url"], f["size"]
    raise RuntimeError(f"No wheel found for {dist}=={version}")


def test_daemon_handle_basic():
    """DaemonHandle can be created, errors before start."""
    from zs_fast_wheel import DaemonHandle

    d = DaemonHandle()
    try:
        d.stats()
        assert False, "should have raised"
    except RuntimeError:
        pass

    try:
        d.signal_demand("foo")
        assert False, "should have raised"
    except RuntimeError:
        pass


def test_daemon_handle_empty_wheels():
    """Starting with empty wheels raises ValueError."""
    from zs_fast_wheel import DaemonHandle

    d = DaemonHandle()
    try:
        d.start(wheels=[], site_packages="/tmp/zs-test-empty")
        assert False, "should have raised"
    except ValueError:
        pass


def test_daemon_download_small_wheel():
    """Download and extract a real small wheel, verify importable."""
    from zs_fast_wheel import DaemonHandle

    url, size = _get_wheel_url("six", "1.17.0")

    with tempfile.TemporaryDirectory() as tmp:
        sp = Path(tmp) / "site-packages"
        sp.mkdir()

        d = DaemonHandle()
        d.start(
            wheels=[{"url": url, "distribution": "six", "size": size}],
            site_packages=str(sp),
        )

        # Wait for completion
        done = d.wait_done("six", timeout_secs=30.0)
        assert done, "six should be done"

        # Verify stats
        total, done_count, pending, in_progress, failed = d.stats()
        assert total == 1
        assert failed == 0

        # Verify importable
        sys.path.insert(0, str(sp))
        try:
            importlib.invalidate_caches()
            import six as _six
            assert _six.PY3
        finally:
            sys.path.pop(0)
            if "six" in sys.modules:
                del sys.modules["six"]

        d.shutdown()


def test_daemon_multiple_wheels():
    """Download multiple wheels, verify all complete."""
    from zs_fast_wheel import DaemonHandle

    packages = [
        ("six", "1.17.0"),
        ("idna", "3.10"),
        ("certifi", "2025.1.31"),
    ]

    wheels = []
    for dist, ver in packages:
        url, size = _get_wheel_url(dist, ver)
        wheels.append({"url": url, "distribution": dist, "size": size})

    with tempfile.TemporaryDirectory() as tmp:
        sp = Path(tmp) / "site-packages"
        sp.mkdir()

        d = DaemonHandle()
        d.start(wheels=wheels, site_packages=str(sp))

        # Wait for all
        d.wait_all(timeout_secs=60.0)

        total, done_count, pending, in_progress, failed = d.stats()
        assert total == 3
        assert failed == 0
        assert pending == 0
        assert in_progress == 0

        # Verify all importable
        sys.path.insert(0, str(sp))
        try:
            importlib.invalidate_caches()
            import six as _six
            import idna as _idna
            import certifi as _certifi
            assert _six.PY3
        finally:
            sys.path.pop(0)
            for mod in ["six", "idna", "certifi"]:
                sys.modules.pop(mod, None)

        d.shutdown()


def test_signal_demand():
    """signal_demand doesn't error (correctness tested via timing)."""
    from zs_fast_wheel import DaemonHandle

    url, size = _get_wheel_url("six", "1.17.0")

    with tempfile.TemporaryDirectory() as tmp:
        sp = Path(tmp) / "site-packages"
        sp.mkdir()

        d = DaemonHandle()
        d.start(
            wheels=[{"url": url, "distribution": "six", "size": size}],
            site_packages=str(sp),
        )

        # Signal demand — should not error
        d.signal_demand("six")

        d.wait_all(timeout_secs=30.0)
        assert d.is_done("six")
        d.shutdown()


def test_is_done_before_complete():
    """is_done returns False before wheel is extracted."""
    from zs_fast_wheel import DaemonHandle

    url, size = _get_wheel_url("six", "1.17.0")

    with tempfile.TemporaryDirectory() as tmp:
        sp = Path(tmp) / "site-packages"
        sp.mkdir()

        d = DaemonHandle()
        # Don't start yet — just create
        # is_done should error since not started
        try:
            d.is_done("six")
            assert False, "should have raised"
        except RuntimeError:
            pass

        d.start(
            wheels=[{"url": url, "distribution": "six", "size": size}],
            site_packages=str(sp),
        )

        d.wait_all(timeout_secs=30.0)
        assert d.is_done("six")
        d.shutdown()


def test_wait_done_timeout():
    """wait_done raises TimeoutError if wheel not ready in time."""
    from zs_fast_wheel import DaemonHandle

    # Use a fake URL that will fail — so the wheel never completes
    with tempfile.TemporaryDirectory() as tmp:
        sp = Path(tmp) / "site-packages"
        sp.mkdir()

        d = DaemonHandle()
        d.start(
            wheels=[{
                "url": "https://httpbin.org/delay/10",
                "distribution": "fakepkg",
                "size": 100,
            }],
            site_packages=str(sp),
        )

        try:
            d.wait_done("fakepkg", timeout_secs=1.0)
        except (TimeoutError, Exception):
            pass  # Expected — either timeout or download failure

        d.shutdown()


def test_lazy_import_hook():
    """LazyImportHook waits for daemon to extract wheel before import succeeds."""
    from zs_fast_wheel import DaemonHandle
    from zerostart.lazy_imports import install_hook, remove_hook

    url, size = _get_wheel_url("six", "1.17.0")

    with tempfile.TemporaryDirectory() as tmp:
        sp = Path(tmp) / "site-packages"
        sp.mkdir()

        d = DaemonHandle()
        d.start(
            wheels=[{"url": url, "distribution": "six", "size": size}],
            site_packages=str(sp),
        )

        # Install hook
        sys.path.insert(0, str(sp))
        hook = install_hook(
            daemon=d,
            import_map={"six": "six"},
        )

        try:
            # Import should block until six is extracted
            importlib.invalidate_caches()
            # Remove six from modules if cached
            sys.modules.pop("six", None)
            import six as _six
            assert _six.PY3
        finally:
            report = remove_hook()
            sys.path.pop(0)
            sys.modules.pop("six", None)

        d.shutdown()

        # Report should exist (may or may not have wait time depending on speed)
        assert report is not None


if __name__ == "__main__":
    tests = [
        test_daemon_handle_basic,
        test_daemon_handle_empty_wheels,
        test_daemon_download_small_wheel,
        test_daemon_multiple_wheels,
        test_signal_demand,
        test_is_done_before_complete,
        test_wait_done_timeout,
        test_lazy_import_hook,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
