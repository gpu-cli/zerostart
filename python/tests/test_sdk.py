"""Tests for zerostart SDK: ready() and on_restore."""

from __future__ import annotations

import os
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path

# Add the package to sys.path for testing
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_ready_noop_when_unmanaged():
    """ready() should be a no-op when ZEROSTART_MANAGED is not set."""
    # Ensure env vars are clean
    env_backup = {}
    for key in ("ZEROSTART_MANAGED", "ZEROSTART_READY_PATH"):
        env_backup[key] = os.environ.pop(key, None)

    try:
        # Must reimport to get clean module state
        import zerostart.sdk as sdk

        # Should not raise, should not create any files
        sdk.ready()
    finally:
        for key, val in env_backup.items():
            if val is not None:
                os.environ[key] = val


def test_ready_touches_file_when_managed():
    """ready() should touch ZEROSTART_READY_PATH when managed."""
    with tempfile.TemporaryDirectory() as tmp:
        ready_path = Path(tmp) / "subdir" / "ready"
        os.environ["ZEROSTART_MANAGED"] = "1"
        os.environ["ZEROSTART_READY_PATH"] = str(ready_path)

        try:
            import zerostart.sdk as sdk

            assert not ready_path.exists()
            sdk.ready()
            assert ready_path.exists()
        finally:
            os.environ.pop("ZEROSTART_MANAGED", None)
            os.environ.pop("ZEROSTART_READY_PATH", None)


def test_ready_creates_parent_dirs():
    """ready() should create parent directories if they don't exist."""
    with tempfile.TemporaryDirectory() as tmp:
        ready_path = Path(tmp) / "a" / "b" / "c" / "ready"
        os.environ["ZEROSTART_MANAGED"] = "1"
        os.environ["ZEROSTART_READY_PATH"] = str(ready_path)

        try:
            import zerostart.sdk as sdk

            sdk.ready()
            assert ready_path.exists()
            assert ready_path.parent.is_dir()
        finally:
            os.environ.pop("ZEROSTART_MANAGED", None)
            os.environ.pop("ZEROSTART_READY_PATH", None)


def test_ready_noop_when_managed_but_no_path():
    """ready() should be a no-op if ZEROSTART_MANAGED is set but READY_PATH is not."""
    os.environ["ZEROSTART_MANAGED"] = "1"
    os.environ.pop("ZEROSTART_READY_PATH", None)

    try:
        import zerostart.sdk as sdk

        # Should not raise
        sdk.ready()
    finally:
        os.environ.pop("ZEROSTART_MANAGED", None)


def test_on_restore_registers_callback():
    """on_restore should register the callback in the callbacks list."""
    import zerostart.sdk as sdk

    initial_count = len(sdk._on_restore_callbacks)

    @sdk.on_restore
    def my_callback() -> None:
        pass

    assert len(sdk._on_restore_callbacks) == initial_count + 1
    assert sdk._on_restore_callbacks[-1] is my_callback

    # Clean up
    sdk._on_restore_callbacks.remove(my_callback)


def test_on_restore_returns_original_function():
    """on_restore decorator should return the original function unchanged."""
    import zerostart.sdk as sdk

    def my_func() -> None:
        pass

    result = sdk.on_restore(my_func)
    assert result is my_func

    # Clean up
    sdk._on_restore_callbacks.remove(my_func)


def test_on_restore_callback_dispatch_via_event():
    """Setting the restore event should trigger registered callbacks."""
    import zerostart.sdk as sdk

    # Ensure restore thread is running
    sdk._ensure_restore_thread()

    called = threading.Event()

    def callback() -> None:
        called.set()

    sdk._on_restore_callbacks.append(callback)

    try:
        # Simulate what SIGUSR1 handler does
        sdk._restore_event.set()

        # Wait for callback to fire
        assert called.wait(timeout=2.0), "on_restore callback was not called within timeout"
    finally:
        sdk._on_restore_callbacks.remove(callback)


def test_on_restore_exception_does_not_stop_later_callbacks():
    """A failing callback should not prevent later callbacks from running."""
    import zerostart.sdk as sdk

    sdk._ensure_restore_thread()

    second_called = threading.Event()

    def failing_callback() -> None:
        raise RuntimeError("intentional test failure")

    def second_callback() -> None:
        second_called.set()

    sdk._on_restore_callbacks.append(failing_callback)
    sdk._on_restore_callbacks.append(second_callback)

    try:
        sdk._restore_event.set()
        assert second_called.wait(timeout=2.0), "Second callback was not called after first failed"
    finally:
        sdk._on_restore_callbacks.remove(failing_callback)
        sdk._on_restore_callbacks.remove(second_callback)


def test_on_restore_sigusr1_triggers_callbacks():
    """SIGUSR1 should trigger on_restore callbacks (Unix only)."""
    if sys.platform == "win32":
        return  # SIGUSR1 not available on Windows

    import zerostart.sdk as sdk

    # Directly register signal handler (since _ensure_restore_thread may have
    # already been called without ZEROSTART_MANAGED in earlier tests)
    signal.signal(signal.SIGUSR1, sdk._restore_signal_handler)
    sdk._ensure_restore_thread()

    called = threading.Event()

    def callback() -> None:
        called.set()

    sdk._on_restore_callbacks.append(callback)

    try:
        # Send SIGUSR1 to ourselves
        os.kill(os.getpid(), signal.SIGUSR1)

        assert called.wait(timeout=2.0), "on_restore callback not triggered by SIGUSR1"
    finally:
        sdk._on_restore_callbacks.remove(callback)


def test_exports():
    """ready and on_restore should be importable from zerostart package."""
    import zerostart

    assert hasattr(zerostart, "ready")
    assert hasattr(zerostart, "on_restore")
    assert callable(zerostart.ready)
    assert callable(zerostart.on_restore)


if __name__ == "__main__":
    # Simple test runner
    import traceback

    tests = [
        test_ready_noop_when_unmanaged,
        test_ready_touches_file_when_managed,
        test_ready_creates_parent_dirs,
        test_ready_noop_when_managed_but_no_path,
        test_on_restore_registers_callback,
        test_on_restore_returns_original_function,
        test_on_restore_callback_dispatch_via_event,
        test_on_restore_exception_does_not_stop_later_callbacks,
        test_on_restore_sigusr1_triggers_callbacks,
        test_exports,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL: {test.__name__}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
