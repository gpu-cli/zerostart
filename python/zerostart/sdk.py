"""Minimal Tier 1 SDK hooks for zerostart process snapshots."""

import logging
import os
import signal
import threading
from pathlib import Path
from typing import Callable

_log = logging.getLogger("zerostart")

_on_restore_callbacks: list[Callable[[], None]] = []
_restore_event = threading.Event()
_restore_thread_started = False
_restore_lock = threading.Lock()


def ready() -> None:
    """Signal that the application is ready for snapshotting.

    No-op when not running under the zerostart supervisor.
    When managed, touches the ready file so the supervisor knows
    the app is initialized and can proceed with CRIU dump.
    """
    if not os.environ.get("ZEROSTART_MANAGED"):
        return
    ready_path_str = os.environ.get("ZEROSTART_READY_PATH")
    if not ready_path_str:
        return
    ready_path = Path(ready_path_str)
    ready_path.parent.mkdir(parents=True, exist_ok=True)
    ready_path.touch()


def on_restore(fn: Callable[[], None]) -> Callable[[], None]:
    """Register a callback to run after process restore.

    Callbacks are dispatched on a background thread when the supervisor
    sends SIGUSR1 after CRIU restore. Callbacks must be idempotent —
    they may run more than once and should tolerate partial restore state.
    Failures are logged without stopping later callbacks.
    """
    _ensure_restore_thread()
    _on_restore_callbacks.append(fn)
    return fn


def _restore_signal_handler(signum: int, frame: object) -> None:
    """Minimal signal handler — only sets the event."""
    _restore_event.set()


def _restore_loop() -> None:
    """Background thread that waits for restore events and runs callbacks."""
    while True:
        _restore_event.wait()
        _restore_event.clear()
        for fn in list(_on_restore_callbacks):
            try:
                fn()
            except Exception:
                _log.exception("on_restore callback failed: %s", fn)


def _ensure_restore_thread() -> None:
    """Lazily start the restore background thread and register SIGUSR1."""
    global _restore_thread_started
    with _restore_lock:
        if _restore_thread_started:
            return
        _restore_thread_started = True

    # Only register signal handler when running under the supervisor
    if os.environ.get("ZEROSTART_MANAGED"):
        signal.signal(signal.SIGUSR1, _restore_signal_handler)

    thread = threading.Thread(target=_restore_loop, daemon=True, name="zerostart-restore")
    thread.start()
