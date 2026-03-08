"""Tests for ready detection module."""

from __future__ import annotations

import http.server
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from zerostart.ready import ReadyEvent, make_run_dir, wait_for_ready


def test_make_run_dir_creates_unique_dirs():
    """Each call to make_run_dir should create a unique directory."""
    dir1 = make_run_dir()
    dir2 = make_run_dir()
    assert dir1 != dir2
    assert dir1.exists()
    assert dir2.exists()
    assert dir1.parent == dir2.parent  # Both under /tmp/zerostart/


def test_make_run_dir_structure():
    """Run dir should be under /tmp/zerostart/."""
    run_dir = make_run_dir()
    assert run_dir.parent == Path("/tmp/zerostart")
    assert len(run_dir.name) == 12  # hex UUID prefix


def test_wait_for_ready_signal_mode():
    """Signal mode should detect when the ready file is touched."""
    run_dir = make_run_dir()
    ready_path = run_dir / "ready"

    # Touch the file after a short delay
    def touch_later():
        time.sleep(0.1)
        ready_path.touch()

    t = threading.Thread(target=touch_later, daemon=True)
    t.start()

    event = wait_for_ready("signal", run_dir, timeout_s=2.0)
    assert event.mode == "signal"
    assert str(ready_path) in event.details


def test_wait_for_ready_signal_timeout():
    """Signal mode should raise TimeoutError when file never appears."""
    run_dir = make_run_dir()

    try:
        wait_for_ready("signal", run_dir, timeout_s=0.2)
        assert False, "Should have raised TimeoutError"
    except TimeoutError:
        pass


def test_wait_for_ready_file_mode():
    """File mode should detect when the specified file appears."""
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "marker.txt"

        def touch_later():
            time.sleep(0.1)
            target.touch()

        t = threading.Thread(target=touch_later, daemon=True)
        t.start()

        run_dir = make_run_dir()  # Not used for file mode but required arg
        event = wait_for_ready(f"file:{target}", run_dir, timeout_s=2.0)
        assert event.mode == "file"
        assert str(target) in event.details


def test_wait_for_ready_file_timeout():
    """File mode should timeout when file never appears."""
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "never.txt"
        run_dir = make_run_dir()

        try:
            wait_for_ready(f"file:{target}", run_dir, timeout_s=0.2)
            assert False, "Should have raised TimeoutError"
        except TimeoutError:
            pass


def test_wait_for_ready_url_mode():
    """URL mode should detect when endpoint returns 200."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, format, *args):
            pass  # Suppress logs

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        run_dir = make_run_dir()
        event = wait_for_ready(f"url:http://127.0.0.1:{port}/health", run_dir, timeout_s=2.0)
        assert event.mode == "url"
        assert "200" in event.details
    finally:
        server.shutdown()


def test_wait_for_ready_url_timeout():
    """URL mode should timeout when endpoint never responds 2xx."""
    run_dir = make_run_dir()

    try:
        # Use a port that nothing is listening on
        wait_for_ready("url:http://127.0.0.1:19999/health", run_dir, timeout_s=0.5)
        assert False, "Should have raised TimeoutError"
    except TimeoutError:
        pass


def test_wait_for_ready_invalid_mode():
    """Invalid mode should raise ValueError."""
    run_dir = make_run_dir()

    try:
        wait_for_ready("invalid", run_dir, timeout_s=1.0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown ready mode" in str(e)


if __name__ == "__main__":
    import traceback

    tests = [
        test_make_run_dir_creates_unique_dirs,
        test_make_run_dir_structure,
        test_wait_for_ready_signal_mode,
        test_wait_for_ready_signal_timeout,
        test_wait_for_ready_file_mode,
        test_wait_for_ready_file_timeout,
        test_wait_for_ready_url_mode,
        test_wait_for_ready_url_timeout,
        test_wait_for_ready_invalid_mode,
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
