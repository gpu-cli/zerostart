#!/usr/bin/env python3
"""Test 2.4: Restore callback delivery via SIGUSR1.

Starts a Python process with on_restore registered, dumps it,
kills it, restores, sends SIGUSR1, and verifies callback ran.

Must run on Linux with CRIU installed (via gpu run).
"""

import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def skip(reason: str) -> None:
    print(f"SKIP: {reason}")
    sys.exit(0)


def fail(reason: str) -> None:
    print(f"FAIL: {reason}")
    sys.exit(1)


def find_zs_snapshot() -> str:
    """Find the zs-snapshot binary."""
    root = Path(__file__).parent.parent.parent
    candidates = [
        root / "crates" / "target" / "release" / "zs-snapshot",
        root / "crates" / "target" / "debug" / "zs-snapshot",
        root / "bin" / "zs-snapshot-linux-x86_64",
    ]
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return str(c)
    return shutil.which("zs-snapshot") or ""


# Script that registers on_restore and writes a marker file
CHILD_SCRIPT_TEMPLATE = '''
import os
import signal
import sys
import threading
import time

sys.path.insert(0, "{python_path}")

from zerostart.sdk import on_restore, ready, _ensure_restore_thread

MARKER = "{marker_path}"

# Register signal handler and start restore thread
os.environ["ZEROSTART_MANAGED"] = "1"
os.environ["ZEROSTART_READY_PATH"] = "{ready_path}"
signal.signal(signal.SIGUSR1, lambda s, f: __import__("zerostart.sdk", fromlist=["_restore_event"])._restore_event.set())
_ensure_restore_thread()

@on_restore
def write_marker():
    with open(MARKER, "w") as f:
        f.write("restored")
    print("on_restore callback executed!", flush=True)

# Signal ready
ready()

# Stay alive
while True:
    time.sleep(1)
'''


def main() -> None:
    print("=== Test: Restore callback delivery ===")

    if platform.system() != "Linux":
        skip("Not Linux")

    if not shutil.which("criu"):
        skip("criu not installed")

    zs_snapshot = find_zs_snapshot()
    if not zs_snapshot:
        skip("zs-snapshot binary not found")

    print(f"Using zs-snapshot: {zs_snapshot}")

    cache_dir = tempfile.mkdtemp()
    run_dir = tempfile.mkdtemp()
    intent_hash = f"test-callback-{int(time.time())}"
    marker_path = os.path.join(run_dir, "callback_marker")
    ready_path = os.path.join(run_dir, "ready")
    python_path = str(Path(__file__).parent.parent.parent / "python")

    script_content = CHILD_SCRIPT_TEMPLATE.format(
        python_path=python_path,
        marker_path=marker_path,
        ready_path=ready_path,
    )

    try:
        # Write child script
        script_file = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        script_file.write(script_content)
        script_file.close()

        # Start child
        child = subprocess.Popen(
            [sys.executable, script_file.name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(f"Started child PID: {child.pid}")

        # Wait for ready signal
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if Path(ready_path).exists():
                break
            time.sleep(0.1)
        else:
            fail("Child never signaled ready")

        print("Child is ready")
        time.sleep(0.25)  # Settle

        # Write metadata
        metadata_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump({
            "intent_hash": intent_hash,
            "env_fingerprint": "test",
            "entrypoint": script_file.name,
            "argv": [sys.executable, script_file.name],
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": f"{platform.machine()}-linux",
        }, metadata_file)
        metadata_file.close()

        # Dump
        print("--- Dump ---")
        result = subprocess.run(
            [zs_snapshot, "dump",
             "--intent-hash", intent_hash,
             "--pid", str(child.pid),
             "--metadata", metadata_file.name,
             "--cache-dir", cache_dir],
            capture_output=True, text=True, timeout=60,
        )

        if result.returncode != 0:
            print(f"Dump stderr: {result.stderr}")
            fail(f"Dump failed: {result.returncode}")

        print("Dump succeeded")

        # Remove marker if it exists (shouldn't, but clean slate)
        if Path(marker_path).exists():
            os.unlink(marker_path)

        # Kill original
        child.terminate()
        child.wait(timeout=5)
        print("Original killed")

        # Restore
        pidfile = os.path.join(run_dir, "restored.pid")
        print("--- Restore ---")
        result = subprocess.run(
            [zs_snapshot, "restore",
             "--intent-hash", intent_hash,
             "--pidfile", pidfile,
             "--cache-dir", cache_dir],
            capture_output=True, text=True, timeout=60,
        )

        if result.returncode != 0:
            print(f"Restore stderr: {result.stderr}")
            fail(f"Restore failed: {result.returncode}")

        restored_pid = int(Path(pidfile).read_text().strip())
        print(f"Restored PID: {restored_pid}")

        # Send SIGUSR1 to trigger on_restore callbacks
        time.sleep(0.5)  # Let process fully restore
        os.kill(restored_pid, signal.SIGUSR1)
        print("Sent SIGUSR1")

        # Wait for marker file
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if Path(marker_path).exists():
                break
            time.sleep(0.1)
        else:
            fail("on_restore callback did not write marker file")

        content = Path(marker_path).read_text()
        if content != "restored":
            fail(f"Marker content wrong: {content!r}")

        print("on_restore callback verified!")

        # Cleanup
        os.kill(restored_pid, signal.SIGTERM)

    finally:
        os.unlink(script_file.name)
        os.unlink(metadata_file.name)
        import shutil as sh
        sh.rmtree(cache_dir, ignore_errors=True)
        sh.rmtree(run_dir, ignore_errors=True)

    print("=== PASS: Restore callback delivery ===")


if __name__ == "__main__":
    main()
