#!/usr/bin/env python3
"""Test 2.2: Python HTTP service dump/restore via CRIU.

Starts a tiny HTTP server, dumps it with --leave-running,
kills the original, restores, and verifies /health works.

Must run on Linux with CRIU installed (via gpu run).
"""

import http.client
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

# Add zerostart to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))


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

    # Try building
    result = subprocess.run(
        ["cargo", "build", "-p", "zs-snapshot", "--release"],
        cwd=str(root / "crates"),
        capture_output=True,
    )
    if result.returncode == 0:
        return str(candidates[0])

    return shutil.which("zs-snapshot") or ""


def check_health(port: int, timeout: float = 2.0) -> bool:
    """Check if HTTP server responds 200 on /health."""
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        conn.request("GET", "/health")
        resp = conn.getresponse()
        conn.close()
        return resp.status == 200
    except (ConnectionRefusedError, OSError, http.client.HTTPException):
        return False


def wait_for_health(port: int, timeout: float = 10.0) -> bool:
    """Poll /health until 200 or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check_health(port):
            return True
        time.sleep(0.1)
    return False


# Simple HTTP server script to run as child process
SERVER_SCRIPT = '''
import http.server
import sys

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()
    def log_message(self, format, *args):
        pass

port = int(sys.argv[1])
server = http.server.HTTPServer(("127.0.0.1", port), Handler)
print(f"Serving on port {port}", flush=True)
server.serve_forever()
'''


def main() -> None:
    print("=== Test: Python HTTP service dump/restore ===")

    if platform.system() != "Linux":
        skip("Not Linux")

    if not shutil.which("criu"):
        skip("criu not installed")

    zs_snapshot = find_zs_snapshot()
    if not zs_snapshot:
        skip("zs-snapshot binary not found")

    print(f"Using zs-snapshot: {zs_snapshot}")

    # Use a random high port
    import random
    port = random.randint(18000, 19000)

    cache_dir = tempfile.mkdtemp()
    run_dir = tempfile.mkdtemp()
    intent_hash = f"test-http-{int(time.time())}"

    try:
        # Start HTTP server
        server_proc = subprocess.Popen(
            [sys.executable, "-c", SERVER_SCRIPT, str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(f"Started server PID: {server_proc.pid}")

        # Wait for health
        if not wait_for_health(port):
            fail("Server did not become healthy")

        print("Server is healthy")

        # Write metadata
        metadata_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump({
            "intent_hash": intent_hash,
            "env_fingerprint": "test",
            "entrypoint": "server.py",
            "argv": [sys.executable, "-c", SERVER_SCRIPT, str(port)],
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": f"{platform.machine()}-linux",
        }, metadata_file)
        metadata_file.close()

        # Dump with --leave-running
        print("--- Dump ---")
        result = subprocess.run(
            [zs_snapshot, "dump",
             "--intent-hash", intent_hash,
             "--pid", str(server_proc.pid),
             "--metadata", metadata_file.name,
             "--cache-dir", cache_dir],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            print(f"Dump stderr: {result.stderr}")
            fail(f"Dump failed with exit code {result.returncode}")

        print("Dump succeeded")

        # Verify server still works after dump (--leave-running)
        if not check_health(port):
            fail("Server not healthy after dump")
        print("Server still healthy after dump")

        # Kill original
        server_proc.terminate()
        server_proc.wait(timeout=5)
        print("Original server killed")

        # Verify it's dead
        if check_health(port):
            fail("Server still responding after kill")

        # Restore
        pidfile = os.path.join(run_dir, "restored.pid")
        print("--- Restore ---")
        result = subprocess.run(
            [zs_snapshot, "restore",
             "--intent-hash", intent_hash,
             "--pidfile", pidfile,
             "--cache-dir", cache_dir],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            print(f"Restore stderr: {result.stderr}")
            fail(f"Restore failed with exit code {result.returncode}")

        print("Restore succeeded")

        # Read restored PID
        restored_pid = int(Path(pidfile).read_text().strip())
        print(f"Restored PID: {restored_pid}")

        # Give it a moment to fully restore
        time.sleep(0.5)

        # Verify /health works on restored process
        if not wait_for_health(port, timeout=5.0):
            fail("Restored server not responding on /health")

        print("Restored server is healthy!")

        # Cleanup restored process
        os.kill(restored_pid, signal.SIGTERM)

    finally:
        # Cleanup
        os.unlink(metadata_file.name)
        import shutil as sh
        sh.rmtree(cache_dir, ignore_errors=True)
        sh.rmtree(run_dir, ignore_errors=True)

    print("=== PASS: Python HTTP service dump/restore ===")


if __name__ == "__main__":
    main()
