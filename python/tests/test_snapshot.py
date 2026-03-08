"""Tests for snapshot identity computation."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from zerostart.snapshot import compute_intent_hash


def test_deterministic():
    """Same inputs should always produce the same hash."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("print('hello')")
        f.flush()

        h1 = compute_intent_hash(f.name, ["python", f.name], "3.11.8", "x86_64-linux")
        h2 = compute_intent_hash(f.name, ["python", f.name], "3.11.8", "x86_64-linux")
        assert h1 == h2


def test_different_argv():
    """Different argv should produce different hashes."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("print('hello')")
        f.flush()

        h1 = compute_intent_hash(f.name, ["python", f.name], "3.11.8", "x86_64-linux")
        h2 = compute_intent_hash(f.name, ["python", f.name, "--port", "8000"], "3.11.8", "x86_64-linux")
        assert h1 != h2


def test_different_python_version():
    """Different Python version should produce different hashes."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("print('hello')")
        f.flush()

        h1 = compute_intent_hash(f.name, ["python", f.name], "3.11.8", "x86_64-linux")
        h2 = compute_intent_hash(f.name, ["python", f.name], "3.12.0", "x86_64-linux")
        assert h1 != h2


def test_different_platform():
    """Different platform should produce different hashes."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("print('hello')")
        f.flush()

        h1 = compute_intent_hash(f.name, ["python", f.name], "3.11.8", "x86_64-linux")
        h2 = compute_intent_hash(f.name, ["python", f.name], "3.11.8", "aarch64-linux")
        assert h1 != h2


def test_content_change_invalidates():
    """Changing the entrypoint content should change the hash."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("version = 1")
        f.flush()
        h1 = compute_intent_hash(f.name, ["python", f.name], "3.11.8", "x86_64-linux")

    with open(f.name, "w") as f2:
        f2.write("version = 2")

    h2 = compute_intent_hash(f.name, ["python", f.name], "3.11.8", "x86_64-linux")
    assert h1 != h2


def test_hash_length():
    """Hash should be 16 hex characters."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("x = 1")
        f.flush()
        h = compute_intent_hash(f.name, ["python", f.name], "3.11.8", "x86_64-linux")
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)


def test_nonexistent_entrypoint_uses_path():
    """If entrypoint file doesn't exist, hash uses the path string."""
    h1 = compute_intent_hash("/nonexistent/serve.py", ["python", "serve.py"], "3.11.8", "x86_64-linux")
    h2 = compute_intent_hash("/nonexistent/other.py", ["python", "other.py"], "3.11.8", "x86_64-linux")
    assert h1 != h2
    assert len(h1) == 16


def test_defaults():
    """Default python_version and platform should use current interpreter."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("x = 1")
        f.flush()
        h = compute_intent_hash(f.name, ["python", f.name])
        assert len(h) == 16


if __name__ == "__main__":
    import traceback

    tests = [
        test_deterministic,
        test_different_argv,
        test_different_python_version,
        test_different_platform,
        test_content_change_invalidates,
        test_hash_length,
        test_nonexistent_entrypoint_uses_path,
        test_defaults,
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
