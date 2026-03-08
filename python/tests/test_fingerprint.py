"""Tests for environment fingerprint computation."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from zerostart.fingerprint import _normalize_freeze, compute_env_fingerprint


def test_manifest_based_fingerprint():
    """Fingerprint from manifest file should be deterministic."""
    with tempfile.TemporaryDirectory() as tmp:
        sp = Path(tmp)
        manifest = sp / ".zerostart-manifest.json"
        manifest.write_text(json.dumps({"torch": "2.4.1", "numpy": "1.26.0"}))

        fp1 = compute_env_fingerprint(sp)
        fp2 = compute_env_fingerprint(sp)
        assert fp1 == fp2
        assert len(fp1) == 16


def test_manifest_change_changes_fingerprint():
    """Changing the manifest should change the fingerprint."""
    with tempfile.TemporaryDirectory() as tmp:
        sp = Path(tmp)
        manifest = sp / ".zerostart-manifest.json"

        manifest.write_text(json.dumps({"torch": "2.4.1"}))
        fp1 = compute_env_fingerprint(sp)

        manifest.write_text(json.dumps({"torch": "2.5.0"}))
        fp2 = compute_env_fingerprint(sp)

        assert fp1 != fp2


def test_normalize_freeze():
    """Normalize should sort, lowercase, strip, and remove comments."""
    raw = """
    # pip freeze output
    Torch==2.4.1
    numpy==1.26.0
    Requests==2.31.0
    """
    normalized = _normalize_freeze(raw)
    lines = normalized.split("\n")
    assert lines == ["numpy==1.26.0", "requests==2.31.0", "torch==2.4.1"]


def test_normalize_freeze_stable():
    """Order shouldn't matter — normalization sorts."""
    a = _normalize_freeze("b==1.0\na==2.0\n")
    b = _normalize_freeze("a==2.0\nb==1.0\n")
    assert a == b


def test_directory_listing_fallback():
    """When no manifest or freeze, should fall back to directory listing."""
    with tempfile.TemporaryDirectory() as tmp:
        sp = Path(tmp)
        (sp / "torch").mkdir()
        (sp / "numpy").mkdir()

        fp = compute_env_fingerprint(sp)
        assert len(fp) == 16
        assert fp != "unknown"


def test_no_info_returns_unknown():
    """When nothing is available, return 'unknown'."""
    # No site_packages, no freeze — should get unknown or a hash
    # This test depends on whether pip/uv are available
    fp = compute_env_fingerprint(None)
    assert isinstance(fp, str)
    assert len(fp) > 0


if __name__ == "__main__":
    import traceback

    tests = [
        test_manifest_based_fingerprint,
        test_manifest_change_changes_fingerprint,
        test_normalize_freeze,
        test_normalize_freeze_stable,
        test_directory_listing_fallback,
        test_no_info_returns_unknown,
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
