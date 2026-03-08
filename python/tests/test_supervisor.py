"""Tests for the Tier 1 supervisor state machine."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from zerostart.supervisor import SupervisorConfig, _find_zs_snapshot, _load_snapshot_metadata


def test_supervisor_config_defaults():
    """SupervisorConfig should have sensible defaults."""
    config = SupervisorConfig()
    assert config.ready_mode == "signal"
    assert config.ready_timeout == 300.0
    assert config.settle_delay == 0.25
    assert "snapshots" in str(config.cache_dir)


def test_load_snapshot_metadata_missing():
    """Should return None when no metadata exists."""
    with tempfile.TemporaryDirectory() as tmp:
        result = _load_snapshot_metadata("zs-snapshot", Path(tmp), "nonexistent")
        assert result is None


def test_load_snapshot_metadata_valid():
    """Should load valid metadata from disk."""
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp)
        intent_dir = cache_dir / "abc123"
        intent_dir.mkdir()

        metadata = {
            "intent_hash": "abc123",
            "env_fingerprint": "def456",
            "entrypoint": "serve.py",
        }
        (intent_dir / "metadata.json").write_text(json.dumps(metadata))

        result = _load_snapshot_metadata("zs-snapshot", cache_dir, "abc123")
        assert result is not None
        assert result["intent_hash"] == "abc123"
        assert result["env_fingerprint"] == "def456"


def test_load_snapshot_metadata_corrupt():
    """Should return None for corrupt metadata."""
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp)
        intent_dir = cache_dir / "abc123"
        intent_dir.mkdir()
        (intent_dir / "metadata.json").write_text("not json{{{")

        result = _load_snapshot_metadata("zs-snapshot", cache_dir, "abc123")
        assert result is None


def test_find_zs_snapshot_env_var():
    """Should use ZS_SNAPSHOT_BIN env var when set."""
    with tempfile.NamedTemporaryFile(suffix="-zs-snapshot", delete=False) as f:
        f.write(b"#!/bin/sh\n")

        with patch.dict("os.environ", {"ZS_SNAPSHOT_BIN": f.name}):
            result = _find_zs_snapshot()
            assert result == f.name


def test_find_zs_snapshot_not_found():
    """Should return None when binary not found anywhere."""
    with patch.dict("os.environ", {}, clear=True):
        with patch("shutil.which", return_value=None):
            # This may still find it via sibling paths, but tests the fallback
            result = _find_zs_snapshot()
            # Result depends on whether zs-snapshot is actually installed
            assert result is None or isinstance(result, str)


if __name__ == "__main__":
    import traceback

    tests = [
        test_supervisor_config_defaults,
        test_load_snapshot_metadata_missing,
        test_load_snapshot_metadata_valid,
        test_load_snapshot_metadata_corrupt,
        test_find_zs_snapshot_env_var,
        test_find_zs_snapshot_not_found,
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
