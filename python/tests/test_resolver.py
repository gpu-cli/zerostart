"""Tests for resolver and cache modules."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from zerostart.resolver import (
    ArtifactPlan,
    WheelArtifact,
    generate_manifest,
    resolve_requirements,
)
from zerostart.cache import EnvironmentCache, CachedEnv
from zerostart.run import parse_inline_metadata


def test_artifact_plan_classification():
    """Wheels below UV_THRESHOLD go to uv, above go to fast-wheel."""
    artifacts = [
        WheelArtifact("six", "1.17.0", "https://example.com/six.whl", size=12_000),
        WheelArtifact("requests", "2.32.5", "https://example.com/requests.whl", size=500_000),
        WheelArtifact("torch", "2.4.1", "https://example.com/torch.whl", size=900_000_000),
        WheelArtifact("numpy", "1.26.4", "https://example.com/numpy.whl", size=15_000_000),
    ]
    plan = ArtifactPlan(artifacts=artifacts, python_version="3.11", platform="linux")

    assert len(plan.uv_wheels) == 2  # six, requests
    assert len(plan.fast_wheels) == 2  # torch, numpy
    assert {w.distribution for w in plan.uv_wheels} == {"six", "requests"}
    assert {w.distribution for w in plan.fast_wheels} == {"torch", "numpy"}


def test_import_to_distribution_mapping():
    """Import name mapping resolves correctly."""
    artifacts = [
        WheelArtifact("PyYAML", "6.0", "https://example.com/pyyaml.whl",
                       import_roots=["yaml"]),
        WheelArtifact("requests", "2.32.5", "https://example.com/requests.whl",
                       import_roots=["requests"]),
        WheelArtifact("nopkg", "1.0", "https://example.com/nopkg.whl"),
    ]
    plan = ArtifactPlan(artifacts=artifacts, python_version="3.11", platform="linux")

    mapping = plan.import_to_distribution
    assert mapping["yaml"] == "PyYAML"
    assert mapping["requests"] == "requests"
    assert mapping["nopkg"] == "nopkg"  # fallback to distribution name


def test_cache_key_deterministic():
    """Same artifacts produce same cache key regardless of order."""
    a1 = WheelArtifact("torch", "2.4.1", "https://example.com/torch.whl")
    a2 = WheelArtifact("numpy", "1.26.4", "https://example.com/numpy.whl")

    plan1 = ArtifactPlan(artifacts=[a1, a2], python_version="3.11", platform="linux")
    plan2 = ArtifactPlan(artifacts=[a2, a1], python_version="3.11", platform="linux")

    assert plan1.cache_key == plan2.cache_key


def test_cache_key_differs_on_version():
    """Different versions produce different cache keys."""
    a1 = WheelArtifact("torch", "2.4.1", "https://example.com/torch-2.4.1.whl")
    a2 = WheelArtifact("torch", "2.5.0", "https://example.com/torch-2.5.0.whl")

    plan1 = ArtifactPlan(artifacts=[a1], python_version="3.11", platform="linux")
    plan2 = ArtifactPlan(artifacts=[a2], python_version="3.11", platform="linux")

    assert plan1.cache_key != plan2.cache_key


def test_generate_manifest():
    """Manifest JSON matches Rust daemon's expected format."""
    artifacts = [
        WheelArtifact("six", "1.17.0", "https://example.com/six.whl",
                       size=12_000, import_roots=["six"]),
        WheelArtifact("torch", "2.4.1", "https://example.com/torch.whl",
                       size=900_000_000, import_roots=["torch"]),
    ]
    plan = ArtifactPlan(artifacts=artifacts, python_version="3.11", platform="linux")

    with tempfile.TemporaryDirectory() as tmp:
        sp = Path(tmp) / "site-packages"
        out = Path(tmp) / "output"
        sp.mkdir()
        out.mkdir()

        manifest_path = generate_manifest(plan, sp, out)

        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text())

        assert data["site_packages"] == str(sp)
        assert "status_dir" not in data
        # Only fast wheels in manifest (torch, not six)
        assert len(data["wheels"]) == 1
        assert data["wheels"][0]["distribution"] == "torch"
        assert data["wheels"][0]["url"] == "https://example.com/torch.whl"
        assert data["wheels"][0]["size"] == 900_000_000
        assert data["wheels"][0]["import_roots"] == ["torch"]


def test_generate_manifest_empty_fast_wheels():
    """Manifest with no fast wheels produces empty wheels list."""
    artifacts = [
        WheelArtifact("six", "1.17.0", "https://example.com/six.whl", size=12_000),
    ]
    plan = ArtifactPlan(artifacts=artifacts, python_version="3.11", platform="linux")

    with tempfile.TemporaryDirectory() as tmp:
        sp = Path(tmp) / "site-packages"
        out = Path(tmp) / "output"
        sp.mkdir()
        out.mkdir()

        manifest_path = generate_manifest(plan, sp, out)
        data = json.loads(manifest_path.read_text())
        assert data["wheels"] == []


def test_env_cache_miss():
    """Cache lookup returns None for unknown plan."""
    with tempfile.TemporaryDirectory() as tmp:
        cache = EnvironmentCache(Path(tmp))
        plan = ArtifactPlan(
            artifacts=[WheelArtifact("six", "1.17.0", "https://example.com/six.whl")],
            python_version="3.11",
            platform="linux",
        )

        result = cache.lookup(plan)
        assert result is None


def test_env_cache_create_and_lookup():
    """Create env, then lookup finds it (incomplete)."""
    with tempfile.TemporaryDirectory() as tmp:
        cache = EnvironmentCache(Path(tmp))
        plan = ArtifactPlan(
            artifacts=[WheelArtifact("six", "1.17.0", "https://example.com/six.whl")],
            python_version="3.11",
            platform="linux",
        )

        env = cache.create_env(plan)
        assert env.env_dir.exists()
        assert not env.is_complete
        assert env.site_packages is not None

        # Lookup should find it (incomplete)
        found = cache.lookup(plan)
        assert found is not None
        assert not found.is_complete


def test_env_cache_complete():
    """Mark complete, then lookup returns complete env."""
    with tempfile.TemporaryDirectory() as tmp:
        cache = EnvironmentCache(Path(tmp))
        plan = ArtifactPlan(
            artifacts=[WheelArtifact("six", "1.17.0", "https://example.com/six.whl")],
            python_version="3.11",
            platform="linux",
        )

        env = cache.create_env(plan)
        cache.mark_complete(env)

        found = cache.lookup(plan)
        assert found is not None
        assert found.is_complete


def test_resolve_empty():
    """Empty requirements returns empty plan."""
    plan = resolve_requirements([])
    assert len(plan.artifacts) == 0


def test_resolve_real_packages():
    """Resolve real PyPI packages (requires network + uv)."""
    if not shutil.which("uv"):
        print("SKIP: uv not found")
        return

    plan = resolve_requirements(["six", "idna"], python_version="3.11", platform="linux")

    assert len(plan.artifacts) >= 2
    dist_names = {a.distribution.lower() for a in plan.artifacts}
    assert "six" in dist_names
    assert "idna" in dist_names

    for a in plan.artifacts:
        assert a.url.startswith("https://")
        assert a.version
        assert a.size > 0


def test_inline_metadata_basic():
    """Parse PEP 723 inline script metadata."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write('''\
# /// script
# dependencies = ["torch", "transformers>=4.0"]
# ///

import torch
print(torch.__version__)
''')
        f.flush()
        deps = parse_inline_metadata(f.name)
    Path(f.name).unlink()
    assert deps == ["torch", "transformers>=4.0"]


def test_inline_metadata_single_quotes():
    """Parse inline metadata with single-quoted strings."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("""\
# /// script
# dependencies = ['requests', 'flask>=2.0']
# ///

import requests
""")
        f.flush()
        deps = parse_inline_metadata(f.name)
    Path(f.name).unlink()
    assert deps == ["requests", "flask>=2.0"]


def test_inline_metadata_multiline():
    """Parse inline metadata with deps on separate lines."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write('''\
# /// script
# dependencies = [
#     "torch>=2.0",
#     "numpy",
#     "pyyaml",
# ]
# ///

print("hello")
''')
        f.flush()
        deps = parse_inline_metadata(f.name)
    Path(f.name).unlink()
    assert deps == ["torch>=2.0", "numpy", "pyyaml"]


def test_inline_metadata_none():
    """Script without inline metadata returns empty list."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write('print("hello")\n')
        f.flush()
        deps = parse_inline_metadata(f.name)
    Path(f.name).unlink()
    assert deps == []


def test_inline_metadata_no_dependencies():
    """Script block without dependencies key returns empty list."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write('''\
# /// script
# requires-python = ">=3.10"
# ///

print("hello")
''')
        f.flush()
        deps = parse_inline_metadata(f.name)
    Path(f.name).unlink()
    assert deps == []


def test_inline_metadata_missing_file():
    """Non-existent file returns empty list."""
    deps = parse_inline_metadata("/tmp/nonexistent_script_xyz.py")
    assert deps == []


def test_inline_metadata_mixed_with_other_comments():
    """Inline metadata parsed correctly when other comments present."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write('''\
#!/usr/bin/env python3
# This is my script

# /// script
# dependencies = ["requests"]
# ///

# More comments here
import requests
''')
        f.flush()
        deps = parse_inline_metadata(f.name)
    Path(f.name).unlink()
    assert deps == ["requests"]


if __name__ == "__main__":
    tests = [
        test_artifact_plan_classification,
        test_import_to_distribution_mapping,
        test_cache_key_deterministic,
        test_cache_key_differs_on_version,
        test_generate_manifest,
        test_generate_manifest_empty_fast_wheels,
        test_env_cache_miss,
        test_env_cache_create_and_lookup,
        test_env_cache_complete,
        test_resolve_empty,
        test_resolve_real_packages,
        test_inline_metadata_basic,
        test_inline_metadata_single_quotes,
        test_inline_metadata_multiline,
        test_inline_metadata_none,
        test_inline_metadata_no_dependencies,
        test_inline_metadata_missing_file,
        test_inline_metadata_mixed_with_other_comments,
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
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
