"""Test lazy import hook with a simulated background installer.

Creates a temporary site-packages dir, installs the hook, then drops
real-ish package stubs on disk in a background thread while the main
thread tries to import them.
"""

from __future__ import annotations

import importlib
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from pathlib import Path

# Add the package to sys.path for testing
sys.path.insert(0, str(Path(__file__).parent.parent))

from zerostart.lazy_imports import install_hook, remove_hook

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")


def test_basic_lazy_import():
    """Simulate: installer drops 'fakepkg' after 0.5s, app imports it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        status_dir = Path(tmpdir) / "status"
        site_packages = Path(tmpdir) / "site-packages"
        status_dir.mkdir()
        site_packages.mkdir()

        # Add site-packages to sys.path so Python can find modules there
        sys.path.insert(0, str(site_packages))

        # Signal: installer is running
        (status_dir / "installing").touch()

        def background_installer():
            """Drop 'fakepkg' after a delay, then signal done."""
            time.sleep(0.5)

            # Create a simple package
            pkg_dir = site_packages / "fakepkg"
            pkg_dir.mkdir()
            (pkg_dir / "__init__.py").write_text('VERSION = "1.0.0"\n')
            (pkg_dir / "utils.py").write_text('def hello(): return "hello from fakepkg"\n')

            time.sleep(0.2)

            # Signal: done
            (status_dir / "installing").unlink()
            (status_dir / "__done__").touch()

        # Start background installer
        installer = threading.Thread(target=background_installer, daemon=True)
        installer.start()

        # Install hook and try importing
        hook = install_hook(status_dir=status_dir)

        start = time.monotonic()
        import fakepkg  # should block ~0.5s

        elapsed = time.monotonic() - start
        print(f"fakepkg imported in {elapsed:.2f}s")
        print(f"fakepkg.VERSION = {fakepkg.VERSION}")

        # Submodule should be immediately available (progressive loading)
        import fakepkg.utils

        print(f"fakepkg.utils.hello() = {fakepkg.utils.hello()}")

        report = remove_hook()
        print(f"Wait report: {report}")

        assert elapsed >= 0.4, f"Should have waited ~0.5s, got {elapsed:.2f}s"
        assert elapsed < 2.0, f"Waited too long: {elapsed:.2f}s"
        assert fakepkg.VERSION == "1.0.0"
        assert fakepkg.utils.hello() == "hello from fakepkg"
        assert "fakepkg" in report

        # Cleanup
        sys.path.remove(str(site_packages))
        del sys.modules["fakepkg"]
        del sys.modules["fakepkg.utils"]

        installer.join(timeout=5)
        print("PASSED: test_basic_lazy_import")


def test_demand_signaling():
    """Verify the hook writes demand signals for the installer to read."""
    with tempfile.TemporaryDirectory() as tmpdir:
        status_dir = Path(tmpdir) / "status"
        site_packages = Path(tmpdir) / "site-packages"
        status_dir.mkdir()
        site_packages.mkdir()
        sys.path.insert(0, str(site_packages))

        (status_dir / "installing").touch()

        demanded: list[str] = []

        def demand_watcher():
            """Watch for demand signals and install on demand."""
            demand_path = status_dir / "demand"
            seen_lines = 0
            while (status_dir / "installing").exists():
                if demand_path.exists():
                    lines = demand_path.read_text().strip().splitlines()
                    new_lines = lines[seen_lines:]
                    seen_lines = len(lines)
                    for module_name in new_lines:
                        demanded.append(module_name)
                        # "Install" the demanded package immediately
                        pkg_dir = site_packages / module_name
                        pkg_dir.mkdir(exist_ok=True)
                        (pkg_dir / "__init__.py").write_text(
                            f'NAME = "{module_name}"\n'
                        )
                time.sleep(0.05)

        def finish_installer():
            time.sleep(2.0)
            (status_dir / "installing").unlink(missing_ok=True)
            (status_dir / "__done__").touch()

        watcher = threading.Thread(target=demand_watcher, daemon=True)
        finisher = threading.Thread(target=finish_installer, daemon=True)
        watcher.start()
        finisher.start()

        hook = install_hook(status_dir=status_dir)

        # These packages don't exist yet — the hook should signal demand
        # and the watcher should create them
        start = time.monotonic()
        import demandpkg_a  # noqa: F811

        a_time = time.monotonic() - start

        start = time.monotonic()
        import demandpkg_b  # noqa: F811

        b_time = time.monotonic() - start

        print(f"demandpkg_a: {demandpkg_a.NAME} (waited {a_time:.2f}s)")
        print(f"demandpkg_b: {demandpkg_b.NAME} (waited {b_time:.2f}s)")
        print(f"Demanded packages: {demanded}")

        report = remove_hook()

        assert "demandpkg_a" in demanded, "Should have signaled demand for demandpkg_a"
        assert "demandpkg_b" in demanded, "Should have signaled demand for demandpkg_b"
        assert demandpkg_a.NAME == "demandpkg_a"
        assert demandpkg_b.NAME == "demandpkg_b"

        # Cleanup
        sys.path.remove(str(site_packages))
        for mod in list(sys.modules):
            if mod.startswith("demandpkg_"):
                del sys.modules[mod]

        (status_dir / "installing").unlink(missing_ok=True)
        (status_dir / "__done__").touch()
        watcher.join(timeout=2)
        finisher.join(timeout=3)
        print("PASSED: test_demand_signaling")


def test_already_installed():
    """Packages already on sys.path should import instantly, no blocking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        status_dir = Path(tmpdir) / "status"
        site_packages = Path(tmpdir) / "site-packages"
        status_dir.mkdir()
        site_packages.mkdir()
        sys.path.insert(0, str(site_packages))

        # Pre-install a package
        pkg_dir = site_packages / "preinst"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text('X = 42\n')

        (status_dir / "installing").touch()

        hook = install_hook(status_dir=status_dir)

        start = time.monotonic()
        import preinst

        elapsed = time.monotonic() - start

        report = remove_hook()
        print(f"preinst imported in {elapsed:.4f}s (should be near-instant)")

        assert elapsed < 0.1, f"Pre-installed package took too long: {elapsed:.2f}s"
        assert preinst.X == 42
        assert "preinst" not in report, "Should not appear in wait report"

        sys.path.remove(str(site_packages))
        del sys.modules["preinst"]
        (status_dir / "installing").unlink()
        print("PASSED: test_already_installed")


def test_installer_not_running():
    """If installer isn't running, missing imports fail immediately."""
    with tempfile.TemporaryDirectory() as tmpdir:
        status_dir = Path(tmpdir) / "status"
        status_dir.mkdir()
        # No "installing" sentinel — installer not running

        hook = install_hook(status_dir=status_dir)

        start = time.monotonic()
        try:
            import nonexistent_pkg_xyz  # noqa: F401
            assert False, "Should have raised ImportError"
        except ImportError:
            elapsed = time.monotonic() - start
            print(f"ImportError raised in {elapsed:.4f}s (should be instant)")
            assert elapsed < 0.2, f"Took too long to fail: {elapsed:.2f}s"

        remove_hook()
        print("PASSED: test_installer_not_running")


def test_progressive_submodules():
    """Submodules should resolve progressively as files land."""
    with tempfile.TemporaryDirectory() as tmpdir:
        status_dir = Path(tmpdir) / "status"
        site_packages = Path(tmpdir) / "site-packages"
        status_dir.mkdir()
        site_packages.mkdir()
        sys.path.insert(0, str(site_packages))

        (status_dir / "installing").touch()

        def progressive_installer():
            # Phase 1: drop package with __init__.py only
            time.sleep(0.3)
            pkg_dir = site_packages / "progpkg"
            pkg_dir.mkdir()
            (pkg_dir / "__init__.py").write_text('PHASE = 1\n')

            # Phase 2: add a submodule
            time.sleep(0.5)
            (pkg_dir / "sub.py").write_text('PHASE = 2\ndef work(): return "done"\n')

            # Done
            (status_dir / "installing").unlink()
            (status_dir / "__done__").touch()

        installer = threading.Thread(target=progressive_installer, daemon=True)
        installer.start()

        hook = install_hook(status_dir=status_dir)

        # Phase 1: top-level import blocks until __init__.py lands
        start = time.monotonic()
        import progpkg

        phase1_time = time.monotonic() - start
        print(f"progpkg imported in {phase1_time:.2f}s (phase 1)")
        assert progpkg.PHASE == 1

        # Phase 2: submodule isn't there yet — hook blocks again
        start = time.monotonic()
        import progpkg.sub

        phase2_time = time.monotonic() - start
        print(f"progpkg.sub imported in {phase2_time:.2f}s (phase 2)")
        assert progpkg.sub.work() == "done"

        report = remove_hook()
        print(f"Report: {report}")

        sys.path.remove(str(site_packages))
        for mod in list(sys.modules):
            if mod.startswith("progpkg"):
                del sys.modules[mod]
        installer.join(timeout=5)
        print("PASSED: test_progressive_submodules")


if __name__ == "__main__":
    test_basic_lazy_import()
    print()
    test_demand_signaling()
    print()
    test_already_installed()
    print()
    test_installer_not_running()
    print()
    test_progressive_submodules()
    print()
    print("ALL TESTS PASSED")
