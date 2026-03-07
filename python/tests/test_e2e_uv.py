#!/usr/bin/env python3
"""End-to-end test: uv installs packages in background, app imports lazily.

This test creates a real venv, starts `uv pip install` in the background,
and runs an app that imports those packages. The lazy import hook blocks
each import until the package is actually installed.

Run locally:  python3 python/tests/test_e2e_uv.py
Run on GPU:   gpu run "python3 python/tests/test_e2e_uv.py"
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from zerostart.lazy_imports import install_hook, remove_hook

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(name)-10s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test")

# Packages to install — ordered small-to-large so we can see the hook
# blocking on the bigger ones while small ones install quickly
PACKAGES = [
    "six",           # ~11KB, instant
    "idna",          # ~60KB
    "certifi",       # ~160KB
    "requests",      # ~63KB but has deps (urllib3, charset-normalizer)
    "pyyaml",        # ~180KB, imports as 'yaml'
]


def find_uv() -> str:
    """Find the uv binary."""
    for candidate in ["uv", os.path.expanduser("~/.cargo/bin/uv"), "/usr/local/bin/uv"]:
        if shutil.which(candidate):
            return candidate
    raise RuntimeError("uv not found — install with: curl -LsSf https://astral.sh/uv/install.sh | sh")


def main() -> None:
    uv = find_uv()
    log.info("Using uv: %s", uv)

    with tempfile.TemporaryDirectory(prefix="zs-e2e-") as tmpdir:
        tmpdir = Path(tmpdir)
        venv_dir = tmpdir / "venv"
        status_dir = tmpdir / "status"
        status_dir.mkdir()

        # Create venv
        log.info("Creating venv...")
        subprocess.run([uv, "venv", str(venv_dir), "--python", "3.10"], check=True, capture_output=True)

        # Find site-packages
        site_packages = list(venv_dir.glob("lib/python*/site-packages"))[0]
        log.info("site-packages: %s", site_packages)

        # Add to sys.path so imports can find packages there
        sys.path.insert(0, str(site_packages))

        # Signal: installer is running
        (status_dir / "installing").touch()

        def background_installer() -> None:
            """Install packages one at a time with uv, writing to the real venv."""
            python = str(venv_dir / "bin" / "python")
            for pkg in PACKAGES:
                log.info("[installer] Installing %s...", pkg)
                t0 = time.monotonic()
                result = subprocess.run(
                    [uv, "pip", "install", "--python", python, "--no-deps", pkg],
                    capture_output=True,
                    text=True,
                )
                elapsed = time.monotonic() - t0
                if result.returncode == 0:
                    log.info("[installer] %s installed (%.2fs)", pkg, elapsed)
                else:
                    log.error("[installer] %s FAILED: %s", pkg, result.stderr.strip())

            # Install deps that requests needs
            for dep in ["urllib3", "charset-normalizer"]:
                subprocess.run(
                    [uv, "pip", "install", "--python", python, "--no-deps", dep],
                    capture_output=True,
                )

            # Signal: done
            (status_dir / "installing").unlink(missing_ok=True)
            (status_dir / "__done__").touch()
            log.info("[installer] ALL DONE")

        # Start background installer
        installer_thread = threading.Thread(target=background_installer, daemon=True)
        installer_thread.start()

        # Install the lazy import hook
        hook = install_hook(status_dir=status_dir, poll_interval=0.02)

        # --- App code starts here --- (this is what the user's script looks like)
        log.info("[app] Starting app (packages still installing)...")

        app_start = time.monotonic()

        # These imports will block until each package lands
        log.info("[app] Importing six...")
        t0 = time.monotonic()
        import six
        log.info("[app] six=%s (%.2fs)", six.__version__, time.monotonic() - t0)

        log.info("[app] Importing idna...")
        t0 = time.monotonic()
        import idna
        log.info("[app] idna=%s (%.2fs)", idna.__version__, time.monotonic() - t0)

        log.info("[app] Importing certifi...")
        t0 = time.monotonic()
        import certifi
        log.info("[app] certifi=%s (%.2fs)", certifi.__version__, time.monotonic() - t0)

        log.info("[app] Importing yaml (pyyaml)...")
        t0 = time.monotonic()
        import yaml
        log.info("[app] yaml=%s (%.2fs)", yaml.__version__, time.monotonic() - t0)

        log.info("[app] Importing requests...")
        t0 = time.monotonic()
        import requests
        log.info("[app] requests=%s (%.2fs)", requests.__version__, time.monotonic() - t0)

        total = time.monotonic() - app_start
        log.info("[app] All imports done in %.2fs", total)

        # --- App code ends ---

        report = remove_hook()
        log.info("Wait report: %s", report)

        # Verify
        assert six.__version__
        assert requests.__version__
        assert yaml.__version__

        # Cleanup sys.path
        sys.path.remove(str(site_packages))
        for mod_name in ["six", "idna", "certifi", "requests", "yaml", "urllib3", "charset_normalizer"]:
            for key in list(sys.modules):
                if key == mod_name or key.startswith(mod_name + "."):
                    del sys.modules[key]

        installer_thread.join(timeout=30)

        log.info("=== RESULT ===")
        if report:
            waited = sum(report.values())
            log.info("Total import wait: %.2fs", waited)
            log.info("App wall time: %.2fs", total)
            log.info("Packages that blocked: %s", list(report.keys()))
        else:
            log.info("No imports blocked — everything was already installed!")
        log.info("PASSED")


if __name__ == "__main__":
    main()
