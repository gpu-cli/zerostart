#!/usr/bin/env python3
"""GPU test: lazy import torch + transformers while uv installs in background.

Run on GPU pod:
    gpu run "cd python && PYTHONPATH=. python3 tests/test_gpu_lazy.py"

Uses a single Python interpreter and installs into a fresh venv that matches
the running Python version. Compares:
  - Baseline: sequential uv install → import
  - Lazy: overlapped uv install + import via lazy hook
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

# Packages to test with
PACKAGES = [
    "torch",
    "transformers",
    "safetensors",
    "accelerate",
    "huggingface_hub",
    "tokenizers",
    "requests",
    "pyyaml",
    "tqdm",
    "regex",
    "filelock",
    "packaging",
    "numpy",
]


def find_uv() -> str:
    for candidate in ["uv", os.path.expanduser("~/.cargo/bin/uv"),
                       "/root/.cargo/bin/uv", "/root/.local/bin/uv"]:
        if shutil.which(candidate):
            return candidate
    raise RuntimeError("uv not found")


def make_venv(uv: str, venv_dir: Path) -> Path:
    """Create venv matching current Python. Returns site-packages path."""
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    subprocess.run(
        [uv, "venv", str(venv_dir), "--python", py_ver],
        check=True, capture_output=True, text=True,
    )
    candidates = list(venv_dir.glob("lib/python*/site-packages"))
    if not candidates:
        raise RuntimeError(f"No site-packages in {venv_dir}")
    return candidates[0]


def run_in_venv(uv: str, venv_dir: Path, code: str) -> tuple[str, float]:
    """Run Python code in a venv, returns (stdout, elapsed)."""
    python = str(venv_dir / "bin" / "python")
    t0 = time.monotonic()
    result = subprocess.run(
        [python, "-c", code],
        capture_output=True, text=True,
    )
    elapsed = time.monotonic() - t0
    output = result.stdout.strip()
    if result.returncode != 0:
        output += "\nSTDERR: " + result.stderr.strip()
    return output, elapsed


def main() -> None:
    uv = find_uv()
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    log.info("Using uv=%s python=%s", uv, py_ver)

    with tempfile.TemporaryDirectory(prefix="zs-gpu-") as tmpdir:
        tmpdir = Path(tmpdir)

        # === BASELINE ===
        log.info("=== BASELINE: Sequential install → import ===")
        baseline_venv = tmpdir / "baseline"
        make_venv(uv, baseline_venv)
        baseline_python = str(baseline_venv / "bin" / "python")

        t0 = time.monotonic()
        result = subprocess.run(
            [uv, "pip", "install", "--python", baseline_python] + PACKAGES,
            capture_output=True, text=True,
        )
        baseline_install = time.monotonic() - t0
        log.info("Baseline install: %.1fs", baseline_install)
        if result.returncode != 0:
            log.error("Baseline install failed:\n%s", result.stderr[:1000])
            sys.exit(1)

        import_code = (
            "import time; t0=time.monotonic(); "
            "import torch; "
            "import transformers; "
            "import safetensors; "
            "print(f'import_time={time.monotonic()-t0:.2f}'); "
            "print(f'torch={torch.__version__}'); "
            "print(f'transformers={transformers.__version__}'); "
            "print(f'cuda={torch.cuda.is_available()}')"
        )
        output, baseline_import = run_in_venv(uv, baseline_venv, import_code)
        log.info("Baseline import: %s (%.1fs)", output, baseline_import)
        baseline_total = baseline_install + baseline_import

        # === LAZY IMPORT ===
        log.info("")
        log.info("=== LAZY: Overlapped install + import ===")
        lazy_venv = tmpdir / "lazy"
        status_dir = tmpdir / "status"
        status_dir.mkdir()
        site_packages = make_venv(uv, lazy_venv)

        # Add venv site-packages to THIS process's sys.path
        sys.path.insert(0, str(site_packages))

        # Also add the venv's bin to PATH for any console scripts
        venv_bin = str(lazy_venv / "bin")
        os.environ["PATH"] = venv_bin + ":" + os.environ.get("PATH", "")

        (status_dir / "installing").touch()

        def background_installer() -> None:
            python = str(lazy_venv / "bin" / "python")
            log.info("[installer] Starting uv pip install...")
            t0 = time.monotonic()
            result = subprocess.run(
                [uv, "pip", "install", "--python", python] + PACKAGES,
                capture_output=True, text=True,
            )
            elapsed = time.monotonic() - t0
            if result.returncode == 0:
                log.info("[installer] Done (%.1fs)", elapsed)
            else:
                log.error("[installer] Failed (%.1fs):\n%s", elapsed, result.stderr[:500])
            (status_dir / "installing").unlink(missing_ok=True)
            (status_dir / "__done__").touch()

        installer = threading.Thread(target=background_installer, daemon=True)
        installer.start()

        hook = install_hook(
            status_dir=status_dir,
            poll_interval=0.05,
            expected_packages=set(PACKAGES),
        )

        # App starts NOW
        app_start = time.monotonic()
        log.info("[app] Starting (install in background)...")

        log.info("[app] import torch...")
        t0 = time.monotonic()
        import torch
        log.info("[app] torch %s (%.1fs) cuda=%s",
                 torch.__version__, time.monotonic() - t0, torch.cuda.is_available())

        log.info("[app] import transformers...")
        t0 = time.monotonic()
        import transformers
        log.info("[app] transformers %s (%.1fs)",
                 transformers.__version__, time.monotonic() - t0)

        log.info("[app] import safetensors...")
        t0 = time.monotonic()
        import safetensors
        log.info("[app] safetensors %s (%.1fs)",
                 safetensors.__version__, time.monotonic() - t0)

        app_total = time.monotonic() - app_start
        log.info("[app] All imports done in %.1fs", app_total)

        report = remove_hook()
        installer.join(timeout=120)

        # === RESULTS ===
        log.info("")
        log.info("=" * 60)
        log.info("RESULTS")
        log.info("=" * 60)
        log.info("Baseline:    %.1fs install + %.1fs import = %.1fs total",
                 baseline_install, baseline_import, baseline_total)
        log.info("Lazy import: %.1fs (overlapped)", app_total)
        if app_total > 0:
            log.info("Speedup:     %.1fx", baseline_total / app_total)
        if report:
            log.info("")
            log.info("Blocked imports:")
            for pkg, wait in sorted(report.items(), key=lambda x: -x[1]):
                log.info("  %-20s %.2fs", pkg, wait)
        log.info("=" * 60)


if __name__ == "__main__":
    main()
