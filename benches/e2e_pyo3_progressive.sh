#!/usr/bin/env bash
# E2E PyO3 Progressive Loading: in-process DaemonHandle + lazy import hook
#
# Builds the zs_fast_wheel PyO3 module on the GPU pod, then tests:
# 1. Unit tests (test_pyo3.py) — DaemonHandle API correctness
# 2. Progressive loading with lazy import hook — small packages
# 3. Progressive loading with torch — full GPU verification
#
# Usage: gpu run "bash benches/e2e_pyo3_progressive.sh"

set -euo pipefail

echo "========================================"
echo "  E2E PyO3 Progressive Loading"
echo "  $(date)"
echo "========================================"

# ─── Step 1: Build PyO3 module ───────────────────────────────────────────

echo ""
echo "=== Step 1: Build zs_fast_wheel PyO3 module ==="

# Set up PATH — cargo may be in various locations on GPU pods
for d in "$HOME/.cargo/bin" "/gpu-cli-workspaces/.cache/cargo/bin" "$HOME/.local/bin" "/usr/local/bin"; do
    [ -d "$d" ] && export PATH="$d:$PATH"
done
source "$HOME/.cargo/env" 2>/dev/null || true

echo "  cargo: $(which cargo 2>/dev/null || echo 'NOT FOUND')"
echo "  maturin: $(which maturin 2>/dev/null || echo 'NOT FOUND')"
echo "  python: $(python3 --version)"

# Build the extension module
cd crates/zs-fast-wheel
echo "  Building with maturin..."
t0=$(python3 -c "import time; print(time.monotonic())")
# Create venv for maturin (it requires one)
rm -rf ../../.venv 2>/dev/null || python3 -c "import shutil; shutil.rmtree('../../.venv', ignore_errors=True)"
python3 -m venv ../../.venv
source ../../.venv/bin/activate
pip install pytest -q 2>/dev/null
maturin develop --release 2>&1 | tail -10
t1=$(python3 -c "import time; print(time.monotonic())")
build_s=$(python3 -c "print(f'{$t1 - $t0:.1f}')")
echo "  Build time: ${build_s}s"
cd ../..

# Verify import works
python3 -c "from zs_fast_wheel import DaemonHandle; print(f'  zs_fast_wheel imported OK: {DaemonHandle}')"

# ─── Step 2: Run unit tests ─────────────────────────────────────────────

echo ""
echo "=== Step 2: PyO3 Unit Tests ==="

PYTHONPATH="python" python3 -m pytest python/tests/test_pyo3.py -v --tb=short 2>&1
T2_EXIT=$?
echo "  Unit tests: $([ $T2_EXIT -eq 0 ] && echo PASS || echo FAIL)"

# ─── Step 3: In-process progressive loading (small packages) ────────────

echo ""
echo "=== Step 3: In-process Progressive Loading (small packages) ==="

PYTHONPATH="python" python3 << 'PYEOF'
import sys, time, json, os, shutil, importlib

SP = "/tmp/e2e-pyo3-small"
if os.path.exists(SP):
    shutil.rmtree(SP)
os.makedirs(SP)

from zs_fast_wheel import DaemonHandle
from zerostart.lazy_imports import install_hook, remove_hook

import logging
logging.basicConfig(level=logging.INFO, format="  %(message)s")

# Start daemon with real wheels
daemon = DaemonHandle()
daemon.start(
    wheels=[
        {"url": "https://files.pythonhosted.org/packages/b7/ce/149a00dd41f10bc29e5921b496af8b574d8413afcd5e30dfa0ed46c2cc5e/six-1.17.0-py2.py3-none-any.whl",
         "distribution": "six", "size": 11050, "import_roots": ["six"]},
        {"url": "https://files.pythonhosted.org/packages/76/c6/c88e154df9c4e1a2a66ccf0005a88dfb2650c1dffb6f5ce603dfbd452ce3/idna-3.10-py3-none-any.whl",
         "distribution": "idna", "size": 70442, "import_roots": ["idna"]},
        {"url": "https://files.pythonhosted.org/packages/16/e1/3079a9ff9b8e11b846c6ac5c8b5bfb7ff225eee721825310c91b3b50304f/tqdm-4.67.3-py3-none-any.whl",
         "distribution": "tqdm", "size": 78374, "import_roots": ["tqdm"]},
    ],
    site_packages=SP,
)

# Install lazy hook — imports will go through DaemonHandle
sys.path.insert(0, SP)
hook = install_hook(
    daemon=daemon,
    import_map={"six": "six", "idna": "idna", "tqdm": "tqdm"},
)

t0 = time.monotonic()
results = {}

# These imports go through the lazy hook → signal_demand → wait_done
for pkg in ["six", "idna", "tqdm"]:
    t = time.monotonic()
    try:
        mod = __import__(pkg)
        elapsed = time.monotonic() - t
        ver = getattr(mod, "__version__", "?")
        results[pkg] = {"status": "OK", "version": ver, "wait_s": round(elapsed, 3)}
    except Exception as e:
        results[pkg] = {"status": "FAIL", "error": str(e)}

total = time.monotonic() - t0
report = remove_hook()
daemon.shutdown()

for pkg, r in results.items():
    if r["status"] == "OK":
        print(f"  {pkg}: OK v{r['version']} ({r['wait_s']}s)")
    else:
        print(f"  {pkg}: FAIL - {r['error']}")

print(f"  Total: {total:.2f}s")
if report:
    print(f"  Hook wait report: {report}")

shutil.rmtree(SP, ignore_errors=True)

failures = [k for k, v in results.items() if v["status"] != "OK"]
sys.exit(1 if failures else 0)
PYEOF
T3_EXIT=$?
echo "  Small progressive: $([ $T3_EXIT -eq 0 ] && echo PASS || echo FAIL)"

# ─── Step 4: In-process progressive loading (torch + GPU) ───────────────

echo ""
echo "=== Step 4: In-process Progressive Loading (torch + GPU) ==="

PYTHONPATH="python" python3 << 'PYEOF'
import sys, time, json, os, shutil, importlib, urllib.request

# Use the venv's site-packages so torch can find nvidia .so files installed via pip
import site
SP = site.getsitepackages()[0]
print(f"  Using site-packages: {SP}")

from zs_fast_wheel import DaemonHandle
from zerostart.lazy_imports import install_hook, remove_hook

import logging
logging.basicConfig(level=logging.INFO, format="  %(message)s")

PY_VER = f"{sys.version_info.major}.{sys.version_info.minor}"
PY_TAG = f"cp{PY_VER.replace('.', '')}"

def pypi_wheel_url(dist, version):
    """Get the best wheel URL from PyPI."""
    url = f"https://pypi.org/pypi/{dist}/{version}/json"
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read())

    best = None
    for f in data["urls"]:
        if not f["filename"].endswith(".whl"):
            continue
        fn = f["filename"]
        # Prefer x86_64 linux wheel matching python version
        if "linux" in fn and "x86_64" in fn and PY_TAG in fn:
            return f["url"], f["size"]
        if "linux" in fn and "x86_64" in fn and "abi3" in fn:
            best = (f["url"], f["size"])
        if "none-any" in fn and not best:
            best = (f["url"], f["size"])
    if best:
        return best
    # Fallback to first wheel
    for f in data["urls"]:
        if f["filename"].endswith(".whl"):
            return f["url"], f["size"]
    raise ValueError(f"No wheel found for {dist}=={version}")

# Pre-install transitive deps that torch needs (small + CUDA runtime libs)
import subprocess
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "typing_extensions", "filelock", "jinja2", "markupsafe",
    "sympy", "mpmath", "networkx", "fsspec",
    "nvidia-cuda-runtime-cu12", "nvidia-cudnn-cu12",
    "nvidia-cuda-nvrtc-cu12", "nvidia-cublas-cu12",
    "nvidia-cuda-cupti-cu12", "nvidia-cufft-cu12",
    "nvidia-curand-cu12", "nvidia-cusolver-cu12",
    "nvidia-cusparse-cu12", "nvidia-nccl-cu12",
    "nvidia-nvtx-cu12", "nvidia-nvjitlink-cu12",
    "triton",
], check=True, capture_output=True)
print("  Pre-installed torch transitive deps")

# Look up wheels from PyPI — daemon handles only the big packages
print("  Resolving wheel URLs from PyPI...")
wheels = []
packages = [
    ("numpy", "2.2.6"),
    ("torch", "2.4.1"),
]

import_map = {}
for dist, ver in packages:
    try:
        url, size = pypi_wheel_url(dist, ver)
        roots = [dist.replace("-", "_").lower()]
        wheels.append({
            "url": url,
            "distribution": dist,
            "size": size,
            "import_roots": roots,
        })
        for r in roots:
            import_map[r] = dist
        print(f"    {dist}=={ver}: {size/1e6:.1f}MB")
    except Exception as e:
        print(f"    {dist}=={ver}: SKIP ({e})")

if not wheels:
    print("  No wheels resolved!")
    sys.exit(1)

# Start daemon
print("  Starting DaemonHandle...")
daemon = DaemonHandle()
daemon.start(wheels=wheels, site_packages=SP, parallel_downloads=8)

# Install lazy hook
sys.path.insert(0, SP)
hook = install_hook(daemon=daemon, import_map=import_map, timeout=300)

t0 = time.monotonic()
results = {}

# Import numpy (should be quick)
print("  Importing numpy (via lazy hook)...")
t = time.monotonic()
try:
    import numpy as np
    elapsed = time.monotonic() - t
    test = (np.array([1, 2, 3]) * 2).tolist()
    results["numpy"] = {"status": "OK", "version": np.__version__, "wait_s": round(elapsed, 2), "test": test}
    print(f"    numpy {np.__version__}: OK ({elapsed:.1f}s)")
except Exception as e:
    results["numpy"] = {"status": "FAIL", "error": str(e)}
    print(f"    numpy: FAIL ({e})")

# Import torch (the big one)
print("  Importing torch (via lazy hook)...")
t = time.monotonic()
try:
    import torch
    elapsed = time.monotonic() - t
    results["torch"] = {
        "status": "OK",
        "version": torch.__version__,
        "wait_s": round(elapsed, 2),
        "cuda": torch.cuda.is_available(),
    }
    print(f"    torch {torch.__version__}: OK ({elapsed:.1f}s)")

    # GPU verification
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"    GPU: {gpu_name}")

        # Tensor ops
        a = torch.randn(1000, 1000, device="cuda")
        b = torch.randn(1000, 1000, device="cuda")
        c = torch.mm(a, b)
        results["torch"]["gpu_matmul"] = list(c.shape)
        results["torch"]["gpu_name"] = gpu_name
        print(f"    GPU matmul 1000x1000: OK")
except Exception as e:
    import traceback
    results["torch"] = {"status": "FAIL", "error": str(e), "tb": traceback.format_exc()}
    print(f"    torch: FAIL ({e})")

total = time.monotonic() - t0
report = remove_hook()

# Wait for daemon to finish all remaining wheels
try:
    daemon.wait_all(timeout_secs=120)
except Exception:
    pass
stats = daemon.stats()
daemon.shutdown()

print()
print(f"  Total time: {total:.1f}s")
print(f"  Daemon stats: {stats[0]} total, {stats[1]} done, {stats[4]} failed")
if report:
    print(f"  Hook wait times: {report}")

print()
print(json.dumps(results, indent=2))

failures = [k for k, v in results.items() if v.get("status") != "OK"]
sys.exit(1 if failures else 0)
PYEOF
T4_EXIT=$?
echo "  Torch progressive: $([ $T4_EXIT -eq 0 ] && echo PASS || echo FAIL)"

# ─── Summary ────────────────────────────────────────────────────────────

echo ""
echo "========================================"
echo "  Summary"
echo "========================================"
echo "  Build PyO3 module:              OK (${build_s}s)"
echo "  Unit tests (test_pyo3.py):      $([ $T2_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "  Small progressive (in-process): $([ $T3_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "  Torch progressive (GPU):        $([ $T4_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "========================================"

[ $T2_EXIT -eq 0 ] && [ $T3_EXIT -eq 0 ] && [ $T4_EXIT -eq 0 ]
