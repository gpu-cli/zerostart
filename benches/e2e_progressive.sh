#!/usr/bin/env bash
# E2E Progressive Loading Test
#
# Tests the full zerostart flow:
# 1. CLI warm starts extracting wheels in the background
# 2. Python starts immediately with lazy import hook
# 3. Imports block until their wheel is ready, then succeed
# 4. Demand signaling prioritizes what Python needs NOW
#
# Usage: gpu run "bash benches/e2e_progressive.sh"

set -euo pipefail

WARM_BIN="./bin/zs-fast-wheel-linux-x86_64"
chmod +x "$WARM_BIN"
PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

echo "========================================"
echo "  E2E Progressive Loading"
echo "  Python: $PY_VERSION"
echo "  $(date)"
echo "========================================"

# ─── Test 1: Small packages — warm then import ──────────────────────────

echo ""
echo "=== Test 1: Warm + Progressive Import (small packages) ==="

SP1="/tmp/e2e-prog-small-$$"
rm -rf "$SP1"
mkdir -p "$SP1"

# Warm first, then import (verifies correctness)
$WARM_BIN warm \
    --requirements "requests
pyyaml
tqdm
packaging
jinja2" \
    --site-packages "$SP1" \
    --python-version "$PY_VERSION" \
    -j 8 2>&1 | while read -r line; do echo "  [warm] $line"; done

PYTHONPATH="$SP1" python3 -c "
import sys, time, importlib

t0 = time.monotonic()
results = {}

for pkg in ['requests', 'yaml', 'tqdm', 'packaging', 'jinja2']:
    importlib.invalidate_caches()
    try:
        t = time.monotonic()
        __import__(pkg)
        elapsed = time.monotonic() - t
        results[pkg] = f'OK ({elapsed*1000:.0f}ms)'
    except Exception as e:
        results[pkg] = f'FAIL: {e}'

total = time.monotonic() - t0
for pkg, status in results.items():
    print(f'  {pkg}: {status}')
print(f'  Total import time: {total*1000:.0f}ms')

failures = [k for k, v in results.items() if 'FAIL' in v]
sys.exit(1 if failures else 0)
"
T1_EXIT=$?
rm -rf "$SP1"
echo "  Test 1: $([ $T1_EXIT -eq 0 ] && echo PASS || echo FAIL)"

# ─── Test 2: Torch progressive — the real test ──────────────────────────

echo ""
echo "=== Test 2: Torch Progressive Loading ==="

SP2="/tmp/e2e-prog-torch-$$"
rm -rf "$SP2"
mkdir -p "$SP2"

# Start warm for torch stack
echo "  Starting warm (torch + deps)..."
T2_WARM_START=$(python3 -c "import time; print(time.monotonic())")

$WARM_BIN warm \
    --requirements "torch
numpy
safetensors" \
    --site-packages "$SP2" \
    --python-version "$PY_VERSION" \
    -j 8 2>&1 | while read -r line; do echo "  [warm] $line"; done &
WARM_PID2=$!

# Start Python immediately — it should be able to import as packages land
echo "  Starting Python (concurrent with warm)..."

PYTHONPATH="$SP2" python3 << 'PYEOF'
import sys, time, importlib, json

results = {}

def try_import(name, max_wait=120):
    """Try importing, waiting for it to appear on disk."""
    t0 = time.monotonic()
    while True:
        importlib.invalidate_caches()
        try:
            mod = __import__(name)
            elapsed = time.monotonic() - t0
            return mod, elapsed
        except ImportError:
            if time.monotonic() - t0 > max_wait:
                return None, time.monotonic() - t0
            time.sleep(0.2)

# Try numpy first (should be available quickly, it's small)
print("  Waiting for numpy...")
np, np_wait = try_import("numpy")
if np:
    results["numpy"] = {"version": np.__version__, "wait_s": round(np_wait, 2), "test": (np.array([1,2,3]) * 2).tolist()}
    print(f"  numpy: OK ({np_wait:.1f}s)")
else:
    results["numpy"] = {"error": f"timeout after {np_wait:.1f}s"}
    print(f"  numpy: TIMEOUT")

# safetensors should land relatively quickly too
print("  Waiting for safetensors...")
st, st_wait = try_import("safetensors")
if st:
    results["safetensors"] = {"version": st.__version__, "wait_s": round(st_wait, 2)}
    print(f"  safetensors: OK ({st_wait:.1f}s)")
else:
    results["safetensors"] = {"error": f"timeout after {st_wait:.1f}s"}
    print(f"  safetensors: TIMEOUT")

# torch is the big one — may take longer
print("  Waiting for torch (this is the big one)...")
torch_mod, torch_wait = try_import("torch", max_wait=180)
if torch_mod:
    results["torch"] = {
        "version": torch_mod.__version__,
        "wait_s": round(torch_wait, 2),
        "cuda": torch_mod.cuda.is_available(),
    }
    print(f"  torch: OK ({torch_wait:.1f}s)")

    # Do a GPU computation to prove it works
    if torch_mod.cuda.is_available():
        a = torch_mod.randn(1000, 1000, device="cuda")
        b = torch_mod.randn(1000, 1000, device="cuda")
        c = torch_mod.mm(a, b)
        results["torch"]["gpu_matmul"] = list(c.shape)
        results["torch"]["gpu_name"] = torch_mod.cuda.get_device_name(0)
        print(f"  GPU matmul: OK ({torch_mod.cuda.get_device_name(0)})")
else:
    results["torch"] = {"error": f"timeout after {torch_wait:.1f}s"}
    print(f"  torch: TIMEOUT")

# Print full results
print()
print(json.dumps(results, indent=2))

# Check all passed
passed = all("error" not in v for v in results.values())
sys.exit(0 if passed else 1)
PYEOF
T2_EXIT=$?

wait "$WARM_PID2" 2>/dev/null || true

T2_WARM_END=$(python3 -c "import time; print(time.monotonic())")
echo "  Test 2: $([ $T2_EXIT -eq 0 ] && echo PASS || echo FAIL)"

rm -rf "$SP2"

# ─── Test 3: DaemonHandle direct (PyO3 in-process) ──────────────────────

echo ""
echo "=== Test 3: DaemonHandle in-process (if maturin available) ==="

# Check if zs_fast_wheel module is available
if python3 -c "import zs_fast_wheel" 2>/dev/null; then
    python3 << 'PYEOF'
import sys, time, json, importlib
from zs_fast_wheel import DaemonHandle
from zerostart.lazy_imports import install_hook, remove_hook

SP = "/tmp/e2e-daemon-direct"

import os, shutil
if os.path.exists(SP):
    shutil.rmtree(SP)
os.makedirs(SP)

# Start daemon with small wheels
daemon = DaemonHandle()
daemon.start(
    wheels=[
        {"url": "https://files.pythonhosted.org/packages/b7/ce/149a00dd41f10bc29e5921b496af8b574d8413afcd5e30dfa0ed46c2cc5e/six-1.17.0-py2.py3-none-any.whl",
         "distribution": "six", "size": 11050, "import_roots": ["six"]},
        {"url": "https://files.pythonhosted.org/packages/76/c6/c88e154df9c4e1a2a66ccf0005a88dfb2650c1dffb6f5ce603dfbd452ce3/idna-3.10-py3-none-any.whl",
         "distribution": "idna", "size": 70442, "import_roots": ["idna"]},
        {"url": "https://files.pythonhosted.org/packages/38/fc/bce832fd4fd99766c04d1ee0eead6b0ec6486fb100ae5e74c1d91292b982/certifi-2025.1.31-py3-none-any.whl",
         "distribution": "certifi", "size": 166393, "import_roots": ["certifi"]},
    ],
    site_packages=SP,
)

# Install lazy hook
hook = install_hook(daemon=daemon, import_map={"six": "six", "idna": "idna", "certifi": "certifi"})

# Add site-packages to path
sys.path.insert(0, SP)

# Imports should work via hook
t0 = time.monotonic()
results = {}

for pkg in ["six", "idna", "certifi"]:
    importlib.invalidate_caches()
    t = time.monotonic()
    try:
        __import__(pkg)
        elapsed = time.monotonic() - t
        results[pkg] = f"OK ({elapsed*1000:.0f}ms)"
    except ImportError as e:
        results[pkg] = f"FAIL: {e}"

report = remove_hook()
daemon.shutdown()

total = time.monotonic() - t0
for pkg, status in results.items():
    print(f"  {pkg}: {status}")
print(f"  Total: {total*1000:.0f}ms")
if report:
    print(f"  Wait report: {report}")

failures = [k for k, v in results.items() if "FAIL" in v]
shutil.rmtree(SP, ignore_errors=True)
sys.exit(1 if failures else 0)
PYEOF
    T3_EXIT=$?
    echo "  Test 3: $([ $T3_EXIT -eq 0 ] && echo PASS || echo FAIL)"
else
    echo "  Skipped (zs_fast_wheel PyO3 module not built — run maturin develop first)"
    T3_EXIT=0
fi

# ─── Summary ────────────────────────────────────────────────────────────

echo ""
echo "========================================"
echo "  Summary"
echo "========================================"
echo "  Test 1 (small progressive):    $([ $T1_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "  Test 2 (torch progressive):    $([ $T2_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "  Test 3 (DaemonHandle direct):  $([ $T3_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "========================================"

[ $T1_EXIT -eq 0 ] && [ $T2_EXIT -eq 0 ] && [ $T3_EXIT -eq 0 ]
