#!/usr/bin/env bash
# Reproduce the exact benchmark numbers from README.md
#
# Tests cold start (empty cache) and warm start (cached env) for:
#   torch, vllm, transformers+torch, diffusers+torch, triton
#
# Also tests accelerate() model loading speedup.
#
# Usage: gpu run "bash benches/bench_readme.sh"
#
# Disk-aware: cleans caches between workloads to avoid filling ephemeral disk.

set -uo pipefail

echo "============================================================"
echo "  zerostart README Benchmark Reproduction"
echo "  $(date -u)"
echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
echo "  Disk: $(df -h /tmp | tail -1 | awk '{print $4}') free"
echo "============================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Install zerostart Python SDK
cd "$PROJECT_DIR/python" && pip install -q -e . 2>&1 | tail -1
cd "$PROJECT_DIR"

# Ensure binary is available
ZS_BIN="$PROJECT_DIR/bin/zerostart-linux-x86_64"
if [ ! -f "$ZS_BIN" ]; then
    ZS_BIN="$PROJECT_DIR/bin/zs-fast-wheel-linux-x86_64"
fi
chmod +x "$ZS_BIN" 2>/dev/null || true

UV="$HOME/.local/bin/uv"
PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

RESULTS_DIR="$PROJECT_DIR/benches/results"
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="$RESULTS_DIR/readme_bench.csv"
echo "workload,method,time_s" > "$RESULTS_FILE"

BENCH_BASE="/tmp/zs-readme-bench"
rm -rf "$BENCH_BASE"
mkdir -p "$BENCH_BASE"

# Disable shared CUDA wheel cache to save disk (each workload is independent)
export ZS_NO_SHARED_CACHE=1

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ts() { python3 -c "import time; print(f'{time.monotonic():.6f}')"; }

disk_free() { df -h /tmp | tail -1 | awk '{print $4}'; }

clean_all_caches() {
    rm -rf "${ZEROSTART_CACHE:-$HOME/.cache/zerostart}" 2>/dev/null || true
    rm -rf "$HOME/.cache/uv" 2>/dev/null || true
    rm -rf "$BENCH_BASE"/* 2>/dev/null || true
    echo "  [cleanup] disk free: $(disk_free)"
}

run_zs_cold() {
    local label="$1"
    local deps="$2"
    local work="$BENCH_BASE/zs-cold-$label"
    rm -rf "$work"
    mkdir -p "$work"

    local test_pkg
    test_pkg=$(echo "$deps" | awk '{print $1}')
    cat > "$work/test.py" << PYEOF
import $test_pkg
print("ok")
PYEOF

    local pkg_args=""
    for pkg in $deps; do
        pkg_args="$pkg_args -p $pkg"
    done

    # Wipe zerostart cache for true cold start
    rm -rf "${ZEROSTART_CACHE:-$HOME/.cache/zerostart}" 2>/dev/null || true

    echo "  [$label] zerostart cold..."
    local t0 t1
    t0=$(ts)
    $ZS_BIN run $pkg_args "$work/test.py" 2>&1 | tail -3
    t1=$(ts)

    local elapsed
    elapsed=$(python3 -c "print(f'{$t1 - $t0:.1f}')")
    echo "  -> ${elapsed}s"

    echo "$label,zs_cold,$elapsed" >> "$RESULTS_FILE"
    rm -rf "$work"
}

run_zs_warm() {
    local label="$1"
    local deps="$2"
    local work="$BENCH_BASE/zs-warm-$label"
    rm -rf "$work"
    mkdir -p "$work"

    local test_pkg
    test_pkg=$(echo "$deps" | awk '{print $1}')
    cat > "$work/test.py" << PYEOF
import $test_pkg
print("ok")
PYEOF

    local pkg_args=""
    for pkg in $deps; do
        pkg_args="$pkg_args -p $pkg"
    done

    echo "  [$label] zerostart warm..."
    local t0 t1
    t0=$(ts)
    $ZS_BIN run $pkg_args "$work/test.py" 2>&1 | tail -1
    t1=$(ts)

    local elapsed
    elapsed=$(python3 -c "print(f'{$t1 - $t0:.1f}')")
    echo "  -> ${elapsed}s"

    echo "$label,zs_warm,$elapsed" >> "$RESULTS_FILE"
    rm -rf "$work"
}

run_uvx_cold() {
    local label="$1"
    local deps="$2"
    local venv="$BENCH_BASE/uvx-$label"
    rm -rf "$venv"

    local test_pkg
    test_pkg=$(echo "$deps" | awk '{print $1}')

    # Wipe uv cache for true cold start
    rm -rf "$HOME/.cache/uv" 2>/dev/null || true

    echo "  [$label] uv cold..."
    local t0 t1
    t0=$(ts)

    $UV venv "$venv" --python "$PY_VERSION" -q 2>/dev/null || $UV venv "$venv" -q
    echo "$deps" | tr ' ' '\n' > "$BENCH_BASE/reqs-$label.txt"
    $UV pip install -r "$BENCH_BASE/reqs-$label.txt" --python "$venv/bin/python" -q 2>/dev/null
    "$venv/bin/python" -c "import $test_pkg; print('ok')" 2>&1 | tail -1

    t1=$(ts)

    local elapsed
    elapsed=$(python3 -c "print(f'{$t1 - $t0:.1f}')")
    echo "  -> ${elapsed}s"

    echo "$label,uv_cold,$elapsed" >> "$RESULTS_FILE"
    # Clean venv immediately to save disk (keep uv cache for warm test)
    rm -rf "$venv" "$BENCH_BASE/reqs-$label.txt"
}

run_uvx_warm() {
    local label="$1"
    local deps="$2"
    local venv="$BENCH_BASE/uvx-warm-$label"
    rm -rf "$venv"

    local test_pkg
    test_pkg=$(echo "$deps" | awk '{print $1}')

    echo "  [$label] uv warm..."
    local t0 t1
    t0=$(ts)

    $UV venv "$venv" --python "$PY_VERSION" -q 2>/dev/null || $UV venv "$venv" -q
    echo "$deps" | tr ' ' '\n' > "$BENCH_BASE/reqs-warm-$label.txt"
    $UV pip install -r "$BENCH_BASE/reqs-warm-$label.txt" --python "$venv/bin/python" -q 2>/dev/null
    "$venv/bin/python" -c "import $test_pkg; print('ok')" 2>&1 | tail -1

    t1=$(ts)

    local elapsed
    elapsed=$(python3 -c "print(f'{$t1 - $t0:.1f}')")
    echo "  -> ${elapsed}s"

    echo "$label,uv_warm,$elapsed" >> "$RESULTS_FILE"
    rm -rf "$venv" "$BENCH_BASE/reqs-warm-$label.txt"
}

# ---------------------------------------------------------------------------
# Benchmark each workload one at a time, cleaning between them
# ---------------------------------------------------------------------------

bench_workload() {
    local label="$1"
    local deps="$2"

    echo "--- $label ---"
    echo "  disk free: $(disk_free)"

    # 1. uv cold start (empty uv cache)
    run_uvx_cold "$label" "$deps"

    # 2. uv warm start (uv cache populated from cold run)
    run_uvx_warm "$label" "$deps"

    # Clean uv cache + venvs to free disk
    rm -rf "$HOME/.cache/uv" 2>/dev/null || true
    rm -rf "$BENCH_BASE"/* 2>/dev/null || true

    # 3. zerostart cold start (empty zerostart cache)
    run_zs_cold "$label" "$deps"

    # 4. zerostart warm start (cache populated from cold run)
    run_zs_warm "$label" "$deps"

    # Clean everything for next workload
    clean_all_caches
    echo ""
}

echo "============================================================"
echo "BENCHMARKS"
echo "============================================================"
echo ""

# Run torch and vllm (the two with README cold start claims) + triton (quick)
# Skip transformers/diffusers to save disk — they share CUDA libs with torch
bench_workload "torch" "torch"
bench_workload "triton" "triton"
bench_workload "vllm" "vllm"

echo "============================================================"
echo "SECTION 2: accelerate() Model Loading"
echo "============================================================"
echo ""

# Install deps for accelerate test
pip install -q accelerate transformers safetensors 2>&1 | tail -1
# Re-install zerostart SDK (may have been removed by cache cleanup)
cd "$PROJECT_DIR/python" && pip install -q -e . 2>&1 | tail -1
cd "$PROJECT_DIR"

python3 << 'PYEOF'
import subprocess, sys, os, time

# Test 1: Baseline from_pretrained
print("--- Baseline from_pretrained ---")
script = """
import time, os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
t0 = time.monotonic()
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", device_map="cuda")
t1 = time.monotonic()
print(f"BASELINE: {t1-t0:.2f}s")
"""
r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)
print(r.stdout.strip())
if r.returncode != 0:
    print("ERR:", r.stderr[-500:])

# Test 2: accelerate() first load (cold cache)
print("")
print("--- accelerate() first load ---")
import shutil
for d in ["/tmp/zs-models", os.path.expanduser("~/.cache/zerostart/models")]:
    shutil.rmtree(d, ignore_errors=True)

script = """
import time, os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
t0 = time.monotonic()
import zerostart
zerostart.accelerate()
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", device_map="cuda")
t1 = time.monotonic()
print(f"ACCELERATE_COLD: {t1-t0:.2f}s")
"""
r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300,
                    env={**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", "")})
print(r.stdout.strip())
if r.returncode != 0:
    print("ERR:", r.stderr[-500:])

# Test 3: accelerate() cached load
print("")
print("--- accelerate() cached load ---")
r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300,
                    env={**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", "")})
print(r.stdout.strip())
if r.returncode != 0:
    print("ERR:", r.stderr[-500:])
PYEOF

echo ""
echo "============================================================"
echo "RESULTS"
echo "============================================================"
echo ""
cat "$RESULTS_FILE"
echo ""

# Summary table
python3 << PYEOF
import csv

data = {}
with open("$RESULTS_FILE") as f:
    for row in csv.DictReader(f):
        wl = row["workload"]
        if wl not in data:
            data[wl] = {}
        data[wl][row["method"]] = float(row["time_s"])

print("Cold Start:")
print(f"  {'Workload':<16} {'zerostart':>10} {'uv':>10} {'Speedup':>10}")
for wl in ["torch", "triton", "vllm"]:
    if wl in data:
        zs = data[wl].get("zs_cold", 0)
        uv = data[wl].get("uv_cold", 0)
        sp = f"{uv/zs:.1f}x" if zs > 0 and uv > 0 else "—"
        print(f"  {wl:<16} {zs:>9.1f}s {uv:>9.1f}s {sp:>10}")

print("")
print("Warm Start:")
print(f"  {'Workload':<16} {'zerostart':>10} {'uv':>10} {'Speedup':>10}")
for wl in ["torch", "triton", "vllm"]:
    if wl in data:
        zs = data[wl].get("zs_warm", 0)
        uv = data[wl].get("uv_warm", 0)
        sp = f"{uv/zs:.1f}x" if zs > 0 and uv > 0 else "—"
        print(f"  {wl:<16} {zs:>9.1f}s {uv:>9.1f}s {sp:>10}")
PYEOF

echo ""
echo "Done. Raw data in $RESULTS_FILE"

rm -rf "$BENCH_BASE"
