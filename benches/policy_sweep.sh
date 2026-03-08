#!/usr/bin/env bash
# Policy Sweep: find optimal concurrency and compare daemon-only vs uv
#
# Tests:
# 1. Concurrency sweep: -j 2,4,8,16,32 on ML workload
# 2. Daemon-only vs uv-only for small packages
# 3. Mixed workload: all packages through daemon vs uv for small + daemon for large
#
# Usage: gpu run "bash benches/policy_sweep.sh"

set -euo pipefail

WARM_BIN="./bin/zs-fast-wheel-linux-x86_64"
chmod +x "$WARM_BIN"
UV="$HOME/.local/bin/uv"
PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
RESULTS="benches/results/policy_sweep.csv"
mkdir -p benches/results

timestamp() { python3 -c "import time; print(f'{time.time():.6f}')"; }
elapsed_ms() { python3 -c "print(int(($2 - $1) * 1000))"; }

echo "========================================"
echo "  Policy Sweep"
echo "  Python: $PY_VERSION"
echo "  $(date)"
echo "========================================"

echo "test,config,time_ms,wheels,files,size_mb" > "$RESULTS"

# ─── Test 1: Concurrency sweep on ML workload ───────────────────────────

ML_DEPS="torch
transformers
tokenizers
safetensors
accelerate
numpy
pyyaml
requests
tqdm
packaging"

echo ""
echo "=== Test 1: Concurrency Sweep (ML workload) ==="

for j in 2 4 8 16 32; do
    sp="/tmp/sweep-j${j}-$$"
    rm -rf "$sp"; mkdir -p "$sp"

    echo -n "  -j $j: "
    t0=$(timestamp)
    $WARM_BIN warm --requirements "$ML_DEPS" --site-packages "$sp" --python-version "$PY_VERSION" -j "$j" 2>/dev/null
    t1=$(timestamp)
    ms=$(elapsed_ms "$t0" "$t1")

    files=$(find "$sp" -type f | wc -l)
    size=$(du -sm "$sp" | cut -f1)
    wheels=$(ls -d "$sp"/*.dist-info 2>/dev/null | wc -l)
    echo "${ms}ms ($files files, ${size}MB)"

    echo "concurrency,j${j},$ms,$wheels,$files,$size" >> "$RESULTS"
    rm -rf "$sp"
done

# ─── Test 2: Small packages — daemon vs uv ──────────────────────────────

SMALL_DEPS="six
idna
certifi
urllib3
charset-normalizer
requests
packaging
tqdm
pyyaml
filelock
jinja2
markupsafe"

echo ""
echo "=== Test 2: Small Packages — daemon-only vs uv ==="

# Daemon-only
sp="/tmp/sweep-daemon-small-$$"
rm -rf "$sp"; mkdir -p "$sp"
echo -n "  daemon-only: "
t0=$(timestamp)
$WARM_BIN warm --requirements "$SMALL_DEPS" --site-packages "$sp" --python-version "$PY_VERSION" -j 8 2>/dev/null
t1=$(timestamp)
ms=$(elapsed_ms "$t0" "$t1")
files=$(find "$sp" -type f | wc -l)
echo "${ms}ms ($files files)"
echo "small_pkgs,daemon,$ms,12,$files,$(du -sm "$sp" | cut -f1)" >> "$RESULTS"
rm -rf "$sp"

# uv-only
venv="/tmp/sweep-uv-small-$$"
rm -rf "$venv"
$UV venv "$venv" --python "$PY_VERSION" -q 2>/dev/null || $UV venv "$venv" -q
echo "$SMALL_DEPS" > /tmp/sweep-reqs-$$.txt
echo -n "  uv-only:     "
t0=$(timestamp)
$UV pip install -r /tmp/sweep-reqs-$$.txt --python "$venv/bin/python" -q 2>/dev/null
t1=$(timestamp)
ms=$(elapsed_ms "$t0" "$t1")
files=$(find "$venv/lib" -type f | wc -l)
echo "${ms}ms ($files files)"
echo "small_pkgs,uv,$ms,12,$files,$(du -sm "$venv/lib" | cut -f1)" >> "$RESULTS"
rm -rf "$venv" /tmp/sweep-reqs-$$.txt

# ─── Test 3: Hybrid vs daemon-only for full ML stack ────────────────────

echo ""
echo "=== Test 3: Full Stack — daemon-only vs hybrid (uv small + daemon large) ==="

# Define small vs large split (threshold: 1MB)
SMALL_ONLY="six
idna
certifi
urllib3
charset-normalizer
requests
packaging
tqdm
pyyaml
filelock
jinja2
markupsafe"

LARGE_ONLY="torch
transformers
tokenizers
safetensors
accelerate
numpy"

ALL_DEPS="$SMALL_ONLY
$LARGE_ONLY"

# Daemon-only (everything through warm)
sp="/tmp/sweep-daemon-all-$$"
rm -rf "$sp"; mkdir -p "$sp"
echo -n "  daemon-only: "
t0=$(timestamp)
$WARM_BIN warm --requirements "$ALL_DEPS" --site-packages "$sp" --python-version "$PY_VERSION" -j 8 2>/dev/null
t1=$(timestamp)
ms=$(elapsed_ms "$t0" "$t1")
files=$(find "$sp" -type f | wc -l)
size=$(du -sm "$sp" | cut -f1)
echo "${ms}ms ($files files, ${size}MB)"
echo "full_stack,daemon_only,$ms,0,$files,$size" >> "$RESULTS"
rm -rf "$sp"

# Hybrid: uv for small, daemon for large (parallel)
sp="/tmp/sweep-hybrid-$$"
rm -rf "$sp"; mkdir -p "$sp"
venv="/tmp/sweep-hybrid-venv-$$"
rm -rf "$venv"
$UV venv "$venv" --python "$PY_VERSION" -q 2>/dev/null || $UV venv "$venv" -q

echo -n "  hybrid:      "
t0=$(timestamp)

# Run uv and daemon in parallel
echo "$SMALL_ONLY" > /tmp/sweep-small-$$.txt
$UV pip install -r /tmp/sweep-small-$$.txt --python "$venv/bin/python" -q 2>/dev/null &
UV_PID=$!

$WARM_BIN warm --requirements "$LARGE_ONLY" --site-packages "$sp" --python-version "$PY_VERSION" -j 8 2>/dev/null &
WARM_PID=$!

wait "$UV_PID" 2>/dev/null || true
wait "$WARM_PID" 2>/dev/null || true

t1=$(timestamp)
ms=$(elapsed_ms "$t0" "$t1")
daemon_files=$(find "$sp" -type f | wc -l)
uv_files=$(find "$venv/lib" -type f 2>/dev/null | wc -l)
total_files=$((daemon_files + uv_files))
daemon_size=$(du -sm "$sp" | cut -f1)
uv_size=$(du -sm "$venv/lib" 2>/dev/null | cut -f1 || echo 0)
total_size=$((daemon_size + uv_size))
echo "${ms}ms ($total_files files, ${total_size}MB)"
echo "full_stack,hybrid,$ms,0,$total_files,$total_size" >> "$RESULTS"
rm -rf "$sp" "$venv" /tmp/sweep-small-$$.txt

# ─── Results ─────────────────────────────────────────────────────────────

echo ""
echo "========================================"
echo "  Results"
echo "========================================"
echo ""
cat "$RESULTS"
echo ""

python3 << 'PYEOF'
import csv

with open("benches/results/policy_sweep.csv") as f:
    rows = list(csv.DictReader(f))

print("--- Concurrency Sweep ---")
conc = [r for r in rows if r["test"] == "concurrency"]
for r in conc:
    print(f"  {r['config']:>4s}: {int(r['time_ms']):>6d}ms")
if conc:
    best = min(conc, key=lambda r: int(r["time_ms"]))
    print(f"  Best: {best['config']} ({best['time_ms']}ms)")

print()
print("--- Small Packages: daemon vs uv ---")
small = {r["config"]: int(r["time_ms"]) for r in rows if r["test"] == "small_pkgs"}
for k, v in small.items():
    print(f"  {k:>8s}: {v:>6d}ms")
if "daemon" in small and "uv" in small:
    if small["daemon"] < small["uv"]:
        print(f"  Winner: daemon ({small['uv']/small['daemon']:.1f}x faster)")
    else:
        print(f"  Winner: uv ({small['daemon']/small['uv']:.1f}x faster)")

print()
print("--- Full Stack: daemon-only vs hybrid ---")
full = {r["config"]: int(r["time_ms"]) for r in rows if r["test"] == "full_stack"}
for k, v in full.items():
    print(f"  {k:>12s}: {v:>6d}ms")
if "daemon_only" in full and "hybrid" in full:
    if full["daemon_only"] < full["hybrid"]:
        print(f"  Winner: daemon-only ({full['hybrid']/full['daemon_only']:.1f}x faster)")
    else:
        print(f"  Winner: hybrid ({full['daemon_only']/full['hybrid']:.1f}x faster)")

print()
print("--- Recommendations ---")
if conc:
    print(f"  Default concurrency: {best['config']}")
if "daemon" in small and "uv" in small:
    if small["daemon"] < small["uv"]:
        print("  Small packages: use daemon (faster even for small wheels)")
    else:
        ratio = small["daemon"] / small["uv"]
        if ratio > 2:
            print("  Small packages: use uv (significantly faster for pure-python wheels)")
        else:
            print("  Small packages: either works (marginal difference)")
if "daemon_only" in full and "hybrid" in full:
    diff = abs(full["daemon_only"] - full["hybrid"])
    if diff < 500:
        print("  Full stack: daemon-only (simpler, similar speed)")
    elif full["hybrid"] < full["daemon_only"]:
        print("  Full stack: hybrid (faster by offloading small wheels to uv)")
    else:
        print("  Full stack: daemon-only (faster than hybrid)")
PYEOF
