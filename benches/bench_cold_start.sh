#!/usr/bin/env bash
# GPU Cold Start Benchmark: zs-fast-wheel warm vs uv pip install
#
# Measures end-to-end time from "boot" to packages ready on disk.
# This is the metric that matters for GPU cold start: how fast can
# we get wheels extracted to site-packages so Python can import them?
#
# Usage: gpu run "bash benches/bench_cold_start.sh"

set -euo pipefail

RESULTS_DIR="benches/results"
mkdir -p "$RESULTS_DIR"

WARM_BIN="./bin/zs-fast-wheel-linux-x86_64"
chmod +x "$WARM_BIN"
UV="$HOME/.local/bin/uv"

# Detect system Python version for correct wheel resolution
PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Detected Python: $PY_VERSION"

# Test sets
SMALL_DEPS="six
idna
certifi
urllib3
requests"

MEDIUM_DEPS="numpy
pandas
scikit-learn
tqdm
pyyaml
requests
packaging
filelock
jinja2"

# Full ML stack
ML_DEPS="torch
transformers
tokenizers
safetensors
accelerate
tqdm
pyyaml
requests
numpy
packaging
filelock
jinja2"

timestamp() {
    python3 -c "import time; print(f'{time.time():.6f}')"
}

elapsed_ms() {
    python3 -c "print(int(($2 - $1) * 1000))"
}

# ─── Benchmark: uv pip install (baseline) ───────────────────────────────

bench_uv_install() {
    local label="$1"
    local deps="$2"
    local venv_dir="/tmp/bench-uv-$$"

    echo "--- [$label] uv pip install (baseline) ---"

    # Create fresh venv
    rm -rf "$venv_dir"
    $UV venv "$venv_dir" --python "$PY_VERSION" -q 2>/dev/null || $UV venv "$venv_dir" -q

    # Write requirements
    local req_file="/tmp/bench-reqs-$$.txt"
    echo "$deps" > "$req_file"

    # Time the install (resolve + download + extract + link)
    local t0
    t0=$(timestamp)

    $UV pip install -r "$req_file" --python "$venv_dir/bin/python" -q 2>/dev/null

    local t1
    t1=$(timestamp)

    local ms
    ms=$(elapsed_ms "$t0" "$t1")

    # Count installed files
    local file_count
    file_count=$(find "$venv_dir/lib" -type f 2>/dev/null | wc -l)
    local size_mb
    size_mb=$(du -sm "$venv_dir/lib" 2>/dev/null | cut -f1)

    echo "  Time: ${ms}ms"
    echo "  Files: $file_count, Size: ${size_mb:-0}MB"

    # Cleanup
    rm -rf "$venv_dir" "$req_file"

    echo "$label,uv_install,$ms,$file_count,${size_mb:-0}" >> "$RESULTS_DIR/bench.csv"
}

# ─── Benchmark: zs-fast-wheel warm ──────────────────────────────────────

bench_warm() {
    local label="$1"
    local deps="$2"
    local sp_dir="/tmp/bench-warm-$$"

    echo "--- [$label] zs-fast-wheel warm ---"

    rm -rf "$sp_dir"
    mkdir -p "$sp_dir"

    # Time the warm (resolve + download + extract) — all in Rust, no Python
    local t0
    t0=$(timestamp)

    $WARM_BIN warm \
        --requirements "$deps" \
        --site-packages "$sp_dir" \
        --python-version "$PY_VERSION" \
        -j 8 2>&1 | grep -v "^$"

    local t1
    t1=$(timestamp)

    local ms
    ms=$(elapsed_ms "$t0" "$t1")

    # Count extracted files
    local file_count
    file_count=$(find "$sp_dir" -type f | wc -l)
    local size_mb
    size_mb=$(du -sm "$sp_dir" | cut -f1)

    echo "  Time: ${ms}ms"
    echo "  Files: $file_count, Size: ${size_mb}MB"

    # Cleanup
    rm -rf "$sp_dir"

    echo "$label,zs_warm,$ms,$file_count,$size_mb" >> "$RESULTS_DIR/bench.csv"
}

# ─── Main ────────────────────────────────────────────────────────────────

echo "========================================"
echo "  GPU Cold Start Benchmark"
echo "  $(date)"
echo "  Python: $PY_VERSION"
echo "========================================"
echo ""

# CSV header
echo "label,method,time_ms,files,size_mb" > "$RESULTS_DIR/bench.csv"

# Small deps
bench_uv_install "small" "$SMALL_DEPS"
echo ""
bench_warm "small" "$SMALL_DEPS"
echo ""

# Medium deps
bench_uv_install "medium" "$MEDIUM_DEPS"
echo ""
bench_warm "medium" "$MEDIUM_DEPS"
echo ""

# ML deps (the big one)
bench_uv_install "ml" "$ML_DEPS"
echo ""
bench_warm "ml" "$ML_DEPS"
echo ""

echo "========================================"
echo "  Results"
echo "========================================"
echo ""
cat "$RESULTS_DIR/bench.csv" | python3 -c "
import sys, csv
rows = list(csv.reader(sys.stdin))
widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
for r in rows:
    print('  '.join(r[i].ljust(widths[i]) for i in range(len(r))))
"
echo ""

# Compute speedups
python3 << 'PYEOF'
import csv

data = {}
with open("benches/results/bench.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = row["label"]
        if key not in data:
            data[key] = {}
        data[key][row["method"]] = int(row["time_ms"])

print("--- Speedup Summary ---")
for label in ["small", "medium", "ml"]:
    if label in data and "uv_install" in data[label] and "zs_warm" in data[label]:
        uv = data[label]["uv_install"]
        zs = data[label]["zs_warm"]
        speedup = uv / zs if zs > 0 else float("inf")
        print(f"  {label:8s}: uv={uv:>6d}ms  zs-warm={zs:>6d}ms  speedup={speedup:.1f}x")
PYEOF
