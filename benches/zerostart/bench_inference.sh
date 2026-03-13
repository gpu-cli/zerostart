#!/usr/bin/env bash
# Inference server benchmark: time-to-health-check and time-to-first-inference
# Compares uv cold vs zerostart cold on realistic inference server workload
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PATH="$SCRIPT_DIR/bin:$PATH"

ZS_BIN="$SCRIPT_DIR/bin/zerostart-linux-x86_64"
chmod +x "$ZS_BIN" 2>/dev/null || true

export ZEROSTART_CACHE="/gpu-cli-workspaces/.cache/zerostart"

disk() { df -h /gpu-cli-workspaces 2>/dev/null | tail -1 | awk '{print $4}'; }

UV_PKGS="torch>=2.5 transformers fastapi uvicorn accelerate"
ZS_PKGS=(-p "torch>=2.5" -p transformers -p fastapi -p uvicorn -p accelerate)

echo "============================================================"
echo "  Inference Server Benchmark (same pod)"
echo "  $(date -u)"
echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "  Disk free: $(disk)"
echo "  Packages: $UV_PKGS"
echo "============================================================"
echo ""

# ===============================================================
# TEST 1: uv COLD — install all packages, run inference server
# ===============================================================
echo "=== [uv] COLD (no cache) ==="
rm -rf /tmp/zs-bench-venv 2>/dev/null || true
uv cache clean 2>/dev/null || true

T_START=$(date +%s%3N)

# Create venv + install
uv venv /tmp/zs-bench-venv --python python3 -q 2>&1
UV_T0=$(date +%s%3N)
echo "  Installing: $UV_PKGS"
uv pip install --python /tmp/zs-bench-venv/bin/python $UV_PKGS -q 2>&1 | tail -5
UV_T1=$(date +%s%3N)
echo "  [uv] Install: $(( (UV_T1 - UV_T0) / 1000 ))s"

# Run the inference server
BENCH_T_START=$T_START /tmp/zs-bench-venv/bin/python "$SCRIPT_DIR/inference_server.py" 2>&1
echo "  Disk after: $(disk)"
echo ""

# ===============================================================
# TEST 2: uv WARM (cached)
# ===============================================================
echo "=== [uv] WARM (cached) ==="
rm -rf /tmp/zs-bench-venv 2>/dev/null || true

T_START=$(date +%s%3N)
uv venv /tmp/zs-bench-venv --python python3 -q 2>&1
UV_T0=$(date +%s%3N)
uv pip install --python /tmp/zs-bench-venv/bin/python $UV_PKGS -q 2>&1 | tail -5
UV_T1=$(date +%s%3N)
echo "  [uv] Install: $(( (UV_T1 - UV_T0) / 1000 ))s"

BENCH_T_START=$T_START /tmp/zs-bench-venv/bin/python "$SCRIPT_DIR/inference_server.py" 2>&1
echo ""

# Clean up uv
rm -rf /tmp/zs-bench-venv 2>/dev/null || true
uv cache clean 2>/dev/null || true
echo "  Disk after cleanup: $(disk)"
echo ""

# ===============================================================
# TEST 3: zerostart COLD (no cache)
# ===============================================================
echo "=== [zerostart] COLD (no cache) ==="
rm -rf "$ZEROSTART_CACHE" 2>/dev/null || true

T_START=$(date +%s%3N)
BENCH_T_START=$T_START $ZS_BIN run -v "${ZS_PKGS[@]}" "$SCRIPT_DIR/inference_server.py" 2>&1
echo "  Disk after: $(disk)"
echo ""

# ===============================================================
# TEST 4: zerostart WARM (cached env)
# ===============================================================
echo "=== [zerostart] WARM (cached env) ==="
T_START=$(date +%s%3N)
BENCH_T_START=$T_START $ZS_BIN run "${ZS_PKGS[@]}" "$SCRIPT_DIR/inference_server.py" 2>&1
echo ""

# ===============================================================
# TEST 5: zerostart COLD INSTALL (env deleted, shared cache warm)
# ===============================================================
echo "=== [zerostart] COLD INSTALL (shared cache warm) ==="
rm -rf "$ZEROSTART_CACHE/envs" 2>/dev/null || true
T_START=$(date +%s%3N)
BENCH_T_START=$T_START $ZS_BIN run -v "${ZS_PKGS[@]}" "$SCRIPT_DIR/inference_server.py" 2>&1
echo ""

echo "============================================================"
echo "  DONE — Results Summary"
echo "============================================================"
