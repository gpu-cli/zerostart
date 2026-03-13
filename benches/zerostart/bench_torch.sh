#!/usr/bin/env bash
# Torch-only benchmark: zerostart (Rust binary) vs uv on the SAME pod
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PATH="$SCRIPT_DIR/bin:$PATH"

ZS_BIN="$SCRIPT_DIR/bin/zerostart-linux-x86_64"
chmod +x "$ZS_BIN" 2>/dev/null || true

# Use /gpu-cli-workspaces for cache (more disk space than /)
export ZEROSTART_CACHE="/gpu-cli-workspaces/.cache/zerostart"

disk() { df -h /gpu-cli-workspaces 2>/dev/null | tail -1 | awk '{print $4}'; }

echo "============================================================"
echo "  TORCH-ONLY Benchmark — Rust binary (same pod)"
echo "  $(date -u)"
echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "  Root disk: $(df -h / | tail -1 | awk '{print $4}') free"
echo "  Workspace disk: $(disk) free"
echo "============================================================"
echo ""

# ===============================================================
# BASELINE: uv COLD (run first to avoid cache interference)
# ===============================================================
echo "=== [baseline] COLD (uv, no cache) ==="
rm -rf /tmp/zs-bench-venv 2>/dev/null || true
uv cache clean 2>/dev/null || true

T_START=$(date +%s%3N)
uv venv /tmp/zs-bench-venv --python python3 -q 2>&1
echo "  [uv] Installing torch..."
UV_T0=$(date +%s%3N)
uv pip install --python /tmp/zs-bench-venv/bin/python "torch>=2.5" -q 2>&1 | tail -5
UV_T1=$(date +%s%3N)
echo "  [uv] Install took $(( (UV_T1 - UV_T0) / 1000 ))s"

BENCH_T_START=$T_START /tmp/zs-bench-venv/bin/python "$SCRIPT_DIR/torch_only.py"
echo ""

# ===============================================================
# BASELINE: uv WARM
# ===============================================================
echo "=== [baseline] WARM (uv cached) ==="
rm -rf /tmp/zs-bench-venv 2>/dev/null || true
T_START=$(date +%s%3N)
uv venv /tmp/zs-bench-venv --python python3 -q 2>&1
UV_T0=$(date +%s%3N)
uv pip install --python /tmp/zs-bench-venv/bin/python "torch>=2.5" -q 2>&1 | tail -5
UV_T1=$(date +%s%3N)
echo "  [uv] Install took $(( (UV_T1 - UV_T0) / 1000 ))s"

BENCH_T_START=$T_START /tmp/zs-bench-venv/bin/python "$SCRIPT_DIR/torch_only.py"
echo ""

# Clean up uv to free disk
rm -rf /tmp/zs-bench-venv 2>/dev/null || true
uv cache clean 2>/dev/null || true
echo "  Disk after uv cleanup: $(disk)"
echo ""

# ===============================================================
# ZEROSTART: COLD (no cache at all)
# ===============================================================
echo "=== [zerostart] COLD (no cache) ==="
rm -rf "$ZEROSTART_CACHE" 2>/dev/null || true

T_START=$(date +%s%3N)
BENCH_T_START=$T_START $ZS_BIN run -v -p "torch>=2.5" "$SCRIPT_DIR/torch_only.py" 2>&1
echo "  Disk after: $(disk)"
echo ""

# ===============================================================
# ZEROSTART: WARM (env + shared cache)
# ===============================================================
echo "=== [zerostart] WARM (cached env) ==="
T_START=$(date +%s%3N)
BENCH_T_START=$T_START $ZS_BIN run -p "torch>=2.5" "$SCRIPT_DIR/torch_only.py" 2>&1
echo ""

# ===============================================================
# ZEROSTART: COLD INSTALL (env deleted, shared cache warm)
# ===============================================================
echo "=== [zerostart] COLD INSTALL (env deleted, shared cache warm) ==="
rm -rf "$ZEROSTART_CACHE/envs" 2>/dev/null || true
T_START=$(date +%s%3N)
BENCH_T_START=$T_START $ZS_BIN run -v -p "torch>=2.5" "$SCRIPT_DIR/torch_only.py" 2>&1
echo ""

echo "Done."
