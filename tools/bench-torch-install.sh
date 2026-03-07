#!/usr/bin/env bash
#
# Benchmark: torch install methods on GPU pod
#
# Compares: uv cold, uv warm, zs-fast-wheel streaming, zs-fast-wheel + uv local whl
#
# Run: gpu run "bash tools/bench-torch-install.sh"
#
set -euo pipefail

UV="$HOME/.local/bin/uv"
ZS="$(pwd)/bin/zs-fast-wheel-linux-x86_64"
RESULTS_DIR="benches/results"
mkdir -p "$RESULTS_DIR"

echo "========================================"
echo "  torch install benchmark"
echo "========================================"
echo "uv:  $($UV --version)"
echo "zs:  $($ZS --help 2>&1 | head -1)"
echo "cpu: $(nproc) cores"
echo "net: $(curl -so /dev/null -w '%{speed_download}' https://files.pythonhosted.org/packages/b7/ce/149a00dd41f10bc29e5921b496af8b574d8413afcd5e30dfa0ed46c2cc5e/six-1.17.0-py2.py3-none-any.whl | awk '{printf "%.1f MB/s", $1/1048576}')"
echo ""

# Find the right torch wheel URL for this platform
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.minor}')")
TORCH_WHL="torch-2.5.0-cp3${PY_VER}-cp3${PY_VER}-manylinux1_x86_64.whl"
# Use CPU torch to avoid CUDA deps for clean benchmark
TORCH_URL="https://download.pytorch.org/whl/cpu/torch-2.5.0%2Bcpu-cp3${PY_VER}-cp3${PY_VER}-linux_x86_64.whl"

echo "torch URL: $TORCH_URL"
echo "python:    3.${PY_VER}"
echo ""

WORKDIR=$(mktemp -d)

# ============================================
# Method 1: uv cold install (no cache)
# ============================================
echo "========================================" | tee -a "$RESULTS_DIR/bench.log"
echo "Method 1: uv cold install from PyPI" | tee -a "$RESULTS_DIR/bench.log"
echo "========================================" | tee -a "$RESULTS_DIR/bench.log"
$UV cache clean torch 2>/dev/null || true
$UV venv "$WORKDIR/venv" --python "3.${PY_VER}" 2>/dev/null
START=$(date +%s%N)
$UV pip install --python "$WORKDIR/venv/bin/python" "torch==2.5.0+cpu" --index-url https://download.pytorch.org/whl/cpu --no-deps 2>&1
END=$(date +%s%N)
ELAPSED_1=$(( (END - START) / 1000000 ))
echo "Time: ${ELAPSED_1}ms" | tee -a "$RESULTS_DIR/bench.log"
echo ""

# ============================================
# Method 2: uv warm cache install
# ============================================
echo "========================================" | tee -a "$RESULTS_DIR/bench.log"
echo "Method 2: uv warm cache install" | tee -a "$RESULTS_DIR/bench.log"
echo "========================================" | tee -a "$RESULTS_DIR/bench.log"
rm -rf "$WORKDIR/venv"
$UV venv "$WORKDIR/venv" --python "3.${PY_VER}" 2>/dev/null
START=$(date +%s%N)
$UV pip install --python "$WORKDIR/venv/bin/python" "torch==2.5.0+cpu" --index-url https://download.pytorch.org/whl/cpu --no-deps 2>&1
END=$(date +%s%N)
ELAPSED_2=$(( (END - START) / 1000000 ))
echo "Time: ${ELAPSED_2}ms" | tee -a "$RESULTS_DIR/bench.log"
echo ""

# ============================================
# Method 3: zs-fast-wheel streaming extract
# ============================================
echo "========================================" | tee -a "$RESULTS_DIR/bench.log"
echo "Method 3: zs-fast-wheel --stream" | tee -a "$RESULTS_DIR/bench.log"
echo "========================================" | tee -a "$RESULTS_DIR/bench.log"
rm -rf "$WORKDIR/zs-extract"
mkdir -p "$WORKDIR/zs-extract"
START=$(date +%s%N)
$ZS "$TORCH_URL" --target "$WORKDIR/zs-extract" --stream true -j 16 --benchmark 2>&1
END=$(date +%s%N)
ELAPSED_3=$(( (END - START) / 1000000 ))
echo "Time: ${ELAPSED_3}ms" | tee -a "$RESULTS_DIR/bench.log"
echo "Files: $(find "$WORKDIR/zs-extract" -type f | wc -l)"
echo "Size:  $(du -sh "$WORKDIR/zs-extract" | cut -f1)"
echo ""

# ============================================
# Method 4: zs-fast-wheel non-streaming (download whole, extract parallel)
# ============================================
echo "========================================" | tee -a "$RESULTS_DIR/bench.log"
echo "Method 4: zs-fast-wheel download+extract" | tee -a "$RESULTS_DIR/bench.log"
echo "========================================" | tee -a "$RESULTS_DIR/bench.log"
rm -rf "$WORKDIR/zs-extract2"
mkdir -p "$WORKDIR/zs-extract2"
START=$(date +%s%N)
$ZS "$TORCH_URL" --target "$WORKDIR/zs-extract2" -j 16 --benchmark 2>&1
END=$(date +%s%N)
ELAPSED_4=$(( (END - START) / 1000000 ))
echo "Time: ${ELAPSED_4}ms" | tee -a "$RESULTS_DIR/bench.log"
echo ""

# ============================================
# Method 5: curl download .whl + uv install from local
# ============================================
echo "========================================" | tee -a "$RESULTS_DIR/bench.log"
echo "Method 5: curl .whl + uv pip install local" | tee -a "$RESULTS_DIR/bench.log"
echo "========================================" | tee -a "$RESULTS_DIR/bench.log"
START=$(date +%s%N)
# Download phase
curl -sSL -o "$WORKDIR/torch.whl" "$TORCH_URL"
DL_END=$(date +%s%N)
DL_MS=$(( (DL_END - START) / 1000000 ))
echo "Download: ${DL_MS}ms ($(du -h "$WORKDIR/torch.whl" | cut -f1))"
# Install phase
rm -rf "$WORKDIR/venv"
$UV venv "$WORKDIR/venv" --python "3.${PY_VER}" 2>/dev/null
$UV pip install --python "$WORKDIR/venv/bin/python" "$WORKDIR/torch.whl" --no-deps 2>&1
END=$(date +%s%N)
ELAPSED_5=$(( (END - START) / 1000000 ))
INSTALL_MS=$(( ELAPSED_5 - DL_MS ))
echo "Install:  ${INSTALL_MS}ms"
echo "Total:    ${ELAPSED_5}ms" | tee -a "$RESULTS_DIR/bench.log"
echo ""

# ============================================
# Method 6: zs-fast-wheel download .whl + uv install from local
# (simulates: fast download then let uv handle install)
# ============================================
echo "========================================" | tee -a "$RESULTS_DIR/bench.log"
echo "Method 6: zs-fast-wheel .whl + uv local" | tee -a "$RESULTS_DIR/bench.log"
echo "  (would need zs to output .whl — skip if not supported)" | tee -a "$RESULTS_DIR/bench.log"
echo "========================================" | tee -a "$RESULTS_DIR/bench.log"
echo "(skipped — zs-fast-wheel extracts directly, doesn't produce .whl)"
echo ""

# ============================================
# Summary
# ============================================
echo "========================================" | tee -a "$RESULTS_DIR/bench.log"
echo "  SUMMARY" | tee -a "$RESULTS_DIR/bench.log"
echo "========================================" | tee -a "$RESULTS_DIR/bench.log"
echo "" | tee -a "$RESULTS_DIR/bench.log"
printf "%-40s %8s\n" "Method" "Time" | tee -a "$RESULTS_DIR/bench.log"
printf "%-40s %8s\n" "----------------------------------------" "--------" | tee -a "$RESULTS_DIR/bench.log"
printf "%-40s %7dms\n" "1. uv cold (PyPI stream+extract+install)" "$ELAPSED_1" | tee -a "$RESULTS_DIR/bench.log"
printf "%-40s %7dms\n" "2. uv warm cache (link only)" "$ELAPSED_2" | tee -a "$RESULTS_DIR/bench.log"
printf "%-40s %7dms\n" "3. zs-fast-wheel --stream (Range reqs)" "$ELAPSED_3" | tee -a "$RESULTS_DIR/bench.log"
printf "%-40s %7dms\n" "4. zs-fast-wheel (download+extract)" "$ELAPSED_4" | tee -a "$RESULTS_DIR/bench.log"
printf "%-40s %7dms\n" "5. curl .whl + uv local install" "$ELAPSED_5" | tee -a "$RESULTS_DIR/bench.log"
echo "" | tee -a "$RESULTS_DIR/bench.log"
echo "Speedup (zs-stream vs uv cold): $(echo "scale=1; $ELAPSED_1 / $ELAPSED_3" | bc)x" | tee -a "$RESULTS_DIR/bench.log"
echo "" | tee -a "$RESULTS_DIR/bench.log"

# Verify torch actually works
echo "=== Verify ===" | tee -a "$RESULTS_DIR/bench.log"
"$WORKDIR/venv/bin/python" -c "import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>&1 | tee -a "$RESULTS_DIR/bench.log"

rm -rf "$WORKDIR"
echo "Done."
