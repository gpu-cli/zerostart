#!/usr/bin/env bash
#
# Benchmark: How long does torch take to install via different methods?
#
# Run on GPU pod: gpu run "bash tools/test-torch-install-time.sh"
#
set -euo pipefail

UV="$HOME/.local/bin/uv"
if ! [ -x "$UV" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    UV="$HOME/.local/bin/uv"
fi

echo "uv: $($UV --version)"
WORKDIR=$(mktemp -d)

# Build zs-fast-wheel if available
ZS_FAST_WHEEL=""
if [ -f "crates/zs-fast-wheel/Cargo.toml" ]; then
    echo "Building zs-fast-wheel..."
    cd crates && cargo build -p zs-fast-wheel --release 2>&1 | tail -1
    ZS_FAST_WHEEL="$(pwd)/target/release/zs-fast-wheel"
    cd ..
fi

TORCH_URL="https://files.pythonhosted.org/packages/torch/2.5.0/torch-2.5.0-cp310-cp310-manylinux1_x86_64.whl"

echo ""
echo "=== Method 1: uv pip install torch (cold cache, from PyPI) ==="
$UV cache clean torch 2>/dev/null || true
$UV venv "$WORKDIR/venv1" --python 3.10 2>/dev/null
echo "Starting..."
time $UV pip install --python "$WORKDIR/venv1/bin/python" "torch==2.5.0" --no-deps 2>&1
echo ""

echo "=== Method 2: uv pip install torch (warm cache) ==="
rm -rf "$WORKDIR/venv1"
$UV venv "$WORKDIR/venv1" --python 3.10 2>/dev/null
echo "Starting..."
time $UV pip install --python "$WORKDIR/venv1/bin/python" "torch==2.5.0" --no-deps 2>&1
echo ""

if [ -n "$ZS_FAST_WHEEL" ]; then
    echo "=== Method 3: zs-fast-wheel streaming extract ==="
    mkdir -p "$WORKDIR/fast-extract"
    echo "Starting..."
    time $ZS_FAST_WHEEL fetch "$TORCH_URL" --output "$WORKDIR/fast-extract" --parallel 8 2>&1
    echo ""
    echo "Extracted files: $(find "$WORKDIR/fast-extract" -type f | wc -l)"
    echo "Total size: $(du -sh "$WORKDIR/fast-extract" | cut -f1)"
fi

echo ""
echo "=== Method 4: uv pip install from local .whl (pre-downloaded) ==="
# Download .whl first
echo "Downloading torch .whl to local disk..."
time curl -sSL -o "$WORKDIR/torch.whl" \
    "https://download.pytorch.org/whl/cpu/torch-2.5.0%2Bcpu-cp310-cp310-linux_x86_64.whl" 2>&1 \
    || time $UV pip download torch==2.5.0 --no-deps -d "$WORKDIR" 2>&1 \
    || echo "Download failed, skipping"

if [ -f "$WORKDIR/torch.whl" ] || ls "$WORKDIR/"torch*.whl 2>/dev/null; then
    WHEEL=$(ls "$WORKDIR/"torch*.whl 2>/dev/null | head -1)
    echo "Wheel size: $(du -h "$WHEEL" | cut -f1)"
    rm -rf "$WORKDIR/venv1"
    $UV venv "$WORKDIR/venv1" --python 3.10 2>/dev/null
    echo "Installing from local .whl..."
    time $UV pip install --python "$WORKDIR/venv1/bin/python" "$WHEEL" --no-deps 2>&1
fi

echo ""
echo "=== Summary ==="
echo "Method 1 (cold uv):   download + extract + install (sequential stream)"
echo "Method 2 (warm uv):   cache hit, just link"
echo "Method 3 (fast-wheel): parallel Range requests, overlapped extract"
echo "Method 4 (local whl):  already on disk, uv extract + install"
echo ""
echo "The win from zs-fast-wheel is in Method 3 vs Method 1's download time."
echo "For Tier 2 (volume cache), we want Method 2's speed."

rm -rf "$WORKDIR"
