#!/bin/bash
# Cross-compile Rust binaries for Linux x86_64 (musl static).
# Run this BEFORE `gpu run` — pods use pre-built binaries from bin/.
#
# Usage: ./scripts/cross-compile.sh

set -euo pipefail

cd "$(dirname "$0")/.."

TARGET="x86_64-unknown-linux-musl"
OUTDIR="bin"

echo "=== Cross-compiling for $TARGET ==="
echo "Output: $OUTDIR/"
echo ""

# Build zs-fast-wheel (the main binary tests use)
echo "Building zs-fast-wheel..."
cd crates
cargo zigbuild --target "$TARGET" --release -p zs-fast-wheel
cd ..

cp "crates/target/$TARGET/release/zerostart" "$OUTDIR/zerostart-linux-x86_64"
chmod +x "$OUTDIR/zerostart-linux-x86_64"
# Also copy with old name for backwards compat
cp "crates/target/$TARGET/release/zerostart" "$OUTDIR/zs-fast-wheel-linux-x86_64"
chmod +x "$OUTDIR/zs-fast-wheel-linux-x86_64"

echo ""
echo "Done. Binaries in $OUTDIR/:"
ls -lh "$OUTDIR/"*-linux-x86_64

echo ""
echo "Next: gpu run \"bash tests/gpu_test_runner.sh\""
