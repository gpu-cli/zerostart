#!/bin/bash
# Test 2.1: Trivial CRIU dump/restore using sleep
# Must run on Linux with CRIU installed (via gpu run)
set -euo pipefail

echo "=== Test: Trivial CRIU dump/restore ==="

# Check prerequisites
if [[ "$(uname)" != "Linux" ]]; then
    echo "SKIP: Not Linux"
    exit 0
fi

if ! command -v criu &>/dev/null; then
    echo "SKIP: criu not installed"
    exit 0
fi

# Find zs-snapshot binary
ZS_SNAPSHOT=""
for candidate in \
    ./crates/target/release/zs-snapshot \
    ./crates/target/debug/zs-snapshot \
    ./bin/zs-snapshot-linux-x86_64; do
    if [[ -x "$candidate" ]]; then
        ZS_SNAPSHOT="$candidate"
        break
    fi
done

if [[ -z "$ZS_SNAPSHOT" ]]; then
    echo "Building zs-snapshot..."
    cd crates && cargo build -p zs-snapshot --release 2>&1 && cd ..
    ZS_SNAPSHOT="./crates/target/release/zs-snapshot"
fi

echo "Using zs-snapshot: $ZS_SNAPSHOT"

# Run doctor first
echo "--- Doctor ---"
$ZS_SNAPSHOT doctor || true

# Create test dirs
CACHE_DIR=$(mktemp -d)
RUN_DIR=$(mktemp -d)
INTENT_HASH="test-trivial-$(date +%s)"

echo "Cache dir: $CACHE_DIR"
echo "Run dir: $RUN_DIR"

# Start a sleep process
sleep 1000 &
SLEEP_PID=$!
echo "Started sleep PID: $SLEEP_PID"

# Write metadata file
METADATA_FILE=$(mktemp)
cat > "$METADATA_FILE" <<EOF
{
    "intent_hash": "$INTENT_HASH",
    "env_fingerprint": "test",
    "entrypoint": "sleep",
    "argv": ["sleep", "1000"],
    "python_version": "3.11.0",
    "platform": "x86_64-linux"
}
EOF

# Dump with --leave-running
echo "--- Dump ---"
$ZS_SNAPSHOT dump \
    --intent-hash "$INTENT_HASH" \
    --pid "$SLEEP_PID" \
    --metadata "$METADATA_FILE" \
    --cache-dir "$CACHE_DIR"

echo "Dump succeeded"

# Verify metadata was written
if [[ ! -f "$CACHE_DIR/$INTENT_HASH/metadata.json" ]]; then
    echo "FAIL: metadata.json not found"
    kill $SLEEP_PID 2>/dev/null || true
    exit 1
fi
echo "Metadata written OK"

# Kill original process
kill $SLEEP_PID
wait $SLEEP_PID 2>/dev/null || true
echo "Original process killed"

# Verify it's dead
if kill -0 $SLEEP_PID 2>/dev/null; then
    echo "FAIL: Original process still alive"
    exit 1
fi

# Restore
PIDFILE="$RUN_DIR/restored.pid"
echo "--- Restore ---"
$ZS_SNAPSHOT restore \
    --intent-hash "$INTENT_HASH" \
    --pidfile "$PIDFILE" \
    --cache-dir "$CACHE_DIR"

echo "Restore succeeded"

# Verify restored PID
if [[ ! -f "$PIDFILE" ]]; then
    echo "FAIL: pidfile not found"
    exit 1
fi

RESTORED_PID=$(cat "$PIDFILE")
echo "Restored PID: $RESTORED_PID"

if ! kill -0 "$RESTORED_PID" 2>/dev/null; then
    echo "FAIL: Restored process not alive"
    exit 1
fi

echo "Restored process is alive"

# Cleanup
kill "$RESTORED_PID" 2>/dev/null || true
rm -rf "$CACHE_DIR" "$RUN_DIR" "$METADATA_FILE"

echo "=== PASS: Trivial CRIU dump/restore ==="
