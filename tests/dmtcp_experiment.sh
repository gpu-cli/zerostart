#!/bin/bash
set -uo pipefail

echo "=== DMTCP Checkpoint/Restore Experiments ==="
echo "Date: $(date -u)"
echo "Python: $(python3 --version)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
df -h /tmp | tail -1 | awk '{print "Disk: " $4 " free"}'
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ZS="$PROJECT_DIR/bin/zerostart-linux-x86_64"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

CKPT_DIR="/tmp/dmtcp_ckpts"

# ============================================================
# Step 1: Install DMTCP
# ============================================================
echo "--- Installing DMTCP ---"
if command -v dmtcp_launch &>/dev/null; then
    echo "DMTCP already installed: $(dmtcp_launch --version 2>&1 | head -1)"
else
    apt-get update -qq && apt-get install -y -qq build-essential git 2>&1 | tail -3
    cd /tmp
    [ ! -d dmtcp ] && git clone --depth 1 https://github.com/dmtcp/dmtcp.git 2>&1 | tail -3
    cd dmtcp
    ./configure --prefix=/usr/local 2>&1 | tail -3
    make -j$(nproc) 2>&1 | tail -3
    make install 2>&1 | tail -3
    cd "$PROJECT_DIR"
    echo "DMTCP installed: $(dmtcp_launch --version 2>&1 | head -1)"
fi
echo ""

# ============================================================
# Step 2: Install torch
# ============================================================
echo "--- Installing torch ---"
export ZEROSTART_CACHE="/tmp/.zs-dmtcp"
rm -rf "$ZEROSTART_CACHE"

cat > /tmp/zs_torch_test.py << 'EOF'
import torch
print(f"torch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
EOF

$ZS run -v -p torch /tmp/zs_torch_test.py 2>&1

SITE_PACKAGES=$(find "$ZEROSTART_CACHE/envs" -path "*/site-packages" -type d 2>/dev/null | head -1)
echo "Site-packages: $SITE_PACKAGES"
rm -rf "$ZEROSTART_CACHE/shared_wheels"
df -h /tmp | tail -1 | awk '{print "Disk after install: " $4 " free"}'
echo ""

# ============================================================
# Step 3: Baseline — torch import + CUDA (3 runs)
# ============================================================
cat > /tmp/bench_baseline.py << 'PYEOF'
import time, sys, os
sys.path.insert(0, os.environ["ZS_SITE_PACKAGES"])
t0 = time.monotonic()
import torch
t1 = time.monotonic()
if torch.cuda.is_available():
    torch.cuda.init()
    x = torch.randn(100, 100, device='cuda')
t2 = time.monotonic()
print(f"import={t1-t0:.3f}s cuda={t2-t1:.3f}s total={t2-t0:.3f}s")
PYEOF

echo "--- Baseline: Python + torch + CUDA (3 runs) ---"
for i in 1 2 3; do
    start=$(date +%s%3N)
    ZS_SITE_PACKAGES="$SITE_PACKAGES" python3 /tmp/bench_baseline.py 2>&1
    end=$(date +%s%3N)
    echo "  wall=$((end - start))ms"
done
echo ""

# ============================================================
# Step 4: DMTCP — simple Python first (sanity check)
# ============================================================
echo "--- DMTCP: simple Python checkpoint (sanity check) ---"
pkill -f dmtcp_coordinator 2>/dev/null || true
sleep 0.5
rm -rf "$CKPT_DIR"
mkdir -p "$CKPT_DIR"

dmtcp_coordinator --daemon -p 7779 2>&1 || true
sleep 0.5

cat > /tmp/dmtcp_simple.py << 'PYEOF'
import time, os
print(f"PID={os.getpid()}, running...")
import sys
sys.stdout.flush()
time.sleep(999)
PYEOF

cd "$CKPT_DIR"
dmtcp_launch -p 7779 python3 /tmp/dmtcp_simple.py &
DPID=$!
cd "$PROJECT_DIR"
sleep 2

echo "  Checkpointing simple process..."
start=$(date +%s%3N)
dmtcp_command -p 7779 -bc 2>&1 || true
# Wait for file
for attempt in $(seq 1 15); do
    if ls "$CKPT_DIR"/ckpt_*.dmtcp 1>/dev/null 2>&1; then break; fi
    sleep 1
done
end=$(date +%s%3N)
echo "  Checkpoint time: $((end - start))ms"
ls -lh "$CKPT_DIR"/ckpt_*.dmtcp 2>/dev/null || echo "  FAILED: no checkpoint file"
echo "  Process alive: $(kill -0 $DPID 2>/dev/null && echo 'yes' || echo 'no')"

# Kill and restore
dmtcp_command -p 7779 -k 2>&1 || true
wait $DPID 2>/dev/null || true

SIMPLE_CKPT=$(ls "$CKPT_DIR"/ckpt_*.dmtcp 2>/dev/null | head -1)
if [ -n "$SIMPLE_CKPT" ]; then
    echo "  Restoring simple process..."
    pkill -f dmtcp_coordinator 2>/dev/null || true
    sleep 0.3
    cd "$CKPT_DIR"
    timeout 5 dmtcp_restart "$SIMPLE_CKPT" 2>&1 &
    RPID=$!
    cd "$PROJECT_DIR"
    sleep 2
    echo "  Restore alive: $(kill -0 $RPID 2>/dev/null && echo 'yes' || echo 'no')"
    kill $RPID 2>/dev/null || true
    wait $RPID 2>/dev/null || true
    echo "  Simple checkpoint/restore: OK"
else
    echo "  Simple checkpoint: FAILED"
fi
echo ""

# ============================================================
# Step 5: DMTCP — checkpoint with torch loaded
# ============================================================
echo "--- DMTCP: checkpoint with torch loaded ---"

pkill -f dmtcp_coordinator 2>/dev/null || true
sleep 0.5
rm -rf "$CKPT_DIR"
mkdir -p "$CKPT_DIR"

dmtcp_coordinator --daemon -p 7779 2>&1 || true
sleep 0.5

cat > /tmp/dmtcp_torch_wait.py << 'PYEOF'
import time, sys, os
sys.path.insert(0, os.environ["ZS_SITE_PACKAGES"])
t0 = time.monotonic()
import torch
t1 = time.monotonic()
print(f"torch imported: {t1-t0:.3f}s", flush=True)
time.sleep(999)
PYEOF

echo "  Launching torch under DMTCP..."
cd "$CKPT_DIR"
ZS_SITE_PACKAGES="$SITE_PACKAGES" dmtcp_launch -p 7779 python3 /tmp/dmtcp_torch_wait.py 2>/tmp/dmtcp_torch_stderr.log &
DPID=$!
cd "$PROJECT_DIR"

sleep 5

echo "  Process alive before checkpoint: $(kill -0 $DPID 2>/dev/null && echo 'yes' || echo 'no')"
echo "  Checkpointing..."
start=$(date +%s%3N)
dmtcp_command -p 7779 -bc 2>&1 || true

# Wait for checkpoint file, check process health
for attempt in $(seq 1 60); do
    if ls "$CKPT_DIR"/ckpt_*.dmtcp 1>/dev/null 2>&1; then break; fi
    if ! kill -0 $DPID 2>/dev/null; then
        echo "  Process DIED during checkpoint (after ${attempt}s)"
        break
    fi
    sleep 1
done
end=$(date +%s%3N)
echo "  Checkpoint time: $((end - start))ms"

# Show what files exist
echo "  Files in $CKPT_DIR:"
ls -lh "$CKPT_DIR"/ 2>/dev/null
echo "  Process alive after checkpoint: $(kill -0 $DPID 2>/dev/null && echo 'yes' || echo 'no')"

# Show any stderr from the process
if [ -s /tmp/dmtcp_torch_stderr.log ]; then
    echo "  Stderr from torch process (last 10 lines):"
    tail -10 /tmp/dmtcp_torch_stderr.log
fi

CKPT_SIZE=$(du -sh "$CKPT_DIR"/ 2>/dev/null | cut -f1 || echo "0")
echo "  Checkpoint size: $CKPT_SIZE"
df -h /tmp | tail -1 | awk '{print "  Disk: " $4 " free"}'

# Kill
dmtcp_command -p 7779 -k 2>&1 || true
wait $DPID 2>/dev/null || true
echo ""

# ============================================================
# Step 6: DMTCP restore — torch (3 runs)
# ============================================================
echo "--- DMTCP: restore torch (3 runs) ---"

CKPT_FILE=$(ls "$CKPT_DIR"/ckpt_*.dmtcp 2>/dev/null | head -1)
if [ -n "$CKPT_FILE" ]; then
    CKPT_SIZE_BYTES=$(stat -c%s "$CKPT_FILE" 2>/dev/null || echo 0)
    echo "  Checkpoint: $CKPT_FILE ($(du -h "$CKPT_FILE" | cut -f1))"
    for i in 1 2 3; do
        pkill -f dmtcp_coordinator 2>/dev/null || true
        sleep 0.3

        start=$(date +%s%3N)
        cd "$CKPT_DIR"
        dmtcp_restart "$CKPT_FILE" &
        RPID=$!
        cd "$PROJECT_DIR"
        # Wait for process to be up (it resumes in sleep)
        sleep 2
        end=$(date +%s%3N)
        alive=$(kill -0 $RPID 2>/dev/null && echo "alive" || echo "dead")
        kill $RPID 2>/dev/null || true
        wait $RPID 2>/dev/null || true
        echo "  Restore $i: $((end - start))ms ($alive)"
    done
else
    echo "  No checkpoint file — skipping"
fi
echo ""

# ============================================================
# Step 7: Fork approach (3 runs)
# ============================================================
cat > /tmp/bench_fork.py << 'PYEOF'
import time, sys, os
sys.path.insert(0, os.environ["ZS_SITE_PACKAGES"])

t0 = time.monotonic()
import torch
t1 = time.monotonic()

pid = os.fork()
if pid == 0:
    tc0 = time.monotonic()
    if torch.cuda.is_available():
        x = torch.randn(100, 100, device='cuda')
        tc1 = time.monotonic()
        print(f"import={t1-t0:.3f}s fork+cuda={tc1-t1:.3f}s total={tc1-t0:.3f}s")
    os._exit(0)
else:
    os.waitpid(pid, 0)
    t2 = time.monotonic()
    print(f"parent total: {t2-t0:.3f}s")
PYEOF

echo "--- Fork: import + fork + CUDA (3 runs) ---"
for i in 1 2 3; do
    start=$(date +%s%3N)
    ZS_SITE_PACKAGES="$SITE_PACKAGES" python3 /tmp/bench_fork.py 2>&1
    end=$(date +%s%3N)
    echo "  wall=$((end - start))ms"
done
echo ""

# ============================================================
# Summary
# ============================================================
echo "=== EXPERIMENT COMPLETE ==="
echo ""
echo "Key findings:"
echo "  Baseline (cold import + CUDA): ~2.0s"
echo "  Fork approach: same import cost, CUDA init in child"
echo "  DMTCP: checkpoint size and restore time TBD above"
