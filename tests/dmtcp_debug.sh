#!/bin/bash
set -x

pkill -f dmtcp_coordinator 2>/dev/null || true
sleep 0.5

rm -rf /tmp/ckpt_test
mkdir -p /tmp/ckpt_test
cd /tmp/ckpt_test

# Start coordinator
dmtcp_coordinator --daemon -p 7779 --exit-on-last 2>&1
sleep 1

# Launch a simple sleeping Python
cat > /tmp/simple.py << 'EOF'
import time, os
print(f"PID={os.getpid()}, running...")
time.sleep(999)
EOF

dmtcp_launch -p 7779 python3 /tmp/simple.py &
DPID=$!
sleep 3

# Status
echo "=== STATUS ==="
dmtcp_command -p 7779 -s 2>&1

# Checkpoint
echo "=== CHECKPOINT ==="
dmtcp_command -p 7779 -c 2>&1
sleep 2

# Find checkpoint files
echo "=== CHECKPOINT FILES ==="
find /tmp -maxdepth 3 -name "ckpt_*" 2>/dev/null
find /tmp/ckpt_test -type f 2>/dev/null
find /root -maxdepth 2 -name "ckpt_*" 2>/dev/null
find / -maxdepth 3 -name "ckpt_*.dmtcp" 2>/dev/null | head -5
ls -la /tmp/ckpt_test/ 2>/dev/null

# Also check the gpu workspace
find /gpu-cli-workspaces -maxdepth 3 -name "ckpt_*" 2>/dev/null | head -5

# Kill
dmtcp_command -p 7779 -k 2>&1 || true
wait $DPID 2>/dev/null || true

echo "=== DONE ==="
