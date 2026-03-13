#!/usr/bin/env bash
# Progressive loading benchmark: time-to-first-CUDA-op
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PATH="$SCRIPT_DIR/bin:$PATH"

ZS_BIN="$SCRIPT_DIR/bin/zerostart-linux-x86_64"
chmod +x "$ZS_BIN" 2>/dev/null || true

export ZEROSTART_CACHE="/gpu-cli-workspaces/.cache/zerostart"

echo "============================================================"
echo "  Progressive Loading Benchmark (same pod)"
echo "  $(date -u)"
echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "============================================================"
echo ""

# ===============================================================
# TEST 1: uv baseline (cold) — total time start to CUDA op
# ===============================================================
echo "=== [baseline] uv cold ==="
rm -rf /tmp/zs-bench-venv 2>/dev/null || true
uv cache clean 2>/dev/null || true

T_START=$(date +%s%3N)
uv venv /tmp/zs-bench-venv --python python3 -q 2>&1
UV_T0=$(date +%s%3N)
uv pip install --python /tmp/zs-bench-venv/bin/python "torch>=2.5" -q 2>&1 | tail -3
UV_T1=$(date +%s%3N)
echo "  [uv] Install: $(( (UV_T1 - UV_T0) / 1000 ))s"

/tmp/zs-bench-venv/bin/python -c "
import time
t0 = $T_START
import torch
t1 = time.time() * 1000
print(f'  [torch] imported at {(t1 - t0)/1000:.1f}s')
x = torch.randn(100, 100, device='cuda')
t2 = time.time() * 1000
print(f'RESULT: {(t2 - t0)/1000:.1f}s to first CUDA op (uv cold)')
" 2>&1
echo ""

rm -rf /tmp/zs-bench-venv
uv cache clean 2>/dev/null || true

# ===============================================================
# TEST 2: zerostart cold (current: wait-all then run)
# ===============================================================
echo "=== [zerostart] cold (wait-all, current behavior) ==="
rm -rf "$ZEROSTART_CACHE" 2>/dev/null || true

T_START=$(date +%s%3N)
BENCH_T_START=$T_START ZS_NO_SHARED_CACHE=1 $ZS_BIN run -p "torch>=2.5" "$SCRIPT_DIR/torch_only.py" 2>&1
echo ""

# ===============================================================
# TEST 3: Simulated progressive — daemon in background, poll for torch
# ===============================================================
echo "=== [zerostart] simulated progressive ==="
rm -rf "$ZEROSTART_CACHE" 2>/dev/null || true

T_START=$(date +%s%3N)

# Create venv
VENV="$ZEROSTART_CACHE/envs/progressive_test"
uv venv "$VENV" --python python3 -q 2>&1
SITE_PACKAGES=$($VENV/bin/python -c "import site; print(site.getsitepackages()[0])")

# Resolve
uv pip compile --python-version 3.11 --python-platform linux --format pylock.toml \
    <(echo "torch>=2.5") -q 2>/dev/null > /tmp/zs-pylock.toml 2>&1

# Build manifest from pylock — wheels only (daemon can't handle sdists)
$VENV/bin/python -c "
import json, re
text = open('/tmp/zs-pylock.toml').read()
wheels = []
sdists = []
for block in text.split('[[packages]]')[1:]:
    name_m = re.search(r'name\s*=\s*\"([^\"]+)\"', block)
    version_m = re.search(r'version\s*=\s*\"([^\"]+)\"', block)
    url_m = re.search(r'url\s*=\s*\"([^\"]+)\"', block)
    size_m = re.search(r'size\s*=\s*(\d+)', block)
    if name_m and url_m:
        url = url_m.group(1)
        pkg = {
            'name': name_m.group(1),
            'version': version_m.group(1) if version_m else '0.0.0',
            'url': url,
            'size': int(size_m.group(1)) if size_m else 0,
        }
        if url.endswith('.whl'):
            wheels.append({
                'url': url,
                'distribution': pkg['name'],
                'version': pkg['version'],
                'size': pkg['size'],
                'import_roots': [pkg['name'].replace('-', '_')],
                'hash': None,
            })
        else:
            sdists.append(pkg['name'] + '==' + pkg['version'])
manifest = {'site_packages': '$SITE_PACKAGES', 'wheels': wheels}
json.dump(manifest, open('/tmp/zs-manifest.json', 'w'))
# Save sdist list for uv to handle
with open('/tmp/zs-sdists.txt', 'w') as f:
    f.write('\n'.join(sdists))
print(f'  Manifest: {len(wheels)} wheels, {len(sdists)} sdists')
if sdists:
    print(f'  Sdists (via uv): {sdists}')
" 2>&1

T_RESOLVE=$(date +%s%3N)
echo "  Resolve: $(( (T_RESOLVE - T_START) / 1000 ))s"

# Install sdist-only packages via uv first (they're small, daemon can't handle them)
if [ -s /tmp/zs-sdists.txt ]; then
    echo "  Installing sdist packages via uv..."
    uv pip install --python "$VENV/bin/python" -r /tmp/zs-sdists.txt -q 2>&1 | tail -3
fi

# Start daemon in background
$ZS_BIN daemon --manifest /tmp/zs-manifest.json &
DAEMON_PID=$!
echo "  Daemon started (PID $DAEMON_PID)"

# Poll for torch readiness, then import immediately
$VENV/bin/python -c "
import time, os, sys

t0 = $T_START
site = '$SITE_PACKAGES'
sys.path.insert(0, site)

# Poll for torch extraction completion
checks = 0
while True:
    torch_init = os.path.join(site, 'torch', '__init__.py')
    torch_lib = os.path.join(site, 'torch', 'lib')
    # torch needs lib/ dir with .so files to be importable
    if os.path.isfile(torch_init) and os.path.isdir(torch_lib):
        so_files = [f for f in os.listdir(torch_lib) if f.endswith('.so')]
        if len(so_files) > 5:
            break
    checks += 1
    time.sleep(0.2)

t_found = time.time() * 1000
print(f'  [torch] files ready at {(t_found - t0)/1000:.1f}s ({checks} polls)')

# Small delay to let atomic commit finish
time.sleep(0.5)

import torch
t_import = time.time() * 1000
print(f'  [torch] imported at {(t_import - t0)/1000:.1f}s (v{torch.__version__})')

x = torch.randn(100, 100, device='cuda')
t_cuda = time.time() * 1000
print(f'RESULT: {(t_cuda - t0)/1000:.1f}s to first CUDA op (progressive)')
" 2>&1

# Let daemon finish in background
wait $DAEMON_PID 2>/dev/null || true
T_END=$(date +%s%3N)
echo "  Daemon total: $(( (T_END - T_START) / 1000 ))s"
echo ""

echo "Done."
