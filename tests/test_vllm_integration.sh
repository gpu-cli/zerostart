#!/bin/bash
set -uo pipefail

echo "=== vLLM + zerostart Integration Test ==="
echo "Date: $(date -u)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_DIR/python:${PYTHONPATH:-}"

BENCH_DIR="/gpu-cli-workspaces/.bench-vllm"
rm -rf "$BENCH_DIR"
mkdir -p "$BENCH_DIR"

# Use a small model that fits in GPU memory
MODEL_ID="${SNAP_MODEL:-Qwen/Qwen2.5-1.5B}"

pip uninstall -y torchvision 2>/dev/null || true

echo "--- Installing vLLM ---"
pip install -q vllm 2>&1 | tail -5

# Install zerostart in editable mode so entry_points register
cd "$PROJECT_DIR/python" && pip install -q -e . 2>&1 | tail -3
cd "$PROJECT_DIR"
echo ""

echo "============================================================"
echo "TEST 1: Verify registration + plugin system"
echo "============================================================"

python3 << 'PYEOF'
import sys, os

# Test 1a: Manual registration
from zerostart.integrations.vllm import register, ZerostartModelLoader
register()

try:
    from vllm.model_executor.model_loader import get_model_loader
    from vllm.config.load import LoadConfig
    load_config = LoadConfig(load_format="zerostart")
    loader = get_model_loader(load_config)
    assert type(loader).__name__ == "ZerostartModelLoader"
    print("✓ Manual registration works")
except Exception as e:
    print(f"✗ Manual registration: {e}")

# Test 1b: Plugin entry point
from zerostart.integrations.vllm import register_plugin
register_plugin()
print("✓ Plugin entry point works")

# Test 1c: Verify it subclasses DefaultModelLoader
try:
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    assert issubclass(ZerostartModelLoader, DefaultModelLoader)
    print("✓ Subclasses DefaultModelLoader")
except Exception as e:
    print(f"  Subclass check: {e}")

# Test 1d: Network volume detection
from zerostart.integrations.vllm import _is_network_volume
# /gpu-cli-workspaces is typically an overlay on RunPod
result = _is_network_volume("/gpu-cli-workspaces")
print(f"  /gpu-cli-workspaces is network volume: {result}")
result2 = _is_network_volume("/tmp")
print(f"  /tmp is network volume: {result2}")
PYEOF

echo ""
echo "============================================================"
echo "TEST 2: Baseline vLLM (standard loading)"
echo "============================================================"

python3 << PYEOF
import subprocess, sys, os, time

script = """
import time, os
t0 = time.monotonic()

from vllm import LLM, SamplingParams

MODEL_ID = "$MODEL_ID"
t_import = time.monotonic()

llm = LLM(model=MODEL_ID, dtype="bfloat16", gpu_memory_utilization=0.8)
t_load = time.monotonic()

params = SamplingParams(max_tokens=20, temperature=0)
outputs = llm.generate(["The quick brown fox"], params)
result = outputs[0].outputs[0].text
t_gen = time.monotonic()

print(f"RESULT: {result}")
print(f"TIME import={t_import-t0:.2f}s load={t_load-t_import:.2f}s generate={t_gen-t_load:.2f}s total={t_gen-t0:.2f}s")
"""

script_path = "$BENCH_DIR/bench_baseline.py"
with open(script_path, "w") as f:
    f.write(script)

t0 = time.monotonic()
r = subprocess.run(
    [sys.executable, script_path],
    capture_output=True, text=True, timeout=300,
)
elapsed = time.monotonic() - t0
print(r.stdout.strip())
if r.returncode != 0:
    print("STDERR:", r.stderr[-1000:])
print(f"Wall clock: {elapsed:.2f}s")
PYEOF

echo ""
echo "============================================================"
echo "TEST 3: vLLM --load-format zerostart (subprocess loader)"
echo "============================================================"

python3 << PYEOF
import subprocess, sys, os, time

script = """
import time, os, logging
logging.basicConfig(level=logging.INFO, format="%(name)-20s %(message)s")
t0 = time.monotonic()

# Register before creating LLM — plugin would do this automatically
from zerostart.integrations.vllm import register
register()

from vllm import LLM, SamplingParams

MODEL_ID = "$MODEL_ID"
t_import = time.monotonic()

llm = LLM(model=MODEL_ID, dtype="bfloat16", gpu_memory_utilization=0.8, load_format="zerostart")
t_load = time.monotonic()

params = SamplingParams(max_tokens=20, temperature=0)
outputs = llm.generate(["The quick brown fox"], params)
result = outputs[0].outputs[0].text
t_gen = time.monotonic()

print(f"RESULT: {result}")
print(f"TIME import={t_import-t0:.2f}s load={t_load-t_import:.2f}s generate={t_gen-t_load:.2f}s total={t_gen-t0:.2f}s")
"""

script_path = "$BENCH_DIR/bench_zs_loader.py"
with open(script_path, "w") as f:
    f.write(script)

t0 = time.monotonic()
r = subprocess.run(
    [sys.executable, script_path],
    capture_output=True, text=True, timeout=300,
    env={**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", "")},
)
elapsed = time.monotonic() - t0
print(r.stdout.strip())
if r.returncode != 0:
    print("STDERR:", r.stderr[-1000:])
print(f"Wall clock: {elapsed:.2f}s")
PYEOF

echo ""
echo "============================================================"
echo "TEST 4: vLLM --load-format zerostart (second run)"
echo "============================================================"

python3 << PYEOF
import subprocess, sys, os, time

script = """
import time, os, logging
logging.basicConfig(level=logging.INFO, format="%(name)-20s %(message)s")
t0 = time.monotonic()

from zerostart.integrations.vllm import register
register()

from vllm import LLM, SamplingParams

MODEL_ID = "$MODEL_ID"
t_import = time.monotonic()

llm = LLM(model=MODEL_ID, dtype="bfloat16", gpu_memory_utilization=0.8, load_format="zerostart")
t_load = time.monotonic()

params = SamplingParams(max_tokens=20, temperature=0)
outputs = llm.generate(["The quick brown fox"], params)
result = outputs[0].outputs[0].text
t_gen = time.monotonic()

print(f"RESULT: {result}")
print(f"TIME import={t_import-t0:.2f}s load={t_load-t_import:.2f}s generate={t_gen-t_load:.2f}s total={t_gen-t0:.2f}s")
"""

script_path = "$BENCH_DIR/bench_zs_loader2.py"
with open(script_path, "w") as f:
    f.write(script)

t0 = time.monotonic()
r = subprocess.run(
    [sys.executable, script_path],
    capture_output=True, text=True, timeout=300,
    env={**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", "")},
)
elapsed = time.monotonic() - t0
print(r.stdout.strip())
if r.returncode != 0:
    print("STDERR:", r.stderr[-1000:])
print(f"Wall clock: {elapsed:.2f}s")
PYEOF

echo ""
echo "============================================================"
rm -rf "$BENCH_DIR"
