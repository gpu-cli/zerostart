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
echo ""

echo "============================================================"
echo "TEST 1: Verify registration works"
echo "============================================================"

python3 << 'PYEOF'
import sys, os
sys.path.insert(0, os.environ.get("PYTHONPATH", "").split(":")[0])

from zerostart.integrations.vllm import register, ZerostartModelLoader
register()

# Verify registration
try:
    from vllm.model_executor.model_loader import get_model_loader
    from vllm.config.load import LoadConfig
    load_config = LoadConfig(load_format="zerostart")
    loader = get_model_loader(load_config)
    print(f"✓ Registered: {type(loader).__name__}")
    assert type(loader).__name__ == "ZerostartModelLoader"
    print("✓ Registration verified")
except Exception as e:
    print(f"Registration test: {e}")
    # Try alternate verification
    try:
        import vllm.model_executor.model_loader as ml
        registry = getattr(ml, "_LOAD_FORMAT_TO_MODEL_LOADER", {})
        if "zerostart" in registry:
            print(f"✓ Found in registry: {registry['zerostart']}")
        else:
            print(f"✗ Not in registry. Keys: {list(registry.keys())}")
    except Exception as e2:
        print(f"✗ Could not verify: {e2}")
PYEOF

echo ""
echo "============================================================"
echo "TEST 2: Baseline vLLM serve (standard loading)"
echo "============================================================"

python3 << PYEOF
import subprocess, sys, os, time

script = """
import time, os, logging
logging.basicConfig(level=logging.INFO, format="%(name)-20s %(message)s")
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

script_path = "$BENCH_DIR/bench_vllm_baseline.py"
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
echo "TEST 3: vLLM with zerostart.accelerate() (transparent hook)"
echo "============================================================"

python3 << PYEOF
import subprocess, sys, os, time

script = """
import time, os, logging
logging.basicConfig(level=logging.INFO, format="%(name)-20s %(message)s")
t0 = time.monotonic()

import zerostart
zerostart.accelerate(cache_dir="$BENCH_DIR/model-cache")
t_accel = time.monotonic()

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
print(f"TIME accel={t_accel-t0:.2f}s import={t_import-t_accel:.2f}s load={t_load-t_import:.2f}s generate={t_gen-t_load:.2f}s total={t_gen-t0:.2f}s")
"""

script_path = "$BENCH_DIR/bench_vllm_accel.py"
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
echo "TEST 4: vLLM with accelerate() — second run (cache hit)"
echo "============================================================"

python3 << PYEOF
import subprocess, sys, os, time

script = """
import time, os, logging
logging.basicConfig(level=logging.INFO, format="%(name)-20s %(message)s")
t0 = time.monotonic()

import zerostart
zerostart.accelerate(cache_dir="$BENCH_DIR/model-cache")
t_accel = time.monotonic()

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
print(f"TIME accel={t_accel-t0:.2f}s import={t_import-t_accel:.2f}s load={t_load-t_import:.2f}s generate={t_gen-t_load:.2f}s total={t_gen-t0:.2f}s")
"""

script_path = "$BENCH_DIR/bench_vllm_cached.py"
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
