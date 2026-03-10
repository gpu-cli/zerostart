#!/bin/bash
set -uo pipefail

echo "=== zerostart.accelerate() Benchmark ==="
echo "Date: $(date -u)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_DIR/python:${PYTHONPATH:-}"

MODEL_ID="${SNAP_MODEL:-Qwen/Qwen2.5-7B}"
BENCH_DIR="/gpu-cli-workspaces/.bench-accelerate"
rm -rf "$BENCH_DIR"
mkdir -p "$BENCH_DIR"

pip install -q accelerate transformers 2>/dev/null
pip uninstall -y torchvision 2>/dev/null || true

echo "============================================================"
echo "BENCHMARK 1: Baseline from_pretrained (no acceleration)"
echo "============================================================"

python3 << 'PYEOF'
import subprocess, sys, os, time

script = """
import time, os
t0 = time.monotonic()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ["MODEL_ID"]
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
model.eval()
t_model = time.monotonic()

inputs = tokenizer("The quick brown fox", return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
result = tokenizer.decode(out[0], skip_special_tokens=True)
t_inf = time.monotonic()

print(f"RESULT: {result}")
print(f"TIME model={t_model-t0:.2f}s inference={t_inf-t_model:.2f}s total={t_inf-t0:.2f}s")
"""

script_path = "/gpu-cli-workspaces/.bench-accelerate/bench_baseline.py"
with open(script_path, "w") as f:
    f.write("import os\n" + script)

t0 = time.monotonic()
r = subprocess.run(
    [sys.executable, script_path],
    capture_output=True, text=True, timeout=120,
    env={**os.environ, "MODEL_ID": os.environ.get("SNAP_MODEL", "Qwen/Qwen2.5-7B")},
)
elapsed = time.monotonic() - t0
print(r.stdout.strip())
if r.returncode != 0:
    print("STDERR:", r.stderr[-500:])
print(f"Wall clock: {elapsed:.2f}s")
PYEOF

echo ""
echo "============================================================"
echo "BENCHMARK 2: accelerate() — first load (auto-caches)"
echo "============================================================"

python3 << 'PYEOF'
import subprocess, sys, os, time

script = """
import time, logging, os
logging.basicConfig(level=logging.INFO, format="%(name)-20s %(message)s")
t0 = time.monotonic()

import zerostart
zerostart.accelerate(cache_dir="/gpu-cli-workspaces/.bench-accelerate/model-cache")
t_accel = time.monotonic()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ["MODEL_ID"]
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
model.eval()
t_model = time.monotonic()

inputs = tokenizer("The quick brown fox", return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
result = tokenizer.decode(out[0], skip_special_tokens=True)
t_inf = time.monotonic()

print(f"RESULT: {result}")
print(f"TIME accelerate={t_accel-t0:.2f}s model={t_model-t_accel:.2f}s inference={t_inf-t_model:.2f}s total={t_inf-t0:.2f}s")
"""

script_path = "/gpu-cli-workspaces/.bench-accelerate/bench_accel_first.py"
with open(script_path, "w") as f:
    f.write("import os\n" + script)

t0 = time.monotonic()
r = subprocess.run(
    [sys.executable, script_path],
    capture_output=True, text=True, timeout=180,
    env={
        **os.environ,
        "MODEL_ID": os.environ.get("SNAP_MODEL", "Qwen/Qwen2.5-7B"),
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
    },
)
elapsed = time.monotonic() - t0
print(r.stdout.strip())
if r.returncode != 0:
    print("STDERR:", r.stderr[-500:])
print(f"Wall clock: {elapsed:.2f}s")
PYEOF

echo ""
echo "============================================================"
echo "BENCHMARK 3: accelerate() — second load (cache hit)"
echo "============================================================"

python3 << 'PYEOF'
import subprocess, sys, os, time

script = """
import time, logging, os
logging.basicConfig(level=logging.INFO, format="%(name)-20s %(message)s")
t0 = time.monotonic()

import zerostart
zerostart.accelerate(cache_dir="/gpu-cli-workspaces/.bench-accelerate/model-cache")
t_accel = time.monotonic()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ["MODEL_ID"]
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
model.eval()
t_model = time.monotonic()

inputs = tokenizer("The quick brown fox", return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
result = tokenizer.decode(out[0], skip_special_tokens=True)
t_inf = time.monotonic()

print(f"RESULT: {result}")
print(f"TIME accelerate={t_accel-t0:.2f}s model={t_model-t_accel:.2f}s inference={t_inf-t_model:.2f}s total={t_inf-t0:.2f}s")
"""

script_path = "/gpu-cli-workspaces/.bench-accelerate/bench_accel_cached.py"
with open(script_path, "w") as f:
    f.write("import os\n" + script)

t0 = time.monotonic()
r = subprocess.run(
    [sys.executable, script_path],
    capture_output=True, text=True, timeout=120,
    env={
        **os.environ,
        "MODEL_ID": os.environ.get("SNAP_MODEL", "Qwen/Qwen2.5-7B"),
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
    },
)
elapsed = time.monotonic() - t0
print(r.stdout.strip())
if r.returncode != 0:
    print("STDERR:", r.stderr[-500:])
print(f"Wall clock: {elapsed:.2f}s")
PYEOF

echo ""
echo "============================================================"
rm -rf "$BENCH_DIR"
