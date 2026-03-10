#!/bin/bash
set -uo pipefail

echo "=== GPU Hydrate vs from_pretrained ==="
echo "Date: $(date -u)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_DIR/python:${PYTHONPATH:-}"

MODEL_ID="${SNAP_MODEL:-Qwen/Qwen2.5-7B}"
BENCH_DIR="/gpu-cli-workspaces/.bench-gpu"
rm -rf "$BENCH_DIR"
mkdir -p "$BENCH_DIR"

pip install -q accelerate transformers 2>/dev/null
pip uninstall -y torchvision 2>/dev/null || true

echo "--- Setup: download model + create snapshot ---"

python3 << PYEOF
import time, logging, torch, os
logging.basicConfig(level=logging.INFO, format="%(name)-20s %(message)s")
from transformers import AutoModelForCausalLM, AutoTokenizer
from zerostart.snapshot import snapshot

MODEL_ID = "$MODEL_ID"
BENCH_DIR = "$BENCH_DIR"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cpu")
model.eval()

import shutil
shutil.rmtree(f"{BENCH_DIR}/snapshot", ignore_errors=True)
snapshot(state={"model": model, "tokenizer": tokenizer}, path=f"{BENCH_DIR}/snapshot")
print("Snapshot created")
PYEOF

echo ""
echo "============================================================"
echo "BENCHMARK: from_pretrained (GPU) — cold subprocess"
echo "============================================================"

# Run from_pretrained in a fresh subprocess
python3 << 'PYEOF'
import subprocess, sys, os, time

script = """
import time, logging
logging.basicConfig(level=logging.INFO, format="%(name)-20s %(message)s")
t0 = time.monotonic()

import torch
t_torch = time.monotonic()

from transformers import AutoModelForCausalLM, AutoTokenizer
t_import = time.monotonic()

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
print(f"TIME torch={t_torch-t0:.2f}s import={t_import-t_torch:.2f}s model={t_model-t_import:.2f}s inference={t_inf-t_model:.2f}s total={t_inf-t0:.2f}s")
"""

script_path = "/gpu-cli-workspaces/.bench-gpu/bench_pretrained.py"
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
echo "BENCHMARK: hydrate (GPU) — cold subprocess"
echo "============================================================"

python3 << 'PYEOF'
import subprocess, sys, os, time

script = """
import time, logging
logging.basicConfig(level=logging.INFO, format="%(name)-20s %(message)s")
t0 = time.monotonic()

import torch
t_torch = time.monotonic()

from zerostart.snapshot import hydrate
t_import = time.monotonic()

restored = hydrate(os.environ["SNAP_PATH"], device="cuda")
model = restored["model"]
model.eval()
tokenizer = restored["tokenizer"]
t_hydrate = time.monotonic()

inputs = tokenizer("The quick brown fox", return_tensors="pt")
# Move input tensors to cuda
input_ids = inputs["input_ids"].to("cuda")
attention_mask = inputs["attention_mask"].to("cuda")
with torch.no_grad():
    out = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                         max_new_tokens=10, do_sample=False)
result = tokenizer.decode(out[0], skip_special_tokens=True)
t_inf = time.monotonic()

print(f"RESULT: {result}")
print(f"TIME torch={t_torch-t0:.2f}s import={t_import-t_torch:.2f}s hydrate={t_hydrate-t_import:.2f}s inference={t_inf-t_hydrate:.2f}s total={t_inf-t0:.2f}s")
"""

script_path = "/gpu-cli-workspaces/.bench-gpu/bench_hydrate.py"
with open(script_path, "w") as f:
    f.write("import os\n" + script)

t0 = time.monotonic()
r = subprocess.run(
    [sys.executable, script_path],
    capture_output=True, text=True, timeout=120,
    env={
        **os.environ,
        "SNAP_PATH": "/gpu-cli-workspaces/.bench-gpu/snapshot",
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
