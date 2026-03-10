#!/bin/bash
set -uo pipefail

echo "=== End-to-End Cold Start Benchmark ==="
echo "Date: $(date -u)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ZS="$PROJECT_DIR/bin/zerostart-linux-x86_64"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR/python:${PYTHONPATH:-}"

MODEL_ID="${SNAP_MODEL:-Qwen/Qwen2.5-7B}"

# All temp data on the volume (more space than /tmp)
BENCH_DIR="/gpu-cli-workspaces/.bench-e2e"
rm -rf "$BENCH_DIR"
mkdir -p "$BENCH_DIR"

# Remove system torchvision that conflicts with fresh torch installs
pip uninstall -y torchvision 2>/dev/null || true

# Clear pip cache so scenario 1 is a true cold install
pip cache purge 2>/dev/null || true

# Clear HF cache so model download is truly cold
export HF_HOME="$BENCH_DIR/hf-cache"

df -h /gpu-cli-workspaces | tail -1 | awk '{print "Disk: " $4 " free on /gpu-cli-workspaces"}'
echo ""

# ============================================================
# Scenario 1: pip install + from_pretrained (traditional)
# ============================================================
echo "--- Scenario 1: pip install + from_pretrained ---"
BENCH_START=$(date +%s%3N)

cat > "$BENCH_DIR/bench_pip.py" << PYEOF
import time
t_script = time.monotonic()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

t_import = time.monotonic()

tokenizer = AutoTokenizer.from_pretrained("$MODEL_ID")
model = AutoModelForCausalLM.from_pretrained("$MODEL_ID", dtype=torch.bfloat16, device_map="cpu")
model.eval()
t_model = time.monotonic()

inputs = tokenizer("The quick brown fox", return_tensors="pt")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
result = tokenizer.decode(out[0], skip_special_tokens=True)
t_inf = time.monotonic()

print(f"RESULT: {result}")
print(f"TIME import={t_import-t_script:.2f}s model={t_model-t_import:.2f}s inference={t_inf-t_model:.2f}s total={t_inf-t_script:.2f}s")
PYEOF

# Fresh venv, no pip cache — true cold install
python3 -m venv "$BENCH_DIR/pip-venv"
"$BENCH_DIR/pip-venv/bin/pip" install --no-cache-dir -q torch transformers accelerate 2>&1 | tail -3
PIP_DONE=$(date +%s%3N)
echo "  pip install: $(( PIP_DONE - BENCH_START ))ms"

"$BENCH_DIR/pip-venv/bin/python" "$BENCH_DIR/bench_pip.py" 2>&1 | tail -30
BENCH_END=$(date +%s%3N)
echo "  Total wall clock (install + load + inference): $(( BENCH_END - BENCH_START ))ms"
rm -rf "$BENCH_DIR/pip-venv"
echo ""

# Clear HF cache so scenario 2 also downloads fresh
rm -rf "$HF_HOME"

# ============================================================
# Scenario 2: zerostart cold + from_pretrained
# ============================================================
echo "--- Scenario 2: zerostart cold + from_pretrained ---"
export ZEROSTART_CACHE="$BENCH_DIR/zs-cache"
export ZS_NO_SHARED_CACHE=1
rm -rf "$ZEROSTART_CACHE"

cat > "$BENCH_DIR/bench_zs_cold.py" << PYEOF
import time
t_script = time.monotonic()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

t_import = time.monotonic()

tokenizer = AutoTokenizer.from_pretrained("$MODEL_ID")
model = AutoModelForCausalLM.from_pretrained("$MODEL_ID", dtype=torch.bfloat16, device_map="cpu")
model.eval()
t_model = time.monotonic()

inputs = tokenizer("The quick brown fox", return_tensors="pt")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
result = tokenizer.decode(out[0], skip_special_tokens=True)
t_inf = time.monotonic()

print(f"RESULT: {result}")
print(f"TIME import={t_import-t_script:.2f}s model={t_model-t_import:.2f}s inference={t_inf-t_model:.2f}s total={t_inf-t_script:.2f}s")
PYEOF

ZS_START=$(date +%s%3N)
$ZS run -p torch -p transformers -p accelerate "$BENCH_DIR/bench_zs_cold.py" 2>&1 | tail -30
ZS_END=$(date +%s%3N)
echo "  Total wall clock (zerostart cold + load + inference): $(( ZS_END - ZS_START ))ms"
echo ""

# ============================================================
# Scenario 3: zerostart warm + from_pretrained
# ============================================================
echo "--- Scenario 3: zerostart warm + from_pretrained ---"
# zerostart package cache is warm from Scenario 2
# HF model cache is warm from Scenario 2

ZS_WARM_START=$(date +%s%3N)
$ZS run -p torch -p transformers -p accelerate "$BENCH_DIR/bench_zs_cold.py" 2>&1 | tail -30
ZS_WARM_END=$(date +%s%3N)
echo "  Total wall clock (zerostart warm + load + inference): $(( ZS_WARM_END - ZS_WARM_START ))ms"
echo ""

# ============================================================
# Scenario 4: zerostart warm + hydrate (snapshot)
# ============================================================
echo "--- Scenario 4: Create snapshot for hydrate ---"

cat > "$BENCH_DIR/bench_create_snap.py" << PYEOF
import time, logging
logging.basicConfig(level=logging.INFO, format="%(name)-20s %(message)s")
t0 = time.monotonic()
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from zerostart.snapshot import snapshot

tokenizer = AutoTokenizer.from_pretrained("$MODEL_ID")
model = AutoModelForCausalLM.from_pretrained("$MODEL_ID", dtype=torch.bfloat16, device_map="cpu")
model.eval()

import shutil
shutil.rmtree("$BENCH_DIR/e2e-snapshot", ignore_errors=True)
snapshot(state={"model": model, "tokenizer": tokenizer}, path="$BENCH_DIR/e2e-snapshot")
t1 = time.monotonic()
print(f"Snapshot created in {t1-t0:.2f}s")
PYEOF

$ZS run -p torch -p transformers -p accelerate "$BENCH_DIR/bench_create_snap.py" 2>&1 | tail -30

echo ""
echo "--- Scenario 4: zerostart warm + hydrate + inference ---"

# Reuse the warm zerostart package cache from scenarios 2/3 —
# the comparison is model loading (hydrate vs from_pretrained),
# not package installation.
export ZEROSTART_CACHE="$BENCH_DIR/zs-cache"

cat > "$BENCH_DIR/bench_hydrate.py" << PYEOF
import time, logging
logging.basicConfig(level=logging.INFO, format="%(name)-20s %(message)s")
t_script = time.monotonic()

import torch
from zerostart.snapshot import hydrate

t_import = time.monotonic()

restored = hydrate("$BENCH_DIR/e2e-snapshot")
model = restored["model"]
model.eval()
tokenizer = restored["tokenizer"]
t_hydrate = time.monotonic()

inputs = tokenizer("The quick brown fox", return_tensors="pt")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
result = tokenizer.decode(out[0], skip_special_tokens=True)
t_inf = time.monotonic()

print(f"RESULT: {result}")
print(f"TIME import={t_import-t_script:.2f}s hydrate={t_hydrate-t_import:.2f}s inference={t_inf-t_hydrate:.2f}s total={t_inf-t_script:.2f}s")
PYEOF

ZS_HYD_START=$(date +%s%3N)
$ZS run -p torch -p transformers -p accelerate "$BENCH_DIR/bench_hydrate.py" 2>&1 | tail -30
ZS_HYD_END=$(date +%s%3N)
echo "  Total wall clock (zerostart warm + hydrate + inference): $(( ZS_HYD_END - ZS_HYD_START ))ms"
echo ""

# ============================================================
# Summary
# ============================================================
echo "============================================================"
echo "MODEL: $MODEL_ID"
echo ""
echo "Boot-to-inference wall clock:"
echo "  1. pip install + from_pretrained:        $(( BENCH_END - BENCH_START ))ms"
echo "  2. zerostart cold + from_pretrained:     $(( ZS_END - ZS_START ))ms"
echo "  3. zerostart warm + from_pretrained:     $(( ZS_WARM_END - ZS_WARM_START ))ms"
echo "  4. zerostart warm + hydrate (snapshot):  $(( ZS_HYD_END - ZS_HYD_START ))ms"
echo "============================================================"

# Cleanup
rm -rf "$BENCH_DIR"
