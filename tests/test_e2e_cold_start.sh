#!/bin/bash
set -uo pipefail

echo "=== End-to-End Cold Start Benchmark ==="
echo "Date: $(date -u)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
df -h /tmp | tail -1 | awk '{print "Disk: " $4 " free"}'
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ZS="$PROJECT_DIR/bin/zerostart-linux-x86_64"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR/python:${PYTHONPATH:-}"

MODEL_ID="${SNAP_MODEL:-Qwen/Qwen2.5-7B}"

# ============================================================
# Scenario 1: pip install + from_pretrained (traditional)
# ============================================================
echo "--- Scenario 1: pip install + from_pretrained ---"
# Clean slate
rm -rf /tmp/.pip-bench-venv
BENCH_START=$(date +%s%3N)

cat > /tmp/bench_pip.py << PYEOF
import time
t_script = time.monotonic()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

t_import = time.monotonic()

tokenizer = AutoTokenizer.from_pretrained("$MODEL_ID")
model = AutoModelForCausalLM.from_pretrained("$MODEL_ID", torch_dtype=torch.bfloat16, device_map="cpu")
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

# Install into a fresh venv (simulates cold container)
python3 -m venv /tmp/.pip-bench-venv
/tmp/.pip-bench-venv/bin/pip install -q torch transformers accelerate 2>&1 | tail -3
PIP_DONE=$(date +%s%3N)
echo "  pip install: $(( PIP_DONE - BENCH_START ))ms"

/tmp/.pip-bench-venv/bin/python /tmp/bench_pip.py 2>&1 | grep -E "^(RESULT|TIME)"
BENCH_END=$(date +%s%3N)
echo "  Total wall clock (install + load + inference): $(( BENCH_END - BENCH_START ))ms"
rm -rf /tmp/.pip-bench-venv
echo ""

# ============================================================
# Scenario 2: zerostart cold + from_pretrained
# ============================================================
echo "--- Scenario 2: zerostart cold + from_pretrained ---"
export ZEROSTART_CACHE="/tmp/.zs-e2e-bench"
export ZS_NO_SHARED_CACHE=1
rm -rf "$ZEROSTART_CACHE"

cat > /tmp/bench_zs_cold.py << PYEOF
import time
t_script = time.monotonic()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

t_import = time.monotonic()

tokenizer = AutoTokenizer.from_pretrained("$MODEL_ID")
model = AutoModelForCausalLM.from_pretrained("$MODEL_ID", torch_dtype=torch.bfloat16, device_map="cpu")
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
$ZS run -p torch -p transformers -p accelerate /tmp/bench_zs_cold.py 2>&1 | grep -E "^(RESULT|TIME|Resolved|Daemon|Environment|Cache)"
ZS_END=$(date +%s%3N)
echo "  Total wall clock (zerostart cold + load + inference): $(( ZS_END - ZS_START ))ms"
echo ""

# ============================================================
# Scenario 3: zerostart warm + from_pretrained
# ============================================================
echo "--- Scenario 3: zerostart warm + from_pretrained ---"
# Cache is now populated from Scenario 2

ZS_WARM_START=$(date +%s%3N)
$ZS run -p torch -p transformers -p accelerate /tmp/bench_zs_cold.py 2>&1 | grep -E "^(RESULT|TIME|Cache)"
ZS_WARM_END=$(date +%s%3N)
echo "  Total wall clock (zerostart warm + load + inference): $(( ZS_WARM_END - ZS_WARM_START ))ms"
echo ""

# ============================================================
# Scenario 4: zerostart warm + hydrate (snapshot)
# ============================================================
echo "--- Scenario 4: Create snapshot for hydrate ---"

cat > /tmp/bench_create_snap.py << PYEOF
import time
t0 = time.monotonic()
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from zerostart.snapshot import snapshot

tokenizer = AutoTokenizer.from_pretrained("$MODEL_ID")
model = AutoModelForCausalLM.from_pretrained("$MODEL_ID", torch_dtype=torch.bfloat16, device_map="cpu")
model.eval()

import shutil
shutil.rmtree("/tmp/e2e-snapshot", ignore_errors=True)
snapshot(state={"model": model, "tokenizer": tokenizer}, path="/tmp/e2e-snapshot")
t1 = time.monotonic()
print(f"Snapshot created in {t1-t0:.2f}s")
PYEOF

$ZS run -p torch -p transformers -p accelerate -p cloudpickle /tmp/bench_create_snap.py 2>&1 | grep -E "^(Snapshot|Cache)"

echo ""
echo "--- Scenario 4: zerostart warm + hydrate + inference ---"

cat > /tmp/bench_hydrate.py << PYEOF
import time
t_script = time.monotonic()

import torch
from zerostart.snapshot import hydrate

t_import = time.monotonic()

restored = hydrate("/tmp/e2e-snapshot")
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
$ZS run -p torch -p transformers -p accelerate -p cloudpickle /tmp/bench_hydrate.py 2>&1 | grep -E "^(RESULT|TIME|Cache)"
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
