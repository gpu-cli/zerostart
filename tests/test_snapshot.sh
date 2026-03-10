#!/bin/bash
set -uo pipefail

echo "=== Snapshot/Hydrate Benchmark ==="
echo "Date: $(date -u)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ZS="$PROJECT_DIR/bin/zerostart-linux-x86_64"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# Install the zerostart Python SDK + cloudpickle + safetensors
pip install -q cloudpickle accelerate 2>/dev/null

# Install zerostart SDK
cd "$PROJECT_DIR/python" && pip install -q -e . 2>/dev/null
cd "$PROJECT_DIR"

# Use a small model for testing: GPT-2 (500MB, fast to download)
# This proves the snapshot/hydrate cycle works end-to-end
cat > /tmp/test_snapshot.py << 'PYEOF'
"""Test snapshot/hydrate cycle with a real model."""
import time
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-20s %(message)s")
log = logging.getLogger("test")

# ---------------------------------------------------------------------------
# Step 1: Load model normally (the slow path)
# ---------------------------------------------------------------------------
log.info("--- Step 1: Normal model load ---")
t0 = time.monotonic()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "gpt2"  # 500MB, good for testing

t1 = time.monotonic()
log.info("Imports: %.2fs", t1 - t0)

tokenizer = AutoTokenizer.from_pretrained(model_id)
t2 = time.monotonic()
log.info("Tokenizer loaded: %.2fs", t2 - t1)

model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
t3 = time.monotonic()
log.info("Model loaded: %.2fs", t3 - t2)

# Quick inference to verify model works
inputs = tokenizer("Hello, world!", return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=20)
result_normal = tokenizer.decode(outputs[0], skip_special_tokens=True)
t4 = time.monotonic()
log.info("Inference: %.2fs", t4 - t3)
log.info("Normal load total: %.2fs", t4 - t0)
log.info("Output: %s", result_normal[:100])
print("")

# ---------------------------------------------------------------------------
# Step 2: Snapshot the loaded state
# ---------------------------------------------------------------------------
log.info("--- Step 2: Snapshot ---")
t5 = time.monotonic()

from zerostart.snapshot import snapshot, hydrate, snapshot_exists, snapshot_info

snap_path = "/tmp/gpt2-snapshot"
snapshot(
    state={"model": model, "tokenizer": tokenizer},
    path=snap_path,
)

t6 = time.monotonic()
log.info("Snapshot created: %.2fs", t6 - t5)

# Show snapshot info
info = snapshot_info(snap_path)
log.info("  Tensors: %d (%d matched to safetensors)",
         info["tensor_count"], info["matched_tensors"])
log.info("  Safetensors files: %s", info["safetensors_files"])
print("")

# ---------------------------------------------------------------------------
# Step 3: Clear everything and hydrate from snapshot
# ---------------------------------------------------------------------------
log.info("--- Step 3: Hydrate from snapshot ---")

# Delete model and tokenizer to prove we're restoring from snapshot
del model
del tokenizer
torch.cuda.empty_cache() if torch.cuda.is_available() else None
import gc; gc.collect()

t7 = time.monotonic()

restored = hydrate(snap_path)
t8 = time.monotonic()
log.info("Hydrate: %.2fs", t8 - t7)

model2 = restored["model"]
tokenizer2 = restored["tokenizer"]

# Verify restored model produces same output
inputs2 = tokenizer2("Hello, world!", return_tensors="pt")
with torch.no_grad():
    outputs2 = model2.generate(**inputs2, max_new_tokens=20)
result_hydrated = tokenizer2.decode(outputs2[0], skip_special_tokens=True)
t9 = time.monotonic()
log.info("Inference after hydrate: %.2fs", t9 - t8)
log.info("Output: %s", result_hydrated[:100])

# Verify outputs match
if result_normal == result_hydrated:
    log.info("PASS: Hydrated model produces identical output")
else:
    log.error("FAIL: Outputs differ!")
    log.error("  Normal:   %s", result_normal[:100])
    log.error("  Hydrated: %s", result_hydrated[:100])
print("")

# ---------------------------------------------------------------------------
# Step 4: Hydrate again (measure pure restore speed)
# ---------------------------------------------------------------------------
log.info("--- Step 4: Hydrate speed (3 runs) ---")
del model2, tokenizer2
gc.collect()

for i in range(3):
    t_start = time.monotonic()
    restored = hydrate(snap_path)
    t_end = time.monotonic()
    log.info("  Hydrate %d: %.3fs", i + 1, t_end - t_start)
    del restored
    gc.collect()

print("")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("=== SUMMARY ===")
print(f"  Normal load:    {t4-t0:.2f}s (imports + tokenizer + model + inference)")
print(f"  Snapshot:       {t6-t5:.2f}s")
print(f"  Hydrate:        {t8-t7:.2f}s")
print(f"  Speedup:        {(t4-t0)/(t8-t7):.1f}x")
print(f"  Output match:   {result_normal == result_hydrated}")
print("=== DONE ===")
PYEOF

# First install deps, then run the test
echo "Installing test dependencies..."
export ZEROSTART_CACHE="/tmp/.zs-snap-test"
export ZS_NO_SHARED_CACHE=1
rm -rf "$ZEROSTART_CACHE"

# Remove system torchvision that conflicts with our torch version
pip uninstall -y torchvision 2>/dev/null || true

$ZS run -v -p torch -p transformers -p accelerate -p cloudpickle /tmp/test_snapshot.py 2>&1
echo ""
echo "Exit: $?"
