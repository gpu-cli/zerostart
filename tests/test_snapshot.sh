#!/bin/bash
set -uo pipefail

echo "=== Snapshot/Hydrate Benchmark (Realistic Model) ==="
echo "Date: $(date -u)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
df -h /tmp | tail -1 | awk '{print "Disk: " $4 " free"}'
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_DIR/python:${PYTHONPATH:-}"

# Install deps — use system torch if available, otherwise install
pip install -q cloudpickle transformers accelerate 2>/dev/null

cat > /tmp/test_snapshot_real.py << 'PYEOF'
"""Snapshot/hydrate benchmark: hydrate vs from_pretrained on a real model.

Both paths assume packages are already installed. The comparison is:
  - from_pretrained (warm HF cache): download once, then load from disk
  - hydrate: mmap safetensors + restore Python state

This measures the model loading bottleneck, not package installation.
"""
import time
import sys
import os
import gc
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-20s %(message)s")
log = logging.getLogger("test")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# Pick a realistic model — Qwen2.5-7B is ~14GB in bf16, ungated
MODEL_ID = os.environ.get("SNAP_MODEL", "Qwen/Qwen2.5-7B")
DTYPE = torch.bfloat16
SNAP_PATH = "/tmp/model-snapshot"
PROMPT = "The quick brown fox jumps over"

log.info("Model: %s (dtype: %s)", MODEL_ID, DTYPE)
log.info("")

# ============================================================
# Benchmark 1: from_pretrained cold (first download)
# ============================================================
log.info("=== from_pretrained (cold — downloading) ===")
t0 = time.monotonic()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
t_tok = time.monotonic()
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map="cpu")
model.eval()
t_load = time.monotonic()
log.info("  Tokenizer: %.2fs", t_tok - t0)
log.info("  Model load (cold): %.2fs", t_load - t_tok)

# Quick inference sanity check
inputs = tokenizer(PROMPT, return_tensors="pt")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
result_original = tokenizer.decode(out[0], skip_special_tokens=True)
t_inf = time.monotonic()
log.info("  Inference: %.2fs", t_inf - t_load)
log.info("  Output: %s", result_original[:120])

# Save state dict for later comparison
original_sd = {k: v.clone() for k, v in model.state_dict().items()}
n_params = sum(p.numel() for p in model.parameters())
weight_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
log.info("  Params: %.1fB (%.1f GB on disk)", n_params / 1e9, weight_bytes / 1e9)
print("")

# ============================================================
# Benchmark 2: from_pretrained warm (HF cache hit)
# ============================================================
log.info("=== from_pretrained (warm — cached on disk) ===")
del model
gc.collect()

times_warm = []
for i in range(3):
    t0w = time.monotonic()
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map="cpu")
    m.eval()
    t1w = time.monotonic()
    times_warm.append(t1w - t0w)
    log.info("  from_pretrained warm %d: %.2fs", i + 1, t1w - t0w)
    del m
    gc.collect()
warm_avg = sum(times_warm) / len(times_warm)
print("")

# ============================================================
# Benchmark 3: Snapshot
# ============================================================
log.info("=== Snapshot ===")

# Reload model for snapshot
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map="cpu")
model.eval()

from zerostart.snapshot import snapshot, hydrate, snapshot_info

import shutil
shutil.rmtree(SNAP_PATH, ignore_errors=True)

t_snap0 = time.monotonic()
snapshot(
    state={"model": model, "tokenizer": tokenizer},
    path=SNAP_PATH,
)
t_snap1 = time.monotonic()
log.info("  Snapshot: %.2fs", t_snap1 - t_snap0)

info = snapshot_info(SNAP_PATH)
log.info("  Tensors: %d total, %d matched to safetensors, %d serialized",
         info["tensor_count"], info["matched_tensors"], info["unmatched_tensors"])

# Show snapshot size on disk
import subprocess
snap_size = subprocess.run(["du", "-sh", SNAP_PATH], capture_output=True, text=True)
log.info("  Snapshot dir size: %s", snap_size.stdout.strip().split("\t")[0])
print("")

# ============================================================
# Benchmark 4: Hydrate
# ============================================================
log.info("=== Hydrate ===")
del model
gc.collect()

times_hydrate = []
for i in range(3):
    t_h0 = time.monotonic()
    restored = hydrate(SNAP_PATH)
    t_h1 = time.monotonic()
    times_hydrate.append(t_h1 - t_h0)
    log.info("  Hydrate %d: %.2fs", i + 1, t_h1 - t_h0)

    if i == 0:
        # Verify weights on first run
        model2 = restored["model"]
        model2.eval()
        hydrated_sd = model2.state_dict()
        mismatches = []
        for key in original_sd:
            if key not in hydrated_sd:
                mismatches.append(f"missing: {key}")
            elif not torch.equal(original_sd[key], hydrated_sd[key]):
                diff = (original_sd[key].float() - hydrated_sd[key].float()).abs().max().item()
                mismatches.append(f"{key} (max diff: {diff:.2e})")
        if mismatches:
            log.warning("  Weight mismatches: %s", mismatches[:5])
        else:
            log.info("  Weights: PASS (%d/%d match)", len(hydrated_sd), len(original_sd))

        # Verify inference
        inputs2 = tokenizer(PROMPT, return_tensors="pt")
        with torch.no_grad():
            out2 = model2.generate(**inputs2, max_new_tokens=20, do_sample=False)
        result_hydrated = tokenizer.decode(out2[0], skip_special_tokens=True)
        log.info("  Output match: %s", result_original == result_hydrated)
        if result_original != result_hydrated:
            log.info("    Original: %s", result_original[:120])
            log.info("    Hydrated: %s", result_hydrated[:120])

    del restored
    gc.collect()

hydrate_avg = sum(times_hydrate) / len(times_hydrate)
print("")

# ============================================================
# Summary
# ============================================================
print("=" * 60)
print(f"MODEL: {MODEL_ID} ({n_params/1e9:.1f}B params, {weight_bytes/1e9:.1f} GB weights)")
print(f"")
print(f"  from_pretrained (warm avg): {warm_avg:.2f}s")
print(f"  hydrate (avg):              {hydrate_avg:.2f}s")
print(f"  speedup:                    {warm_avg/hydrate_avg:.1f}x")
print(f"  snapshot creation:          {t_snap1-t_snap0:.2f}s")
print(f"  tensors matched:            {info['matched_tensors']}/{info['tensor_count']}")
print(f"  output match:               {result_original == result_hydrated}")
print("=" * 60)
PYEOF

df -h /tmp | tail -1 | awk '{print "Disk before model download: " $4 " free"}'
python3 /tmp/test_snapshot_real.py 2>&1
echo ""
echo "Exit: $?"
