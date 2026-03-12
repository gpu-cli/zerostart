#!/usr/bin/env bash
# Benchmark: Model loading cold (download+load) vs accelerate()
# Tests with Qwen3.5-35B-A3B (MoE, ~18GB weights)
set -uo pipefail

echo "============================================================"
echo "  Model Loading Benchmark (Cold = download + load)"
echo "  $(date -u)"
echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "  Disk: $(df -h / | tail -1 | awk '{print $4}') free"
echo "============================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Install deps first, then zerostart SDK last
pip install -q accelerate transformers safetensors 2>&1 | tail -3
cp "$PROJECT_DIR/README.md" "$PROJECT_DIR/python/README.md" 2>/dev/null || true
sed -i 's|readme = "../README.md"|readme = "README.md"|' "$PROJECT_DIR/python/pyproject.toml" 2>/dev/null || true
cd "$PROJECT_DIR/python" && pip install . 2>&1 | tail -3
cd "$PROJECT_DIR"

python3 -c "import zerostart; print('zerostart SDK: ok')" || { echo "FATAL: zerostart not importable"; exit 1; }

disk_free() { df -h / | tail -1 | awk '{print $4}'; }

# Find all possible HF cache locations and remove the model dir from each
clear_hf_cache() {
    local MODEL="$1"
    local SAFE_ID="${MODEL//\//--}"
    local MODEL_DIR="models--${SAFE_ID}"

    # Check all known HF cache locations
    for cache_dir in \
        "${HF_HUB_CACHE:-}" \
        "${HF_HOME:+${HF_HOME}/hub}" \
        "$HOME/.cache/huggingface/hub" \
        "/gpu-cli-workspaces/.cache/huggingface/hub" \
        "/workspace/.cache/huggingface/hub" \
        "/root/.cache/huggingface/hub"; do
        if [ -n "$cache_dir" ] && [ -d "$cache_dir/$MODEL_DIR" ]; then
            echo "  [cleanup] Removing HF cache: $cache_dir/$MODEL_DIR"
            rm -rf "$cache_dir/$MODEL_DIR"
        fi
    done
}

bench_cold() {
    local MODEL="$1"
    local LABEL="$2"

    echo "============================================================"
    echo "  $LABEL: $MODEL"
    echo "  Disk: $(disk_free) free"
    echo "============================================================"

    # ---------------------------------------------------------------
    # Test 1: COLD baseline — no HF cache, download + load
    # ---------------------------------------------------------------
    echo "--- Test 1: Baseline COLD (download + load, no cache) ---"
    clear_hf_cache "$MODEL"
    python3 << PYEOF
import time, os, gc, torch
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
gc.collect(); torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
t0 = time.monotonic()
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('$MODEL', device_map='auto', dtype=torch.bfloat16)
t1 = time.monotonic()
param_b = sum(p.numel() for p in model.parameters()) / 1e9
peak_mb = torch.cuda.max_memory_allocated() / 1e6
print(f'BASELINE_COLD: {t1-t0:.2f}s  (params={param_b:.1f}B, peak_gpu={peak_mb:.0f}MB)')
del model; gc.collect(); torch.cuda.empty_cache()
PYEOF
    echo "  Disk after baseline cold: $(disk_free) free"
    echo ""

    # ---------------------------------------------------------------
    # Test 2: WARM baseline — HF cache populated, just load
    # ---------------------------------------------------------------
    echo "--- Test 2: Baseline WARM (HF cached, just load) ---"
    python3 << PYEOF
import time, os, gc, torch
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
gc.collect(); torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
t0 = time.monotonic()
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('$MODEL', device_map='auto', dtype=torch.bfloat16)
t1 = time.monotonic()
peak_mb = torch.cuda.max_memory_allocated() / 1e6
print(f'BASELINE_WARM: {t1-t0:.2f}s  (peak_gpu={peak_mb:.0f}MB)')
del model; gc.collect(); torch.cuda.empty_cache()
PYEOF
    echo ""

    # ---------------------------------------------------------------
    # Test 3: accelerate() COLD — no zs cache, HF cache present
    # ---------------------------------------------------------------
    echo "--- Test 3: accelerate() COLD (no zs cache, HF cached) ---"
    rm -rf /tmp/zs-models ~/.cache/zerostart/models /gpu-cli-workspaces/zs-models 2>/dev/null || true
    python3 << PYEOF
import time, os, gc, torch, logging
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
logging.basicConfig(level=logging.INFO, format='  %(name)s: %(message)s')
gc.collect(); torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
t0 = time.monotonic()
import zerostart
zerostart.accelerate()
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('$MODEL', device_map='auto', dtype=torch.bfloat16)
t1 = time.monotonic()
peak_mb = torch.cuda.max_memory_allocated() / 1e6
print(f'ACCELERATE_COLD: {t1-t0:.2f}s  (peak_gpu={peak_mb:.0f}MB)')
zerostart.decelerate()
del model; gc.collect(); torch.cuda.empty_cache()
PYEOF
    echo ""

    # ---------------------------------------------------------------
    # Test 4: accelerate() WARM — zs cache populated
    # ---------------------------------------------------------------
    echo "--- Test 4: accelerate() WARM (zs cache populated) ---"
    python3 << PYEOF
import time, os, gc, torch, logging
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
logging.basicConfig(level=logging.INFO, format='  %(name)s: %(message)s')
gc.collect(); torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
t0 = time.monotonic()
import zerostart
zerostart.accelerate()
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('$MODEL', device_map='auto', dtype=torch.bfloat16)
t1 = time.monotonic()
peak_mb = torch.cuda.max_memory_allocated() / 1e6
print(f'ACCELERATE_WARM: {t1-t0:.2f}s  (peak_gpu={peak_mb:.0f}MB)')
zerostart.decelerate()
del model; gc.collect(); torch.cuda.empty_cache()
PYEOF
    echo ""

    # ---------------------------------------------------------------
    # Test 5: accelerate() FULL COLD — no HF cache, no zs cache
    # ---------------------------------------------------------------
    echo "--- Test 5: accelerate() FULL COLD (download + load, no cache) ---"
    clear_hf_cache "$MODEL"
    rm -rf /tmp/zs-models ~/.cache/zerostart/models /gpu-cli-workspaces/zs-models 2>/dev/null || true
    python3 << PYEOF
import time, os, gc, torch, logging
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
logging.basicConfig(level=logging.INFO, format='  %(name)s: %(message)s')
gc.collect(); torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
t0 = time.monotonic()
import zerostart
zerostart.accelerate()
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('$MODEL', device_map='auto', dtype=torch.bfloat16)
t1 = time.monotonic()
peak_mb = torch.cuda.max_memory_allocated() / 1e6
print(f'ACCELERATE_FULL_COLD: {t1-t0:.2f}s  (peak_gpu={peak_mb:.0f}MB)')
zerostart.decelerate()
del model; gc.collect(); torch.cuda.empty_cache()
PYEOF
    echo ""

    # Cleanup
    clear_hf_cache "$MODEL"
    rm -rf /tmp/zs-models ~/.cache/zerostart/models /gpu-cli-workspaces/zs-models 2>/dev/null || true
    echo "  [cleanup] Disk: $(disk_free) free"
    echo ""
}

# Qwen3.5-35B-A3B: MoE model, ~35B total params, ~3B active, ~18GB weights
bench_cold "Qwen/Qwen3.5-35B-A3B" "Large MoE (35B-A3B)"

echo ""
echo "============================================"
echo "  SUMMARY"
echo "============================================"
echo "  BASELINE_COLD    = download + from_pretrained (no cache)"
echo "  BASELINE_WARM    = from_pretrained (HF cached)"
echo "  ACCELERATE_COLD  = accelerate() first load (HF cached)"
echo "  ACCELERATE_WARM  = accelerate() cached load (mmap hydrate)"
echo "  ACCELERATE_FULL  = accelerate() download + load (no cache)"
echo "============================================"
echo ""
echo "Done."
