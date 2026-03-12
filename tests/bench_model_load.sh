#!/usr/bin/env bash
# End-to-end benchmark: Full cold start (install deps + download model + load)
# Compares: uv baseline vs zerostart run --accelerate
set -uo pipefail

echo "============================================================"
echo "  End-to-End Cold Start Benchmark"
echo "  $(date -u)"
echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "  Disk: $(df -h / | tail -1 | awk '{print $4}') free"
echo "============================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL="${1:-Qwen/Qwen3.5-35B-A3B}"
DEPS=("torch>=2.5" "transformers" "safetensors" "accelerate")

disk_free() { df -h / | tail -1 | awk '{print $4}'; }

clear_hf_cache() {
    local M="$1"
    local SAFE_ID="${M//\//--}"
    local MODEL_DIR="models--${SAFE_ID}"
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

clear_all_caches() {
    echo "  [cleanup] Clearing all caches..."
    rm -rf /tmp/zs-bench-venv 2>/dev/null || true
    rm -rf ~/.cache/zerostart 2>/dev/null || true
    rm -rf /gpu-cli-workspaces/.cache/zerostart 2>/dev/null || true
    uv cache clean 2>/dev/null || true
}

clear_zs_env_cache() {
    rm -rf ~/.cache/zerostart/envs 2>/dev/null || true
    rm -rf /gpu-cli-workspaces/.cache/zerostart/envs 2>/dev/null || true
}

clear_zs_model_cache() {
    rm -rf /tmp/zs-models ~/.cache/zerostart/models /gpu-cli-workspaces/zs-models 2>/dev/null || true
}

# ---------------------------------------------------------------
# Setup
# ---------------------------------------------------------------
echo "--- Setup ---"
which uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
echo "uv: $(uv --version)"

# Install zerostart CLI into system python
cp "$PROJECT_DIR/README.md" "$PROJECT_DIR/python/README.md" 2>/dev/null || true
sed -i 's|readme = "../README.md"|readme = "README.md"|' "$PROJECT_DIR/python/pyproject.toml" 2>/dev/null || true
pip install -q "$PROJECT_DIR/python" 2>&1 | tail -3
python3 -c "import zerostart; print('zerostart SDK: ok')" || { echo "FATAL: zerostart not importable"; exit 1; }

DAEMON_BIN="$PROJECT_DIR/bin/zs-fast-wheel-linux-x86_64"
if [ -f "$DAEMON_BIN" ]; then
    export PATH="$(dirname "$DAEMON_BIN"):$PATH"
    echo "daemon: $DAEMON_BIN"
fi

# Create the self-timing serve script used by zerostart tests.
# Uses PEP 723 inline metadata for dep detection.
# Measures wall time from BENCH_T_START env var (set before zerostart invocation)
# through model load completion.
cat > /tmp/zs-bench-serve.py << 'PYEOF'
# /// script
# dependencies = ["torch>=2.5", "transformers", "safetensors", "accelerate"]
# ///
import time, os, gc

t_wall_start = float(os.environ.get('BENCH_T_START', '0'))
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
model_id = os.environ.get('BENCH_MODEL', 'Qwen/Qwen3.5-35B-A3B')

import torch
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', dtype=torch.bfloat16)

t_done = time.time() * 1000
param_b = sum(p.numel() for p in model.parameters()) / 1e9
peak_mb = torch.cuda.max_memory_allocated() / 1e6

if t_wall_start > 0:
    total = (t_done - t_wall_start) / 1000
    print(f'RESULT: {total:.1f}s  (params={param_b:.1f}B, peak_gpu={peak_mb:.0f}MB)')
else:
    print(f'RESULT: unknown (no BENCH_T_START)')

del model; gc.collect(); torch.cuda.empty_cache()
PYEOF

echo "deps: ${DEPS[*]}"
echo "model: $MODEL"
echo ""

echo "============================================================"
echo "  Disk: $(disk_free) free"
echo "============================================================"
echo ""

# ---------------------------------------------------------------
# Test 1: BASELINE FULL COLD — uv install + download + load
# No uv cache, no HF cache
# ---------------------------------------------------------------
echo "=== Test 1: Baseline FULL COLD (uv install + download + load) ==="
clear_all_caches
clear_hf_cache "$MODEL"
echo "  Disk before: $(disk_free) free"

T_START=$(date +%s%3N)

uv venv /tmp/zs-bench-venv --python python3 -q 2>&1
uv pip install --python /tmp/zs-bench-venv/bin/python "${DEPS[@]}" -q 2>&1 | tail -5

BENCH_T_START=$T_START BENCH_MODEL="$MODEL" \
    /tmp/zs-bench-venv/bin/python /tmp/zs-bench-serve.py

echo "  Disk after: $(disk_free) free"
echo ""

# ---------------------------------------------------------------
# Test 2: BASELINE WARM — uv install (cached) + model load (HF cached)
# ---------------------------------------------------------------
echo "=== Test 2: Baseline WARM (uv cached + HF cached) ==="
rm -rf /tmp/zs-bench-venv 2>/dev/null || true

T_START=$(date +%s%3N)

uv venv /tmp/zs-bench-venv --python python3 -q 2>&1
uv pip install --python /tmp/zs-bench-venv/bin/python "${DEPS[@]}" -q 2>&1 | tail -5

BENCH_T_START=$T_START BENCH_MODEL="$MODEL" \
    /tmp/zs-bench-venv/bin/python /tmp/zs-bench-serve.py

echo ""

# Clean up baseline venv
rm -rf /tmp/zs-bench-venv

# ---------------------------------------------------------------
# Test 3: ZEROSTART FULL COLD — no uv cache, no HF cache, no zs cache
# ---------------------------------------------------------------
echo "=== Test 3: zerostart FULL COLD (install + download + load) ==="
clear_all_caches
clear_hf_cache "$MODEL"
clear_zs_model_cache
echo "  Disk before: $(disk_free) free"

T_START=$(date +%s%3N)
BENCH_T_START=$T_START BENCH_MODEL="$MODEL" \
    zerostart --accelerate /tmp/zs-bench-serve.py 2>&1

echo "  Disk after: $(disk_free) free"
echo ""

# ---------------------------------------------------------------
# Test 4: ZEROSTART WARM — env cached + HF cached
# ---------------------------------------------------------------
echo "=== Test 4: zerostart WARM (env cached + HF cached) ==="

T_START=$(date +%s%3N)
BENCH_T_START=$T_START BENCH_MODEL="$MODEL" \
    zerostart --accelerate /tmp/zs-bench-serve.py 2>&1

echo ""

# ---------------------------------------------------------------
# Test 5: ZEROSTART COLD INSTALL — no env cache, HF + uv cached
# This tests: how fast does zerostart re-install when wheels are
# already in uv's cache but the env needs rebuilding?
# ---------------------------------------------------------------
echo "=== Test 5: zerostart COLD INSTALL (no env, HF+uv cached) ==="
clear_zs_env_cache

T_START=$(date +%s%3N)
BENCH_T_START=$T_START BENCH_MODEL="$MODEL" \
    zerostart --accelerate /tmp/zs-bench-serve.py 2>&1

echo ""

# ---------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------
clear_hf_cache "$MODEL"
clear_zs_model_cache
rm -rf /tmp/zs-bench-venv /tmp/zs-bench-serve.py
echo "  [cleanup] Disk: $(disk_free) free"

echo ""
echo "============================================"
echo "  SUMMARY"
echo "============================================"
echo "  Test 1: BASELINE_FULL_COLD  = uv pip install (no cache) + HF download + load"
echo "  Test 2: BASELINE_WARM       = uv pip install (cached)   + load (HF cached)"
echo "  Test 3: ZEROSTART_FULL_COLD = zerostart (no cache)      + HF download + load"
echo "  Test 4: ZEROSTART_WARM      = zerostart (env cached)    + load (HF cached)"
echo "  Test 5: ZEROSTART_COLD_INST = zerostart (uv cached)     + load (HF cached)"
echo "============================================"
echo ""
echo "Done."
