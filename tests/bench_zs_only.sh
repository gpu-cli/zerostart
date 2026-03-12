#!/usr/bin/env bash
# Benchmark: zerostart-only tests (Tests 3-5)
# Run after bench_model_load.sh baseline tests, or standalone
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL="${1:-Qwen/Qwen3.5-35B-A3B}"
SERVE_SCRIPT="$SCRIPT_DIR/bench_serve.py"

export PATH="$PROJECT_DIR/bin:$PATH"

echo "============================================================"
echo "  Zerostart Benchmark"
echo "  $(date -u)"
echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "============================================================"

# Install latest SDK
pip install -q "$PROJECT_DIR/python" 2>&1 | tail -1
python3 -c "import zerostart; print('SDK: ok')"

disk_free() { df -h / | tail -1 | awk '{print $4}'; }

clear_hf_cache() {
    local SAFE_ID="${1//\//-}"  # Qwen/Foo → Qwen-Foo
    SAFE_ID="${SAFE_ID//-/--}"  # Qwen-Foo → Qwen--Foo (HF format: org--model)
    # Actually HF uses the exact format: models--Org--Model
    SAFE_ID="${1//\//__TMP__}"
    SAFE_ID="${SAFE_ID//__TMP__/--}"
    local MODEL_DIR="models--${SAFE_ID}"
    for cache_dir in \
        "${HF_HUB_CACHE:-}" \
        "${HF_HOME:+${HF_HOME}/hub}" \
        "$HOME/.cache/huggingface/hub" \
        "/gpu-cli-workspaces/.cache/huggingface/hub" \
        "/root/.cache/huggingface/hub"; do
        if [ -n "$cache_dir" ] && [ -d "$cache_dir/$MODEL_DIR" ]; then
            echo "  [cleanup] Removing HF cache: $cache_dir/$MODEL_DIR"
            rm -rf "$cache_dir/$MODEL_DIR"
        fi
    done
}

clear_all_zs() {
    echo "  [cleanup] Clearing all zerostart caches..."
    rm -rf ~/.cache/zerostart 2>/dev/null || true
    rm -rf /gpu-cli-workspaces/.cache/zerostart 2>/dev/null || true
    rm -rf /root/.cache/zerostart 2>/dev/null || true
}

clear_zs_env_cache() {
    rm -rf ~/.cache/zerostart/envs 2>/dev/null || true
    rm -rf /gpu-cli-workspaces/.cache/zerostart/envs 2>/dev/null || true
    rm -rf /root/.cache/zerostart/envs 2>/dev/null || true
}

# ---------------------------------------------------------------
# Test 3: ZEROSTART FULL COLD
# ---------------------------------------------------------------
echo ""
echo "=== Test 3: zerostart FULL COLD (install + download + load) ==="
clear_all_zs
uv cache clean 2>/dev/null || true
clear_hf_cache "$MODEL"
echo "  Disk before: $(disk_free) free"

T_START=$(date +%s%3N)
BENCH_T_START=$T_START BENCH_MODEL="$MODEL" \
    zerostart --accelerate "$SERVE_SCRIPT" 2>&1

echo "  Disk after: $(disk_free) free"
echo ""

# ---------------------------------------------------------------
# Test 4: ZEROSTART WARM
# ---------------------------------------------------------------
echo "=== Test 4: zerostart WARM (env cached + HF cached) ==="

T_START=$(date +%s%3N)
BENCH_T_START=$T_START BENCH_MODEL="$MODEL" \
    zerostart --accelerate "$SERVE_SCRIPT" 2>&1

echo ""

# ---------------------------------------------------------------
# Test 5: ZEROSTART COLD INSTALL (no env cache, HF+uv cached)
# ---------------------------------------------------------------
echo "=== Test 5: zerostart COLD INSTALL (no env, HF+uv cached) ==="
clear_zs_env_cache

T_START=$(date +%s%3N)
BENCH_T_START=$T_START BENCH_MODEL="$MODEL" \
    zerostart --accelerate "$SERVE_SCRIPT" 2>&1

echo ""
echo "============================================"
echo "  DONE"
echo "============================================"
