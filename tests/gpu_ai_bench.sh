#!/bin/bash
set -euo pipefail

# Heavy AI package benchmark: zerostart vs uvx on GPU pod
# Tests realistic AI cold-start scenarios
# NOTE: Each torch-based test uses ~7-15GB disk. We clean between every run.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ZS="$PROJECT_DIR/bin/zerostart-linux-x86_64"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export ZEROSTART_CACHE="/tmp/.zs-bench"

echo "=== AI Package Cold Start Benchmark ==="
echo "Date: $(date -u)"
echo "Python: $(python3 --version)"
echo "uv: $(uv --version)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Binary: $ZS"
df -h /tmp | tail -1 | awk '{print "Disk: " $4 " free"}'
echo ""

# Results file
RESULTS="$PROJECT_DIR/benches/results/ai_bench.csv"
mkdir -p "$PROJECT_DIR/benches/results"
echo "test,method,time_s,exit_code" > "$RESULTS"

measure() {
    local label=$1
    local method=$2
    shift 2
    echo "--- $label ($method) ---"
    local start=$(date +%s%3N)
    if "$@" 2>&1; then
        local rc=0
    else
        local rc=$?
    fi
    local end=$(date +%s%3N)
    local elapsed_ms=$((end - start))
    local elapsed_s=$((elapsed_ms / 1000)).$((elapsed_ms % 1000 / 100))
    echo "  >> ${elapsed_s}s (exit=$rc)"
    echo "$label,$method,${elapsed_s},$rc" >> "$RESULTS"
    echo ""
}

cleanup() {
    # Aggressively clean ALL caches to free disk for next test
    rm -rf /tmp/.zs-bench /tmp/.zs-vllm /tmp/.zs-test
    uv cache clean 2>/dev/null || true
    rm -rf ~/.local/share/uv/tools 2>/dev/null || true
    # Clean uv archive cache (this is where extracted venvs live)
    rm -rf /gpu-cli-workspaces/.cache/uv/archive-v0/* 2>/dev/null || true
    df -h /tmp | tail -1 | awk '{print "  (disk: " $4 " free)"}'
}

# Create test scripts
cat > /tmp/torch_test.py << 'EOF'
import torch
print(f"torch {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    x = torch.randn(1000, 1000, device='cuda')
    print(f"CUDA tensor OK: {x.shape}")
EOF

cat > /tmp/vllm_test.py << 'EOF'
import vllm
print(f"vllm {vllm.__version__}")
import torch
print(f"torch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
EOF

cat > /tmp/hf_test.py << 'EOF'
import transformers
import torch
print(f"transformers {transformers.__version__}")
print(f"torch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
EOF

cat > /tmp/diff_test.py << 'EOF'
import diffusers
import torch
print(f"diffusers {diffusers.__version__}")
print(f"torch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
EOF

cat > /tmp/triton_test.py << 'EOF'
import triton
print(f"triton {triton.__version__}")
EOF

# ============================================================
# Test 1: torch (the big one — ~900MB wheel + CUDA deps)
# ============================================================
echo "========== TEST 1: torch =========="

cleanup
measure "torch" "zs_cold" $ZS run -v -p torch /tmp/torch_test.py
measure "torch" "zs_warm" $ZS run -p torch /tmp/torch_test.py

cleanup
measure "torch" "uvx_cold" uvx --from torch --with torch python -c "import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
measure "torch" "uvx_warm" uvx --from torch --with torch python -c "import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# ============================================================
# Test 2: vllm (torch + vllm + many deps — realistic LLM serving)
# ============================================================
echo "========== TEST 2: vllm =========="

cleanup
measure "vllm" "zs_cold" $ZS run -v -p vllm /tmp/vllm_test.py
measure "vllm" "zs_warm" $ZS run -p vllm /tmp/vllm_test.py

cleanup
measure "vllm" "uvx_cold" uvx --from vllm --with vllm python -c "import vllm; print(f'vllm {vllm.__version__}')"
measure "vllm" "uvx_warm" uvx --from vllm --with vllm python -c "import vllm; print(f'vllm {vllm.__version__}')"

# ============================================================
# Test 3: transformers + torch (common fine-tuning setup)
# ============================================================
echo "========== TEST 3: transformers+torch =========="

cleanup
measure "hf_torch" "zs_cold" $ZS run -v -p torch -p transformers -p tokenizers /tmp/hf_test.py
measure "hf_torch" "zs_warm" $ZS run -p torch -p transformers -p tokenizers /tmp/hf_test.py

# ============================================================
# Test 4: diffusers + torch (image generation)
# ============================================================
echo "========== TEST 4: diffusers+torch =========="

cleanup
measure "diffusers" "zs_cold" $ZS run -v -p torch -p diffusers -p transformers -p accelerate /tmp/diff_test.py
measure "diffusers" "zs_warm" $ZS run -p torch -p diffusers -p transformers -p accelerate /tmp/diff_test.py

# ============================================================
# Test 5: triton (GPU kernel compilation)
# ============================================================
echo "========== TEST 5: triton =========="

cleanup
measure "triton" "zs_cold" $ZS run -v -p triton /tmp/triton_test.py
measure "triton" "zs_warm" $ZS run -p triton /tmp/triton_test.py

# ============================================================
# Summary
# ============================================================
echo ""
echo "========== RESULTS =========="
cat "$RESULTS"
echo ""
echo "Done."
