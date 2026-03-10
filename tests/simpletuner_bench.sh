#!/bin/bash
set -uo pipefail

echo "=== SimpleTuner Install Benchmark ==="
echo "Date: $(date -u)"
echo "Python: $(python3 --version)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
df -h /tmp | tail -1 | awk '{print "Disk: " $4 " free"}'
echo ""

# Install system deps that some Python packages need
apt-get install -y -qq libsndfile1 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ZS="$PROJECT_DIR/bin/zerostart-linux-x86_64"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# SimpleTuner base + cuda deps as a requirements file
# SimpleTuner deps adapted for Python 3.11 (RunPod default)
# Removed: skrample (3.12+), torchcodec (3.12+), sdnq (3.12+),
#          trainingsample (3.12+), peft-singlora (3.12+), ramtorch (3.12+)
# These are SimpleTuner-specific; core ML stack is the same
cat > /tmp/simpletuner-requirements.txt << 'EOF'
diffusers>=0.36.0
transformers>=4.55.0
hf_transfer>=0.1.0
datasets>=3.0.1
wandb>=0.21.0
requests>=2.32.4
pillow>=11.3.0
accelerate>=1.5.2
safetensors>=0.5.3
compel>=2.1.1
clip-interrogator>=0.6.0
open-clip-torch>=2.26.1
scipy>=1.11.1
boto3>=1.35.83
pandas>=2.2.3
botocore>=1.35.83
torchsde>=0.2.6
torchmetrics>=1.1.1
colorama>=0.4.6
numpy>=2.2.0
num2words>=0.5.13
peft>=0.17.0
tensorboard>=2.18.0
sentencepiece>=0.2.0
spacy>=3.7.4
optimum-quanto>=0.2.7
lycoris-lora>=3.4.0
torch-optimi>=0.2.1
librosa>=0.10.2
loguru>=0.7.2
toml>=0.10.2
fastapi>=0.115.0
sse-starlette>=1.6.5
beautifulsoup4>=4.12.3
tokenizers>=0.21.0
huggingface-hub>=0.34.3
imageio>=2.37.0
hf-xet>=1.1.5
vector-quantize-pytorch>=1.27.15
cryptography>=41.0.0
aiosqlite>=0.19.0
httpx>=0.28.0
psutil>=5.9.0
torch>=2.10.0
torchvision>=0.25.0
torchaudio>=2.10.0
triton>=3.3.0
bitsandbytes>=0.45.0
deepspeed>=0.17.2
torchao>=0.14.0,<0.16.0
nvidia-cudnn-cu12
nvidia-nccl-cu12
nvidia-ml-py>=12.555
lm-eval>=0.4.4
EOF

REQS=$(cat /tmp/simpletuner-requirements.txt | wc -l)
echo "Requirements: $REQS packages"
echo ""

# Test script that imports key SimpleTuner deps
cat > /tmp/st_test_imports.py << 'PYEOF'
import time
t0 = time.monotonic()

import torch
t1 = time.monotonic()
import diffusers
t2 = time.monotonic()
import transformers
t3 = time.monotonic()
import accelerate
import safetensors
import numpy
import pandas
t4 = time.monotonic()

print(f"torch={t1-t0:.2f}s diffusers={t2-t1:.2f}s transformers={t3-t2:.2f}s rest={t4-t3:.2f}s total={t4-t0:.2f}s")
print(f"  torch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
print(f"  diffusers {diffusers.__version__}")
print(f"  transformers {transformers.__version__}")
print("SimpleTuner core deps OK")
PYEOF

# ============================================================
# Test 1: zerostart cold start
# ============================================================
echo "--- Test 1: zerostart cold start ---"
export ZEROSTART_CACHE="/tmp/.zs-st-bench"
export ZS_NO_SHARED_CACHE=1
rm -rf "$ZEROSTART_CACHE"
df -h /tmp | tail -1 | awk '{print "Disk before: " $4 " free"}'

start=$(date +%s%3N)
$ZS run -v -r /tmp/simpletuner-requirements.txt /tmp/st_test_imports.py 2>&1
zs_exit=$?
end=$(date +%s%3N)
zs_cold=$((end - start))
echo "  zerostart cold: ${zs_cold}ms (exit: $zs_exit)"
df -h /tmp | tail -1 | awk '{print "Disk after: " $4 " free"}'
echo ""

# ============================================================
# Test 2: zerostart warm start
# ============================================================
echo "--- Test 2: zerostart warm start ---"
for i in 1 2 3; do
    start=$(date +%s%3N)
    $ZS run -r /tmp/simpletuner-requirements.txt /tmp/st_test_imports.py 2>&1
    end=$(date +%s%3N)
    echo "  zerostart warm $i: $((end - start))ms"
done
echo ""

# ============================================================
# Summary
# ============================================================
echo "=== SUMMARY ==="
echo "  Direct requirements: $REQS"
echo "  zerostart cold: ${zs_cold}ms"
df -h /tmp | tail -1 | awk '{print "  Disk remaining: " $4 " free"}'
echo ""
echo "=== DONE ==="
