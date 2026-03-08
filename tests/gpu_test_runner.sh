#!/bin/bash
# GPU Test Runner — runs zerostart test plan on a GPU pod
# Usage: gpu run "bash tests/gpu_test_runner.sh"
set -euo pipefail

PASS=0
FAIL=0
SKIP=0

pass() { echo "  ✓ PASS: $1"; ((PASS++)); }
fail() { echo "  ✗ FAIL: $1 — $2"; ((FAIL++)); }
skip() { echo "  - SKIP: $1 — $2"; ((SKIP++)); }

echo "=== zerostart GPU Test Runner ==="
echo "Date: $(date)"
echo "Python: $(python3 --version)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo ""

# --- Setup ---
echo "--- Setup ---"

# Source cargo env (rustup installs to non-standard paths on GPU pods)
for p in "$HOME/.cargo/env" "/gpu-cli-workspaces/.cache/cargo/env"; do
    [ -f "$p" ] && source "$p"
done

# Navigate to project root
cd /workspace/zerostart 2>/dev/null || cd "$(dirname "$0")/.."
echo "project root: $(pwd)"

# Create a fresh venv (the synced .venv has macOS python which breaks maturin)
echo "Creating fresh venv..."
rm -rf /tmp/zs-venv
python3 -m venv /tmp/zs-venv
source /tmp/zs-venv/bin/activate
pip install maturin 2>&1 | tail -1

# Build PyO3 module first (zs-fast-wheel must exist before zerostart can import it)
echo "Building zs-fast-wheel PyO3 module..."
cd crates/zs-fast-wheel
maturin develop --release 2>&1 | tail -3
cd ../..
echo "zs_fast_wheel built"

# Install zerostart from source
echo "Installing zerostart..."
pip install -e python/ 2>&1 | tail -1
echo "zerostart installed: $(which zerostart)"

# Verify imports
python3 -c "import zerostart; import zs_fast_wheel; print('imports ok')"

echo ""
echo "=== 1. Basic Functionality ==="

# --- 1.1 Smoke test — single small package ---
echo ""
echo "--- 1.1 Smoke test — single small package ---"
cat > /tmp/test_smoke.py << 'PYEOF'
import requests
print(f"requests: {requests.__version__}")
PYEOF
echo 'requests' > /tmp/reqs_smoke.txt
if zerostart -r /tmp/reqs_smoke.txt /tmp/test_smoke.py 2>&1; then
    pass "1.1 smoke test"
else
    fail "1.1 smoke test" "exit $?"
fi

# --- 1.2 No requirements ---
echo ""
echo "--- 1.2 No requirements ---"
echo 'print("hello from zerostart")' > /tmp/test_noreqs.py
if output=$(zerostart /tmp/test_noreqs.py 2>&1); then
    if echo "$output" | grep -q "hello from zerostart"; then
        pass "1.2 no requirements"
    else
        fail "1.2 no requirements" "output missing expected string"
    fi
else
    fail "1.2 no requirements" "exit $?"
fi

# --- 1.3 Inline packages with -p ---
echo ""
echo "--- 1.3 Inline packages with -p ---"
cat > /tmp/test_inline.py << 'PYEOF'
import yaml
print(f"yaml: {yaml.__version__}")
PYEOF
if zerostart -p pyyaml /tmp/test_inline.py 2>&1; then
    pass "1.3 inline packages"
else
    fail "1.3 inline packages" "exit $?"
fi

# --- 1.4 Script args passthrough ---
echo ""
echo "--- 1.4 Script args passthrough ---"
cat > /tmp/test_args.py << 'PYEOF'
import sys
print(f"args: {sys.argv[1:]}")
assert sys.argv[1] == "--port", f"expected --port, got {sys.argv[1]}"
assert sys.argv[2] == "8000", f"expected 8000, got {sys.argv[2]}"
print("args ok")
PYEOF
if zerostart /tmp/test_args.py -- --port 8000 2>&1; then
    pass "1.4 script args"
else
    fail "1.4 script args" "exit $?"
fi

# --- 1.5 Auto-detect requirements.txt ---
echo ""
echo "--- 1.5 Auto-detect requirements.txt ---"
mkdir -p /tmp/test_autodetect
echo 'six' > /tmp/test_autodetect/requirements.txt
echo 'import six; print(f"six: {six.__version__}")' > /tmp/test_autodetect/app.py
if (cd /tmp/test_autodetect && zerostart app.py 2>&1); then
    pass "1.5 auto-detect requirements.txt"
else
    fail "1.5 auto-detect requirements.txt" "exit $?"
fi

# --- 1.6 PEP 723 inline script metadata ---
echo ""
echo "--- 1.6 PEP 723 inline metadata ---"
cat > /tmp/test_pep723.py << 'PYEOF'
# /// script
# dependencies = ["requests", "six"]
# ///

import requests
import six
print(f"requests: {requests.__version__}, six: {six.__version__}")
PYEOF
if zerostart /tmp/test_pep723.py 2>&1; then
    pass "1.6 PEP 723 inline metadata"
else
    fail "1.6 PEP 723 inline metadata" "exit $?"
fi

# --- 1.7 PEP 723 with version constraints ---
echo ""
echo "--- 1.7 PEP 723 with version constraints ---"
cat > /tmp/test_pep723_ver.py << 'PYEOF'
# /// script
# dependencies = [
#     "numpy>=1.24",
#     "pyyaml~=6.0",
# ]
# ///

import numpy as np
import yaml
print(f"numpy: {np.__version__}, yaml: {yaml.__version__}")
PYEOF
if zerostart /tmp/test_pep723_ver.py 2>&1; then
    pass "1.7 PEP 723 version constraints"
else
    fail "1.7 PEP 723 version constraints" "exit $?"
fi

# --- 1.8 PEP 723 multiline ---
echo ""
echo "--- 1.8 PEP 723 multiline ---"
cat > /tmp/test_pep723_multi.py << 'PYEOF'
# /// script
# dependencies = [
#     "requests",
#     "click",
#     "rich",
# ]
# ///

import requests, click, rich
print(f"requests: {requests.__version__}, click: {click.__version__}, rich: {rich.__version__}")
PYEOF
if zerostart /tmp/test_pep723_multi.py 2>&1; then
    pass "1.8 PEP 723 multiline"
else
    fail "1.8 PEP 723 multiline" "exit $?"
fi

# --- 1.9 Explicit -p overrides inline metadata ---
echo ""
echo "--- 1.9 -p overrides inline metadata ---"
cat > /tmp/test_override.py << 'PYEOF'
# /// script
# dependencies = ["requests"]
# ///

import six
print(f"six: {six.__version__}")
PYEOF
if zerostart -p six /tmp/test_override.py 2>&1; then
    pass "1.9 -p overrides inline"
else
    fail "1.9 -p overrides inline" "exit $?"
fi

echo ""
echo "=== 2. Package Variety ==="

# --- 2.1 Pure Python packages ---
echo ""
echo "--- 2.1 Pure Python packages ---"
cat > /tmp/test_pure.py << 'PYEOF'
import requests, click, rich, six, idna, certifi
print(f"requests={requests.__version__} click={click.__version__} rich={rich.__version__}")
print(f"six={six.__version__} idna={idna.__version__} certifi={certifi.__version__}")
PYEOF
if zerostart -p 'requests' -p 'click' -p 'rich' -p 'six' -p 'idna' -p 'certifi' /tmp/test_pure.py 2>&1; then
    pass "2.1 pure Python packages"
else
    fail "2.1 pure Python packages" "exit $?"
fi

# --- 2.2 Compiled extensions ---
echo ""
echo "--- 2.2 Compiled extensions ---"
cat > /tmp/test_compiled.py << 'PYEOF'
import numpy as np
import orjson
arr = np.array([1, 2, 3])
print(f"numpy: {np.__version__}, array: {arr}")
print(f"orjson: {orjson.__version__}")
PYEOF
if zerostart -p numpy -p orjson /tmp/test_compiled.py 2>&1; then
    pass "2.2 compiled extensions"
else
    fail "2.2 compiled extensions" "exit $?"
fi

# --- 2.3 Large ML packages ---
echo ""
echo "--- 2.3 Large ML packages (torch + transformers) ---"
cat > /tmp/test_ml.py << 'PYEOF'
import torch
import transformers
import safetensors
print(f"torch: {torch.__version__}, cuda: {torch.cuda.is_available()}")
print(f"transformers: {transformers.__version__}")
print(f"safetensors: {safetensors.__version__}")
if torch.cuda.is_available():
    x = torch.randn(10, device="cuda")
    print(f"GPU tensor: {x.shape} on {x.device}")
PYEOF
if zerostart -p torch -p transformers -p safetensors /tmp/test_ml.py 2>&1; then
    pass "2.3 large ML packages"
else
    fail "2.3 large ML packages" "exit $?"
fi

# --- 2.4 Non-obvious import names ---
echo ""
echo "--- 2.4 Non-obvious import names ---"
cat > /tmp/test_names.py << 'PYEOF'
import yaml       # from pyyaml
from PIL import Image  # from pillow
import sklearn     # from scikit-learn
import dateutil    # from python-dateutil
import bs4         # from beautifulsoup4
print(f"yaml={yaml.__version__} PIL=ok sklearn={sklearn.__version__}")
print(f"dateutil={dateutil.__version__} bs4={bs4.__version__}")
PYEOF
if zerostart -p pyyaml -p pillow -p scikit-learn -p python-dateutil -p beautifulsoup4 /tmp/test_names.py 2>&1; then
    pass "2.4 non-obvious import names"
else
    fail "2.4 non-obvious import names" "exit $?"
fi

# --- 2.5 Packages with many transitive deps ---
echo ""
echo "--- 2.5 Transitive deps (boto3) ---"
cat > /tmp/test_transitive.py << 'PYEOF'
import boto3
import botocore
print(f"boto3: {boto3.__version__}, botocore: {botocore.__version__}")
PYEOF
if zerostart -p boto3 /tmp/test_transitive.py 2>&1; then
    pass "2.5 transitive deps"
else
    fail "2.5 transitive deps" "exit $?"
fi

echo ""
echo "=== 3. Version Pinning & Changes ==="

# --- 3.1 Pinned versions ---
echo ""
echo "--- 3.1 Pinned versions ---"
cat > /tmp/test_pinned.py << 'PYEOF'
import numpy as np
import requests
# Check exact versions
assert np.__version__ == "1.26.4", f"expected numpy 1.26.4, got {np.__version__}"
assert requests.__version__ == "2.31.0", f"expected requests 2.31.0, got {requests.__version__}"
print(f"numpy={np.__version__} requests={requests.__version__} — versions correct")
PYEOF
echo -e 'numpy==1.26.4\nrequests==2.31.0' > /tmp/reqs_pinned.txt
rm -rf /tmp/zs_cache_pinned
if zerostart -r /tmp/reqs_pinned.txt --cache-dir /tmp/zs_cache_pinned /tmp/test_pinned.py 2>&1; then
    pass "3.1 pinned versions"
else
    fail "3.1 pinned versions" "exit $?"
fi

# --- 3.2 Version upgrade — cache invalidation ---
echo ""
echo "--- 3.2 Version upgrade cache invalidation ---"
rm -rf /tmp/zs_cache_upgrade

# Run 1: numpy 1.26.4
echo 'numpy==1.26.4' > /tmp/reqs_v1.txt
cat > /tmp/test_v1.py << 'PYEOF'
import numpy as np
print(f"v1: numpy={np.__version__}")
assert np.__version__ == "1.26.4"
PYEOF
zerostart -r /tmp/reqs_v1.txt --cache-dir /tmp/zs_cache_upgrade /tmp/test_v1.py 2>&1
ENVS_AFTER_V1=$(ls /tmp/zs_cache_upgrade/envs/ 2>/dev/null | wc -l)

# Run 2: numpy 1.26.3 (different version)
echo 'numpy==1.26.3' > /tmp/reqs_v2.txt
cat > /tmp/test_v2.py << 'PYEOF'
import numpy as np
print(f"v2: numpy={np.__version__}")
assert np.__version__ == "1.26.3"
PYEOF
zerostart -r /tmp/reqs_v2.txt --cache-dir /tmp/zs_cache_upgrade /tmp/test_v2.py 2>&1
ENVS_AFTER_V2=$(ls /tmp/zs_cache_upgrade/envs/ 2>/dev/null | wc -l)

if [ "$ENVS_AFTER_V2" -gt "$ENVS_AFTER_V1" ]; then
    pass "3.2 version upgrade creates new cache"
else
    fail "3.2 version upgrade" "expected more envs after v2 ($ENVS_AFTER_V2 <= $ENVS_AFTER_V1)"
fi

# --- 3.3 Adding a package ---
echo ""
echo "--- 3.3 Adding a package to requirements ---"
rm -rf /tmp/zs_cache_add

echo 'six' > /tmp/reqs_add1.txt
echo 'import six; print(f"six={six.__version__}")' > /tmp/test_add.py
zerostart -r /tmp/reqs_add1.txt --cache-dir /tmp/zs_cache_add /tmp/test_add.py 2>&1
ENVS_1=$(ls /tmp/zs_cache_add/envs/ 2>/dev/null | wc -l)

echo -e 'six\nidna' > /tmp/reqs_add2.txt
cat > /tmp/test_add2.py << 'PYEOF'
import six, idna
print(f"six={six.__version__} idna={idna.__version__}")
PYEOF
zerostart -r /tmp/reqs_add2.txt --cache-dir /tmp/zs_cache_add /tmp/test_add2.py 2>&1
ENVS_2=$(ls /tmp/zs_cache_add/envs/ 2>/dev/null | wc -l)

if [ "$ENVS_2" -gt "$ENVS_1" ]; then
    pass "3.3 adding package creates new cache"
else
    fail "3.3 adding package" "expected new env ($ENVS_2 <= $ENVS_1)"
fi

# --- 3.4 Warm cache hit ---
echo ""
echo "--- 3.4 Warm cache hit ---"
rm -rf /tmp/zs_cache_warm
echo 'import six; print(f"six={six.__version__}")' > /tmp/test_warm.py
echo 'six' > /tmp/reqs_warm.txt

# Cold run
T0=$(date +%s%N)
zerostart -r /tmp/reqs_warm.txt --cache-dir /tmp/zs_cache_warm /tmp/test_warm.py 2>&1
T1=$(date +%s%N)
COLD_MS=$(( (T1 - T0) / 1000000 ))

# Warm run
T0=$(date +%s%N)
zerostart -r /tmp/reqs_warm.txt --cache-dir /tmp/zs_cache_warm /tmp/test_warm.py 2>&1
T1=$(date +%s%N)
WARM_MS=$(( (T1 - T0) / 1000000 ))

echo "  cold: ${COLD_MS}ms, warm: ${WARM_MS}ms"
if [ "$WARM_MS" -lt "$COLD_MS" ]; then
    pass "3.4 warm cache faster than cold"
else
    fail "3.4 warm cache" "warm (${WARM_MS}ms) not faster than cold (${COLD_MS}ms)"
fi

echo ""
echo "=== 4. Progressive Loading ==="

# --- 4.1 Import while installing ---
echo ""
echo "--- 4.1 Import while installing ---"
cat > /tmp/test_progressive.py << 'PYEOF'
import time
t0 = time.monotonic()
import torch
t_torch = time.monotonic() - t0

t1 = time.monotonic()
import numpy
t_numpy = time.monotonic() - t1

print(f"torch import: {t_torch:.2f}s")
print(f"numpy import: {t_numpy:.2f}s")
print(f"torch: {torch.__version__}, cuda: {torch.cuda.is_available()}")
PYEOF
rm -rf /tmp/zs_cache_prog
if zerostart -p torch -p numpy --cache-dir /tmp/zs_cache_prog /tmp/test_progressive.py 2>&1; then
    pass "4.1 progressive loading"
else
    fail "4.1 progressive loading" "exit $?"
fi

# --- 4.2 Submodule imports ---
echo ""
echo "--- 4.2 Submodule imports ---"
cat > /tmp/test_submod.py << 'PYEOF'
import torch
import torch.nn
import torch.nn.functional as F
from torch import Tensor
print(f"torch={torch.__version__} nn={torch.nn} F={F}")
PYEOF
rm -rf /tmp/zs_cache_submod
if zerostart -p torch --cache-dir /tmp/zs_cache_submod /tmp/test_submod.py 2>&1; then
    pass "4.2 submodule imports"
else
    fail "4.2 submodule imports" "exit $?"
fi

# --- 4.3 Try/except imports ---
echo ""
echo "--- 4.3 Speculative imports (try/except) ---"
cat > /tmp/test_speculative.py << 'PYEOF'
import time
t0 = time.monotonic()
try:
    import flash_attn
    HAS_FLASH = True
except ImportError:
    HAS_FLASH = False
elapsed = time.monotonic() - t0
print(f"flash_attn import took {elapsed:.1f}s, available: {HAS_FLASH}")
# Should fail within speculative timeout (~5s), not 300s
assert elapsed < 10, f"speculative import took too long: {elapsed:.1f}s"
print("speculative timeout ok")
PYEOF
rm -rf /tmp/zs_cache_spec
if zerostart -p torch --cache-dir /tmp/zs_cache_spec /tmp/test_speculative.py 2>&1; then
    pass "4.3 speculative imports"
else
    fail "4.3 speculative imports" "exit $?"
fi

echo ""
echo "=== 5. Error Handling ==="

# --- 5.1 Script that exits non-zero ---
echo ""
echo "--- 5.1 Script exits non-zero ---"
echo 'import sys; print("exiting 1"); sys.exit(1)' > /tmp/test_exit1.py
if ! zerostart /tmp/test_exit1.py 2>&1; then
    pass "5.1 non-zero exit propagated"
else
    fail "5.1 non-zero exit" "expected non-zero exit"
fi

# --- 5.2 Script that raises ---
echo ""
echo "--- 5.2 Script raises exception ---"
echo 'raise RuntimeError("boom")' > /tmp/test_raise.py
if ! zerostart /tmp/test_raise.py 2>&1; then
    pass "5.2 exception propagated"
else
    fail "5.2 exception" "expected non-zero exit"
fi

# --- 5.3 Non-existent script ---
echo ""
echo "--- 5.3 Non-existent script ---"
if ! zerostart /tmp/this_script_does_not_exist.py 2>&1; then
    pass "5.3 non-existent script fails"
else
    fail "5.3 non-existent script" "expected failure"
fi

echo ""
echo "=== 6. Real Application Stacks ==="

# --- 6.1 HuggingFace pipeline ---
echo ""
echo "--- 6.1 HuggingFace transformers pipeline ---"
cat > /tmp/test_hf.py << 'PYEOF'
from transformers import pipeline
pipe = pipeline("text-generation", model="gpt2", device="cuda")
result = pipe("Hello, I am", max_new_tokens=20)
print(result[0]["generated_text"])
print("HF pipeline ok")
PYEOF
rm -rf /tmp/zs_cache_hf
if timeout 300 zerostart -p torch -p transformers -p accelerate --cache-dir /tmp/zs_cache_hf /tmp/test_hf.py 2>&1; then
    pass "6.1 HuggingFace pipeline"
else
    fail "6.1 HuggingFace pipeline" "exit $?"
fi

# --- 6.2 Training loop ---
echo ""
echo "--- 6.2 PyTorch training loop ---"
cat > /tmp/test_train.py << 'PYEOF'
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(10, 1).cuda()
optimizer = optim.Adam(model.parameters())
x = torch.randn(32, 10).cuda()
y = torch.randn(32, 1).cuda()

for i in range(10):
    loss = nn.functional.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

print(f"Training complete. Final loss: {loss.item():.4f}")
PYEOF
rm -rf /tmp/zs_cache_train
if zerostart -p torch --cache-dir /tmp/zs_cache_train /tmp/test_train.py 2>&1; then
    pass "6.2 training loop"
else
    fail "6.2 training loop" "exit $?"
fi

# --- 6.3 FastAPI + torch ---
echo ""
echo "--- 6.3 FastAPI + torch ---"
cat > /tmp/test_api.py << 'PYEOF'
import torch
from fastapi import FastAPI

app = FastAPI()

@app.get("/predict")
def predict():
    x = torch.randn(10, device="cuda")
    return {"result": x.tolist()}

print(f"torch: {torch.__version__}, cuda: {torch.cuda.is_available()}")
print("FastAPI app created ok")
PYEOF
rm -rf /tmp/zs_cache_api
if zerostart -p torch -p fastapi -p uvicorn --cache-dir /tmp/zs_cache_api /tmp/test_api.py 2>&1; then
    pass "6.3 FastAPI + torch"
else
    fail "6.3 FastAPI + torch" "exit $?"
fi

# --- 6.4 Data processing (no GPU) ---
echo ""
echo "--- 6.4 Data processing (pandas + numpy) ---"
cat > /tmp/test_data.py << 'PYEOF'
import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.randn(1000, 4), columns=list("ABCD"))
print(f"Shape: {df.shape}")
print(f"Mean A: {df['A'].mean():.4f}")
PYEOF
rm -rf /tmp/zs_cache_data
if zerostart -p pandas -p numpy --cache-dir /tmp/zs_cache_data /tmp/test_data.py 2>&1; then
    pass "6.4 data processing"
else
    fail "6.4 data processing" "exit $?"
fi

# --- 6.5 PEP 723 real-world script ---
echo ""
echo "--- 6.5 PEP 723 real-world ML script ---"
cat > /tmp/test_pep723_real.py << 'PYEOF'
# /// script
# dependencies = [
#     "torch>=2.0",
#     "numpy",
#     "pyyaml",
# ]
# ///

import torch
import numpy as np
import yaml

config = yaml.safe_load('lr: 0.001\nepochs: 10')
x = torch.tensor(np.random.randn(5, 3), dtype=torch.float32)
if torch.cuda.is_available():
    x = x.cuda()
print(f"tensor shape: {x.shape}, device: {x.device}")
print(f"config: {config}")
print("PEP 723 real-world ok")
PYEOF
rm -rf /tmp/zs_cache_pep723
if zerostart --cache-dir /tmp/zs_cache_pep723 /tmp/test_pep723_real.py 2>&1; then
    pass "6.5 PEP 723 real-world"
else
    fail "6.5 PEP 723 real-world" "exit $?"
fi

echo ""
echo "=== 7. Performance ==="

# --- 7.1 Install speed: zs-fast-wheel warm vs uv ---
echo ""
echo "--- 7.1 Install speed comparison ---"
SITE_PKG=$(mktemp -d)

echo "  uv pip install (medium workload):"
T0=$(date +%s%N)
uv pip install numpy pandas scikit-learn --target "$SITE_PKG" 2>&1 | tail -1
T1=$(date +%s%N)
UV_MS=$(( (T1 - T0) / 1000000 ))
echo "  uv: ${UV_MS}ms"
rm -rf "$SITE_PKG"/*

echo "  zs-fast-wheel warm (medium workload):"
T0=$(date +%s%N)
bin/zs-fast-wheel-linux-x86_64 warm --requirements "numpy
pandas
scikit-learn" --site-packages "$SITE_PKG" 2>&1 | tail -3
T1=$(date +%s%N)
ZS_MS=$(( (T1 - T0) / 1000000 ))
echo "  zs-fast-wheel: ${ZS_MS}ms"
rm -rf "$SITE_PKG"

if [ "$ZS_MS" -lt "$UV_MS" ]; then
    echo "  speedup: $(echo "scale=1; $UV_MS / $ZS_MS" | bc)x"
    pass "7.1 zs-fast-wheel faster than uv"
else
    echo "  uv was faster this time (${UV_MS}ms vs ${ZS_MS}ms)"
    skip "7.1 speed comparison" "uv faster (may vary by network)"
fi

echo ""
echo "==========================================="
echo "RESULTS: $PASS passed, $FAIL failed, $SKIP skipped"
echo "==========================================="

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
