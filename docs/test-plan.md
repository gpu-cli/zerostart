# zerostart Test Plan

Comprehensive test plan covering every real-world scenario. All GPU tests run via `gpu run`.

## Test Categories

1. [Basic Functionality](#1-basic-functionality)
2. [Package Variety](#2-package-variety)
3. [Version Pinning & Changes](#3-version-pinning--changes)
4. [Progressive Loading](#4-progressive-loading)
5. [Cache Behavior](#5-cache-behavior)
6. [Error Handling & Edge Cases](#6-error-handling--edge-cases)
7. [Real Application Stacks](#7-real-application-stacks)
8. [Performance & Benchmarks](#8-performance--benchmarks)

---

## 1. Basic Functionality

### 1.1 Smoke test — single small package
```bash
# Verify the simplest possible case works
echo 'import requests; print(requests.get("https://httpbin.org/get").status_code)' > /tmp/test_smoke.py
echo 'requests' > /tmp/requirements.txt
zerostart -r /tmp/requirements.txt /tmp/test_smoke.py
```
**Expected:** prints `200`, exits 0

### 1.2 Smoke test — no requirements
```bash
echo 'print("hello")' > /tmp/test_noreqs.py
zerostart /tmp/test_noreqs.py
```
**Expected:** prints `hello`, no install step, exits 0

### 1.3 Inline packages with -p
```bash
echo 'import yaml; print(yaml.__version__)' > /tmp/test_inline.py
zerostart -p pyyaml /tmp/test_inline.py
```
**Expected:** prints pyyaml version

### 1.4 Script args passthrough
```bash
cat > /tmp/test_args.py << 'EOF'
import sys
print(f"args: {sys.argv[1:]}")
assert sys.argv[1] == "--port"
assert sys.argv[2] == "8000"
EOF
zerostart /tmp/test_args.py --port 8000
```
**Expected:** prints `args: ['--port', '8000']`

### 1.5 Auto-detect requirements.txt
```bash
mkdir -p /tmp/test_autodetect && cd /tmp/test_autodetect
echo 'six' > requirements.txt
echo 'import six; print(six.__version__)' > app.py
zerostart app.py
```
**Expected:** finds requirements.txt automatically, prints six version

### 1.6 PEP 723 inline script metadata
```bash
cat > /tmp/test_pep723.py << 'EOF'
# /// script
# dependencies = ["requests", "six"]
# ///

import requests
import six
print(f"requests: {requests.__version__}, six: {six.__version__}")
EOF
zerostart /tmp/test_pep723.py
```
**Expected:** reads deps from the script block, no -p or -r needed

### 1.7 PEP 723 with version constraints
```bash
cat > /tmp/test_pep723_versions.py << 'EOF'
# /// script
# dependencies = [
#     "numpy>=1.24",
#     "pyyaml~=6.0",
# ]
# ///

import numpy as np
import yaml
print(f"numpy: {np.__version__}, yaml: {yaml.__version__}")
EOF
zerostart /tmp/test_pep723_versions.py
```
**Expected:** installs packages satisfying constraints

### 1.8 PEP 723 multiline with trailing commas
```bash
cat > /tmp/test_pep723_multi.py << 'EOF'
# /// script
# dependencies = [
#     "requests",
#     "click",
#     "rich",
# ]
# ///

import requests, click, rich
print("all imported")
EOF
zerostart /tmp/test_pep723_multi.py
```

### 1.9 Explicit -p overrides inline metadata
```bash
cat > /tmp/test_override.py << 'EOF'
# /// script
# dependencies = ["requests"]
# ///

import six
print(six.__version__)
EOF
# -p should take priority, inline metadata ignored
zerostart -p six /tmp/test_override.py
```
**Expected:** installs six (from -p), not requests (from inline)

---

## 2. Package Variety

### 2.1 Pure Python packages (small, no compiled extensions)
```
requests, flask, click, rich, httpx, fastapi, pydantic
```
**Test:** Install all, import all, verify versions
**Why:** These should all go through uv (< 1MB wheels)

### 2.2 Compiled extensions (medium)
```
numpy, pandas, pillow, cryptography, orjson, msgpack
```
**Test:** Install all, verify compiled extensions load (e.g., `numpy.array([1,2,3])`)
**Why:** Platform-specific wheels, .so files

### 2.3 Large ML packages
```
torch, transformers, tokenizers, safetensors, accelerate
```
**Test:** Install all, verify `torch.cuda.is_available()`, load a small model
**Why:** These are the primary use case — multi-GB wheels

### 2.4 Packages with non-obvious import names
```
pyyaml → yaml
pillow → PIL
scikit-learn → sklearn
python-dateutil → dateutil
beautifulsoup4 → bs4
```
**Test:** Install by distribution name, import by module name
**Why:** Tests import_map resolution in resolver.py and lazy_imports.py

### 2.5 Packages with native CUDA extensions
```
triton, flash-attn, bitsandbytes, xformers
```
**Test:** Install and import on GPU pod, verify CUDA functionality
**Why:** These have CUDA-specific wheels and runtime GPU checks

### 2.6 Namespace packages
```
google-cloud-storage, google-auth, azure-storage-blob
```
**Test:** Install and import (e.g., `from google.cloud import storage`)
**Why:** Namespace packages use implicit namespace — no `__init__.py` in parent

### 2.7 Packages with many transitive deps
```
boto3 (pulls ~15 deps), django (pulls ~10), jupyter (pulls ~50+)
```
**Test:** Install top-level, verify transitive deps resolve correctly
**Why:** Tests that resolver handles deep dependency trees

---

## 3. Version Pinning & Changes

### 3.1 Pinned versions
```
torch==2.1.0
numpy==1.26.4
transformers==4.38.0
```
**Test:** Verify exact versions installed, not latest
**Why:** Production deployments pin versions

### 3.2 Version range specifiers
```
torch>=2.0,<2.3
numpy~=1.26.0
requests>=2.28
```
**Test:** Verify resolved versions satisfy constraints
**Why:** Common in requirements.txt

### 3.3 Version upgrade — cache invalidation
```bash
# Run 1: install torch==2.1.0
echo 'torch==2.1.0' > /tmp/reqs.txt
zerostart -r /tmp/reqs.txt /tmp/test.py

# Run 2: upgrade to torch==2.2.0
echo 'torch==2.2.0' > /tmp/reqs.txt
zerostart -r /tmp/reqs.txt /tmp/test.py
```
**Expected:** Run 2 creates a NEW cache entry (different cache key), installs 2.2.0
**Why:** Version bumps in requirements.txt must not serve stale cached env

### 3.4 Version downgrade
```bash
# Same as above but go from 2.2.0 → 2.1.0
```
**Expected:** New cache entry, old version restored correctly
**Why:** Rollbacks are common in production

### 3.5 Adding a new package to existing requirements
```bash
# Run 1
echo -e 'torch==2.1.0\nnumpy' > /tmp/reqs.txt
zerostart -r /tmp/reqs.txt /tmp/test.py

# Run 2 — add transformers
echo -e 'torch==2.1.0\nnumpy\ntransformers' > /tmp/reqs.txt
zerostart -r /tmp/reqs.txt /tmp/test.py
```
**Expected:** Run 2 gets cache miss (different artifact set), installs all 3
**Why:** Common workflow — add dep, re-run

### 3.6 Removing a package from requirements
```bash
# Run 1: torch + numpy
# Run 2: just torch
```
**Expected:** Cache miss, new env with only torch
**Why:** Ensure removed deps don't leak from old cached envs

### 3.7 Same packages, different Python version
```bash
# Verify cache key includes python_version
# Same requirements resolved for 3.10 vs 3.11 should produce different cache keys
```
**Why:** Different Python versions may resolve to different wheel builds

---

## 4. Progressive Loading

### 4.1 Import while installing — basic
```python
# Script imports torch first (large), then small packages
import time
t0 = time.monotonic()
import torch  # should block until torch wheel extracted
t1 = time.monotonic()
import numpy  # may already be done
t2 = time.monotonic()
print(f"torch: {t1-t0:.1f}s, numpy: {t2-t1:.1f}s")
print(f"torch version: {torch.__version__}")
print(f"cuda: {torch.cuda.is_available()}")
```
**Expected:** torch import blocks briefly, numpy instant or near-instant

### 4.2 Demand prioritization
```python
# Import a large package that would normally be late in the queue
# It should get reprioritized when demanded
import safetensors  # small but may be queued behind torch
import torch        # large, gets reprioritized to front
```
**Expected:** Both imports succeed, torch reprioritized ahead of queue

### 4.3 Submodule imports
```python
import torch.nn
import torch.nn.functional as F
import transformers.models.auto
from PIL import Image
```
**Expected:** Top-level resolved once, submodules import without re-waiting

### 4.4 Try/except imports (speculative)
```python
# Common pattern in ML code
try:
    import flash_attn
    HAS_FLASH = True
except ImportError:
    HAS_FLASH = False

# Should timeout quickly (5s) if flash-attn not in requirements
```
**Expected:** Speculative timeout (5s), not full 300s timeout

### 4.5 Conditional imports
```python
import sys
if sys.platform == "linux":
    import torch
else:
    torch = None
```
**Expected:** Works correctly, import blocks only on Linux

### 4.6 Import ordering — small before large
```python
# Import small packages first, they should be instant
import requests    # small, installed via uv
import six         # small
import torch       # large, may still be downloading
```
**Expected:** requests and six instant (uv path), torch blocks until ready

### 4.7 Rapid successive imports of same package
```python
import torch
import torch  # second import — should be instant (cached in sys.modules)
import torch.nn  # submodule — should not re-wait
```
**Expected:** No double-wait, Python's module cache handles subsequent imports

---

## 5. Cache Behavior

### 5.1 Cold start — no cache
```bash
rm -rf .zerostart/
zerostart -p requests /tmp/test.py
```
**Expected:** Full resolve + install, creates cache entry, marks complete

### 5.2 Warm start — cache hit
```bash
# Run same command again
zerostart -p requests /tmp/test.py
```
**Expected:** "Cache hit" log message, instant start, no download

### 5.3 Cache key stability
```bash
# Same requirements in different order should produce same cache key
echo -e 'torch\nnumpy' > /tmp/reqs1.txt
echo -e 'numpy\ntorch' > /tmp/reqs2.txt
# Both should resolve to same cache key
```
**Expected:** Identical cache keys regardless of requirement order

### 5.4 Interrupted install — resume
```bash
# Start install, kill midway
zerostart -p torch /tmp/test.py &
sleep 3 && kill %1

# Re-run — should detect incomplete env
zerostart -p torch /tmp/test.py
```
**Expected:** Detects incomplete env, rebuilds cleanly

### 5.5 Corrupt cache — recovery
```bash
# Manually corrupt a cached env
rm -rf .zerostart/envs/*/lib/python*/site-packages/torch/
zerostart -p torch /tmp/test.py
```
**Expected:** Either detects corruption (import fails) or re-creates env

### 5.6 Multiple concurrent runs
```bash
# Two processes with same requirements
zerostart -p numpy /tmp/test1.py &
zerostart -p numpy /tmp/test2.py &
wait
```
**Expected:** No corruption, both succeed (may race on cache creation)

---

## 6. Error Handling & Edge Cases

### 6.1 Non-existent package
```bash
zerostart -p this-package-does-not-exist-xyz /tmp/test.py
```
**Expected:** Clear error message from resolver, exits non-zero

### 6.2 Package with no wheel (sdist only)
```bash
# Some packages only have source distributions
zerostart -p some-sdist-only-package /tmp/test.py
```
**Expected:** Graceful fallback or clear error

### 6.3 Network failure during download
```bash
# Simulate by using a bad URL in manifest
# Or disconnect network mid-download
```
**Expected:** Timeout, clear error, no partial corruption

### 6.4 Script that exits non-zero
```bash
echo 'import sys; sys.exit(1)' > /tmp/test_exit.py
zerostart -p requests /tmp/test_exit.py
```
**Expected:** Exits 1, cleanup still happens (hook removed, daemon shutdown)

### 6.5 Script that raises exception
```bash
echo 'raise RuntimeError("boom")' > /tmp/test_crash.py
zerostart -p requests /tmp/test_crash.py
```
**Expected:** Traceback printed, cleanup still happens

### 6.6 KeyboardInterrupt (Ctrl+C)
```bash
echo 'import time; time.sleep(999)' > /tmp/test_hang.py
zerostart -p torch /tmp/test_hang.py
# Press Ctrl+C during install or script execution
```
**Expected:** Clean shutdown, no zombie processes or corrupt cache

### 6.7 Import of package not in requirements
```python
# Script imports a package that wasn't in requirements
import some_random_package  # not requested
```
**Expected:** Normal ImportError (speculative timeout 5s, then fail)

### 6.8 Conflicting dependencies
```bash
echo -e 'numpy==1.24.0\nscipy==1.12.0' > /tmp/reqs.txt
# scipy 1.12 requires numpy>=1.22.4 — compatible
# But what about genuinely conflicting versions?
```
**Expected:** uv pip compile catches conflicts and reports error

### 6.9 Very large number of packages
```bash
# 100+ packages (e.g., full data science stack)
echo -e 'torch\ntransformers\npandas\nscipy\nmatplotlib\njupyter\nscikit-learn\nseaborn\nplotly\nbokeh\ndask\npolars\narrow\nfastapi\nuvicorn\nsqlalchemy\nalembic\ncelery\nredis\nboto3' > /tmp/reqs_big.txt
zerostart -r /tmp/reqs_big.txt /tmp/test.py
```
**Expected:** Resolves and installs all, progressive loading works

### 6.10 Package with post-install scripts/entry points
```bash
zerostart -p jupyter /tmp/test.py
```
**Expected:** Packages install correctly even without running setup.py entry_points

### 6.11 uv not installed
```bash
# Temporarily hide uv from PATH
PATH=/usr/bin:/bin zerostart -p requests /tmp/test.py
```
**Expected:** Clear error message that uv is required

---

## 7. Real Application Stacks

### 7.1 vLLM inference server
```bash
cat > /tmp/test_vllm.py << 'EOF'
from vllm import LLM, SamplingParams
llm = LLM(model="facebook/opt-125m")  # small model for testing
output = llm.generate(["Hello, world"], SamplingParams(max_tokens=20))
print(output[0].outputs[0].text)
EOF
zerostart -p vllm /tmp/test_vllm.py
```
**Expected:** Model loads, generates text, GPU memory used
**Why:** Primary use case

### 7.2 SGLang inference
```bash
cat > /tmp/test_sglang.py << 'EOF'
import sglang as sgl
# Basic smoke test
print(f"sglang version: {sgl.__version__}")
EOF
zerostart -p sglang /tmp/test_sglang.py
```

### 7.3 HuggingFace Transformers pipeline
```bash
cat > /tmp/test_hf.py << 'EOF'
from transformers import pipeline
pipe = pipeline("text-generation", model="gpt2", device="cuda")
result = pipe("Hello, I am", max_new_tokens=20)
print(result[0]["generated_text"])
EOF
zerostart -p 'torch transformers accelerate' /tmp/test_hf.py
```
**Expected:** Downloads gpt2, generates text on GPU

### 7.4 FastAPI + torch inference server
```bash
cat > /tmp/test_api.py << 'EOF'
import torch
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/predict")
def predict():
    x = torch.randn(10, device="cuda")
    return {"result": x.tolist()}

print(f"torch: {torch.__version__}, cuda: {torch.cuda.is_available()}")
print("FastAPI app created successfully")
# Don't actually serve — just verify imports work
EOF
zerostart -p 'torch fastapi uvicorn' /tmp/test_api.py
```

### 7.5 Training script (PyTorch)
```bash
cat > /tmp/test_train.py << 'EOF'
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
EOF
zerostart -p torch /tmp/test_train.py
```

### 7.6 Diffusion model (ComfyUI-style)
```bash
cat > /tmp/test_diffusion.py << 'EOF'
import torch
from diffusers import StableDiffusionPipeline

# Just verify imports and CUDA, don't actually run inference (needs >4GB VRAM)
print(f"torch: {torch.__version__}")
print(f"cuda: {torch.cuda.is_available()}")
print(f"diffusers imported successfully")
EOF
zerostart -p 'torch diffusers transformers accelerate safetensors' /tmp/test_diffusion.py
```

### 7.7 Data processing (no GPU)
```bash
cat > /tmp/test_data.py << 'EOF'
import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(1000, 4), columns=list("ABCD"))
print(f"Shape: {df.shape}")
print(f"Mean:\n{df.mean()}")
EOF
zerostart -p 'pandas numpy' /tmp/test_data.py
```
**Why:** Not everything is GPU — verify non-GPU workloads work too

### 7.8 Multi-framework (torch + tensorflow)
```bash
cat > /tmp/test_multi.py << 'EOF'
import torch
import tensorflow as tf
print(f"torch: {torch.__version__}, cuda: {torch.cuda.is_available()}")
print(f"tf: {tf.__version__}, gpus: {tf.config.list_physical_devices('GPU')}")
EOF
zerostart -p 'torch tensorflow' /tmp/test_multi.py
```
**Why:** Some users mix frameworks — verify no conflicts

---

## 8. Performance & Benchmarks

### 8.1 Install speed vs uv (small packages)
```bash
# Baseline: uv pip install
time uv pip install requests flask click rich httpx

# zerostart
time zerostart -p 'requests flask click rich httpx' /tmp/noop.py
```
**Expected:** Similar speed (bottleneck is resolution, not download)

### 8.2 Install speed vs uv (ML stack)
```bash
# Baseline
time uv pip install torch transformers safetensors accelerate tokenizers

# zerostart
time zerostart -p 'torch transformers safetensors accelerate tokenizers' /tmp/noop.py
```
**Expected:** zerostart 6-9x faster on large wheels

### 8.3 Time to first import (progressive)
```python
import time
t0 = time.monotonic()
import torch
t_torch = time.monotonic() - t0

t1 = time.monotonic()
import transformers
t_transformers = time.monotonic() - t1

print(f"torch: {t_torch:.2f}s")
print(f"transformers: {t_transformers:.2f}s")
```
**Expected:** torch < 3s even though full stack is 7GB

### 8.4 Warm cache performance
```bash
# Second run should be near-instant
time zerostart -p 'torch transformers' /tmp/noop.py  # cold
time zerostart -p 'torch transformers' /tmp/noop.py  # warm
```
**Expected:** Warm run < 1s (just cache lookup + path setup)

### 8.5 Memory usage during install
```bash
# Monitor RSS during large install
zerostart -p 'torch transformers' /tmp/noop.py &
PID=$!
while kill -0 $PID 2>/dev/null; do
    ps -o rss= -p $PID
    sleep 1
done
```
**Expected:** Reasonable memory (< 500MB RSS) — streaming should avoid loading full wheels into memory

---

## Execution Order

Run tests in this order (builds on previous results):

1. **1.1-1.5** — Smoke tests (catch fundamental breakage)
2. **2.1-2.2** — Small/medium packages (verify basic install pipeline)
3. **4.1, 4.6** — Progressive loading basics
4. **5.1-5.2** — Cache cold/warm
5. **2.3-2.5** — Large ML packages on GPU
6. **3.1-3.6** — Version pinning and cache invalidation
7. **4.2-4.7** — Advanced progressive loading
8. **6.1-6.11** — Error handling
9. **7.1-7.8** — Real application stacks
10. **8.1-8.5** — Performance benchmarks

## Test Runner

Each test should capture:
- Exit code
- Wall clock time
- stdout/stderr
- Cache state before/after (ls .zerostart/)

```bash
# Template for each test
echo "=== Test X.Y: description ==="
time zerostart ... 2>&1 | tee /tmp/test_X_Y.log
echo "exit: $?"
ls -la .zerostart/envs/ 2>/dev/null
echo "==="
```
