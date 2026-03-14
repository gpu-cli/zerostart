# zerostart

Fast cold starts for Python GPU applications. Drop-in wrapper that installs packages in the background while your app starts running.

```bash
zerostart run serve.py
```

Auto-detects dependencies from PEP 723 inline metadata, `pyproject.toml`, or `requirements.txt`. Works on any container GPU provider — RunPod, Vast.ai, Lambda, etc.

## Benchmarks

### Inference Server Cold Start

Time-to-health-check for an inference server (torch + transformers + fastapi + uvicorn) on RTX A6000:

| Metric | uv | zerostart | Speedup |
|--------|-----|-----------|---------|
| Health check ready | 46s | **1.8s** | **25x** |
| First CUDA inference | 55s | **33s** | 1.7x |

With progressive loading, your server accepts health probes in under 2 seconds while torch installs in the background. Kubernetes/load balancers see a healthy pod almost immediately.

### AI Pipeline Cold Starts

Cold start benchmarks across real AI workloads on RTX A6000:

| Workload | uv | zerostart | Speedup |
|----------|-----|-----------|---------|
| Diffusion (torch+diffusers+transformers) | 51s | **40s** | 22% faster |
| LLM Fine-tuning (torch+transformers+peft+trl) | 59s | **38s** | 36% faster |
| Computer Vision (torch+torchvision+opencv) | 54s | **40s** | 26% faster |
| Audio/Speech (torch+transformers+numpy) | 48s | **31s** | 35% faster |
| Data Science (pandas+sklearn+xgboost) | 8s | 8s | ~1x |

### Warm Start (cached environment)

Warm starts are where zerostart consistently wins. uv re-resolves and rebuilds the environment on every invocation. zerostart checks a cache marker and exec's Python directly.

| Workload | zerostart | uv | Speedup |
|----------|-----------|-----|---------|
| torch | 1.8s | 13.2s | 7x |
| vllm | 3.3s | 14.5s | 4x |
| triton | 0.2s | 1.0s | 5x |

All measured on RunPod (RTX 4090 / A6000).

## How It Works

### Progressive loading

Python starts immediately while packages install in the background. A lazy import hook blocks each `import` only until that specific package is extracted:

```
uv (sequential):
  [====== install all packages ======] then [start Python]
                                             46s to health check

zerostart (progressive):
  [start Python immediately]
  import fastapi  → ready in 0.3s   → /health responding at 1.8s
  import torch    → blocks 23s      → first inference at 33s
  [======= packages installing in background =======]
```

Small packages (fastapi, uvicorn) are available in under a second. Large packages (torch, transformers) block on import until extracted. Your app runs as soon as its first imports resolve.

### GET+pipeline architecture

Downloads full wheels via single GET requests (32 parallel connections), then pipelines extraction through 4 parallel workers. Biggest wheels download first to maximize overlap:

```
Download:  torch [==============>]  numpy [=>]  six [>]
Extract:         [worker 0: torch ======>]
                 [worker 1: numpy =>]
                 [worker 2: six >]
```

### System CUDA detection

On pods with CUDA pre-installed, zerostart detects the system CUDA version and skips downloading nvidia-* wheels when the system already provides compatible libraries. This saves ~2-6GB of downloads depending on the workload.

### Shared CUDA layer cache

CUDA libraries (nvidia-cublas, nvidia-cudnn, nvidia-nccl, etc.) are ~6GB and identical across torch, vllm, and diffusers environments. zerostart caches extracted wheels and hardlinks them into new environments — so the second torch-based environment skips re-extracting those 6GB.

### Warm starts: Rust cache check

zerostart's warm path is three operations in Rust:
1. `stat(".complete")` — does the cached environment exist?
2. `find("lib/python*/site-packages")` — locate it
3. `exec(python)` — run directly

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/gpu-cli/zerostart/main/install.sh | sh
```

Or manually:

```bash
# Binary only
curl -fsSL https://github.com/gpu-cli/zerostart/releases/latest/download/zerostart-linux-x86_64 \
  -o /usr/local/bin/zerostart && chmod +x /usr/local/bin/zerostart

# Python SDK (for accelerate(), vLLM integration)
pip install git+https://github.com/gpu-cli/zerostart.git#subdirectory=python
```

Requires Linux + Python 3.10+ + `uv` (pre-installed on most GPU containers).

## Quick Start

```bash
# Auto-detect deps from PEP 723 metadata, pyproject.toml, or requirements.txt
zerostart run serve.py

# Add extra packages on top of auto-detected deps
zerostart run -p torch serve.py

# Explicit requirements file
zerostart run -r requirements.txt serve.py

# Run a package directly
zerostart run torch -- -c "import torch; print(torch.cuda.is_available())"

# Pass args to your script
zerostart run serve.py -- --port 8000
```

### Dependency Detection

zerostart automatically finds dependencies — no flags needed:

1. **PEP 723 inline metadata** (checked first):
```python
# /// script
# dependencies = ["torch>=2.0", "transformers", "safetensors"]
# ///
import torch
```

2. **pyproject.toml** `[project.dependencies]` in the script's directory or parents

3. **requirements.txt** in the script's directory or parents

`-p` and `-r` flags add packages on top of whatever is auto-detected.

## Model Loading Acceleration

`zerostart.accelerate()` patches `from_pretrained` to speed up model loading. Sets `low_cpu_mem_usage=True` by default (skips random weight initialization), and auto-caches models for faster repeat loads on models that fit in GPU memory.

```python
import zerostart
zerostart.accelerate()

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", device_map="cuda")
```

| Model | Cold (download+load) | Baseline (HF cached) | accelerate() | Notes |
|-------|---------------------|---------------------|--------------|-------|
| Qwen3.5-35B-A3B | 299s | 10.1s | 10.1s | MoE, 34.7B params, device_map='auto' |
| Qwen2.5-7B | — | 5.5s | 3.3s | Fits in GPU, cache provides speedup |
| Qwen2.5-1.5B | — | 3.5s | 3.2s | Small model, minimal difference |

All measured on RTX A6000 (48GB). For models requiring `device_map='auto'` (model > VRAM), accelerate() matches baseline by eliminating random weight initialization. For models that fit entirely in GPU memory, the mmap cache provides additional speedup.

Or via CLI:

```bash
zerostart run --accelerate -p torch -p transformers serve.py
```

### How it works

| Hook | What it does |
|------|-------------|
| `low_cpu_mem_usage` | Sets `low_cpu_mem_usage=True` by default — skips random weight initialization |
| Auto-cache | Snapshots model on first load, mmap hydrate via `safe_open` on repeat (models that fit in GPU) |
| Parallel shard loading | Loads multiple safetensors shards concurrently during cache hydration |
| Suffix tensor matching | Handles MoE models where state_dict and safetensors use different key prefixes |
| Network volume fix | Eager read instead of mmap on NFS/JuiceFS (cold reads only*) |
| .bin conversion | Converts legacy checkpoints to safetensors, mmaps on repeat |

*Network volume fix only helps on cold reads from network-backed filesystems where mmap page faults trigger network round-trips. On FUSE with warm page cache (most container providers), mmap is already fast.

For `device_map='auto'` (model larger than VRAM), caching is skipped — HF's shard-by-shard loading directly to the right device is faster than our load-to-CPU-then-dispatch path.

### Model Cache

Models are automatically cached after first load:

```python
from zerostart.model_cache import ModelCache

cache = ModelCache("/volume/models")
cache.list_entries()                    # Show cached models
cache.auto_evict(max_size_bytes=50e9)   # LRU eviction to stay under 50GB
```

### vLLM Integration

A custom model loader for vLLM that switches safetensors loading from mmap to eager read on network filesystems (NFS, JuiceFS, CIFS).

**Not enabled by default.** On most container providers, the kernel page cache makes mmap fast enough. The eager path only helps on cold reads from slow network storage.

```bash
vllm serve Qwen/Qwen2.5-7B --load-format zerostart
```

Auto-registers via vLLM's plugin system when zerostart is installed.

## Architecture

The entire cold path runs in Rust with progressive loading:

```
zerostart run -p torch serve.py

  1. Find Python          (uv python find || which python3)
  2. Check warm cache     (stat .complete marker — instant)
  3. Resolve deps         (uv pip compile --format pylock.toml)
  4. Detect system CUDA   (skip nvidia wheels if system libs match)
  5. Check shared cache   (hardlink cached CUDA libs)
  6. Start daemon         (GET+pipeline download + parallel extract)
  7. Start Python         (immediately, with lazy import hook)
     └─ imports block only until their package is extracted
```

Key design decisions:

- **Progressive loading** — Python starts immediately. Imports block per-package, not globally. Health checks respond in seconds, not minutes.
- **GET+pipeline** — single GET per wheel (maximizes bandwidth), 4 parallel extract workers (prevents small wheels queuing behind large ones).
- **System CUDA detection** — skips nvidia-* wheels when the pod already has compatible CUDA libraries, saving ~2-6GB of downloads.
- **Atomic extraction** — each wheel extracts to a staging directory, then renames into site-packages. Partial extractions never corrupt the target.
- **No venv overhead** — uses a flat site-packages directory with a content-addressed cache key.

## Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `ZS_PARALLEL_DOWNLOADS` | 32 | Concurrent HTTP connections |
| `ZS_EXTRACT_THREADS` | num_cpus * 2 | Parallel extraction threads |
| `ZEROSTART_CACHE` | `~/.cache/zerostart` | Cache directory |
| `ZS_NO_SYSTEM_CUDA` | unset | Disable system CUDA detection |
| `ZS_NO_SHARED_CACHE` | unset | Disable shared wheel cache |

## When to Use It

**Good fit:**
- Inference servers — health check in <2s on cold start vs 45s+ with uv
- Repeated runs on the same pod — warm starts are 4-7x faster than uv
- Spot instances, CI/CD, autoscaling where you restart often
- Large GPU packages (torch, vllm, diffusers) — parallel downloads + progressive loading

**Not worth it:**
- Small packages only — uv is faster, zerostart adds startup overhead
- One-off scripts that don't need to respond to health checks during install

## Requirements

- Linux (container GPU providers: RunPod, Vast.ai, Lambda, etc.)
- `uv` for dependency resolution (pre-installed on most GPU containers)
- Python 3.10+

macOS works for development (same CLI, no GPU optimization).

## gpu-cli Integration

If you use [gpu-cli](https://gpu-cli.sh):

```bash
gpu run "zerostart run -p torch serve.py"
```

## License

MIT

---

Built by the [gpu-cli](https://gpu-cli.sh) team.
