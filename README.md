# zerostart

Parallel streaming wheel extraction for installing large Python packages on remote GPUs.

```bash
zerostart run serve.py
```

Auto-detects dependencies from PEP 723 inline metadata, `pyproject.toml`, or `requirements.txt`. Works on any container GPU provider — RunPod, Vast.ai, Lambda, etc.

## Benchmarks

### Cold Start (first run, empty cache)

Cold start speedup depends on pod network bandwidth. zerostart opens multiple parallel HTTP connections per wheel — this helps when a single connection can't saturate the link, but doesn't help when one connection already maxes out the pipe.

| Pod network | Workload | zerostart | uv | Speedup |
|-------------|----------|-----------|-----|---------|
| Moderate (~200 Mbps) | torch (6.8 GB) | 33s | 98s | 3x |
| Moderate (~200 Mbps) | triton (638 MB) | 3.4s | 1.0s | uv faster |
| Fast (~1 Gbps) | diffusers+torch (7 GB) | 57s | 57s | ~1x |

On bandwidth-constrained pods (common with cheaper providers), parallel Range requests download large wheels 3x faster. On fast-network pods, a single connection already saturates the link and both tools finish in about the same time. For small packages, zerostart's startup overhead makes uv faster — just use uv directly.

### Warm Start (cached environment)

Warm starts are where zerostart consistently wins regardless of network speed. uv re-resolves dependencies and rebuilds the environment on every invocation. zerostart checks a cache marker and exec's Python directly.

| Workload | zerostart | uv | Speedup |
|----------|-----------|-----|---------|
| torch | 1.8s | 13.2s | 7x |
| vllm | 3.3s | 14.5s | 4x |
| triton | 0.2s | 1.0s | 5x |

All measured on RunPod (RTX 4090 / A6000).

### End-to-End: Install + Download + Load Model

Full cold-to-inference benchmark with Qwen3.5-35B-A3B (34.7B params, MoE) on RTX A6000:

| Test | Description | Time |
|------|-------------|------|
| Baseline full cold | uv install + HF download + model load | 349s |
| Baseline warm | uv cached + HF cached | 59s |
| zerostart full cold | install + HF download + model load | 428s |
| **zerostart warm** | **env cached + HF cached** | **15s** |
| zerostart cold install | env rebuild, uv cached | 98s |

The big win is warm starts: **15s vs 59s** (4x faster). uv re-resolves and re-links 53 packages on every run; zerostart checks a cache marker and runs immediately. For full cold starts, HF model download (~270s) dominates both paths.

## How It Works

### Cold starts: parallel Range-request streaming

uv downloads each wheel as a single HTTP connection. zerostart uses HTTP Range requests to download multiple chunks of each wheel in parallel, and starts extracting files while chunks are still arriving:

```
uv (sequential per wheel):
  torch.whl  [=========downloading=========>] then [==extracting==]
  numpy.whl  [=====>] then [=]

zerostart (parallel chunks, overlapped extraction):
  torch.whl  chunk1 [====>]──extract──►
             chunk2 [====>]──extract──►     ← 4 concurrent Range requests
             chunk3 [====>]──extract──►       per large wheel
             chunk4 [====>]──extract──►
  numpy.whl  [=>]──extract──►               ← all wheels in parallel
```

### Warm starts: Rust cache check vs re-resolve

uv re-resolves dependencies and rebuilds the tool environment on every invocation — even when packages are cached. For vllm (177 packages), that means metadata checks and linking for each one.

zerostart's warm path is three operations in Rust:
1. `stat(".complete")` — does the cached environment exist?
2. `find("lib/python*/site-packages")` — locate it
3. `exec(python)` — run directly

### Shared CUDA layer cache

CUDA libraries (nvidia-cublas, nvidia-cudnn, nvidia-nccl, etc.) are ~6GB and identical across torch, vllm, and diffusers environments. zerostart caches extracted wheels and hardlinks them into new environments — so the second torch-based environment skips re-extracting those 6GB.

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

The entire cold path runs in Rust — no Python orchestrator:

```
zerostart run -p torch serve.py

  1. Find Python          (uv python find || which python3)
  2. Check warm cache     (stat .complete marker — instant)
  3. Resolve deps         (uv pip compile --format pylock.toml)
  4. Check shared cache   (hardlink cached CUDA libs)
  5. Stream wheels        (parallel Range-request download + extract)
  6. exec(python)         (replaces process, no overhead)
```

Key design decisions:

- **All wheels through the streaming daemon** — every package with a wheel URL goes through parallel download+extract. Only sdist-only packages (rare) fall back to `uv pip install`.
- **Atomic extraction** — each wheel extracts to a staging directory, then renames into site-packages. Partial extractions never corrupt the target.
- **No venv overhead** — uses a flat site-packages directory with a content-addressed cache key.
- **Demand-driven scheduling** — when Python hits `import torch`, the daemon reprioritizes torch to the front of the download queue.

## Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `ZS_PARALLEL_DOWNLOADS` | 16 | Concurrent HTTP connections |
| `ZS_EXTRACT_THREADS` | num_cpus * 2 | Parallel extraction threads |
| `ZS_CHUNK_MB` | 16 | Streaming chunk size (MB) for Range requests |
| `ZEROSTART_CACHE` | `~/.cache/zerostart` | Cache directory |

## When to Use It

**Good fit:**
- Repeated runs on the same pod — warm starts are 4-7x faster than uv
- Large GPU packages on bandwidth-constrained pods — parallel downloads help when a single connection is slow
- Spot instances, CI/CD, autoscaling where you restart often and warm cache pays off

**Not worth it:**
- One-off cold starts on fast-network pods — uv is just as fast
- Small packages — uv is faster, zerostart adds startup overhead
- Local NVMe with models in page cache

## Requirements

- Linux (container GPU providers: RunPod, Vast.ai, Lambda, etc.)
- `uv` for dependency resolution (pre-installed on most GPU containers)
- Python 3.10+

macOS works for development (same CLI, no streaming optimization).

## gpu-cli Integration

If you use [gpu-cli](https://gpu-cli.sh):

```bash
gpu run "zerostart run -p torch serve.py"
```

## License

MIT

---

Built by the [gpu-cli](https://gpu-cli.sh) team.
