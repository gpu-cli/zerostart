# zerostart

Parallel streaming wheel extraction for installing large Python packages on remote GPUs.

```bash
zerostart run -p torch serve.py
```

Works on any container GPU provider — RunPod, Vast.ai, Lambda, etc.

## Benchmarks

Measured on RTX 4090 pods (RunPod). Results vary with network speed — slower pod networks show a larger advantage for zerostart because there's more room for parallel downloads to help.

### Cold Start (first run, empty cache)

| Workload | zerostart | uv | Speedup |
|----------|-----------|-----|---------|
| torch + CUDA (6.8 GB) | 33s | 98s | 3x |
| vllm (9.4 GB) | 60s | 58s | ~1x |
| triton (638 MB) | 3.4s | 1.0s | uv faster |

zerostart's cold start advantage comes from parallel HTTP Range requests. For large packages like torch on a bandwidth-limited connection, this matters. For small packages like triton, the overhead isn't worth it — just use uv.

vllm cold starts are roughly comparable. The package set is large (177 wheels) but many are small, so uv's single-connection approach keeps up.

### Warm Start (cached environment)

| Workload | zerostart | uv | Speedup |
|----------|-----------|-----|---------|
| torch | 1.8s | 13.2s | 7x |
| vllm | 3.3s | 14.5s | 4x |
| triton | 0.2s | 1.0s | 5x |

Warm starts are where zerostart consistently wins. uv re-resolves dependencies and rebuilds the environment on every run. zerostart checks a cache marker and exec's Python directly — no resolution, no environment setup.

### Network speed matters

On pods with slower network (common with cheaper providers), the cold start advantage grows because parallel Range requests can saturate the link where a single connection can't. On fast-network pods (1Gbps+), uv downloads quickly enough that the parallel approach helps less.

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
# Run a script with dependencies
zerostart run -p torch -p transformers serve.py

# Run inline
zerostart run torch -- -c "import torch; print(torch.cuda.is_available())"

# With a requirements file
zerostart run -r requirements.txt serve.py

# Pass args to your script
zerostart run serve.py -- --port 8000
```

### PEP 723 Inline Script Metadata

Embed dependencies directly in your script — no `requirements.txt` needed:

```python
# /// script
# dependencies = ["torch>=2.0", "transformers", "safetensors"]
# ///

import torch
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-uncased")
print(f"Loaded on {model.device}")
```

```bash
zerostart run serve.py  # deps auto-detected from script
```

## Model Loading Acceleration

`zerostart.accelerate()` patches `from_pretrained` to speed up model loading by skipping unnecessary work (random weight init, repeated downloads). Add one line:

```python
import zerostart
zerostart.accelerate()

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", device_map="cuda")
```

Or via CLI:

```bash
zerostart run --accelerate -p torch -p transformers serve.py
```

### How it works

| Hook | What it does |
|------|-------------|
| Meta device init | Skips random weight initialization during `from_pretrained` |
| Auto-cache | Snapshots model on first load, mmap hydrate on repeat |
| Network volume fix | Eager read instead of mmap on NFS/JuiceFS (cold reads only*) |
| .bin conversion | Converts legacy checkpoints to safetensors, mmaps on repeat |

*Network volume fix only helps on cold reads from network-backed filesystems where mmap page faults trigger network round-trips. On FUSE with warm page cache (most container providers), mmap is already fast.

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
- Large GPU packages (torch, vllm, diffusers) on container providers with moderate network
- Repeated runs where warm start time matters
- Spot instances, CI/CD, autoscaling where cold starts add up

**Not worth it:**
- Small packages — uv is already fast, zerostart adds overhead
- One-off scripts that don't repeat
- Pods with very fast network (1Gbps+) where uv cold starts are already quick
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
