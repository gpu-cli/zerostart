# zerostart

**Fast cold starts for GPU Python.** Drop-in replacement for `uvx` that's 10-12x faster on cold starts and 10-30x faster on warm starts.

```bash
# Instead of: uvx --from torch python serve.py
zerostart run -p torch serve.py
```

No code changes. No platform lock-in. Works on any container GPU provider.

## Benchmarks

Measured on RTX 4090 pods (RunPod). Both tools download the same wheels from PyPI — zerostart is faster because of *how* it downloads.

### Cold Start (first run, empty cache)

| Workload | Packages | Size | zerostart | uvx | Speedup |
|----------|----------|------|-----------|-----|---------|
| torch + CUDA | 27 wheels | 6.8 GB | **47.9s** | 512.2s | **10.7x** |
| vllm (full LLM stack) | 177 wheels | 9.4 GB | **69.6s** | 862.1s | **12.4x** |
| transformers + torch | 51 wheels | 7.0 GB | **45.4s** | — | — |
| diffusers + torch | 60 wheels | 7.0 GB | **50.4s** | — | — |
| triton | 1 wheel | 638 MB | **7.0s** | — | — |

### Warm Start (cached environment)

| Workload | zerostart | uvx | Speedup |
|----------|-----------|-----|---------|
| torch | **2.5s** | 25.5s | **10.2x** |
| vllm | **3.6s** | 114.0s | **31.7x** |
| transformers + torch | **3.9s** | — | — |
| diffusers + torch | **5.3s** | — | — |
| triton | **0.9s** | — | — |

On a faster pod (1Gbps+ network), cold starts drop further:

| Workload | zerostart | uvx | Speedup |
|----------|-----------|-----|---------|
| torch | **23.1s** | 90.9s | **3.9x** |
| vllm | **33.3s** | 138.1s | **4.1x** |

## Why Is It Faster?

### Cold starts: parallel Range-request streaming

uvx downloads each wheel as a single HTTP connection. A 873MB torch wheel = one TCP stream.

zerostart uses **HTTP Range requests** to download multiple chunks of each wheel in parallel, and starts extracting files while chunks are still arriving:

```
uvx (sequential per wheel):
  torch.whl  [=========downloading=========>] then [==extracting==]
  numpy.whl  [=====>] then [=]

zerostart (parallel chunks, overlapped extraction):
  torch.whl  chunk1 [====>]──extract──►
             chunk2 [====>]──extract──►     ← 4 concurrent Range requests
             chunk3 [====>]──extract──►       per large wheel
             chunk4 [====>]──extract──►
  numpy.whl  [=>]──extract──►               ← all wheels in parallel
```

On a slow network, this is the difference between 1 connection at 15 MB/s (60s for torch) and 16+ connections saturating the link.

### Warm starts: Rust cache check vs full re-resolve

uvx re-resolves dependencies and rebuilds the tool environment on every invocation — even when packages are cached. For vllm (177 packages), that means 177 cache lookups + metadata checks + links.

zerostart's warm path is three operations in Rust:
1. `stat(".complete")` — does the cached environment exist?
2. `find("lib/python*/site-packages")` — locate it
3. `exec(python)` — run directly

No resolution, no environment setup, no uv involved.

### Shared CUDA layer cache

CUDA libraries (nvidia-cublas, nvidia-cudnn, nvidia-nccl, etc.) are ~6GB and identical across torch, vllm, and diffusers environments. zerostart caches extracted wheels at `$ZEROSTART_CACHE/shared_wheels/` and hardlinks them into new environments — so the second torch-based environment skips downloading those 6GB entirely.

## Quick Start

```bash
# Run a package (like uvx)
zerostart run torch -- -c "import torch; print(torch.cuda.is_available())"

# Run a script with dependencies
zerostart run -p torch -p transformers serve.py

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

## Architecture

The entire cold path runs in Rust — no Python orchestrator:

```
zerostart run -p torch serve.py

  1. Find Python          (uv python find || which python3)
  2. Check warm cache     (stat .complete marker — instant)
  3. Resolve deps         (uv pip compile --format pylock.toml)
  4. Check shared cache   (hardlink cached CUDA libs — parallel via rayon)
  5. Stream wheels        (parallel Range-request download + extract)
  6. exec(python)         (replaces process, no overhead)
```

Key design decisions:

- **All wheels through the streaming daemon** — every package with a wheel URL goes through parallel download+extract. Only sdist-only packages (rare) fall back to `uv pip install`.
- **Atomic extraction** — each wheel extracts to a staging directory, then renames into site-packages. Partial extractions never corrupt the target.
- **No venv overhead** — uses a flat site-packages directory with a content-addressed cache key. No `uv venv` on the critical path.
- **Demand-driven scheduling** — when Python hits `import torch`, the daemon reprioritizes torch to the front of the download queue.

## Tuning

Performance knobs via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ZS_PARALLEL_DOWNLOADS` | 16 | Concurrent HTTP connections |
| `ZS_EXTRACT_THREADS` | num_cpus * 2 | Parallel extraction threads |
| `ZS_CHUNK_MB` | 16 | Streaming chunk size (MB) for Range requests |
| `ZEROSTART_CACHE` | `~/.cache/zerostart` | Cache directory |

```bash
# Crank up parallelism on a fast network
ZS_PARALLEL_DOWNLOADS=32 ZS_CHUNK_MB=32 zerostart run -v -p torch test.py
```

## Requirements

- Linux (container GPU providers: RunPod, Vast.ai, Lambda, etc.)
- `uv` for requirement resolution (pre-installed on most GPU containers)
- Python 3.10+

macOS works for development (same CLI, no streaming optimization).

## gpu-cli Integration

If you use [gpu-cli](https://gpu-cli.sh):

```bash
# Your script runs on a GPU pod with fast package loading
gpu run "zerostart run -p torch serve.py"
```

## License

MIT

---

Built by the [gpu-cli](https://gpu-cli.sh) team.
