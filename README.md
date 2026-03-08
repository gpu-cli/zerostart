# zerostart

**2-second cold starts for GPU Python.**

```bash
# First run: installs packages, runs your app, snapshots automatically (~30s)
zerostart run vllm serve meta-llama/Llama-3-8B

# Second run: restores from snapshot (~2s)
zerostart run vllm serve meta-llama/Llama-3-8B
```

No code changes. No platform lock-in. Works on any Linux machine with NVIDIA GPUs.

## The Problem

Every GPU Python cold start pays a tax:

| Layer | Time |
|-------|------|
| Package install (pip install torch) | 60-180s |
| Module imports (import torch = 80+ .so files) | 5-15s |
| Model weight loading | 10-60s |
| GPU warmup (torch.compile, CUDA graphs) | 5-30s |
| **Total** | **~2-5 min** |

## How It Works

zerostart uses a tiered cache — like CPU L1/L2/L3. Each tier is faster but narrower. Miss one, fall through to the next:

| Tier | What | Cold start |
|------|------|------------|
| 0 | GPU snapshot (CRIU + cuda-checkpoint) | **~2s** |
| 1 | Process snapshot (CRIU, no VRAM) | ~5s |
| 2 | Cached packages (pre-extracted on volume) | ~10s |
| 3 | Fast install (uv parallel download+extract) | ~30s |

Every run that misses a tier **builds it for next time**. You don't configure anything — it just gets faster.

## Quick Start

```bash
# Install
curl -sSL https://github.com/gpu-cli/zerostart/releases/latest/download/install.sh | bash

# Check your system
zerostart doctor

# Run any Python app
zerostart run python serve.py

# Run vLLM, SGLang, ComfyUI — anything
zerostart run vllm serve meta-llama/Llama-3-8B
```

## Ready Detection

zerostart needs to know when your app is initialized. By default, it figures this out automatically:

1. **Port binding** — detects when your server starts listening
2. **Health probes** — probes `/health`, `/v1/models`, `/ready`
3. **GPU stabilization** — watches VRAM allocations settle

For explicit control:

```bash
# Poll a specific health endpoint
zerostart run --ready=url:http://localhost:8000/health python serve.py

# Wait for a file touch (any language, no SDK needed)
zerostart run --ready=signal python train.py
# In your app: touch /tmp/zerostart/ready
```

### Python SDK (optional)

For precise snapshot control, add one line:

```python
import zerostart
zerostart.ready()  # snapshot happens here
```

If zerostart isn't installed or you're not running under the orchestrator, `ready()` is a no-op. Safe to leave in production code.

## Snapshot Modes

```bash
# Full GPU snapshot (default for GPU apps — includes VRAM)
zerostart run --snapshot=gpu python serve.py

# Process-only (skip VRAM — load model fresh from disk or network)
zerostart run --snapshot=process python serve.py

# No snapshots (just fast package install + caching)
zerostart run --snapshot=off python serve.py
```

## Snapshot Management

```bash
# List cached snapshots
zerostart list

# Inspect details
zerostart inspect <key>

# Pre-warm (build snapshot without serving)
zerostart warm vllm serve model

# Garbage collect
zerostart gc --max-age=7d --max-size=50G

# Delete specific snapshot
zerostart delete <key>
```

## gpu-cli Integration

If you use [gpu-cli](https://gpu-cli.sh), add one line to `gpu.jsonc`:

```jsonc
{
  "zerostart": true
}
```

gpu-cli automatically wraps your commands with zerostart. Your second `gpu run` is instant.

## Requirements

- Linux (kernel 3.11+)
- NVIDIA GPU with driver r550+
- CRIU 4.0+ (`apt install criu`)
- cuda-checkpoint ([NVIDIA/cuda-checkpoint](https://github.com/NVIDIA/cuda-checkpoint))

On macOS: zerostart gracefully degrades — runs your app normally without snapshots.

## Testing on Real GPUs

Build locally with cross-compilation to avoid burning GPU time on compilation, then use [gpu-cli](https://gpu-cli.sh) to test on real NVIDIA GPUs:

```bash
# Cross-compile locally for Linux
cargo build --target x86_64-unknown-linux-gnu --release

# Test on a real GPU (binary synced automatically)
gpu run "zerostart doctor"
gpu run "zerostart run python serve.py"
```

## Benchmarks

Measured on an RTX 4090 pod, comparing `uv pip install` (current fastest Python installer) against `zs-fast-wheel warm` (zerostart's progressive wheel installer):

### Package Install Speed

| Workload | Packages | Size | uv pip install | zs-fast-wheel | Speedup |
|----------|----------|------|---------------|---------------|---------|
| Small (requests, six, etc.) | 6 wheels | 3 MB | 767ms | 775ms | 1.0x |
| Medium (numpy, pandas, scikit-learn) | 19 wheels | 251 MB | 12.0s | 1.3s | **9.3x** |
| ML (torch, transformers, triton + CUDA) | 56 wheels | 7 GB | 97.8s | 16.2s | **6.0x** |

For small packages there's no difference — the bottleneck is resolution, not download. For real ML stacks (hundreds of MB to multiple GB), zs-fast-wheel is **6-9x faster** than uv because it streams and extracts in parallel instead of download-then-extract.

### Progressive Loading

zs-fast-wheel supports **progressive loading** — Python can start importing packages while the rest are still downloading. On the same RTX 4090 pod:

| Package | Size | Time to first import |
|---------|------|---------------------|
| numpy | 16 MB | 0.2s |
| safetensors | 0.5 MB | 5.2s (queued behind large wheels) |
| torch | 873 MB | 1.3s (demand-prioritized) |

With demand signaling, `import torch` completes in **1.3 seconds** even though the full 7GB ML stack takes 16s to install. Your code starts running before the install finishes.

### How It Works

1. **Parallel streaming** — downloads and extracts wheels simultaneously across 8 connections
2. **Demand signaling** — when Python hits `import torch`, the daemon reprioritizes torch to the front of the queue
3. **Streaming extraction** — large wheels (>50MB) use HTTP Range requests to extract while downloading
4. **No temp files** — wheels extract directly to site-packages, no intermediate copies

## Architecture: zs-fast-wheel

The package install layer is a standalone Rust binary + PyO3 module called `zs-fast-wheel`. It can be used independently of the rest of zerostart.

```
┌─────────────────────────────────────────────────┐
│  Python process                                 │
│                                                 │
│  import torch  ──► lazy import hook             │
│                     │                           │
│                     ├─ signal_demand("torch")   │
│                     └─ wait_done("torch")       │
│                          │                      │
│  ┌───────────────────────┼────────────────────┐ │
│  │  DaemonEngine (Rust, in-process via PyO3)  │ │
│  │                       │                    │ │
│  │  ┌─── download ───┐   │   ┌── extract ──┐ │ │
│  │  │ wheel 1  ████░░ │──┼──►│ site-pkgs/  │ │ │
│  │  │ wheel 2  ██████ │  │   │ ✓ done      │ │ │
│  │  │ torch    ░░░░░░ │◄─┘   │ ...         │ │ │
│  │  │  (reprioritized) │     └─────────────┘ │ │
│  │  └─────────────────┘                      │ │
│  └───────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

Key design decisions:

- **Pure `std::sync` for cross-runtime safety** — the DaemonEngine uses `Mutex` + `Condvar` (not tokio sync primitives) so `wait_done()` works correctly when the PyO3 caller and the download tokio runtime are on separate threads. This was a critical bug fix — `tokio::sync::watch` channels silently hang when polled from a different runtime.
- **In-process via PyO3** — no subprocess, no IPC, no sockets. The Rust engine runs as a native Python extension. `signal_demand()` and `wait_done()` are direct function calls with ~0 overhead.
- **Atomic extraction** — each wheel extracts to a staging directory, then atomically renames into site-packages. Partial extractions never corrupt the target.
- **Streaming for large wheels** — wheels >50MB use HTTP Range requests to start extracting before the full download completes.

## How It Compares

| | Modal | zerostart |
|---|---|---|
| Cold start | ~2s | ~2s |
| Code changes | Rewrite as Modal class | None (or 1 line) |
| Platform | Modal only | Any Linux + NVIDIA GPU |
| Open source | No | Yes (Apache 2.0) |
| GPU snapshots | Proprietary | NVIDIA cuda-checkpoint (open) |
| Pricing | Per-second compute | Free |

## License

Apache 2.0

---

Built by the [gpu-cli](https://gpu-cli.sh) team.
