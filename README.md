# zerostart

**Fast cold starts for GPU Python.** Install packages 6-9x faster than pip/uv and start importing before the install finishes.

```bash
# Install and run — packages load progressively as your app starts
uvx zerostart serve.py
```

No code changes. No platform lock-in. Works on any container GPU provider.

## The Problem

Every GPU Python cold start wastes minutes on package installation:

| Step | Time |
|------|------|
| `pip install torch` (download + extract 800MB wheel) | 60-180s |
| `pip install transformers tokenizers safetensors ...` | 30-60s |
| Module imports (`import torch` loads 80+ .so files) | 5-15s |
| **Total before your code runs** | **~2-5 min** |

On container GPU providers (RunPod, Vast.ai, Lambda), you pay for GPU time during this entire wait.

## How It Works

zerostart does two things:

1. **Fast parallel install** — downloads and extracts wheels simultaneously across 8 connections, streaming large wheels directly to site-packages with no temp files
2. **Progressive loading** — your app starts immediately; `import torch` blocks only until torch is extracted, not until everything is done

```
┌─ Your Python app ──────────────────────────────────┐
│                                                     │
│  import torch        # blocks 1.3s (873MB wheel)    │
│  import transformers # blocks 0s (already done)     │
│  model = load(...)   # runs while deps still land   │
│                                                     │
│  ┌─ zs-fast-wheel (Rust, in-process) ────────────┐  │
│  │  downloading:  ████████░░ tokenizers           │  │
│  │  extracting:   ██████████ torch ✓              │  │
│  │  queued:       safetensors, triton, ...        │  │
│  └────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

When your code hits `import torch`, the daemon reprioritizes torch to the front of the queue. Your app runs in parallel with the install — not after it.

## Quick Start

```bash
# Run any Python script with progressive loading
uvx zerostart serve.py

# PEP 723 inline deps — just works (reads from script header)
uvx zerostart serve.py

# With explicit requirements
uvx zerostart -r requirements.txt serve.py

# With inline packages
uvx zerostart -p torch transformers serve.py

# Pass args to your script
uvx zerostart serve.py --port 8000
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
uvx zerostart serve.py  # deps auto-detected from script
```

Or install it:

```bash
pip install zerostart
zerostart serve.py
```

## Benchmarks

Measured on an RTX 4090 pod (RunPod), comparing `uv pip install` against `zs-fast-wheel`:

### Install Speed

| Workload | Packages | Size | uv pip install | zs-fast-wheel | Speedup |
|----------|----------|------|---------------|---------------|---------|
| Small (requests, six, etc.) | 6 wheels | 3 MB | 767ms | 775ms | 1.0x |
| Medium (numpy, pandas, scikit-learn) | 19 wheels | 251 MB | 12.0s | 1.3s | **9.3x** |
| ML (torch, transformers, triton + CUDA) | 56 wheels | 7 GB | 97.8s | 16.2s | **6.0x** |

For small packages there's no difference. For real ML stacks (hundreds of MB to GB), zs-fast-wheel is **6-9x faster** because it streams and extracts in parallel instead of download-then-extract.

### Time to First Import

With progressive loading, your code doesn't wait for the full install:

| Package | Size | Time to first import |
|---------|------|---------------------|
| numpy | 16 MB | 0.2s |
| torch | 873 MB | 1.3s (demand-prioritized) |
| safetensors | 0.5 MB | 5.2s (queued behind large wheels) |

`import torch` completes in **1.3 seconds** even though the full 7GB ML stack takes 16s to install.

## How It Works (Details)

1. **Parallel streaming** — downloads and extracts wheels simultaneously across 8 connections
2. **Demand signaling** — when Python hits `import torch`, the daemon reprioritizes torch to the front of the queue
3. **Streaming extraction** — large wheels (>50MB) start extracting before the full download completes
4. **No temp files** — wheels extract directly to site-packages, no intermediate copies
5. **Lazy import hook** — a `sys.meta_path` finder that gates imports until the package is on disk, then lets normal Python machinery do the loading

### Architecture

The core is `zs-fast-wheel`, a Rust binary + PyO3 module:

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

- **In-process via PyO3** — no subprocess, no IPC, no sockets. The Rust engine runs as a native Python extension. `signal_demand()` and `wait_done()` are direct function calls.
- **`std::sync` for cross-runtime safety** — uses `Mutex` + `Condvar` (not tokio sync) so `wait_done()` works correctly across threads.
- **Atomic extraction** — each wheel extracts to a staging directory, then atomically renames into site-packages. Partial extractions never corrupt the target.

## Environment Caching

zerostart caches completed environments in `.zerostart/`. If the same requirements are resolved again, it reuses the cached environment (~0s install). The cache key is a hash of the resolved artifact set.

## Requirements

- Python 3.10+
- Linux (container GPU providers: RunPod, Vast.ai, Lambda, etc.)
- `uv` for requirement resolution (pre-installed on most GPU containers)

On macOS: zerostart runs your script directly without progressive loading (useful for development).

## gpu-cli Integration

If you use [gpu-cli](https://gpu-cli.sh):

```bash
# Your script runs with progressive loading on a GPU pod
gpu run "uvx zerostart serve.py"
```

## License

MIT

---

Built by the [gpu-cli](https://gpu-cli.sh) team.
