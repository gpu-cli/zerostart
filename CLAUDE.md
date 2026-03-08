# CLAUDE.md

## What is zerostart

Open-source Rust binary + Python SDK that eliminates GPU cold starts for Python applications. Fast parallel package installation (uv + streaming wheel extraction) and progressive imports let your app start using packages while they're still installing.

Drop-in wrapper — `zerostart run python serve.py` — that makes cold starts fast (~10-30s vs minutes).

Repository: `github.com/gpu-cli/zerostart` (Apache 2.0)

## Build & Development

Rust workspace rooted at `crates/`:

```bash
cd crates && cargo check --workspace
cd crates && cargo nextest run --workspace
cd crates && cargo clippy --workspace -- -D warnings
cd crates && cargo fmt --workspace
```

Cross-compile for Linux (musl static binary):
```bash
cd crates && cargo zigbuild --target x86_64-unknown-linux-musl --release -p zs-fast-wheel
cp crates/target/x86_64-unknown-linux-musl/release/zs-fast-wheel bin/zs-fast-wheel-linux-x86_64
```

Python SDK:
```bash
cd python && pip install -e .
```

## Architecture

Fast package installation + progressive loading for container GPU environments:

| Strategy | Cold start | How |
|----------|------------|-----|
| Volume-cached packages | ~10s | Reuse prior install from persistent volume |
| Fast parallel install | ~30s | Streaming wheel extraction + demand-driven scheduling |
| Progressive loading | overlap | App imports resolve as packages land (lazy import hook) |

First run installs everything (~30s). Subsequent runs reuse the cached environment (~10s). Progressive loading means your app can start executing while large wheels (torch, transformers) are still extracting.

## Key Decision: No CRIU

CRIU process snapshots (Tier 0/1 in the original design) are **deferred indefinitely**. Reason: container GPU providers (RunPod, Vast.ai) don't grant `CAP_SYS_ADMIN` needed for CRIU. VM providers that support CRIU take 3-5min to boot, negating the ~2s restore benefit. The plans remain in `../private-zerostart/planning/criu-process-snapshots/` for future reference if container providers add privileged mode.

Focus is on making Tier 2/3 (fast install + progressive loading) as fast as possible on container providers.

## Core API

### CLI

```bash
# Run with progressive package loading
zerostart run python serve.py

# With explicit requirements
zerostart run -r requirements.txt python serve.py

# Direct wheel installation
zs-fast-wheel install --wheels <urls> --target <dir>

# Daemon mode (background install, demand-driven)
zs-fast-wheel daemon --manifest manifest.json

# Resolve + warm cache
zs-fast-wheel warm --requirements "torch>=2.0\ntransformers"
```

### Python SDK

```python
from zerostart.lazy_imports import install_hook, remove_hook

# Install hook — imports block until the package is extracted
hook = install_hook(daemon=daemon_handle, import_map={"torch": "torch"})

# Your code runs immediately, imports resolve progressively
import torch  # blocks only until torch wheel is extracted
model = torch.load(...)

report = remove_hook()  # returns per-package wait times
```

## Workspace Crates

| Crate | Purpose |
|-------|---------|
| `zs-fast-wheel` | CLI + library: parallel wheel download, streaming extraction, daemon mode, PyO3 bindings |

## Key Constraints

- **Python + GPU focused** — Go/Rust/Node cold starts are 30-60ms, not worth solving
- **Container-first** — designed for RunPod, Vast.ai, and similar container GPU providers
- **Linux target** — cross-compile from macOS, test on Linux via `gpu run`

## Coding Conventions

- **Edition 2024** for all crates
- **No `unwrap()` or `expect()`** in production code — use `?`, `thiserror`, `match`
- **No `println!`/`eprintln!`** — use `tracing` (except CLI user-facing output)
- Error handling: `thiserror` for library errors, `anyhow` for CLI errors
- Async runtime: tokio
- No backwards compat — cut over and remove old stuff
- No `any` type in TypeScript — use proper types

## Planning & Tracking

- Private planning/research: `../private-zerostart/`
- Issue tracking: Beads (`bd list`, `bd show`, `bd ready`)
- Related: safetensors-streaming (`../safetensors-streaming/`)
- Parent: gpu-cli (`gpu-cli.sh`)

## GPU Testing

Cross-compile locally, test on container GPUs via `gpu run`. Binaries in `bin/` are gitignored and synced to pods via `gpu.jsonc` `include: ["bin/"]`.

```bash
# Cross-compile locally (don't burn GPU time on compilation)
cd crates && cargo zigbuild --target x86_64-unknown-linux-musl --release -p zs-fast-wheel
cp crates/target/x86_64-unknown-linux-musl/release/zs-fast-wheel bin/zs-fast-wheel-linux-x86_64

# Test on a GPU pod
gpu run "bin/zs-fast-wheel-linux-x86_64 warm --requirements 'torch>=2.0'"
```
