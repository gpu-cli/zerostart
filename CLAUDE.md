# CLAUDE.md

## What is zerostart

Open-source Rust binary that eliminates GPU cold starts for Python applications. Combines CRIU process snapshots, NVIDIA cuda-checkpoint GPU snapshots, fast package installation (uv), and volume caching into a tiered cache hierarchy.

Drop-in wrapper — `zerostart run python serve.py` — that makes second runs instant (~2s cold start).

Repository: `github.com/gpu-cli/zerostart` (Apache 2.0)

## Build & Development

Rust workspace rooted at `crates/`:

```bash
cd crates && cargo check --workspace
cd crates && cargo nextest run --workspace
cd crates && cargo clippy --workspace -- -D warnings
cd crates && cargo fmt --workspace
```

Python SDK:
```bash
cd python && pip install -e .
```

## Architecture

Tiered cache hierarchy (like CPU L1/L2/L3). Miss one tier, fall through to the next:

| Tier | Strategy | Cold start |
|------|----------|------------|
| 0 | GPU snapshot restore (CRIU + cuda-checkpoint) | ~2s |
| 1 | Process snapshot restore (CRIU, no VRAM) | ~5s |
| 2 | Volume-cached packages | ~10s |
| 3 | Fast parallel install (uv) | ~30s |

Every run that misses a tier builds it for next time. First run is Tier 3 (~30s), second run is Tier 0 (~2s).

## Core API

### CLI

```bash
# Auto-detect mode (default) — works with any server framework
zerostart run vllm serve meta-llama/Llama-3-8B

# Explicit health check
zerostart run --ready=url:http://localhost:8000/health python serve.py

# Signal mode (app touches /tmp/zerostart/ready)
zerostart run --ready=signal python train.py

# Snapshot management
zerostart list                    # show cached snapshots
zerostart gc                     # garbage collect old snapshots
zerostart doctor                 # check CRIU, cuda-checkpoint, drivers
```

### Ready Detection (--ready flag)

```
auto      (default) Watch port binding + health probes + GPU memory stabilization
signal    Wait for file touch at /tmp/zerostart/ready
url:<url> Poll URL until 200
file:<p>  Watch for file creation
```

### Python SDK (optional)

```python
import zerostart
zerostart.ready()                 # signal snapshot point (no-op if not under orchestrator)

@zerostart.on_restore             # run after restore (reinit RNG, reconnect DB, etc.)
def reinit(): ...

@zerostart.on_snapshot            # run before snapshot (close connections, flush logs)
def cleanup(): ...
```

## Workspace Crates

| Crate | Purpose |
|-------|---------|
| `zerostart` | CLI binary — `zerostart run`, `zerostart list`, `zerostart doctor` |
| `zs-snapshot` | CRIU dump/restore orchestration + cuda-checkpoint integration |
| `zs-cache` | Volume cache management, package installation (uv integration) |
| `zs-detect` | Auto-detection of ready state (port binding, health check, GPU stabilization) |

## Key Constraints

- **Python + GPU focused** — Go/Rust/Node cold starts are 30-60ms, not worth solving
- **Single-GPU only** — cuda-checkpoint does NOT support NCCL (multi-GPU)
- **Same GPU type required** for snapshot restore
- **Driver r550+** required for cuda-checkpoint
- **Linux only** — CRIU is Linux-only (graceful degradation on macOS: just runs the app)

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

This project requires NVIDIA GPUs for snapshot testing. Use gpu-cli:

```bash
gpu run "zerostart doctor"
gpu run "zerostart run python serve.py"
```

Do NOT attempt CRIU/cuda-checkpoint tests locally on macOS.
