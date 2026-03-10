# zerostart.accelerate — Transparent Model Loading Acceleration

## Problem

GPU Python cold starts are dominated by three bottlenecks:

| Bottleneck | Time (7B model) | Root cause |
|------------|-----------------|------------|
| Package install | 120-240s | pip resolves + downloads + extracts sequentially |
| Python imports | 15-20s | transformers/diffusers import chains |
| Model loading | 35-40s | Random weight init (75%), disk I/O (15%), CPU→GPU (10%) |
| **Total** | **3-5 min** | |

Zerostart already solves #1 (fast-install) and #2 (progressive imports). This design covers #3.

## Design Principle

**User code unchanged.** One line enables acceleration:

```python
import zerostart
zerostart.accelerate()

# Everything downstream is faster — no code changes needed
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", device_map="cuda")
# 4x faster, identical result
```

Or via CLI (no code changes at all):

```bash
zerostart --accelerate serve.py
```

## Architecture

Three layers, each independent but composable:

```
┌─────────────────────────────────────────────────┐
│  Layer 3: Serving Integrations                   │
│  ComfyUI · vLLM · FastAPI · custom               │
├─────────────────────────────────────────────────┤
│  Layer 2: Model Cache                            │
│  Auto-snapshot · mmap hydrate · eviction          │
├─────────────────────────────────────────────────┤
│  Layer 1: Transparent Hooks                      │
│  from_pretrained · safetensors · torch.load       │
├─────────────────────────────────────────────────┤
│  Layer 0: Fast Install + Progressive Imports     │
│  (exists today)                                   │
└─────────────────────────────────────────────────┘
```

## Layer 1: Transparent Hooks (`zerostart/accelerate.py`)

### Hook 1: `PreTrainedModel.from_pretrained`

Monkey-patches `transformers.PreTrainedModel.from_pretrained` to:
1. Force `low_cpu_mem_usage=True` (meta device init, skips random weight init)
2. On second load of same model, serve from mmap cache instead
3. Log timing breakdown

```python
_original_from_pretrained = PreTrainedModel.from_pretrained.__func__

@classmethod
def _fast_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
    # Check model cache first
    cache_key = _model_cache_key(pretrained_model_name_or_path, kwargs)
    if _model_cache.has(cache_key):
        return _model_cache.load(cache_key, device=kwargs.get("device_map"))

    # Force meta device init (eliminates 75% of load time)
    kwargs.setdefault("low_cpu_mem_usage", True)

    t0 = time.monotonic()
    model = _original_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs)
    log.info("from_pretrained: %.2fs (accelerated)", time.monotonic() - t0)

    # Auto-snapshot for next load
    _model_cache.save(cache_key, model)
    return model
```

### Hook 2: `ModelMixin.from_pretrained` (diffusers)

Same pattern as Hook 1, but for diffusers models (UNet, VAE, etc.):
- Force `low_cpu_mem_usage=True`
- Cache snapshots per model
- Also patch `DiffusionPipeline.from_pretrained` which loads multiple sub-models

### Hook 3: `safetensors.torch.load_file`

Patch to handle network volume slowdown:
- Detect if path is on a network/FUSE filesystem
- If so, use eager read (`load(open(f,'rb').read())`) instead of mmap
- On local NVMe, keep mmap (it's fast)

```python
def _is_network_volume(path: str) -> bool:
    """Detect FUSE/NFS mounts where mmap is 30-50x slower."""
    # Check /proc/mounts or statfs for filesystem type
    ...

_original_load_file = safetensors.torch.load_file

def _fast_load_file(filename, device="cpu"):
    if _is_network_volume(filename):
        # Eager read avoids mmap penalty on network volumes
        with open(filename, "rb") as f:
            return safetensors.torch.load(f.read(), device=device)
    return _original_load_file(filename, device=device)
```

### Hook 4: `torch.load`

For legacy `.bin` checkpoints:
- First load: convert to safetensors, cache
- Subsequent loads: mmap from safetensors cache
- Transparent — caller gets same tensors

### API

```python
def accelerate(
    cache_dir: str | None = None,     # Default: /volume/zs-model-cache or ~/.cache/zerostart/models
    hooks: list[str] | None = None,   # ["transformers", "diffusers", "safetensors", "torch"]
    auto_cache: bool = True,          # Auto-snapshot models after first load
    network_volume_fix: bool = True,  # Eager read on network volumes
) -> None:
    """Enable transparent model loading acceleration. Call once at startup."""
    ...

def decelerate() -> None:
    """Remove all hooks, restore original functions."""
    ...
```

## Layer 2: Model Cache (`zerostart/model_cache.py`)

Manages cached model snapshots for repeat loading.

```python
class ModelCache:
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)

    def has(self, key: str) -> bool:
        """Check if model is cached."""
        ...

    def save(self, key: str, model_or_state: Any) -> Path:
        """Snapshot a model or state dict for fast hydration.

        Accepts:
        - nn.Module (uses existing snapshot() logic)
        - dict of {name: nn.Module | Tensor | tokenizer}
        - DiffusionPipeline (extracts sub-models)
        """
        ...

    def load(self, key: str, device: str = "cuda") -> Any:
        """Hydrate a cached model. Returns same type as was saved."""
        ...

    def evict(self, key: str) -> None:
        """Remove a cached model."""
        ...

    def list(self) -> list[CacheEntry]:
        """List all cached models with sizes and ages."""
        ...

    def auto_evict(self, max_size_gb: float = 50.0) -> None:
        """LRU eviction to stay under size limit."""
        ...
```

### Cache layout

```
{cache_dir}/
  models/
    {cache_key}/
      manifest.json        # Model config, tensor refs, metadata
      tokenizer_*/          # Tokenizer files (save_pretrained output)
      tensors/              # Any tensors not in safetensors files
    index.json              # All cached models, sizes, last access times
```

### Cache key derivation

```python
def _model_cache_key(model_id: str, kwargs: dict) -> str:
    """Deterministic key from model ID + loading kwargs."""
    relevant = {
        "model_id": model_id,
        "dtype": str(kwargs.get("torch_dtype", kwargs.get("dtype", "auto"))),
        "revision": kwargs.get("revision", "main"),
    }
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:16]
```

## Layer 3: Serving Integrations

### ComfyUI Integration

```python
# zerostart/integrations/comfyui.py

def patch_comfyui():
    """Patch ComfyUI's model loader for fast loading.

    ComfyUI uses comfy.sd.load_checkpoint_guess_config() which calls
    safetensors loading internally. We intercept at two levels:
    1. safetensors.torch.load_file (Layer 1 hook)
    2. ComfyUI's model management — cache loaded models across workflow runs
    """
    ...
```

Usage:
```bash
# CLI: zero code changes to ComfyUI
zerostart --accelerate --integration comfyui comfyui/main.py

# Or in a custom launcher:
import zerostart
zerostart.accelerate()
zerostart.integrations.comfyui.patch()
import comfyui.main
```

### vLLM Integration

```python
# zerostart/integrations/vllm.py

class ZerostartModelLoader(BaseModelLoader):
    """vLLM model loader that uses zerostart's mmap hydrate.

    Register as a custom loader:
        --load-format zerostart

    First load: delegates to default loader, auto-snapshots
    Subsequent: mmap hydrate (4x faster)
    """

    def download_model(self, model_config) -> None:
        # Use HF hub download (standard)
        ...

    def load_weights(self, model, model_config) -> None:
        cache_key = _model_cache_key(model_config.model, ...)
        if _cache.has(cache_key):
            # Fast path: mmap hydrate
            state = _cache.load(cache_key, device="cuda")
            model.load_state_dict(state, strict=False, assign=True)
        else:
            # First load: standard, then cache
            default_load_weights(model, model_config)
            _cache.save(cache_key, model)
```

### FastAPI / Custom Serving

```python
# zerostart/integrations/serving.py

class ModelServer:
    """Pre-load models at startup, serve via any framework."""

    def __init__(self, cache_dir: str = "/volume/models"):
        self.cache = ModelCache(cache_dir)
        self.loaded: dict[str, Any] = {}

    def preload(self, models: dict[str, str], device: str = "cuda"):
        """Pre-load multiple models at startup.

        Args:
            models: {name: model_id_or_path}
        """
        for name, model_id in models.items():
            if self.cache.has(model_id):
                self.loaded[name] = self.cache.load(model_id, device=device)
            else:
                # Load normally, cache for next time
                model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device)
                self.cache.save(model_id, model)
                self.loaded[name] = model

    def get(self, name: str) -> Any:
        return self.loaded[name]
```

## CLI Surface

```bash
# Basic: accelerate any Python script
zerostart --accelerate serve.py

# With packages + acceleration
zerostart --accelerate -p torch -p transformers serve.py

# Pre-warm model cache (download + snapshot, no inference)
zerostart warm-model Qwen/Qwen2.5-7B --device cuda --cache-dir /volume/models

# Pre-warm multiple models
zerostart warm-model stabilityai/stable-diffusion-xl-base-1.0 \
                     lllyasviel/control_v11p_sd15_canny \
                     --cache-dir /volume/models

# Cache management
zerostart cache list                    # Show cached models
zerostart cache evict --max-size 50GB   # LRU eviction
zerostart cache clear                   # Remove all

# Integration shortcuts
zerostart comfyui --accelerate          # Run ComfyUI with acceleration
```

## Expected Performance

### Qwen2.5-7B on RTX 4090 (measured + projected)

| Scenario | Traditional | Zerostart | Speedup |
|----------|-------------|-----------|---------|
| Cold (no packages, no model) | ~300s | ~40s | 7.5x |
| Warm packages, cold model | ~60s | ~15s | 4x |
| Warm packages, warm model (cache) | ~45s | ~11s | 4x |
| Everything warm | ~45s | ~11s | 4x |

### SDXL Pipeline on RTX 4090 (projected)

| Scenario | Traditional | Zerostart | Speedup |
|----------|-------------|-----------|---------|
| Cold start | ~5 min | ~45s | 6.7x |
| Warm start | ~60s | ~15s | 4x |

## Implementation Order

1. **`accelerate.py`** — Hook 1 (transformers from_pretrained) + Hook 3 (safetensors network fix)
2. **`model_cache.py`** — Cache with auto-snapshot + hydrate
3. **Hook 2** — diffusers from_pretrained
4. **Hook 4** — torch.load conversion
5. **CLI** — `--accelerate` flag, `warm-model` command, `cache` subcommand
6. **ComfyUI integration** — patch loader
7. **vLLM integration** — custom model loader

## Files

```
python/zerostart/
  accelerate.py          # Layer 1: transparent hooks
  model_cache.py         # Layer 2: model cache management
  snapshot.py            # Core snapshot/hydrate (exists, extend)
  integrations/
    __init__.py
    comfyui.py           # ComfyUI model loader patch
    vllm.py              # vLLM BaseModelLoader impl
    serving.py           # Generic ModelServer helper
```
