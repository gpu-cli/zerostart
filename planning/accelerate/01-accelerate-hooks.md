# 01: Transparent Hooks (`accelerate.py`)

## Overview

Single entry point that monkey-patches framework loading functions for transparent acceleration. No user code changes required.

## API

```python
import zerostart

zerostart.accelerate(
    cache_dir=None,            # Auto-detect: /volume first, then ~/.cache/zerostart/models
    auto_cache=True,           # Snapshot models after first load for faster second load
    network_volume_fix=True,   # Detect FUSE/NFS and use eager read instead of mmap
)
```

## Hook Registry

Each hook is a pair: `(patch_fn, unpatch_fn)`. Hooks are only installed if the target module is importable.

```python
_hooks: list[tuple[str, Callable, Callable]] = []

def accelerate(**kwargs):
    _try_hook_transformers(kwargs)
    _try_hook_diffusers(kwargs)
    _try_hook_safetensors(kwargs)
    _try_hook_torch_load(kwargs)

def decelerate():
    for name, _, unpatch in reversed(_hooks):
        unpatch()
    _hooks.clear()
```

## Hook 1: transformers `from_pretrained`

**Target:** `transformers.PreTrainedModel.from_pretrained`

**What it does:**
1. Checks model cache — if cached, hydrate and return (skip everything)
2. Forces `low_cpu_mem_usage=True` to use meta device init
3. Calls original `from_pretrained`
4. Auto-snapshots result for next load

**Edge cases:**
- `quantization_config` (GPTQ, AWQ, BitsAndBytes) — don't cache quantized models, quantization is hardware-specific
- `device_map="auto"` — already uses meta device, but we still cache
- `revision` parameter — include in cache key
- `from_pretrained(local_path)` — still cache, key by resolved path

**Implementation:**

```python
def _try_hook_transformers(config):
    try:
        from transformers import PreTrainedModel
    except ImportError:
        return

    original = PreTrainedModel.from_pretrained.__func__
    cache = config.get("_cache")

    @classmethod
    def patched(cls, pretrained_model_name_or_path, *args, **kwargs):
        # Skip cache for quantized models
        if kwargs.get("quantization_config"):
            kwargs.setdefault("low_cpu_mem_usage", True)
            return original(cls, pretrained_model_name_or_path, *args, **kwargs)

        # Try cache
        cache_key = _cache_key(pretrained_model_name_or_path, kwargs)
        device = kwargs.get("device_map", kwargs.get("device"))
        if cache and cache.has(cache_key):
            log.info("Cache hit for %s", pretrained_model_name_or_path)
            state = cache.load(cache_key, device=device)
            return state["model"]

        # Accelerated load
        kwargs.setdefault("low_cpu_mem_usage", True)
        t0 = time.monotonic()
        model = original(cls, pretrained_model_name_or_path, *args, **kwargs)
        elapsed = time.monotonic() - t0
        log.info("from_pretrained(%s): %.2fs", pretrained_model_name_or_path, elapsed)

        # Auto-cache
        if cache and config.get("auto_cache", True):
            cache.save(cache_key, {"model": model})

        return model

    PreTrainedModel.from_pretrained = patched
    _hooks.append(("transformers", None, lambda: setattr(PreTrainedModel, "from_pretrained", classmethod(original))))
```

## Hook 2: diffusers `from_pretrained`

**Target:** `diffusers.ModelMixin.from_pretrained` and `diffusers.DiffusionPipeline.from_pretrained`

Same pattern as transformers. DiffusionPipeline is special because it loads multiple sub-models — we cache the entire pipeline state as one snapshot.

## Hook 3: safetensors network volume fix

**Target:** `safetensors.torch.load_file`

**Problem:** mmap is 30-50x slower on FUSE/NFS volumes (RunPod persistent volumes, vast.ai volumes). This affects ALL model loading, not just from_pretrained.

**Detection:**

```python
import os
import subprocess

def _is_network_volume(path: str) -> bool:
    """Check if path is on a FUSE/NFS filesystem where mmap is slow."""
    try:
        # Linux: check /proc/mounts
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                mount_point = parts[1]
                fs_type = parts[2]
                if path.startswith(mount_point) and fs_type in (
                    "fuse", "fuse.juicefs", "fuse.gcsfuse", "nfs", "nfs4",
                    "cifs", "smbfs", "9p", "overlay",
                ):
                    return True
    except FileNotFoundError:
        pass
    return False
```

**Patch:**

```python
def _try_hook_safetensors(config):
    try:
        import safetensors.torch
    except ImportError:
        return

    original = safetensors.torch.load_file

    def patched(filename, device="cpu"):
        if config.get("network_volume_fix", True) and _is_network_volume(str(filename)):
            # Eager read: read entire file into memory, then deserialize
            # Avoids mmap page fault penalty on network volumes
            with open(filename, "rb") as f:
                data = f.read()
            return safetensors.torch.load(data, device=device)
        return original(filename, device=device)

    safetensors.torch.load_file = patched
    _hooks.append(("safetensors", None, lambda: setattr(safetensors.torch, "load_file", original)))
```

## Hook 4: torch.load → safetensors conversion

**Target:** `torch.load`

For legacy `.bin` checkpoints, convert to safetensors on first load, mmap on subsequent loads.

```python
def _try_hook_torch_load(config):
    original = torch.load
    cache = config.get("_cache")

    def patched(f, *args, **kwargs):
        if not isinstance(f, (str, Path)) or not str(f).endswith(".bin"):
            return original(f, *args, **kwargs)

        # Check if we have a safetensors conversion cached
        sf_path = cache.safetensors_path_for(str(f))
        if sf_path and sf_path.exists():
            from safetensors.torch import load_file
            device = kwargs.get("map_location", "cpu")
            if isinstance(device, torch.device):
                device = str(device)
            return load_file(str(sf_path), device=device or "cpu")

        # Load normally, convert to safetensors for next time
        result = original(f, *args, **kwargs)
        if cache and isinstance(result, dict):
            cache.save_as_safetensors(str(f), result)
        return result

    torch.load = patched
    _hooks.append(("torch.load", None, lambda: setattr(torch, "load", original)))
```

## Testing

```python
def test_accelerate_transformers():
    """from_pretrained is faster with acceleration."""
    import zerostart
    zerostart.accelerate(cache_dir="/tmp/zs-test")

    from transformers import AutoModelForCausalLM

    # First load: normal speed + auto-cache
    t0 = time.monotonic()
    model1 = AutoModelForCausalLM.from_pretrained("gpt2")
    first_load = time.monotonic() - t0

    del model1

    # Second load: should be much faster (cache hit)
    t0 = time.monotonic()
    model2 = AutoModelForCausalLM.from_pretrained("gpt2")
    second_load = time.monotonic() - t0

    assert second_load < first_load * 0.5  # At least 2x faster

    zerostart.decelerate()
```
