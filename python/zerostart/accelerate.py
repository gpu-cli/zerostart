"""Transparent model loading acceleration.

One line enables faster model loading — no user code changes needed:

    import zerostart
    zerostart.accelerate()

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", device_map="cuda")

Hooks:
    1. transformers PreTrainedModel.from_pretrained — meta device init + cache
    2. diffusers ModelMixin/DiffusionPipeline.from_pretrained — same
    3. safetensors.torch.load_file — eager read on network volumes
    4. torch.load — .bin → safetensors conversion
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable

from zerostart.model_cache import ModelCache, cache_key

log = logging.getLogger("zerostart.accelerate")

_hooks: list[tuple[str, Callable[[], None]]] = []
_cache: ModelCache | None = None
_bg_save_thread: threading.Thread | None = None


def accelerate(
    cache_dir: str | None = None,
    auto_cache: bool = True,
    network_volume_fix: bool = True,
    hooks: list[str] | None = None,
) -> None:
    """Enable transparent model loading acceleration. Call once at startup.

    Args:
        cache_dir: Model cache directory. Auto-detected if None.
        auto_cache: Auto-snapshot models after first load for faster repeat loads.
        hooks: Which hooks to install. Default: all available.
            Options: "transformers", "diffusers", "safetensors", "torch"
        network_volume_fix: Use eager read instead of mmap on network volumes.
    """
    global _cache

    if _hooks:
        log.warning("accelerate() already called — call decelerate() first to re-configure")
        return

    _cache = ModelCache(cache_dir)

    config = {
        "auto_cache": auto_cache,
        "network_volume_fix": network_volume_fix,
        "cache": _cache,
    }

    enabled = hooks or ["transformers", "diffusers", "safetensors", "torch"]

    if "safetensors" in enabled:
        _try_hook_safetensors(config)
    if "transformers" in enabled:
        _try_hook_transformers(config)
    if "diffusers" in enabled:
        _try_hook_diffusers(config)
    if "torch" in enabled:
        _try_hook_torch_load(config)

    installed = [name for name, _ in _hooks]
    if installed:
        log.info("Acceleration enabled: %s (cache: %s)", ", ".join(installed), _cache.cache_dir)
    else:
        log.warning("No hooks installed — no supported frameworks found")


def decelerate() -> None:
    """Remove all hooks, restore original functions."""
    global _cache, _bg_save_thread
    # Wait for any background save to complete
    if _bg_save_thread is not None and _bg_save_thread.is_alive():
        _bg_save_thread.join(timeout=30)
        _bg_save_thread = None
    for name, unpatch in reversed(_hooks):
        unpatch()
        log.debug("Removed hook: %s", name)
    _hooks.clear()
    _cache = None
    log.info("Acceleration disabled")


def model_cache() -> ModelCache | None:
    """Get the active model cache (None if not accelerated)."""
    return _cache


# ---------------------------------------------------------------------------
# Hook 1: transformers PreTrainedModel.from_pretrained
# ---------------------------------------------------------------------------

def _try_hook_transformers(config: dict[str, Any]) -> None:
    try:
        from transformers import PreTrainedModel
    except ImportError:
        return

    original = PreTrainedModel.from_pretrained

    mc = config["cache"]
    auto = config["auto_cache"]

    @classmethod  # type: ignore[misc]
    def patched(cls: type, pretrained_model_name_or_path: str, *args: Any, **kwargs: Any) -> Any:
        # Skip cache for quantized models (hardware-specific)
        if kwargs.get("quantization_config"):
            kwargs.setdefault("low_cpu_mem_usage", True)
            return original.__func__(cls, pretrained_model_name_or_path, *args, **kwargs)

        # Try cache
        key = cache_key(str(pretrained_model_name_or_path), kwargs)
        device = kwargs.get("device_map", kwargs.get("device"))

        # Skip cache for device_map='auto' — HF's shard-by-shard loading to
        # the right device is faster than our load-to-CPU-then-dispatch path.
        # We still benefit from low_cpu_mem_usage=True below.
        use_cache = device != "auto"

        if use_cache and mc.has(key):
            t0 = time.monotonic()
            state = mc.load(key, device=_resolve_device(device))
            model = state.get("model")
            if model is not None:
                log.info(
                    "Cache hit: %s (%.2fs)",
                    pretrained_model_name_or_path,
                    time.monotonic() - t0,
                )
                return model

        # Accelerated load: force meta device init
        kwargs.setdefault("low_cpu_mem_usage", True)
        t0 = time.monotonic()
        model = original.__func__(cls, pretrained_model_name_or_path, *args, **kwargs)
        elapsed = time.monotonic() - t0
        log.info("from_pretrained(%s): %.2fs (accelerated)", pretrained_model_name_or_path, elapsed)

        # Auto-cache in background thread (non-blocking)
        # Skip for device_map='auto' — cache won't be used anyway
        if auto and use_cache:
            _bg_cache_save(mc, key, model, str(pretrained_model_name_or_path), kwargs)

        return model

    PreTrainedModel.from_pretrained = patched
    _hooks.append(("transformers", lambda: setattr(PreTrainedModel, "from_pretrained", original)))


def _bg_cache_save(
    mc: ModelCache,
    key: str,
    model: Any,
    model_id: str,
    kwargs: dict[str, Any],
) -> None:
    """Save model to cache in a background thread."""
    global _bg_save_thread

    # Capture state_dict eagerly on main thread (safe reference to tensor memory)
    try:
        import torch
        # Verify model has parameters before attempting save
        param_count = sum(1 for _ in model.parameters())
        if param_count == 0:
            return
    except Exception:
        return

    def _save() -> None:
        try:
            mc.save(
                key,
                {"model": model},
                model_id=model_id,
                dtype=str(kwargs.get("torch_dtype", kwargs.get("dtype", "auto"))),
                revision=kwargs.get("revision", "main"),
            )
        except Exception as e:
            log.warning("Background cache save failed for %s: %s", model_id, e)

    _bg_save_thread = threading.Thread(target=_save, daemon=True)
    _bg_save_thread.start()


# ---------------------------------------------------------------------------
# Hook 2: diffusers ModelMixin.from_pretrained + DiffusionPipeline
# ---------------------------------------------------------------------------

def _try_hook_diffusers(config: dict[str, Any]) -> None:
    try:
        from diffusers import ModelMixin
    except ImportError:
        return

    original_model = ModelMixin.from_pretrained

    mc = config["cache"]
    auto = config["auto_cache"]

    @classmethod  # type: ignore[misc]
    def patched_model(cls: type, pretrained_model_name_or_path: str, *args: Any, **kwargs: Any) -> Any:
        key = cache_key(f"diffusers:{pretrained_model_name_or_path}", kwargs)
        if mc.has(key):
            t0 = time.monotonic()
            state = mc.load(key, device=_resolve_device(kwargs.get("device")))
            model = state.get("model")
            if model is not None:
                log.info("Cache hit: %s (%.2fs)", pretrained_model_name_or_path, time.monotonic() - t0)
                return model

        kwargs.setdefault("low_cpu_mem_usage", True)
        t0 = time.monotonic()
        model = original_model.__func__(cls, pretrained_model_name_or_path, *args, **kwargs)
        log.info("diffusers.from_pretrained(%s): %.2fs", pretrained_model_name_or_path, time.monotonic() - t0)

        if auto:
            _bg_cache_save(mc, key, model, f"diffusers:{pretrained_model_name_or_path}", kwargs)

        return model

    ModelMixin.from_pretrained = patched_model
    _hooks.append(("diffusers.ModelMixin", lambda: setattr(ModelMixin, "from_pretrained", original_model)))

    # Also patch DiffusionPipeline if available
    try:
        from diffusers import DiffusionPipeline

        original_pipeline = DiffusionPipeline.from_pretrained

        @classmethod  # type: ignore[misc]
        def patched_pipeline(cls: type, pretrained_model_name_or_path: str, *args: Any, **kwargs: Any) -> Any:
            key = cache_key(f"pipeline:{pretrained_model_name_or_path}", kwargs)
            if mc.has(key):
                t0 = time.monotonic()
                state = mc.load(key, device=_resolve_device(kwargs.get("device")))
                log.info("Pipeline cache hit: %s (%.2fs)", pretrained_model_name_or_path, time.monotonic() - t0)
                # Reconstruct pipeline from components — caller must reassemble
                return state

            t0 = time.monotonic()
            pipeline = original_pipeline.__func__(cls, pretrained_model_name_or_path, *args, **kwargs)
            log.info("DiffusionPipeline.from_pretrained(%s): %.2fs", pretrained_model_name_or_path, time.monotonic() - t0)

            if auto:
                try:
                    mc.save(key, pipeline, model_id=f"pipeline:{pretrained_model_name_or_path}")
                except Exception as e:
                    log.warning("Auto-cache failed: %s", e)

            return pipeline

        DiffusionPipeline.from_pretrained = patched_pipeline
        _hooks.append(("diffusers.Pipeline", lambda: setattr(DiffusionPipeline, "from_pretrained", original_pipeline)))
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Hook 3: safetensors.torch.load_file — network volume fix
# ---------------------------------------------------------------------------

def _try_hook_safetensors(config: dict[str, Any]) -> None:
    if not config.get("network_volume_fix", True):
        return

    try:
        import safetensors.torch
    except ImportError:
        return

    original = safetensors.torch.load_file

    def patched(filename: str, device: str = "cpu") -> dict[str, Any]:
        path = str(filename)
        if _is_network_volume(path):
            t0 = time.monotonic()
            with open(path, "rb") as f:
                data = f.read()
            # safetensors.torch.load() (bytes) doesn't accept device kwarg
            result = safetensors.torch.load(data)
            if device and device != "cpu":
                import torch
                result = {k: v.to(device) for k, v in result.items()}
            log.debug("Eager read %s (%.2fs, network volume)", Path(path).name, time.monotonic() - t0)
            return result
        return original(filename, device=device)

    safetensors.torch.load_file = patched
    _hooks.append(("safetensors", lambda: setattr(safetensors.torch, "load_file", original)))


# ---------------------------------------------------------------------------
# Hook 4: torch.load — .bin → safetensors conversion
# ---------------------------------------------------------------------------

def _try_hook_torch_load(config: dict[str, Any]) -> None:
    try:
        import torch
    except ImportError:
        return

    mc = config.get("cache")
    if mc is None:
        return

    original = torch.load

    def patched(f: Any, *args: Any, **kwargs: Any) -> Any:
        # Only intercept .bin file paths
        if not isinstance(f, (str, Path)) or not str(f).endswith(".bin"):
            return original(f, *args, **kwargs)

        # Check for cached safetensors conversion
        sf_path = mc.safetensors_path_for(str(f))
        if sf_path is not None:
            try:
                from safetensors.torch import load_file
                device = kwargs.get("map_location", "cpu")
                if hasattr(device, "__str__"):
                    device = str(device)
                if not isinstance(device, str):
                    device = "cpu"
                t0 = time.monotonic()
                result = load_file(str(sf_path), device=device)
                log.info("Loaded %s from safetensors cache (%.2fs)", Path(f).name, time.monotonic() - t0)
                return result
            except Exception as e:
                log.warning("Safetensors cache load failed, falling back: %s", e)

        # Load normally
        result = original(f, *args, **kwargs)

        # Cache as safetensors for next time (background)
        if isinstance(result, dict):
            # Only cache if all values are tensors
            all_tensors = True
            for v in result.values():
                if not hasattr(v, "shape"):
                    all_tensors = False
                    break
            if all_tensors:
                try:
                    mc.save_as_safetensors(str(f), result)
                except Exception as e:
                    log.debug("Could not cache as safetensors: %s", e)

        return result

    torch.load = patched
    _hooks.append(("torch.load", lambda: setattr(torch, "load", original)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_network_volume_cache: dict[str, bool] = {}


def _is_network_volume(path: str) -> bool:
    """Check if path is on a FUSE/NFS filesystem where mmap is slow."""
    # Cache results per mount point
    if path in _network_volume_cache:
        return _network_volume_cache[path]

    result = _check_network_volume(path)
    _network_volume_cache[path] = result
    return result


def _check_network_volume(path: str) -> bool:
    """Detect FUSE/NFS mounts via /proc/mounts.

    overlay is intentionally excluded — most container providers (RunPod, etc.)
    use overlay backed by local SSD where mmap is fast.
    """
    slow_fs_types = frozenset({
        "fuse", "fuse.juicefs", "fuse.gcsfuse", "fuse.sshfs",
        "nfs", "nfs4", "cifs", "smbfs", "9p",
    })

    try:
        best_match = ""
        best_fs = ""
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mount_point = parts[1]
                fs_type = parts[2]
                if path.startswith(mount_point) and len(mount_point) > len(best_match):
                    best_match = mount_point
                    best_fs = fs_type
        return best_fs in slow_fs_types
    except FileNotFoundError:
        return False


def _resolve_device(device: str | None) -> str:
    """Resolve device string for hydrate()."""
    if device is None:
        return "cpu"
    if device == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device
