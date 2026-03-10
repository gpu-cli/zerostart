"""ComfyUI integration for accelerated model loading.

Patches ComfyUI's checkpoint loader for cache-backed loading.

Usage:
    # CLI: zero code changes to ComfyUI
    zerostart run --accelerate -p comfyui main.py

    # Programmatic:
    from zerostart.integrations.comfyui import patch
    patch()
    import comfyui.main
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("zerostart.comfyui")

_patched = False


def patch(cache_dir: str | None = None) -> None:
    """Patch ComfyUI for accelerated model loading.

    1. Enables zerostart.accelerate() (safetensors network fix, etc.)
    2. Patches comfy.sd.load_checkpoint_guess_config for cache-backed loading
    """
    global _patched
    if _patched:
        return

    import zerostart
    zerostart.accelerate(cache_dir=cache_dir)

    try:
        import comfy.sd as sd
    except ImportError:
        log.warning("ComfyUI not installed — skipping checkpoint loader patch")
        _patched = True
        return

    original_load = sd.load_checkpoint_guess_config
    cache = zerostart.model_cache()

    def _fast_load(ckpt_path: str, *args: Any, **kwargs: Any) -> Any:
        key = _comfy_cache_key(ckpt_path)

        if cache and cache.has(key):
            t0 = time.monotonic()
            state = cache.load(key, device="cpu")
            log.info(
                "Cache hit: %s (%.2fs)",
                Path(ckpt_path).name,
                time.monotonic() - t0,
            )
            return _wrap_as_checkpoint_result(state, ckpt_path)

        t0 = time.monotonic()
        result = original_load(ckpt_path, *args, **kwargs)
        elapsed = time.monotonic() - t0
        log.info("Loaded %s (%.2fs)", Path(ckpt_path).name, elapsed)

        # Cache for next time
        if cache:
            try:
                extracted = _extract_checkpoint_state(result)
                cache.save(key, extracted, model_id=Path(ckpt_path).name)
            except Exception as e:
                log.warning("Auto-cache failed for %s: %s", Path(ckpt_path).name, e)

        return result

    sd.load_checkpoint_guess_config = _fast_load
    _patched = True
    log.info("ComfyUI checkpoint loader patched")


def preload(model_paths: list[str], cache_dir: str | None = None) -> None:
    """Pre-snapshot ComfyUI model files for fast loading.

    Run once after downloading models to pre-populate the cache.
    """
    from zerostart.model_cache import ModelCache

    cache = ModelCache(cache_dir)

    for path in model_paths:
        key = _comfy_cache_key(path)
        if cache.has(key):
            log.info("Already cached: %s", Path(path).name)
            continue

        try:
            from safetensors.torch import load_file
            t0 = time.monotonic()
            state_dict = load_file(path)
            cache.save(key, {"state_dict": state_dict}, model_id=Path(path).name)
            log.info("Cached %s (%.2fs)", Path(path).name, time.monotonic() - t0)
        except Exception as e:
            log.warning("Failed to cache %s: %s", path, e)


def _comfy_cache_key(ckpt_path: str) -> str:
    """Cache key from checkpoint file path + modification time."""
    p = Path(ckpt_path)
    try:
        mtime = str(p.stat().st_mtime)
    except OSError:
        mtime = "0"
    raw = f"{p.resolve()}|{mtime}"
    return f"comfy-{hashlib.sha256(raw.encode()).hexdigest()[:12]}"


def _extract_checkpoint_state(result: Any) -> dict[str, Any]:
    """Extract state from ComfyUI's load_checkpoint result for caching."""
    # ComfyUI returns a tuple: (ModelPatcher, CLIP, VAE, ...)
    state: dict[str, Any] = {}
    if isinstance(result, (list, tuple)):
        for i, item in enumerate(result):
            if item is not None and hasattr(item, "model"):
                state[f"component_{i}"] = item.model
            elif item is not None and hasattr(item, "state_dict"):
                state[f"component_{i}"] = item
    return state


def _wrap_as_checkpoint_result(state: dict[str, Any], ckpt_path: str) -> Any:
    """Wrap cached state back into ComfyUI's expected format.

    This is a best-effort reconstruction — ComfyUI's internal types
    may need more specific handling per version.
    """
    # Return the raw state for now — integrators should override this
    # based on their ComfyUI version
    return state
