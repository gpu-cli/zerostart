"""zerostart — eliminate GPU cold starts for Python applications.

Fast parallel package installation + progressive imports + model loading acceleration.

Usage:
    import zerostart
    zerostart.accelerate()  # Transparent 4x faster model loading

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", device_map="cuda")
"""

from zerostart.lazy_imports import LazyImportHook, install_hook, remove_hook

__all__ = [
    "LazyImportHook",
    "install_hook",
    "remove_hook",
    "accelerate",
    "decelerate",
    "model_cache",
    "ModelCache",
]


def accelerate(
    cache_dir: str | None = None,
    auto_cache: bool = True,
    network_volume_fix: bool = True,
    hooks: list[str] | None = None,
) -> None:
    """Enable transparent model loading acceleration. Call once at startup."""
    from zerostart.accelerate import accelerate as _accelerate
    _accelerate(
        cache_dir=cache_dir,
        auto_cache=auto_cache,
        network_volume_fix=network_volume_fix,
        hooks=hooks,
    )


def decelerate() -> None:
    """Remove all hooks, restore original functions."""
    from zerostart.accelerate import decelerate as _decelerate
    _decelerate()


def model_cache():
    """Get the active model cache (None if not accelerated)."""
    from zerostart.accelerate import model_cache as _model_cache
    return _model_cache()


def ModelCache(cache_dir: str | None = None):
    """Create a standalone model cache."""
    from zerostart.model_cache import ModelCache as _ModelCache
    return _ModelCache(cache_dir)
