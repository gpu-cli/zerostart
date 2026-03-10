"""vLLM integration for accelerated model loading.

Provides a custom model loader that uses zerostart's mmap hydrate.

Usage:
    # Register and use with vLLM
    from zerostart.integrations.vllm import register
    register()
    # Then: vllm serve model --load-format zerostart

    # Or via zerostart CLI
    zerostart run --accelerate -p vllm -- python -m vllm.entrypoints.openai.api_server ...
"""

from __future__ import annotations

import logging
import time
from typing import Any

from zerostart.model_cache import ModelCache, cache_key

log = logging.getLogger("zerostart.vllm")


def register() -> None:
    """Register the zerostart model loader with vLLM.

    After calling this, you can use --load-format zerostart with vLLM.
    """
    try:
        from vllm.model_executor.model_loader import loader
        loader._MODEL_LOADER_REGISTRY["zerostart"] = ZerostartModelLoader
        log.info("Registered zerostart model loader with vLLM")
    except ImportError:
        log.warning("vLLM not installed — cannot register model loader")
    except AttributeError:
        log.warning("vLLM version does not support custom model loaders")


class ZerostartModelLoader:
    """vLLM model loader using zerostart's mmap hydrate.

    First load: delegates to default loader, auto-snapshots.
    Subsequent loads: mmap hydrate from cache (4x faster).
    """

    def __init__(self, load_config: Any):
        self.load_config = load_config
        self.cache = ModelCache()

    def download_model(self, model_config: Any) -> None:
        """Download model via HF hub (standard path)."""
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                model_config.model,
                revision=getattr(model_config, "revision", None),
            )
        except Exception as e:
            log.warning("HF download failed, vLLM will handle: %s", e)

    def load_weights(self, model: Any, model_config: Any) -> None:
        """Load weights from cache or standard path."""
        key = cache_key(model_config.model, {
            "dtype": str(getattr(model_config, "dtype", "auto")),
            "revision": getattr(model_config, "revision", "main"),
        })

        if self.cache.has(key):
            t0 = time.monotonic()
            state = self.cache.load(key, device="cuda")
            cached_model = state.get("model")
            if cached_model is not None:
                # Transfer weights from cached model to vLLM's model
                try:
                    model.load_weights(cached_model.state_dict().items())
                except AttributeError:
                    model.load_state_dict(cached_model.state_dict(), strict=False)
                log.info(
                    "Loaded from zerostart cache (%.2fs)",
                    time.monotonic() - t0,
                )
                return

        # Standard load, then cache
        t0 = time.monotonic()
        try:
            from vllm.model_executor.model_loader.loader import DefaultModelLoader
            default = DefaultModelLoader(self.load_config)
            default.load_weights(model, model_config)
        except ImportError:
            log.warning("Cannot import DefaultModelLoader — weights not loaded")
            return

        elapsed = time.monotonic() - t0
        log.info("Standard load (%.2fs), caching for next time", elapsed)

        try:
            self.cache.save(
                key,
                {"model": model},
                model_id=model_config.model,
                dtype=str(getattr(model_config, "dtype", "auto")),
            )
        except Exception as e:
            log.warning("Auto-cache failed: %s", e)
