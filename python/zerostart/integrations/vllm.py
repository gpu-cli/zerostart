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
from typing import TYPE_CHECKING, Any

from zerostart.model_cache import ModelCache, cache_key

if TYPE_CHECKING:
    import torch.nn as nn
    from vllm.config import ModelConfig
    from vllm.config.load import LoadConfig

log = logging.getLogger("zerostart.vllm")


def register() -> None:
    """Register the zerostart model loader with vLLM.

    After calling this, you can use --load-format zerostart with vLLM.
    """
    try:
        from vllm.model_executor.model_loader import register_model_loader
        register_model_loader("zerostart")(ZerostartModelLoader)
        log.info("Registered zerostart model loader with vLLM")
    except ImportError:
        # Fallback for older vLLM versions
        try:
            import vllm.model_executor.model_loader as ml
            registry = getattr(ml, "_LOAD_FORMAT_TO_MODEL_LOADER", None)
            if registry is None:
                registry = getattr(ml, "_MODEL_LOADER_REGISTRY", None)
            if registry is not None:
                registry["zerostart"] = ZerostartModelLoader
                log.info("Registered zerostart model loader with vLLM (legacy)")
            else:
                log.warning("Cannot find vLLM model loader registry")
        except ImportError:
            log.warning("vLLM not installed — cannot register model loader")
    except Exception as e:
        log.warning("Failed to register with vLLM: %s", e)


def _get_base_class() -> type:
    """Get BaseModelLoader, falling back to object if not available."""
    try:
        from vllm.model_executor.model_loader.base_loader import BaseModelLoader
        return BaseModelLoader
    except ImportError:
        return object


# Dynamically set base class so we don't fail on import if vLLM isn't installed
_Base = _get_base_class()


class ZerostartModelLoader(_Base):  # type: ignore[misc]
    """vLLM model loader using zerostart's mmap hydrate.

    First load: delegates to default loader, auto-snapshots.
    Subsequent loads: mmap hydrate from cache (4x faster).
    """

    def __init__(self, load_config: LoadConfig):
        if _Base is not object:
            super().__init__(load_config)
        self.load_config = load_config
        self.cache = ModelCache()

    def download_model(self, model_config: ModelConfig) -> None:
        """Download model via HF hub (standard path)."""
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                model_config.model,
                revision=getattr(model_config, "revision", None),
            )
        except Exception as e:
            log.warning("HF download failed, vLLM will handle: %s", e)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
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
                sd = cached_model.state_dict()
                if hasattr(model, "load_weights"):
                    model.load_weights(sd.items())
                else:
                    model.load_state_dict(sd, strict=False)
                log.info(
                    "Loaded from zerostart cache (%.2fs)",
                    time.monotonic() - t0,
                )
                return

        # Standard load, then cache
        t0 = time.monotonic()
        default_loader = self._get_default_loader()
        if default_loader is None:
            log.warning("Cannot import DefaultModelLoader — weights not loaded")
            return

        default_loader.load_weights(model, model_config)
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

    def _get_default_loader(self) -> Any:
        """Get vLLM's default model loader as fallback."""
        try:
            from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
            return DefaultModelLoader(self.load_config)
        except ImportError:
            pass
        try:
            from vllm.model_executor.model_loader.loader import DefaultModelLoader
            return DefaultModelLoader(self.load_config)
        except ImportError:
            pass
        return None
