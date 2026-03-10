"""Generic model serving helper for custom stacks.

Usage:
    from zerostart.integrations.serving import ModelServer

    server = ModelServer("/volume/models")
    server.preload({
        "llm": "Qwen/Qwen2.5-7B",
        "embedder": "BAAI/bge-small-en-v1.5",
    }, device="cuda")

    @app.post("/generate")
    def generate(prompt: str):
        model = server.get("llm")
        ...
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from zerostart.model_cache import ModelCache, cache_key

log = logging.getLogger("zerostart.serving")


class ModelServer:
    """Pre-load and serve models from cache."""

    def __init__(self, cache_dir: str | Path | None = None):
        self.cache = ModelCache(cache_dir)
        self._loaded: dict[str, dict[str, Any]] = {}

    def preload(
        self,
        models: dict[str, str],
        device: str = "cuda",
        dtype: str = "bfloat16",
    ) -> dict[str, float]:
        """Pre-load models. Returns {name: load_time_seconds}.

        Args:
            models: Mapping of {name: model_id_or_path}.
            device: Target device ("cuda", "cpu").
            dtype: Model dtype ("bfloat16", "float16", "float32").
        """
        import zerostart
        zerostart.accelerate(cache_dir=str(self.cache.cache_dir))

        times: dict[str, float] = {}
        for name, model_id in models.items():
            t0 = time.monotonic()
            key = cache_key(model_id, {"dtype": dtype})

            if self.cache.has(key):
                state = self.cache.load(key, device=device)
                self._loaded[name] = state
            else:
                model = self._load_model(model_id, device, dtype)
                state = {"model": model}
                self.cache.save(key, state, model_id=model_id, dtype=dtype)
                self._loaded[name] = state

            elapsed = time.monotonic() - t0
            times[name] = elapsed
            log.info("Loaded %s (%s) in %.2fs", name, model_id, elapsed)

        return times

    def get(self, name: str) -> Any:
        """Get a pre-loaded model."""
        state = self._loaded.get(name)
        if state is None:
            raise KeyError(f"Model '{name}' not loaded. Call preload() first.")
        return state.get("model", state)

    def get_state(self, name: str) -> dict[str, Any]:
        """Get full state dict (model + tokenizer + extras)."""
        state = self._loaded.get(name)
        if state is None:
            raise KeyError(f"Model '{name}' not loaded. Call preload() first.")
        return state

    def _load_model(self, model_id: str, device: str, dtype: str) -> Any:
        import torch
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)

        # Try transformers first, then diffusers
        try:
            from transformers import AutoModelForCausalLM
            return AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch_dtype, device_map=device,
            )
        except Exception:
            pass

        try:
            from diffusers import DiffusionPipeline
            return DiffusionPipeline.from_pretrained(
                model_id, torch_dtype=torch_dtype,
            ).to(device)
        except Exception:
            pass

        raise ValueError(
            f"Could not load {model_id} — install transformers or diffusers"
        )
