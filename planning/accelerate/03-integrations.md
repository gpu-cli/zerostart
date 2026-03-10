# 03: Serving Integrations

## Overview

Framework-specific patches that work with Layer 1 (hooks) and Layer 2 (cache) to accelerate specific serving stacks.

## ComfyUI Integration

### How ComfyUI loads models

ComfyUI's model loading path:
```
comfy/sd.py:load_checkpoint_guess_config()
  → safetensors.torch.load_file() or torch.load()
  → model_config.get_model()  # creates model skeleton
  → model.load_state_dict()
  → ModelPatcher(model)  # wraps for memory management
```

ComfyUI also has its own model management:
- `comfy/model_management.py` — tracks VRAM, moves models between CPU/GPU
- `comfy/model_patcher.py` — lazy weight loading, LoRA application
- Models are cached in memory between workflow runs

### What we patch

```python
# zerostart/integrations/comfyui.py

def patch():
    """Patch ComfyUI for accelerated model loading.

    Three levels:
    1. safetensors.load_file — network volume fix (from Layer 1)
    2. load_checkpoint_guess_config — cache full loaded checkpoints
    3. model_management — pre-load models on startup
    """
    import zerostart
    zerostart.accelerate()  # Layer 1 hooks

    # Patch ComfyUI's checkpoint loader
    try:
        import comfy.sd as sd
        _original_load = sd.load_checkpoint_guess_config

        def _fast_load(ckpt_path, *args, **kwargs):
            cache = zerostart.model_cache()
            key = _comfy_cache_key(ckpt_path)

            if cache.has(key):
                log.info("Cache hit: %s", Path(ckpt_path).name)
                state = cache.load(key, device="cpu")  # CPU — ComfyUI manages GPU placement
                return _wrap_as_model_patcher(state)

            result = _original_load(ckpt_path, *args, **kwargs)

            # Cache for next time
            cache.save(key, _extract_state(result))
            return result

        sd.load_checkpoint_guess_config = _fast_load
    except ImportError:
        pass


def preload(model_paths: list[str], cache_dir: str | None = None):
    """Pre-snapshot ComfyUI models for fast loading.

    Run once after downloading models:
        zerostart warm-model --comfyui /models/checkpoints/sdxl_base.safetensors
    """
    cache = ModelCache(cache_dir)
    for path in model_paths:
        key = _comfy_cache_key(path)
        if not cache.has(key):
            # Load and snapshot
            state_dict = safetensors.torch.load_file(path)
            cache.save(key, {"state_dict": state_dict})
            log.info("Cached: %s", Path(path).name)
```

### CLI

```bash
# Run ComfyUI with acceleration
zerostart --accelerate -p comfyui main.py --listen 0.0.0.0

# Pre-warm ComfyUI model cache
zerostart warm-model --comfyui \
    /models/checkpoints/sdxl_base.safetensors \
    /models/controlnet/control_v11p_sd15_canny.safetensors
```

## vLLM Integration

### How vLLM loads models

```
vllm.engine.llm_engine.LLMEngine.__init__()
  → ModelLoader.download_model()  # HF hub download
  → ModelLoader.load_weights()    # Load into model
    → safetensors.torch.load_file() per shard
    → model.load_weights(iter_of_tensors)
```

vLLM supports custom model loaders via `--load-format`. We provide a `ZerostartModelLoader`.

### Implementation

```python
# zerostart/integrations/vllm.py

from vllm.model_executor.model_loader.loader import BaseModelLoader

class ZerostartModelLoader(BaseModelLoader):
    """vLLM model loader using zerostart's mmap hydrate.

    Usage:
        vllm serve model --load-format zerostart

    Or register programmatically:
        from zerostart.integrations.vllm import register
        register()
        # then use --load-format zerostart
    """

    def __init__(self, load_config):
        super().__init__(load_config)
        self.cache = ModelCache()

    def download_model(self, model_config) -> None:
        """Download via HF hub (standard path)."""
        from huggingface_hub import snapshot_download
        snapshot_download(model_config.model, revision=model_config.revision)

    def load_weights(self, model, model_config) -> None:
        """Load weights from cache or standard path."""
        key = cache_key(model_config.model, {
            "dtype": str(model_config.dtype),
            "revision": model_config.revision,
        })

        if self.cache.has(key):
            log.info("Loading from zerostart cache")
            state = self.cache.load(key, device="cuda")
            # vLLM models use load_weights(iter) not load_state_dict
            model.load_weights(state["model"].state_dict().items())
        else:
            # Standard load, then cache
            from vllm.model_executor.model_loader.loader import DefaultModelLoader
            default = DefaultModelLoader(self.load_config)
            default.load_weights(model, model_config)
            self.cache.save(key, {"model": model})


def register():
    """Register zerostart loader with vLLM."""
    from vllm.model_executor.model_loader import loader
    loader._MODEL_LOADER_REGISTRY["zerostart"] = ZerostartModelLoader
```

### CLI

```bash
# vLLM with zerostart acceleration
pip install zerostart
vllm serve Qwen/Qwen2.5-7B --load-format zerostart

# Or via zerostart CLI
zerostart --accelerate -p vllm -- python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B
```

## Generic Serving Helper

For custom serving stacks that aren't ComfyUI or vLLM.

```python
# zerostart/integrations/serving.py

class ModelServer:
    """Pre-load and serve models from cache.

    Usage:
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

    def __init__(self, cache_dir: str | Path | None = None):
        self.cache = ModelCache(cache_dir)
        self._loaded: dict[str, Any] = {}

    def preload(
        self,
        models: dict[str, str],
        device: str = "cuda",
        dtype: str = "bfloat16",
    ) -> dict[str, float]:
        """Pre-load models. Returns {name: load_time_seconds}."""
        times = {}
        for name, model_id in models.items():
            t0 = time.monotonic()
            key = cache_key(model_id, {"dtype": dtype})

            if self.cache.has(key):
                state = self.cache.load(key, device=device)
                self._loaded[name] = state
            else:
                # First time — load via transformers/diffusers, auto-cache
                import zerostart
                zerostart.accelerate(cache_dir=str(self.cache.cache_dir))
                model = self._load_model(model_id, device, dtype)
                self.cache.save(key, {"model": model})
                self._loaded[name] = {"model": model}

            times[name] = time.monotonic() - t0
            log.info("Loaded %s (%s) in %.2fs", name, model_id, times[name])
        return times

    def get(self, name: str) -> Any:
        """Get a pre-loaded model."""
        state = self._loaded.get(name)
        if state is None:
            raise KeyError(f"Model '{name}' not loaded. Call preload() first.")
        return state.get("model", state)

    def _load_model(self, model_id, device, dtype):
        import torch
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        try:
            from transformers import AutoModelForCausalLM
            return AutoModelForCausalLM.from_pretrained(
                model_id, dtype=dtype_map.get(dtype, torch.bfloat16), device_map=device,
            )
        except Exception:
            from diffusers import DiffusionPipeline
            return DiffusionPipeline.from_pretrained(
                model_id, torch_dtype=dtype_map.get(dtype, torch.bfloat16),
            ).to(device)
```

## Extending snapshot.py for diffusers

The existing `_reconstruct_module_from_config` handles transformers models. For diffusers:

```python
# Add to snapshot.py

def _extract_model_config(module: Any) -> dict[str, Any] | None:
    # Existing transformers check
    if hasattr(module, "config"):
        config = module.config
        if hasattr(config, "to_dict"):
            return {
                "_type": "transformers",
                # ...existing code...
            }

    # NEW: diffusers models
    if hasattr(module, "config") and isinstance(getattr(module, "config", None), dict):
        # diffusers stores config as a plain dict
        return {
            "_type": "diffusers",
            "_class": type(module).__name__,
            "_module": type(module).__module__,
            "config_dict": module.config,
        }

    return None


def _reconstruct_module_from_config(model_config, ...):
    mc = model_config
    if mc["_type"] == "transformers":
        # ...existing code...
    elif mc["_type"] == "diffusers":
        model_module = importlib.import_module(mc["_module"])
        model_class = getattr(model_module, mc["_class"])
        with _no_init_weights():
            with torch.device("meta"):
                module = model_class(**mc["config_dict"])
        # ... same load_state_dict flow ...
```
