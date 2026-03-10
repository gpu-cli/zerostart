"""vLLM integration for accelerated model loading.

Provides a custom model loader that subclasses vLLM's DefaultModelLoader
and runs inside vLLM's EngineCore subprocess where weights are actually loaded.

Key optimization:
  Network volume fix: sets safetensors_load_strategy="eager" on FUSE/NFS
  volumes where mmap is 30-50x slower than sequential read.

Usage:
    # Option 1: Auto-registration via entry_points (pip install zerostart)
    vllm serve Qwen/Qwen2.5-7B --load-format zerostart

    # Option 2: Manual registration
    from zerostart.integrations.vllm import register
    register()
    # Then: --load-format zerostart
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

log = logging.getLogger("zerostart.vllm")

if TYPE_CHECKING:
    import torch.nn as nn
    from vllm.config import ModelConfig
    from vllm.config.load import LoadConfig


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register() -> None:
    """Register the zerostart model loader with vLLM.

    After calling this, you can use --load-format zerostart with vLLM.
    """
    try:
        from vllm.model_executor.model_loader import register_model_loader
        register_model_loader("zerostart")(ZerostartModelLoader)
        log.info("Registered zerostart model loader with vLLM")
    except ImportError:
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


def register_plugin() -> None:
    """Entry point for vLLM's general plugin system.

    Register in pyproject.toml:
        [project.entry-points."vllm.general_plugins"]
        zerostart = "zerostart.integrations.vllm:register_plugin"

    This runs in EVERY vLLM process (including EngineCore subprocesses)
    before model loading begins.
    """
    register()
    log.info("zerostart vLLM plugin loaded")


# ---------------------------------------------------------------------------
# Dynamic base class (don't fail if vLLM not installed)
# ---------------------------------------------------------------------------

def _get_default_loader_class() -> type:
    """Get DefaultModelLoader, falling back to BaseModelLoader, then object."""
    try:
        from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
        return DefaultModelLoader
    except ImportError:
        pass
    try:
        from vllm.model_executor.model_loader.base_loader import BaseModelLoader
        return BaseModelLoader
    except ImportError:
        return object


_DefaultLoader = _get_default_loader_class()


# ---------------------------------------------------------------------------
# Network volume detection
# ---------------------------------------------------------------------------

_network_volume_cache: dict[str, bool] = {}


def _is_network_volume(path: str) -> bool:
    """Check if path is on a FUSE/NFS filesystem where mmap is 30-50x slower."""
    if path in _network_volume_cache:
        return _network_volume_cache[path]

    result = False
    # Only truly network-backed filesystems where mmap page faults
    # trigger network round-trips. Overlay is excluded because it's
    # backed by local storage and mmap works fine there.
    slow_fs = frozenset({
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
        result = best_fs in slow_fs
    except FileNotFoundError:
        pass

    _network_volume_cache[path] = result
    return result


# ---------------------------------------------------------------------------
# ZerostartModelLoader
# ---------------------------------------------------------------------------

class ZerostartModelLoader(_DefaultLoader):  # type: ignore[misc]
    """vLLM model loader with network volume acceleration.

    Subclasses DefaultModelLoader. On FUSE/NFS network volumes, sets
    safetensors_load_strategy="eager" so vLLM reads entire files into
    memory instead of using mmap (which is 30-50x slower on these FSes).

    On local NVMe, delegates entirely to DefaultModelLoader (mmap is fast).
    """

    def __init__(self, load_config: LoadConfig):
        import os

        # Rewrite load_format from "zerostart" to "safetensors" so
        # DefaultModelLoader._prepare_weights() doesn't reject it.
        self._zerostart_requested = getattr(load_config, "load_format", None) == "zerostart"
        if self._zerostart_requested:
            load_config.load_format = "safetensors"

        # Switch to eager loading if explicitly requested or on a network FS
        # where mmap page faults trigger expensive network round-trips.
        #
        # Note: on FUSE mounts with warm page cache (e.g. RunPod MFS), mmap
        # is actually faster than eager because it avoids copying data.
        # Eager only helps on cold reads from slow network FSes (NFS, JuiceFS).
        # Use ZEROSTART_EAGER=1 to force eager loading.
        force_eager = os.environ.get("ZEROSTART_EAGER", "").lower() in ("1", "true")
        on_network_volume = any(
            _is_network_volume(p)
            for p in ["/volume", "/gpu-cli-workspaces", "/workspace"]
            if Path(p).exists()
        )

        if force_eager or on_network_volume:
            current = getattr(load_config, "safetensors_load_strategy", "lazy")
            if current != "eager":
                load_config.safetensors_load_strategy = "eager"
                reason = "ZEROSTART_EAGER=1" if force_eager else "network volume detected"
                log.info(
                    "Switched safetensors_load_strategy to 'eager' (%s)", reason,
                )

        if _DefaultLoader is not object:
            super().__init__(load_config)
        else:
            self.load_config = load_config

    def download_model(self, model_config: ModelConfig) -> None:
        """Download model via HF hub (standard path)."""
        if _DefaultLoader is not object and hasattr(super(), "download_model"):
            super().download_model(model_config)
        else:
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(
                    model_config.model,
                    revision=getattr(model_config, "revision", None),
                )
            except Exception as e:
                log.warning("HF download failed: %s", e)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Load weights, delegating to DefaultModelLoader."""
        t0 = time.monotonic()

        if _DefaultLoader is not object and hasattr(super(), "load_weights"):
            super().load_weights(model, model_config)
        else:
            log.warning("DefaultModelLoader not available — basic weight loading")
            self._fallback_load_weights(model, model_config)

        elapsed = time.monotonic() - t0
        strategy = getattr(self.load_config, "safetensors_load_strategy", "unknown")
        log.info("Weight loading complete (%.2fs, strategy=%s)", elapsed, strategy)

    def _fallback_load_weights(
        self, model: nn.Module, model_config: ModelConfig,
    ) -> None:
        """Fallback weight loading when DefaultModelLoader isn't available."""
        from safetensors.torch import load_file

        model_path = Path(model_config.model)
        if not model_path.is_dir():
            try:
                from zerostart.snapshot import _find_hf_cache_dir
                cache_dir = _find_hf_cache_dir(model_config.model)
                if cache_dir:
                    model_path = cache_dir
            except ImportError:
                pass

        sf_files = sorted(model_path.glob("*.safetensors"))
        if not sf_files:
            log.warning("No safetensors files found at %s", model_path)
            return

        for sf_file in sf_files:
            if _is_network_volume(str(sf_file)):
                import safetensors.torch as st
                with open(sf_file, "rb") as f:
                    tensors = st.load(f.read())
            else:
                tensors = load_file(str(sf_file))

            if hasattr(model, "load_weights"):
                model.load_weights(tensors.items())
            else:
                model.load_state_dict(tensors, strict=False)
