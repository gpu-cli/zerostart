"""Snapshot & hydrate: fast checkpoint/restore for GPU Python applications.

Strategy C (JSON-only): No cloudpickle. Tokenizer saved via save_pretrained,
model config as JSON, tensors referenced in safetensors files via mmap.

Cold hydrate of Qwen2.5-7B (15.2GB): ~1.7s (excluding inference)
  - Rust tokenizer: 0.22s
  - mmap 339 tensors: 0.07s
  - import model class: 1.09s
  - reconstruct model: 0.33s

Usage:
    from zerostart.snapshot import snapshot, hydrate

    snapshot(
        state={"model": model, "tokenizer": tokenizer},
        path="/cache/my-model.zsnap",
    )

    state = hydrate("/cache/my-model.zsnap")
    model = state["model"]       # weights are mmap'd, zero-copy
    tokenizer = state["tokenizer"]
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import struct
import sys
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("zerostart.snapshot")

SNAPSHOT_VERSION = 2


# ---------------------------------------------------------------------------
# No-init weights context manager
# ---------------------------------------------------------------------------

class _no_init_weights:
    """Patches torch.nn.init functions to no-ops during model creation."""

    def __init__(self) -> None:
        self._originals: dict[str, Any] = {}

    def __enter__(self) -> None:
        import torch.nn.init as init

        for name in dir(init):
            fn = getattr(init, name)
            if callable(fn) and not name.startswith("_"):
                self._originals[name] = fn
                setattr(init, name, lambda *args, **kwargs: None)

        import torch.nn as nn
        if hasattr(nn.Module, "reset_parameters"):
            self._originals["Module.reset_parameters"] = nn.Module.reset_parameters
            nn.Module.reset_parameters = lambda self: None

    def __exit__(self, *args: object) -> None:
        import torch.nn.init as init
        import torch.nn as nn

        for name, fn in self._originals.items():
            if name == "Module.reset_parameters":
                nn.Module.reset_parameters = fn
            else:
                setattr(init, name, fn)
        self._originals.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_torch():
    import torch
    return torch


def _environment_fingerprint() -> str:
    parts = [
        f"python={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        f"platform={platform.machine()}",
    ]
    try:
        torch = _get_torch()
        parts.append(f"torch={torch.__version__}")
        if torch.cuda.is_available():
            parts.append(f"cuda={torch.version.cuda}")
    except ImportError:
        pass
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _extract_model_config(module: Any) -> dict[str, Any] | None:
    if hasattr(module, "config"):
        config = module.config
        if hasattr(config, "to_dict"):
            return {
                "_type": "transformers",
                "_class": type(module).__name__,
                "_module": type(module).__module__,
                "config_class": type(config).__name__,
                "config_module": type(config).__module__,
                "config_dict": config.to_dict(),
            }
    return None


# ---------------------------------------------------------------------------
# Safetensors file discovery
# ---------------------------------------------------------------------------

def _find_safetensors_for_model(module: Any) -> list[Path]:
    paths: list[Path] = []
    if hasattr(module, "config") and hasattr(module.config, "_name_or_path"):
        model_path = Path(module.config._name_or_path)
        if model_path.is_dir():
            paths.extend(sorted(model_path.glob("*.safetensors")))
        else:
            hf_cache = _find_hf_cache_dir(module.config._name_or_path)
            if hf_cache:
                paths.extend(sorted(hf_cache.glob("*.safetensors")))
    return paths


def _find_hf_cache_dir(model_id: str) -> Path | None:
    safe_id = model_id.replace("/", "--")
    model_subdir = f"models--{safe_id}"

    candidates: list[Path] = []
    try:
        from huggingface_hub import constants
        candidates.append(Path(constants.HF_HUB_CACHE))
    except ImportError:
        pass
    if hf_hub_cache := os.environ.get("HF_HUB_CACHE"):
        candidates.append(Path(hf_hub_cache))
    if hf_home := os.environ.get("HF_HOME"):
        candidates.append(Path(hf_home) / "hub")
    candidates.append(Path(os.path.expanduser("~/.cache/huggingface/hub")))

    seen: set[str] = set()
    for c in candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)

        model_dir = c / model_subdir
        snapshots = model_dir / "snapshots"
        if not snapshots.is_dir():
            continue

        snap_dirs = sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if snap_dirs:
            result = snap_dirs[0]
            sf_count = len(list(result.glob("*.safetensors")))
            log.info("Found HF cache for %s at %s (%d safetensors files)", model_id, result, sf_count)
            return result

    log.warning("Could not find HF cache for %s", model_id)
    return None


def _build_tensor_to_file_map(
    safetensors_files: list[Path],
) -> dict[str, tuple[Path, str]]:
    tensor_to_file: dict[str, tuple[Path, str]] = {}
    for sf_path in safetensors_files:
        try:
            with open(sf_path, "rb") as f:
                header_size_bytes = f.read(8)
                if len(header_size_bytes) < 8:
                    continue
                header_size = struct.unpack("<Q", header_size_bytes)[0]
                header_json = f.read(header_size)
                header = json.loads(header_json)
                for tensor_name in header:
                    if tensor_name != "__metadata__":
                        tensor_to_file[tensor_name] = (sf_path, tensor_name)
        except Exception as e:
            log.warning("Failed to read safetensors header from %s: %s", sf_path, e)
    return tensor_to_file


def _match_tensor_to_safetensors(
    ref_key: str,
    tensor_to_file: dict[str, tuple[Path, str]],
) -> tuple[str, str] | None:
    """Try progressively stripping dot-prefixes to match a tensor name."""
    candidates = [ref_key]
    remaining = ref_key
    while "." in remaining:
        remaining = remaining.split(".", 1)[1]
        candidates.append(remaining)
    for candidate in candidates:
        if candidate in tensor_to_file:
            sf_path, sf_tensor_name = tensor_to_file[candidate]
            return str(sf_path), sf_tensor_name
    return None


# ---------------------------------------------------------------------------
# Snapshot: capture state
# ---------------------------------------------------------------------------

def snapshot(
    state: dict[str, Any],
    path: str | Path,
    safetensors_files: list[str | Path] | None = None,
) -> Path:
    """Snapshot a Python+model state for fast hydration.

    Saves tokenizer via save_pretrained, model config as JSON,
    tensor references into safetensors files. No cloudpickle.
    """
    t0 = time.monotonic()
    torch = _get_torch()
    snap_dir = Path(path)
    snap_dir.mkdir(parents=True, exist_ok=True)

    # 1. Find safetensors files
    sf_files: list[Path] = []
    if safetensors_files:
        sf_files = [Path(f) for f in safetensors_files]
    else:
        for value in state.values():
            if isinstance(value, torch.nn.Module):
                sf_files.extend(_find_safetensors_for_model(value))
    sf_files = list(dict.fromkeys(sf_files))
    log.info("Found %d safetensors files", len(sf_files))

    # 2. Build tensor→file mapping
    tensor_to_file = _build_tensor_to_file_map(sf_files)
    log.info("Mapped %d tensors across safetensors files", len(tensor_to_file))

    # 3. Process each state entry
    model_configs: dict[str, dict[str, Any]] = {}
    tensor_refs: dict[str, dict[str, Any]] = {}
    tokenizer_keys: list[str] = []
    unmatched: list[str] = []

    for key, value in state.items():
        if isinstance(value, torch.nn.Module):
            # Save model config as JSON
            config = _extract_model_config(value)
            if config:
                model_configs[key] = config

            # Build tensor refs from state_dict
            sd = value.state_dict()
            tensor_keys_for_model = []
            for param_name, tensor in sd.items():
                ref_key = f"{key}.{param_name}"
                ref_info: dict[str, Any] = {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                }
                match = _match_tensor_to_safetensors(ref_key, tensor_to_file)
                if match:
                    ref_info["safetensors_file"] = match[0]
                    ref_info["safetensors_tensor"] = match[1]
                else:
                    unmatched.append(ref_key)
                tensor_refs[ref_key] = ref_info
                tensor_keys_for_model.append(ref_key)

        elif _is_tokenizer(value):
            # Save tokenizer via save_pretrained (JSON files)
            tok_dir = snap_dir / f"tokenizer_{key}"
            value.save_pretrained(str(tok_dir))
            tokenizer_keys.append(key)
            log.info("Saved tokenizer '%s' via save_pretrained", key)

        elif isinstance(value, torch.Tensor):
            ref_key = key
            ref_info = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            match = _match_tensor_to_safetensors(ref_key, tensor_to_file)
            if match:
                ref_info["safetensors_file"] = match[0]
                ref_info["safetensors_tensor"] = match[1]
            else:
                unmatched.append(ref_key)
            tensor_refs[ref_key] = ref_info

    if unmatched:
        log.warning("%d tensors not matched to safetensors — will be serialized", len(unmatched))

    # 4. Serialize unmatched tensors
    if unmatched:
        _save_unmatched_tensors(snap_dir, unmatched, state, torch)

    # 5. Write manifest (pure JSON, no pickle)
    manifest = {
        "version": SNAPSHOT_VERSION,
        "created": time.time(),
        "fingerprint": _environment_fingerprint(),
        "model_configs": model_configs,
        "tensor_refs": tensor_refs,
        "tokenizer_keys": tokenizer_keys,
        "safetensors_files": [str(f) for f in sf_files],
        "state_keys": list(state.keys()),
        "tensor_count": len(tensor_refs),
        "matched_tensors": len(tensor_refs) - len(unmatched),
        "unmatched_tensors": len(unmatched),
    }

    with open(snap_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)

    elapsed = time.monotonic() - t0
    total_params = sum(1 for _ in tensor_refs)
    log.info(
        "Snapshot saved to %s (%.1fs, %d tensors, %d matched)",
        snap_dir, elapsed, len(tensor_refs), len(tensor_refs) - len(unmatched),
    )

    return snap_dir


def _is_tokenizer(obj: Any) -> bool:
    """Check if an object is a HuggingFace tokenizer."""
    return hasattr(obj, "save_pretrained") and hasattr(obj, "encode") and hasattr(obj, "decode")


def _save_unmatched_tensors(
    snap_dir: Path,
    unmatched: list[str],
    state: dict[str, Any],
    torch: Any,
) -> None:
    """Save tensors not found in safetensors files."""
    import io
    tensors_dir = snap_dir / "tensors"
    tensors_dir.mkdir(exist_ok=True)

    for ref_key in unmatched:
        parts = ref_key.split(".", 1)
        top_key = parts[0]
        param_name = parts[1] if len(parts) > 1 else None

        obj = state.get(top_key)
        if obj is None:
            continue

        tensor = None
        if isinstance(obj, torch.nn.Module) and param_name:
            sd = obj.state_dict()
            tensor = sd.get(param_name)
        elif isinstance(obj, torch.Tensor):
            tensor = obj

        if tensor is not None:
            buf = io.BytesIO()
            torch.save(tensor, buf)
            safe_name = ref_key.replace("/", "_").replace(".", "_") + ".pt"
            with open(tensors_dir / safe_name, "wb") as f:
                f.write(buf.getvalue())


# ---------------------------------------------------------------------------
# Hydrate: restore state
# ---------------------------------------------------------------------------

def hydrate(
    path: str | Path,
    device: str | None = None,
    verify_fingerprint: bool = True,
) -> dict[str, Any]:
    """Hydrate a snapshot — restore model + tokenizer with mmap'd weights.

    No cloudpickle. Tokenizer loaded via Rust tokenizers library (fast path)
    or transformers AutoTokenizer (fallback). Model reconstructed from JSON
    config + mmap'd safetensors.
    """
    t0 = time.monotonic()
    snap_dir = Path(path)

    # 1. Load manifest
    with open(snap_dir / "manifest.json") as f:
        manifest = json.load(f)

    version = manifest.get("version", 1)
    if version == 1:
        return _hydrate_v1(snap_dir, manifest, device, verify_fingerprint, t0)

    if verify_fingerprint:
        current_fp = _environment_fingerprint()
        snap_fp = manifest.get("fingerprint", "")
        if current_fp != snap_fp:
            log.warning(
                "Environment fingerprint mismatch (snapshot=%s, current=%s)",
                snap_fp, current_fp,
            )

    t_manifest = time.monotonic()

    # 2. Load tokenizers (Rust fast path, no transformers import)
    restored_state: dict[str, Any] = {}
    for tok_key in manifest.get("tokenizer_keys", []):
        tok_dir = snap_dir / f"tokenizer_{tok_key}"
        restored_state[tok_key] = _load_tokenizer(tok_dir)
    t_tok = time.monotonic()

    # 3. Load tensors via mmap (directly to target device if possible)
    torch = _get_torch()
    tensor_device = device or "cpu"
    loaded_tensors = _load_tensors_mmap(manifest, snap_dir, torch, tensor_device)
    t_mmap = time.monotonic()

    # 4. Reconstruct models (tensors already on target device)
    for model_key, model_config in manifest.get("model_configs", {}).items():
        module = _reconstruct_module_from_config(
            model_config, model_key, loaded_tensors, device, torch,
        )
        if module is not None:
            restored_state[model_key] = module
    t_model = time.monotonic()

    # 5. Restore standalone tensors
    for ref_key in manifest.get("tensor_refs", {}):
        if "." not in ref_key and ref_key not in restored_state:
            tensor = loaded_tensors.get(ref_key)
            if tensor is not None:
                if device:
                    tensor = tensor.to(device)
                restored_state[ref_key] = tensor

    elapsed = time.monotonic() - t0
    log.info(
        "Hydration complete (%.3fs total: %.3fs manifest + %.3fs tokenizer + %.3fs mmap + %.3fs model)",
        elapsed, t_manifest - t0, t_tok - t_manifest, t_mmap - t_tok, t_model - t_mmap,
    )

    return restored_state


def _load_tokenizer(tok_dir: Path) -> Any:
    """Load tokenizer — try Rust tokenizers lib first, fall back to transformers."""
    t0 = time.monotonic()

    # Fast path: Rust tokenizers library (no transformers import)
    tokenizer_json = tok_dir / "tokenizer.json"
    if tokenizer_json.exists():
        try:
            from tokenizers import Tokenizer as RustTokenizer
            rust_tok = RustTokenizer.from_file(str(tokenizer_json))
            wrapper = _FastTokenizerWrapper(rust_tok, tok_dir)
            log.info("Loaded tokenizer via Rust tokenizers lib (%.3fs)", time.monotonic() - t0)
            return wrapper
        except ImportError:
            pass

    # Fallback: transformers AutoTokenizer
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(tok_dir))
    log.info("Loaded tokenizer via AutoTokenizer (%.3fs)", time.monotonic() - t0)
    return tok


class _FastTokenizerWrapper:
    """Minimal wrapper around tokenizers.Tokenizer for generate() compat."""

    def __init__(self, rust_tokenizer: Any, tok_dir: Path):
        self._tok = rust_tokenizer
        # Load special tokens from tokenizer_config.json
        config_path = tok_dir / "tokenizer_config.json"
        self._config: dict[str, Any] = {}
        if config_path.exists():
            with open(config_path) as f:
                self._config = json.load(f)

        # Handle eos_token_id that might be a dict or list
        eos = self._config.get("eos_token_id")
        if isinstance(eos, list):
            eos = eos[0] if eos else None
        elif isinstance(eos, dict):
            eos = eos.get("content")
        self.eos_token_id = eos
        self.pad_token_id = self._config.get("pad_token_id", self.eos_token_id)

    def __call__(self, text: str, return_tensors: str | None = None, **kwargs: Any) -> Any:
        import torch
        encoded = self._tok.encode(text)
        ids = encoded.ids
        mask = encoded.attention_mask
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.tensor([mask], dtype=torch.long),
            }
        return {"input_ids": ids, "attention_mask": mask}

    def decode(self, ids: Any, skip_special_tokens: bool = False) -> str:
        import torch
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def encode(self, text: str, **kwargs: Any) -> Any:
        return self._tok.encode(text)


def _load_tensors_mmap(
    manifest: dict[str, Any],
    snap_dir: Path,
    torch: Any,
    device: str = "cpu",
) -> dict[str, Any]:
    """Load all referenced tensors via mmap, directly to target device."""
    tensor_refs = manifest.get("tensor_refs", {})

    # Group by safetensors file
    file_to_tensors: dict[str, list[tuple[str, str]]] = {}
    standalone: list[str] = []

    for ref_key, ref_info in tensor_refs.items():
        sf_file = ref_info.get("safetensors_file")
        sf_tensor = ref_info.get("safetensors_tensor")
        if sf_file and sf_tensor:
            file_to_tensors.setdefault(sf_file, []).append((ref_key, sf_tensor))
        else:
            standalone.append(ref_key)

    loaded: dict[str, Any] = {}

    for sf_file, tensor_pairs in file_to_tensors.items():
        sf_path = Path(sf_file)
        if not sf_path.exists():
            log.warning("Safetensors file not found: %s", sf_file)
            continue

        # Try safetensors-streaming first (Rust mmap)
        try:
            import safetensors_streaming
            handle = safetensors_streaming.safe_open(str(sf_path), framework="pt", device=device)
            for ref_key, sf_tensor_name in tensor_pairs:
                try:
                    loaded[ref_key] = handle.get_tensor(sf_tensor_name)
                except Exception as e:
                    log.warning("Failed to load %s: %s", sf_tensor_name, e)
            continue
        except ImportError:
            pass

        # Standard safetensors — load directly to target device
        try:
            from safetensors.torch import load_file
            all_tensors = load_file(str(sf_path), device=device)
            for ref_key, sf_tensor_name in tensor_pairs:
                if sf_tensor_name in all_tensors:
                    loaded[ref_key] = all_tensors[sf_tensor_name]
            continue
        except ImportError:
            pass

        log.warning("No safetensors loader for %s", sf_file)

    # Standalone (unmatched) tensors
    tensors_dir = snap_dir / "tensors"
    for ref_key in standalone:
        safe_name = ref_key.replace("/", "_").replace(".", "_") + ".pt"
        pt_path = tensors_dir / safe_name
        if pt_path.exists():
            loaded[ref_key] = torch.load(pt_path, map_location=device, weights_only=True)

    return loaded


def _reconstruct_module_from_config(
    model_config: dict[str, Any],
    state_key: str,
    loaded_tensors: dict[str, Any],
    device: str | None,
    torch: Any,
) -> Any:
    """Reconstruct an nn.Module from JSON config + mmap'd tensors."""
    import importlib

    t0 = time.monotonic()

    mc = model_config
    if mc.get("_type") != "transformers":
        log.warning("Unknown model type: %s", mc.get("_type"))
        return None

    try:
        model_module = importlib.import_module(mc["_module"])
        model_class = getattr(model_module, mc["_class"])
        config_module = importlib.import_module(mc["config_module"])
        config_class = getattr(config_module, mc["config_class"])
    except Exception as e:
        log.warning("Failed to import model class: %s", e)
        return None

    t_import = time.monotonic()

    try:
        cfg = config_class.from_dict(mc["config_dict"])
        with _no_init_weights():
            with torch.device("meta"):
                module = model_class(cfg)
    except Exception as e:
        log.warning("Failed to create model on meta device: %s", e)
        return None

    t_meta = time.monotonic()

    # Build state_dict from loaded tensors
    prefix = f"{state_key}."
    state_dict = {}
    for ref_key, tensor in loaded_tensors.items():
        if ref_key.startswith(prefix):
            param_name = ref_key[len(prefix):]
            state_dict[param_name] = tensor

    if state_dict:
        try:
            module.load_state_dict(state_dict, strict=False, assign=True)
        except TypeError:
            module.load_state_dict(state_dict, strict=False)

        if hasattr(module, "tie_weights"):
            module.tie_weights()

    t_load = time.monotonic()

    _materialize_meta_tensors(module)

    if device:
        module = module.to(device)

    t_done = time.monotonic()
    log.info(
        "Reconstructed %s (import=%.2fs, meta=%.2fs, load_sd=%.2fs, materialize=%.2fs)",
        mc["_class"], t_import - t0, t_meta - t_import,
        t_load - t_meta, t_done - t_load,
    )

    return module


def _materialize_meta_tensors(module: Any) -> None:
    """Replace remaining meta-device tensors with properly initialized CPU tensors."""
    torch = _get_torch()
    meta_submodules: list[tuple[str, Any]] = []

    for name, submodule in module.named_modules():
        for buf_name, buf in submodule.named_buffers(recurse=False):
            if buf.device.type == "meta":
                meta_submodules.append((name, submodule))
                break

    if not meta_submodules:
        return

    log.info("Materializing %d submodules with meta tensors", len(meta_submodules))

    for name, submodule in meta_submodules:
        submodule_class = type(submodule)
        try:
            if hasattr(submodule, "config"):
                new_sub = submodule_class(submodule.config)
            elif hasattr(submodule, "dim") and hasattr(submodule, "max_position_embeddings"):
                new_sub = submodule_class(submodule.dim, submodule.max_position_embeddings)
            else:
                new_sub = submodule.to_empty(device="cpu")

            if name:
                parts = name.split(".")
                target = module
                for part in parts[:-1]:
                    target = getattr(target, part)
                setattr(target, parts[-1], new_sub)
        except Exception as e:
            log.warning("Failed to materialize %s (%s): %s", name, submodule_class.__name__, e)
            for buf_name, buf in list(submodule.named_buffers(recurse=False)):
                if buf.device.type == "meta":
                    new_buf = torch.empty(buf.shape, dtype=buf.dtype, device="cpu")
                    submodule.register_buffer(buf_name, new_buf)


# ---------------------------------------------------------------------------
# V1 backward compat (cloudpickle-based snapshots)
# ---------------------------------------------------------------------------

class _ModulePlaceholder:
    def __init__(self, module_class: type, config: dict[str, Any] | None, tensor_keys: list[str]):
        self.module_class = module_class
        self.config = config
        self.tensor_keys = tensor_keys


class _TensorPlaceholder:
    def __init__(self, key: str):
        self.key = key


def _hydrate_v1(
    snap_dir: Path,
    manifest: dict[str, Any],
    device: str | None,
    verify_fingerprint: bool,
    t0: float,
) -> dict[str, Any]:
    """Hydrate a v1 (cloudpickle-based) snapshot."""
    import cloudpickle

    if verify_fingerprint:
        current_fp = _environment_fingerprint()
        snap_fp = manifest.get("fingerprint", "")
        if current_fp != snap_fp:
            log.warning("Environment fingerprint mismatch (snapshot=%s, current=%s)", snap_fp, current_fp)

    with open(snap_dir / "python_state.pkl", "rb") as f:
        cleaned_state = cloudpickle.loads(f.read())
    t_python = time.monotonic() - t0
    log.info("Python state loaded (%.3fs)", t_python)

    torch = _get_torch()
    loaded_tensors = _load_tensors_mmap(manifest, snap_dir, torch)
    t_mmap = time.monotonic() - t0 - t_python
    log.info("Tensors loaded via mmap (%.3fs, %d tensors)", t_mmap, len(loaded_tensors))

    restored_state: dict[str, Any] = {}
    for key, value in cleaned_state.items():
        if isinstance(value, _ModulePlaceholder):
            if value.config and value.config.get("_type") == "transformers":
                module = _reconstruct_module_from_config(
                    value.config, key, loaded_tensors, device, torch,
                )
            else:
                module = None
            restored_state[key] = module
        elif isinstance(value, _TensorPlaceholder):
            tensor = loaded_tensors.get(key)
            if tensor is not None and device:
                tensor = tensor.to(device)
            restored_state[key] = tensor
        else:
            restored_state[key] = value

    elapsed = time.monotonic() - t0
    log.info("Hydration complete (%.3fs)", elapsed)
    return restored_state


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def snapshot_exists(path: str | Path) -> bool:
    snap_dir = Path(path)
    return (snap_dir / "manifest.json").exists()


def snapshot_info(path: str | Path) -> dict[str, Any]:
    snap_dir = Path(path)
    with open(snap_dir / "manifest.json") as f:
        return json.load(f)
