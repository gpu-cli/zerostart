"""Snapshot & hydrate: fast checkpoint/restore for GPU Python applications.

Captures a running Python+model state and restores it in ~1-2s by:
1. Serializing Python objects (config, tokenizer, etc.) via cloudpickle (~1-10MB)
2. Recording tensor references into safetensors files (not the data itself)
3. On hydrate: mmap safetensors files (zero-copy) and wire tensors back into the model

Works with any model that uses safetensors for weights (transformers, diffusers, etc.).
Integrates with safetensors-streaming for zero-copy mmap loading.

Usage:
    # After model is loaded and ready to serve:
    from zerostart.snapshot import snapshot, hydrate

    snapshot(
        state={"model": model, "tokenizer": tokenizer, "config": config},
        path="/cache/my-model.zsnap",
    )

    # On next start (~1-2s instead of 30s+):
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
import sys
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("zerostart.snapshot")


# ---------------------------------------------------------------------------
# Tensor reference extraction
# ---------------------------------------------------------------------------

def _get_torch():
    """Import torch lazily."""
    import torch
    return torch


def _extract_tensor_refs(
    state: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Separate tensors from Python objects.

    Walks the state dict and replaces torch.Tensor values with placeholder
    references. Returns (cleaned_state, tensor_map).

    tensor_map: {dotted_key: {shape, dtype, device, ...}}
    """
    torch = _get_torch()
    tensor_map: dict[str, dict[str, Any]] = {}
    cleaned = {}

    for key, value in state.items():
        if isinstance(value, torch.nn.Module):
            # Extract state_dict tensors, keep the module shell
            module_tensors = {}
            sd = value.state_dict()
            for param_name, tensor in sd.items():
                ref_key = f"{key}.{param_name}"
                module_tensors[param_name] = {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "device": str(tensor.device),
                    "numel": tensor.numel(),
                    "nbytes": tensor.nelement() * tensor.element_size(),
                }
                tensor_map[ref_key] = module_tensors[param_name]

            # Store the module with empty parameter placeholders
            cleaned[key] = _ModulePlaceholder(
                module_class=type(value),
                config=_extract_model_config(value),
                tensor_keys=list(module_tensors.keys()),
            )
        elif isinstance(value, torch.Tensor):
            tensor_map[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "numel": value.numel(),
                "nbytes": value.nelement() * value.element_size(),
            }
            cleaned[key] = _TensorPlaceholder(key)
        else:
            # Non-tensor values get serialized directly
            cleaned[key] = value

    return cleaned, tensor_map


def _extract_model_config(module: Any) -> dict[str, Any] | None:
    """Try to extract a model's config for reconstruction."""
    # transformers models have .config
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


class _ModulePlaceholder:
    """Placeholder for a nn.Module — stores config + param keys, not weights."""
    def __init__(self, module_class: type, config: dict[str, Any] | None, tensor_keys: list[str]):
        self.module_class = module_class
        self.config = config
        self.tensor_keys = tensor_keys


class _TensorPlaceholder:
    """Placeholder for a standalone tensor."""
    def __init__(self, key: str):
        self.key = key


# ---------------------------------------------------------------------------
# Safetensors file discovery
# ---------------------------------------------------------------------------

def _find_safetensors_for_model(module: Any) -> list[Path]:
    """Find the safetensors files that a model's weights came from.

    Checks:
    1. model.config._name_or_path → HF cache directory
    2. Explicit safetensors_files in snapshot() call
    """
    paths: list[Path] = []

    if hasattr(module, "config") and hasattr(module.config, "_name_or_path"):
        model_path = Path(module.config._name_or_path)
        if model_path.is_dir():
            # Local directory — find safetensors files
            paths.extend(sorted(model_path.glob("*.safetensors")))
        else:
            # HF model ID — check the HF cache
            hf_cache = _find_hf_cache_dir(module.config._name_or_path)
            if hf_cache:
                paths.extend(sorted(hf_cache.glob("*.safetensors")))

    return paths


def _find_hf_cache_dir(model_id: str) -> Path | None:
    """Find the HF hub cache directory for a model."""
    hf_home = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    hub_dir = hf_home / "hub"

    # HF cache structure: hub/models--org--name/snapshots/<hash>/
    safe_id = model_id.replace("/", "--")
    model_dir = hub_dir / f"models--{safe_id}"

    if not model_dir.is_dir():
        return None

    # Find the latest snapshot
    snapshots = model_dir / "snapshots"
    if not snapshots.is_dir():
        return None

    # Get the most recent snapshot directory
    snap_dirs = sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    return snap_dirs[0] if snap_dirs else None


def _build_tensor_to_file_map(
    safetensors_files: list[Path],
) -> dict[str, tuple[Path, str]]:
    """Map tensor names → (file_path, tensor_name_in_file).

    Reads safetensors headers (JSON only, no tensor data) to build
    a complete mapping of which tensor lives in which file.
    """
    import struct

    tensor_to_file: dict[str, tuple[Path, str]] = {}

    for sf_path in safetensors_files:
        try:
            with open(sf_path, "rb") as f:
                # safetensors format: 8 bytes (u64 LE) = header size, then JSON header
                header_size_bytes = f.read(8)
                if len(header_size_bytes) < 8:
                    continue
                header_size = struct.unpack("<Q", header_size_bytes)[0]
                header_json = f.read(header_size)
                header = json.loads(header_json)

                for tensor_name, info in header.items():
                    if tensor_name == "__metadata__":
                        continue
                    tensor_to_file[tensor_name] = (sf_path, tensor_name)
        except Exception as e:
            log.warning("Failed to read safetensors header from %s: %s", sf_path, e)

    return tensor_to_file


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------

def _environment_fingerprint() -> str:
    """Hash of environment to detect incompatible restores."""
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


# ---------------------------------------------------------------------------
# Snapshot: capture state
# ---------------------------------------------------------------------------

def snapshot(
    state: dict[str, Any],
    path: str | Path,
    safetensors_files: list[str | Path] | None = None,
) -> Path:
    """Snapshot a Python+model state for fast hydration.

    Args:
        state: Dict of named objects to snapshot. Values can be:
            - torch.nn.Module (model) — tensors extracted, config preserved
            - torch.Tensor — replaced with reference
            - anything else — serialized via cloudpickle
        path: Directory to write the snapshot to (created if needed).
        safetensors_files: Explicit safetensors file paths. If None, auto-detected
            from model configs.

    Returns:
        Path to the snapshot directory.
    """
    import cloudpickle

    t0 = time.monotonic()
    snap_dir = Path(path)
    snap_dir.mkdir(parents=True, exist_ok=True)

    # 1. Find safetensors files
    sf_files: list[Path] = []
    if safetensors_files:
        sf_files = [Path(f) for f in safetensors_files]
    else:
        # Auto-detect from model configs
        for value in state.values():
            torch = _get_torch()
            if isinstance(value, torch.nn.Module):
                found = _find_safetensors_for_model(value)
                sf_files.extend(found)

    sf_files = list(dict.fromkeys(sf_files))  # dedupe preserving order
    log.info("Found %d safetensors files", len(sf_files))

    # 2. Build tensor→file mapping from safetensors headers
    tensor_to_file = _build_tensor_to_file_map(sf_files)
    log.info("Mapped %d tensors across safetensors files", len(tensor_to_file))

    # 3. Separate tensors from Python state
    cleaned_state, tensor_map = _extract_tensor_refs(state)

    # 4. Match model parameters to safetensors file locations
    tensor_file_refs: dict[str, dict[str, Any]] = {}
    unmatched: list[str] = []

    for ref_key, ref_info in tensor_map.items():
        # ref_key is like "model.transformer.h.0.attn.bias"
        # safetensors key might be any suffix: "transformer.h.0.attn.bias",
        # "h.0.attn.bias", or the full key. Try progressively stripping
        # dot-separated prefixes until we find a match.

        candidates = [ref_key]
        remaining = ref_key
        while "." in remaining:
            remaining = remaining.split(".", 1)[1]
            candidates.append(remaining)

        matched = False
        for candidate in candidates:
            if candidate in tensor_to_file:
                sf_path, sf_tensor_name = tensor_to_file[candidate]
                tensor_file_refs[ref_key] = {
                    **ref_info,
                    "safetensors_file": str(sf_path),
                    "safetensors_tensor": sf_tensor_name,
                }
                matched = True
                break

        if not matched:
            unmatched.append(ref_key)
            # Store tensor data directly for unmatched tensors
            tensor_file_refs[ref_key] = ref_info

    if unmatched:
        log.warning(
            "%d tensors not matched to safetensors files — will be serialized directly",
            len(unmatched),
        )

    # 5. Serialize unmatched tensors (those not in any safetensors file)
    torch = _get_torch()
    unmatched_tensors: dict[str, bytes] = {}
    for ref_key in unmatched:
        # Find the actual tensor in the original state
        parts = ref_key.split(".", 1)
        top_key = parts[0]
        param_name = parts[1] if len(parts) > 1 else None

        obj = state.get(top_key)
        if obj is None:
            continue

        if isinstance(obj, torch.nn.Module) and param_name:
            sd = obj.state_dict()
            if param_name in sd:
                import io
                buf = io.BytesIO()
                torch.save(sd[param_name], buf)
                unmatched_tensors[ref_key] = buf.getvalue()
        elif isinstance(obj, torch.Tensor):
            import io
            buf = io.BytesIO()
            torch.save(obj, buf)
            unmatched_tensors[ref_key] = buf.getvalue()

    # 6. Serialize Python state (without tensors)
    python_state_bytes = cloudpickle.dumps(cleaned_state)
    log.info("Python state: %.1f KB", len(python_state_bytes) / 1024)

    # 7. Write snapshot files
    # Manifest
    manifest = {
        "version": 1,
        "created": time.time(),
        "fingerprint": _environment_fingerprint(),
        "tensor_count": len(tensor_map),
        "matched_tensors": len(tensor_map) - len(unmatched),
        "unmatched_tensors": len(unmatched),
        "safetensors_files": [str(f) for f in sf_files],
        "tensor_refs": tensor_file_refs,
        "state_keys": list(state.keys()),
    }

    with open(snap_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    with open(snap_dir / "python_state.pkl", "wb") as f:
        f.write(python_state_bytes)

    # Save unmatched tensor data
    if unmatched_tensors:
        tensors_dir = snap_dir / "tensors"
        tensors_dir.mkdir(exist_ok=True)
        for ref_key, data in unmatched_tensors.items():
            safe_name = ref_key.replace("/", "_").replace(".", "_") + ".pt"
            with open(tensors_dir / safe_name, "wb") as f:
                f.write(data)

    elapsed = time.monotonic() - t0
    total_params = sum(r.get("numel", 0) for r in tensor_map.values())
    log.info(
        "Snapshot saved to %s (%.1fs, %d tensors, %.1fM params, %.1f KB Python state)",
        snap_dir, elapsed, len(tensor_map), total_params / 1e6,
        len(python_state_bytes) / 1024,
    )

    return snap_dir


# ---------------------------------------------------------------------------
# Hydrate: restore state
# ---------------------------------------------------------------------------

def hydrate(
    path: str | Path,
    device: str | None = None,
    verify_fingerprint: bool = True,
) -> dict[str, Any]:
    """Hydrate a snapshot — restore model state with mmap'd safetensors weights.

    Args:
        path: Path to the snapshot directory.
        device: Target device for tensors (e.g., "cuda:0"). If None, uses
            the device recorded in the snapshot.
        verify_fingerprint: Check that the environment matches the snapshot.

    Returns:
        Dict of restored objects (same keys as the original snapshot() call).
    """
    import cloudpickle

    t0 = time.monotonic()
    snap_dir = Path(path)

    # 1. Load manifest
    with open(snap_dir / "manifest.json") as f:
        manifest = json.load(f)

    if verify_fingerprint:
        current_fp = _environment_fingerprint()
        snap_fp = manifest.get("fingerprint", "")
        if current_fp != snap_fp:
            log.warning(
                "Environment fingerprint mismatch (snapshot=%s, current=%s). "
                "Tensors may be incompatible.",
                snap_fp, current_fp,
            )

    # 2. Load Python state
    with open(snap_dir / "python_state.pkl", "rb") as f:
        cleaned_state = cloudpickle.loads(f.read())

    t_python = time.monotonic() - t0
    log.info("Python state loaded (%.3fs)", t_python)

    # 3. Load tensor data via mmap
    torch = _get_torch()
    tensor_refs = manifest.get("tensor_refs", {})

    # Group tensors by safetensors file for efficient loading
    file_to_tensors: dict[str, list[tuple[str, str]]] = {}
    standalone_tensors: list[str] = []

    for ref_key, ref_info in tensor_refs.items():
        sf_file = ref_info.get("safetensors_file")
        sf_tensor = ref_info.get("safetensors_tensor")
        if sf_file and sf_tensor:
            file_to_tensors.setdefault(sf_file, []).append((ref_key, sf_tensor))
        else:
            standalone_tensors.append(ref_key)

    # Load tensors from safetensors files (mmap, zero-copy)
    loaded_tensors: dict[str, Any] = {}
    t_mmap_start = time.monotonic()

    for sf_file, tensor_pairs in file_to_tensors.items():
        sf_path = Path(sf_file)
        if not sf_path.exists():
            log.warning("Safetensors file not found: %s", sf_file)
            continue

        # Try safetensors-streaming first (zero-copy Rust mmap)
        try:
            import safetensors_streaming
            handle = safetensors_streaming.safe_open(str(sf_path), framework="pt", device="cpu")
            for ref_key, sf_tensor_name in tensor_pairs:
                try:
                    tensor = handle.get_tensor(sf_tensor_name)
                    loaded_tensors[ref_key] = tensor
                except Exception as e:
                    log.warning("Failed to load tensor %s from %s: %s", sf_tensor_name, sf_file, e)
            continue
        except ImportError:
            pass

        # Fall back to standard safetensors
        try:
            from safetensors.torch import load_file
            all_tensors = load_file(str(sf_path), device="cpu")
            for ref_key, sf_tensor_name in tensor_pairs:
                if sf_tensor_name in all_tensors:
                    loaded_tensors[ref_key] = all_tensors[sf_tensor_name]
                else:
                    log.warning("Tensor %s not found in %s", sf_tensor_name, sf_file)
            continue
        except ImportError:
            pass

        # Last resort: torch.load from the serialized tensor files
        log.warning("No safetensors loader available for %s", sf_file)

    # Load standalone (unmatched) tensors from serialized files
    tensors_dir = snap_dir / "tensors"
    for ref_key in standalone_tensors:
        safe_name = ref_key.replace("/", "_").replace(".", "_") + ".pt"
        pt_path = tensors_dir / safe_name
        if pt_path.exists():
            loaded_tensors[ref_key] = torch.load(pt_path, map_location="cpu", weights_only=True)

    t_mmap = time.monotonic() - t_mmap_start
    log.info("Tensors loaded via mmap (%.3fs, %d tensors)", t_mmap, len(loaded_tensors))

    # 4. Reconstruct state: wire tensors back into Python objects
    restored_state: dict[str, Any] = {}

    for key, value in cleaned_state.items():
        if isinstance(value, _ModulePlaceholder):
            module = _reconstruct_module(value, key, loaded_tensors, device)
            restored_state[key] = module
        elif isinstance(value, _TensorPlaceholder):
            tensor = loaded_tensors.get(key)
            if tensor is not None:
                if device:
                    tensor = tensor.to(device)
                restored_state[key] = tensor
            else:
                log.warning("Tensor %s not found in snapshot", key)
                restored_state[key] = None
        else:
            restored_state[key] = value

    elapsed = time.monotonic() - t0
    log.info(
        "Hydration complete (%.3fs total: %.3fs python + %.3fs tensors)",
        elapsed, t_python, t_mmap,
    )

    return restored_state


def _reconstruct_module(
    placeholder: _ModulePlaceholder,
    state_key: str,
    loaded_tensors: dict[str, Any],
    device: str | None,
) -> Any:
    """Reconstruct an nn.Module from its placeholder + tensor data."""
    torch = _get_torch()

    config = placeholder.config
    module = None

    # Try to reconstruct from config (transformers-style)
    if config and config.get("_type") == "transformers":
        try:
            import importlib
            model_module = importlib.import_module(config["_module"])
            model_class = getattr(model_module, config["_class"])

            config_module = importlib.import_module(config["config_module"])
            config_class = getattr(config_module, config["config_class"])

            model_config = config_class.from_dict(config["config_dict"])

            # Normal init handles tied weights, attention masks, buffers.
            # Random weight init is ~0.1s, dwarfed by import time.
            module = model_class(model_config)

            log.info("Reconstructed %s from config", config["_class"])
        except Exception as e:
            log.warning("Failed to reconstruct from config: %s", e)

    # Fallback: try cloudpickle'd module class with basic init
    if module is None:
        try:
            module = placeholder.module_class.__new__(placeholder.module_class)
            if hasattr(module, "__init__"):
                torch.nn.Module.__init__(module)
        except Exception as e:
            log.warning("Failed to create empty module: %s", e)
            return None

    # Load state dict from mmap'd tensors
    state_dict = {}
    for param_name in placeholder.tensor_keys:
        ref_key = f"{state_key}.{param_name}"
        tensor = loaded_tensors.get(ref_key)
        if tensor is not None:
            state_dict[param_name] = tensor

    if state_dict:
        try:
            module.load_state_dict(state_dict, strict=False, assign=True)
        except TypeError:
            # older torch doesn't have assign=True
            module.load_state_dict(state_dict, strict=False)

        # Re-tie weights (e.g., lm_head.weight = wte.weight in GPT-2)
        if hasattr(module, "tie_weights"):
            module.tie_weights()

    # Move to device if requested
    if device:
        module = module.to(device)

    return module


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def snapshot_exists(path: str | Path) -> bool:
    """Check if a valid snapshot exists at the given path."""
    snap_dir = Path(path)
    return (snap_dir / "manifest.json").exists() and (snap_dir / "python_state.pkl").exists()


def snapshot_info(path: str | Path) -> dict[str, Any]:
    """Get metadata about a snapshot without loading it."""
    snap_dir = Path(path)
    with open(snap_dir / "manifest.json") as f:
        return json.load(f)
