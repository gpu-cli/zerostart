"""Model cache: auto-snapshot + mmap hydrate for fast repeat loading.

Wraps snapshot.py with cache key derivation, index tracking, and LRU eviction.

Usage:
    cache = ModelCache("/volume/models")
    cache.save("qwen-7b", {"model": model, "tokenizer": tokenizer})
    state = cache.load("qwen-7b", device="cuda")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zerostart.snapshot import hydrate, snapshot

log = logging.getLogger("zerostart.model_cache")


@dataclass
class CacheEntry:
    key: str
    model_id: str
    dtype: str
    revision: str
    size_bytes: int
    created_at: float
    last_accessed: float
    tensor_count: int


def cache_key(model_id: str, kwargs: dict[str, Any] | None = None) -> str:
    """Deterministic key from model ID + relevant kwargs."""
    kwargs = kwargs or {}
    parts = {
        "model_id": model_id,
        "dtype": str(kwargs.get("torch_dtype", kwargs.get("dtype", "auto"))),
        "revision": kwargs.get("revision", "main"),
    }
    raw = json.dumps(parts, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _default_cache_dir() -> Path:
    """Auto-detect best cache location."""
    if env_dir := os.environ.get("ZEROSTART_MODEL_CACHE"):
        return Path(env_dir)
    # RunPod / Vast.ai persistent volume
    for volume in ("/volume", "/gpu-cli-workspaces"):
        if os.path.isdir(volume):
            return Path(volume) / "zs-models"
    return Path(os.path.expanduser("~/.cache/zerostart/models"))


class ModelCache:
    """Manages cached model snapshots on disk."""

    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else _default_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.cache_dir / "index.json"
        self._index = self._load_index()

    def has(self, key: str) -> bool:
        """Check if model snapshot exists and is valid."""
        snap_dir = self.cache_dir / key
        return (snap_dir / "manifest.json").exists()

    def save(self, key: str, state: dict[str, Any] | Any, **metadata: Any) -> Path:
        """Save model state for fast hydration.

        Accepts:
        - dict[str, Any] — state dict with models, tokenizers, tensors
        - nn.Module — single model (wrapped as {"model": module})
        - DiffusionPipeline — extracts all sub-models
        """
        state = self._normalize_state(state)

        snap_dir = self.cache_dir / key
        if snap_dir.exists():
            shutil.rmtree(snap_dir)

        t0 = time.monotonic()
        snapshot(state=state, path=snap_dir)
        elapsed = time.monotonic() - t0

        # Update index
        size_bytes = sum(f.stat().st_size for f in snap_dir.rglob("*") if f.is_file())
        now = time.time()
        self._index["entries"][key] = {
            "model_id": metadata.get("model_id", key),
            "dtype": metadata.get("dtype", "auto"),
            "revision": metadata.get("revision", "main"),
            "size_bytes": size_bytes,
            "created_at": now,
            "last_accessed": now,
            "tensor_count": metadata.get("tensor_count", 0),
        }
        self._save_index()

        log.info("Cached %s (%.2fs, %.1f MB)", key, elapsed, size_bytes / 1e6)
        return snap_dir

    def load(self, key: str, device: str = "cuda") -> dict[str, Any]:
        """Hydrate cached model. Returns state dict."""
        snap_dir = self.cache_dir / key
        if not (snap_dir / "manifest.json").exists():
            raise FileNotFoundError(f"No cached model for key: {key}")

        t0 = time.monotonic()
        result = hydrate(snap_dir, device=device)
        elapsed = time.monotonic() - t0

        self._touch_access(key)
        log.info("Loaded %s from cache (%.2fs, device=%s)", key, elapsed, device)
        return result

    def evict(self, key: str) -> None:
        """Remove cached model."""
        snap_dir = self.cache_dir / key
        if snap_dir.exists():
            shutil.rmtree(snap_dir)
        self._index["entries"].pop(key, None)
        self._save_index()
        log.info("Evicted %s", key)

    def list_entries(self) -> list[CacheEntry]:
        """List all cached models."""
        entries = []
        for key, data in self._index.get("entries", {}).items():
            if not self.has(key):
                continue
            entries.append(CacheEntry(
                key=key,
                model_id=data.get("model_id", key),
                dtype=data.get("dtype", "auto"),
                revision=data.get("revision", "main"),
                size_bytes=data.get("size_bytes", 0),
                created_at=data.get("created_at", 0),
                last_accessed=data.get("last_accessed", 0),
                tensor_count=data.get("tensor_count", 0),
            ))
        return entries

    def auto_evict(self, max_size_bytes: int) -> list[str]:
        """LRU eviction. Removes least recently accessed until under limit."""
        entries = self.list_entries()
        total = sum(e.size_bytes for e in entries)

        if total <= max_size_bytes:
            return []

        entries.sort(key=lambda e: e.last_accessed)

        evicted = []
        for entry in entries:
            if total <= max_size_bytes:
                break
            self.evict(entry.key)
            total -= entry.size_bytes
            evicted.append(entry.key)

        return evicted

    def safetensors_path_for(self, bin_path: str) -> Path | None:
        """Get cached safetensors conversion path for a .bin file."""
        key = hashlib.sha256(bin_path.encode()).hexdigest()[:16]
        sf_path = self.cache_dir / "conversions" / f"{key}.safetensors"
        if sf_path.exists():
            return sf_path
        return None

    def save_as_safetensors(self, bin_path: str, state_dict: dict[str, Any]) -> Path:
        """Convert and cache a .bin state dict as safetensors."""
        try:
            from safetensors.torch import save_file
        except ImportError as e:
            raise ImportError("safetensors required for .bin conversion") from e

        conv_dir = self.cache_dir / "conversions"
        conv_dir.mkdir(parents=True, exist_ok=True)

        key = hashlib.sha256(bin_path.encode()).hexdigest()[:16]
        sf_path = conv_dir / f"{key}.safetensors"
        save_file(state_dict, str(sf_path))
        log.info("Converted %s → %s", bin_path, sf_path)
        return sf_path

    def _normalize_state(self, state: Any) -> dict[str, Any]:
        """Normalize input to a dict suitable for snapshot()."""
        if isinstance(state, dict):
            return state

        # nn.Module
        if hasattr(state, "state_dict") and hasattr(state, "config"):
            return {"model": state}

        # DiffusionPipeline
        if hasattr(state, "components"):
            return {
                name: comp
                for name, comp in state.components.items()
                if comp is not None
            }

        return {"model": state}

    def _load_index(self) -> dict[str, Any]:
        if self._index_path.exists():
            with open(self._index_path) as f:
                return json.load(f)
        return {"version": 1, "entries": {}}

    def _save_index(self) -> None:
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    def _touch_access(self, key: str) -> None:
        if key in self._index.get("entries", {}):
            self._index["entries"][key]["last_accessed"] = time.time()
            self._save_index()
