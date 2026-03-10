# 02: Model Cache (`model_cache.py`)

## Overview

Manages cached model snapshots on disk. Builds on existing `snapshot.py` but adds:
- Cache key derivation (model ID + dtype + revision → deterministic key)
- Index tracking (sizes, last access, metadata)
- LRU eviction
- Generic save (accepts nn.Module, pipeline, state dict)

## API

```python
class ModelCache:
    def __init__(self, cache_dir: str | Path | None = None):
        """Initialize cache. Auto-detects best location:
        1. Explicit cache_dir
        2. ZEROSTART_MODEL_CACHE env var
        3. /volume/zs-models (if /volume exists — RunPod/Vast)
        4. ~/.cache/zerostart/models
        """

    def has(self, key: str) -> bool:
        """Check if model snapshot exists and is valid."""

    def save(self, key: str, state: dict[str, Any] | nn.Module) -> Path:
        """Save model state for fast hydration.

        Accepts:
        - dict[str, Any] — arbitrary state dict (model, tokenizer, etc.)
        - nn.Module — single model (wrapped as {"model": module})
        - DiffusionPipeline — extracts all sub-models + scheduler
        """

    def load(self, key: str, device: str = "cuda") -> dict[str, Any]:
        """Hydrate cached model. Returns state dict."""

    def evict(self, key: str) -> None:
        """Remove cached model."""

    def list_entries(self) -> list[CacheEntry]:
        """List all cached models."""

    def auto_evict(self, max_size_bytes: int) -> list[str]:
        """LRU eviction to stay under size limit. Returns evicted keys."""

    def safetensors_path_for(self, bin_path: str) -> Path | None:
        """Get cached safetensors conversion path for a .bin file."""

    def save_as_safetensors(self, bin_path: str, state_dict: dict) -> Path:
        """Convert and cache a .bin state dict as safetensors."""
```

## Cache Key Derivation

```python
def cache_key(model_id: str, kwargs: dict[str, Any]) -> str:
    """Deterministic key from model ID + relevant kwargs."""
    parts = {
        "model_id": model_id,
        "dtype": str(kwargs.get("torch_dtype", kwargs.get("dtype", "auto"))),
        "revision": kwargs.get("revision", "main"),
    }
    raw = json.dumps(parts, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
```

## Disk Layout

```
{cache_dir}/
  index.json                    # Global index
  {cache_key}/
    manifest.json               # Snapshot manifest (from snapshot.py)
    tokenizer_{name}/           # Tokenizer save_pretrained output
    tensors/                    # Fallback tensors not in safetensors
  conversions/
    {sha256_of_bin_path}.safetensors  # .bin → safetensors conversions
```

## Index Format

```json
{
  "version": 1,
  "entries": {
    "a1b2c3d4e5f6g7h8": {
      "model_id": "Qwen/Qwen2.5-7B",
      "dtype": "torch.bfloat16",
      "revision": "main",
      "size_bytes": 15200000000,
      "created_at": 1710000000.0,
      "last_accessed": 1710100000.0,
      "tensor_count": 339,
      "snapshot_version": 2
    }
  }
}
```

## Changes to `snapshot.py`

The existing `snapshot()` and `hydrate()` functions stay as-is. `ModelCache` wraps them with caching logic:

```python
def save(self, key: str, state: dict[str, Any] | Any) -> Path:
    # Normalize input
    if hasattr(state, "state_dict") and hasattr(state, "config"):
        # nn.Module
        state = {"model": state}
    elif hasattr(state, "components"):
        # DiffusionPipeline — extract sub-models
        state = {name: comp for name, comp in state.components.items() if comp is not None}

    snap_dir = self.cache_dir / key
    snapshot(state=state, path=snap_dir)
    self._update_index(key, state)
    return snap_dir

def load(self, key: str, device: str = "cuda") -> dict[str, Any]:
    snap_dir = self.cache_dir / key
    result = hydrate(snap_dir, device=device)
    self._touch_access(key)
    return result
```

## Eviction

```python
def auto_evict(self, max_size_bytes: int) -> list[str]:
    """LRU eviction. Removes least recently accessed entries until under limit."""
    entries = self.list_entries()
    total = sum(e.size_bytes for e in entries)

    if total <= max_size_bytes:
        return []

    # Sort by last_accessed ascending (oldest first)
    entries.sort(key=lambda e: e.last_accessed)

    evicted = []
    for entry in entries:
        if total <= max_size_bytes:
            break
        self.evict(entry.key)
        total -= entry.size_bytes
        evicted.append(entry.key)

    return evicted
```
