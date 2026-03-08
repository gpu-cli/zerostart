"""zerostart — lazy imports that resolve as packages install in the background."""

from zerostart.lazy_imports import LazyImportHook, install_hook, remove_hook
from zerostart.sdk import on_restore, ready

__all__ = [
    "LazyImportHook",
    "install_hook",
    "remove_hook",
    "ready",
    "on_restore",
]
