"""zerostart — lazy imports that resolve as packages install in the background."""

from zerostart.lazy_imports import LazyImportHook, install_hook, remove_hook

__all__ = ["LazyImportHook", "install_hook", "remove_hook"]
