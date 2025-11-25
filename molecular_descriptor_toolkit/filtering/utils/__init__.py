"""Filtering utilities module"""

# Direct import for logging (no dependencies)
from molecular_descriptor_toolkit.filtering.utils.logging import log

__all__ = [
    "get_optimal_device",
    "log",
]

# Lazy import for gpu (requires torch)
def __getattr__(name):
    if name == "get_optimal_device":
        from molecular_descriptor_toolkit.filtering.utils.gpu import get_optimal_device
        return get_optimal_device
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
