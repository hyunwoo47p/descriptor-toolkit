"""ChemDescriptorML (CDML) - Filtering Utilities Module"""

# Direct import for logging (no dependencies)
from Chem_Descriptor_ML.filtering.utils.logging import log

__all__ = [
    "get_optimal_device",
    "log",
]

# Lazy import for gpu (requires torch)
def __getattr__(name):
    if name == "get_optimal_device":
        from Chem_Descriptor_ML.filtering.utils.gpu import get_optimal_device
        return get_optimal_device
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
