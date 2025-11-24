"""Filtering module for descriptor pipeline - Lazy imports"""

__all__ = ["DescriptorPipeline"]

def __getattr__(name):
    if name == "DescriptorPipeline":
        from molecular_descriptor_toolkit.filtering.pipeline import DescriptorPipeline
        return DescriptorPipeline
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
