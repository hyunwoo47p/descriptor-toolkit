"""
Descriptor Pipeline - Unified GPU-Accelerated Molecular Descriptor Filtering

Version: 1.0.0
"""

__version__ = "1.0.0"

from descriptor_pipeline.core.pipeline import DescriptorPipeline
from descriptor_pipeline.config.settings import PipelineConfig

__all__ = ["DescriptorPipeline", "PipelineConfig"]
