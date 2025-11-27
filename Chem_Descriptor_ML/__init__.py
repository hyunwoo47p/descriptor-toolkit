"""
Molecular Descriptor Toolkit (MDT)
===================================

A comprehensive toolkit for molecular descriptor calculation and filtering
with GPU acceleration.

Workflow:
1. Preprocessing: XML → Parquet, Schema Generation, Descriptor Calculation
2. Filtering: Multi-pass descriptor filtering pipeline (Pass 0-4)

Features:
- GPU-accelerated by default (automatic CPU fallback)
- Checkpoint/resume support
- Memory-optimized processing
- Scalable to billion-compound datasets
"""

__version__ = "1.0.0"
__author__ = "KAERI_UES"

from Chem_Descriptor_ML.config.settings import Config

__all__ = ["Config"]
