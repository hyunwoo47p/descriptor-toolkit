"""
Preprocessing module for ChemDescriptorML (CDML)

Data preprocessing utilities:
- XML Parser: PubChem XML → Parquet conversion
- Schema Generator: Auto-detect descriptor schema from data
- Descriptor Calculator: RDKit + Mordred descriptor calculation
"""

from Chem_Descriptor_ML.preprocessing.pipeline import (
    PreprocessingPipeline,
    run_preprocessing
)

__all__ = [
    'PreprocessingPipeline',
    'run_preprocessing',
]
