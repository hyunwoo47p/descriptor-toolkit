"""
ChemDescriptorML (CDML)
========================

GPU-accelerated molecular descriptor calculation, filtering, and ML training toolkit.

Tracks:
1. Track 1 - Descriptor Extraction & Filtering:
   - Preprocessing: XML → Parquet, Schema Generation, Descriptor Calculation
   - Filtering: 5-Stage Pipeline (Variance → Spearman → VIF → HSIC/RDC → Final)

2. Track 2 - ML Model Training:
   - 8 Regression Models: RandomForest, XGBoost, LightGBM, Ridge, Lasso, etc.
   - K-Fold Cross-Validation + Hold-Out Evaluation
   - Cluster-aware Descriptor Selection

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
