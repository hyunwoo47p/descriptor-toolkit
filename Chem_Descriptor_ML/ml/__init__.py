"""
ML Training Module for ChemDescriptorML (CDML)

Track 2: ML Model Training Pipeline
- 8 Regression Models (RandomForest, XGBoost, LightGBM, Ridge, Lasso, ElasticNet, ExtraTrees, GPR)
- K-Fold Cross-Validation + Hold-Out Evaluation
- Cluster-aware Descriptor Selection (sequential, representative, random_alternative, mixed)
- Extreme Value Prediction Analysis
"""

from .ensemble import OptimalMLEnsemble

__all__ = ['OptimalMLEnsemble']
