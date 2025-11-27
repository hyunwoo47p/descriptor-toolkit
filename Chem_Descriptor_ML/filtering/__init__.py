"""
Filtering module for ChemDescriptorML (CDML) - Lazy imports

Track 1: 5-Stage Descriptor Filtering Pipeline
- Pass 0: Sampling (for large datasets)
- Pass 1: Variance Filtering (remove low-variance descriptors)
- Pass 2: Spearman Correlation Clustering
- Pass 3: VIF (Variance Inflation Factor) Filtering
- Pass 4: Nonlinear Analysis (HSIC + RDC)
"""

__all__ = ["DescriptorPipeline"]

def __getattr__(name):
    if name == "DescriptorPipeline":
        from Chem_Descriptor_ML.filtering.pipeline import DescriptorPipeline
        return DescriptorPipeline
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
