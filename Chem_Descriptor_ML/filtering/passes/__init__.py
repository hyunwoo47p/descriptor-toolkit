"""
ChemDescriptorML (CDML) - Filtering Passes Module

Track 1: 5-Stage Descriptor Filtering
- Pass 0: SamplingPass - Large dataset sampling
- Pass 1: StatisticsAndVarianceFilter - Variance filtering
- Pass 2: SpearmanComputerGPU - Correlation clustering
- Pass 3: VIFFilteringPassGPUWithClusters - VIF filtering
- Pass 4: NonlinearDetectionPassGPU - HSIC/RDC analysis

Lazy imports to avoid dependency issues.
"""

__all__ = [
    "SamplingPass",
    "StatisticsAndVarianceFilter",
    "SpearmanComputerGPU",
    "VIFFilteringPassGPUWithClusters",
    "NonlinearDetectionPassGPU",
    "SpearmanClusteringPass",
    "DisjointSet",
    "GraphBuilder",
    "LeidenClustering",
    "SeedManager",
    "IterativeVIFFiltering",
]

# Lazy imports - only import when accessed
def __getattr__(name):
    if name == "SamplingPass":
        from Chem_Descriptor_ML.filtering.passes.pass0_sampling import SamplingPass
        return SamplingPass
    elif name == "StatisticsAndVarianceFilter":
        from Chem_Descriptor_ML.filtering.passes.pass1_statistics import StatisticsAndVarianceFilter
        return StatisticsAndVarianceFilter
    elif name == "SpearmanComputerGPU":
        from Chem_Descriptor_ML.filtering.passes.pass2_correlation import SpearmanComputerGPU
        return SpearmanComputerGPU
    elif name == "VIFFilteringPassGPUWithClusters":
        from Chem_Descriptor_ML.filtering.passes.pass3_vif import VIFFilteringPassGPUWithClusters
        return VIFFilteringPassGPUWithClusters
    elif name == "NonlinearDetectionPassGPU":
        from Chem_Descriptor_ML.filtering.passes.pass4_nonlinear import NonlinearDetectionPassGPU
        return NonlinearDetectionPassGPU
    elif name == "SpearmanClusteringPass":
        from Chem_Descriptor_ML.filtering.passes.spearman_clustering import SpearmanClusteringPass
        return SpearmanClusteringPass
    elif name == "DisjointSet":
        from Chem_Descriptor_ML.filtering.passes.spearman_clustering import DisjointSet
        return DisjointSet
    elif name == "GraphBuilder":
        from Chem_Descriptor_ML.filtering.passes.graph_builder import GraphBuilder
        return GraphBuilder
    elif name == "LeidenClustering":
        from Chem_Descriptor_ML.filtering.passes.graph_builder import LeidenClustering
        return LeidenClustering
    elif name == "SeedManager":
        from Chem_Descriptor_ML.filtering.passes.seed_manager import SeedManager
        return SeedManager
    elif name == "IterativeVIFFiltering":
        from Chem_Descriptor_ML.filtering.passes.vif_iterative import IterativeVIFFiltering
        return IterativeVIFFiltering
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
