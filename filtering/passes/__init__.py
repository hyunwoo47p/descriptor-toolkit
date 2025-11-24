"""Filtering passes module - Lazy imports to avoid dependency issues"""

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
        from molecular_descriptor_toolkit.filtering.passes.pass0_sampling import SamplingPass
        return SamplingPass
    elif name == "StatisticsAndVarianceFilter":
        from molecular_descriptor_toolkit.filtering.passes.pass1_statistics import StatisticsAndVarianceFilter
        return StatisticsAndVarianceFilter
    elif name == "SpearmanComputerGPU":
        from molecular_descriptor_toolkit.filtering.passes.pass2_correlation import SpearmanComputerGPU
        return SpearmanComputerGPU
    elif name == "VIFFilteringPassGPUWithClusters":
        from molecular_descriptor_toolkit.filtering.passes.pass3_vif import VIFFilteringPassGPUWithClusters
        return VIFFilteringPassGPUWithClusters
    elif name == "NonlinearDetectionPassGPU":
        from molecular_descriptor_toolkit.filtering.passes.pass4_nonlinear import NonlinearDetectionPassGPU
        return NonlinearDetectionPassGPU
    elif name == "SpearmanClusteringPass":
        from molecular_descriptor_toolkit.filtering.passes.spearman_clustering import SpearmanClusteringPass
        return SpearmanClusteringPass
    elif name == "DisjointSet":
        from molecular_descriptor_toolkit.filtering.passes.spearman_clustering import DisjointSet
        return DisjointSet
    elif name == "GraphBuilder":
        from molecular_descriptor_toolkit.filtering.passes.graph_builder import GraphBuilder
        return GraphBuilder
    elif name == "LeidenClustering":
        from molecular_descriptor_toolkit.filtering.passes.graph_builder import LeidenClustering
        return LeidenClustering
    elif name == "SeedManager":
        from molecular_descriptor_toolkit.filtering.passes.seed_manager import SeedManager
        return SeedManager
    elif name == "IterativeVIFFiltering":
        from molecular_descriptor_toolkit.filtering.passes.vif_iterative import IterativeVIFFiltering
        return IterativeVIFFiltering
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
