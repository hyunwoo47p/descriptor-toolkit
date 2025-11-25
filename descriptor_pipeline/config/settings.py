"""
Pipeline Configuration - Unified settings for descriptor filtering pipeline
GPU mode focused - optimized for cluster_gpu_parallel workflow
"""
from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum


class RangeMode(str, Enum):
    """범위 계산 모드"""
    MINMAX = "minmax"
    TRIMMED = "trimmed"
    IQR = "iqr"


@dataclass
class PipelineConfig:
    """Unified descriptor filtering pipeline configuration (GPU-focused)"""
    
    # ====================================================================
    # Input/Output
    # ====================================================================
    parquet_glob: str
    output_dir: str = "pipeline_output"
    descriptor_columns: Optional[List[str]] = None
    
    # Metadata handling
    n_metadata_cols: int = 6
    metadata_columns: Optional[List[str]] = None
    
    # ====================================================================
    # GPU Settings
    # ====================================================================
    prefer_gpu: bool = True
    gpu_id: int = 0
    gpu_batch_size: int = 5000
    max_gpu_memory_gb: float = 40.0  # Maximum GPU memory to use (GB)
    
    # ====================================================================
    # Pass 0: Sampling
    # ====================================================================
    sample_per_file: Optional[int] = None
    file_independent_sampling: bool = False
    random_seed: int = 42
    
    # ====================================================================
    # Pass 1: Statistics & Variance Filtering
    # ====================================================================
    force_recompute: bool = False
    variance_threshold: float = 0.002
    use_robust_variance: bool = True
    range_mode: RangeMode = RangeMode.TRIMMED
    trim_percentiles: Tuple[float, float] = (2.5, 97.5)
    max_missing_ratio: float = 0.5
    min_effective_n: int = 100
    
    # Clipping
    use_percentile_clip: bool = True
    percentile_low: float = 2.5
    percentile_high: float = 97.5
    
    # Binary detection
    binary_skewness_threshold: float = 0.005
    remove_binary: bool = True
    
    # ====================================================================
    # Pass 2: Spearman Correlation
    # ====================================================================
    spearman_threshold: float = 0.95
    m: int = 64  # CountSketch buckets
    r: int = 8   # CountSketch repetitions
    spearman_resolution: int = 2  # Legacy
    
    # ====================================================================
    # Pass 3: VIF Multicollinearity
    # ====================================================================
    vif_threshold: float = 10.0
    vif_batch_size: int = 100
    
    # ====================================================================
    # Pass 4: HSIC + RDC Nonlinear
    # ====================================================================
    nonlinear_threshold: float = 0.3
    w_hsic: float = 0.5
    w_rdc: float = 0.5
    hsic_D: int = 50  # HSIC random features dimension
    rdc_d: int = 20   # RDC random projection dimension
    rdc_seeds: int = 3  # Number of RDC random seeds
    
    # Additional HSIC/RDC parameters
    hsic_block_size: int = 100
    hsic_memory_limit_gb: float = 4.0
    rdc_fisher_weight: float = 0.7
    rdc_max_weight: float = 0.3
    rdc_cdf_resolution: int = 1000
    
    # Legacy parameters (for backward compatibility)
    hsic_kernel_bandwidth: float = 1.0
    hsic_use_gamma: bool = True
    rdc_k: int = 10
    rdc_s: float = 1.0 / 6.0
    nonlinear_resolution: float = 2.0
    
    # ====================================================================
    # Graph & Clustering
    # ====================================================================
    topk: int = 40
    weights: Tuple[float, float] = (0.4, 0.6)  # (HSIC, RDC) for CPU path
    leiden_resolution: float = 1.0
    n_consensus: int = 10
    consensus_memory_limit_p: int = 10000
    
    # ====================================================================
    # Processing Options
    # ====================================================================
    batch_rows: int = 10000
    checkpoint: bool = True
    verbose: bool = True
    
    # ====================================================================
    # Numerical Stability
    # ====================================================================
    var_eps: float = 1e-12
    range_eps: float = 1e-12
    mean_eps: float = 1e-12
    iqr_multiplier: float = 7.0
    hard_cap: float = 1e9
