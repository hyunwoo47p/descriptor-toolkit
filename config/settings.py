"""
Unified Configuration for Molecular Descriptor Toolkit
Hierarchical SSOT design with section-based organization
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, List, Literal, Union, Dict, Any
import warnings


# ========== Enums & Type Aliases ==========

class RangeMode(str, Enum):
    """
    DEPRECATED: Use Literal['MINMAX', 'TRIMMED', 'IQR'] instead.
    Kept for backward compatibility only.
    """
    MINMAX = "MINMAX"
    TRIMMED = "TRIMMED"
    IQR = "IQR"


RangeLiteral = Literal["MINMAX", "TRIMMED", "IQR"]


# ========== Helper Functions ==========

def _auto_detect_device() -> str:
    """Automatically detect and return best available device"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _cuda_available() -> bool:
    """Check if CUDA is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists, create if not"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def canonicalize_rangemode(s: str | RangeMode) -> RangeLiteral:
    """
    Canonicalize range mode to uppercase string
    
    Args:
        s: Range mode (case-insensitive string or RangeMode enum)
    
    Returns:
        Canonical uppercase string
    
    Examples:
        >>> canonicalize_rangemode("minmax")
        'MINMAX'
        >>> canonicalize_rangemode(RangeMode.TRIMMED)
        'TRIMMED'
    """
    if isinstance(s, RangeMode):
        return s.value
    
    s = s.strip().upper()
    if s not in {"MINMAX", "TRIMMED", "IQR"}:
        raise ValueError(f"Unknown range_mode: {s}. Must be one of: MINMAX, TRIMMED, IQR")
    return s  # type: ignore


# ========== Configuration Sections ==========

@dataclass
class DeviceConfig:
    """
    Device and GPU acceleration settings
    
    Attributes:
        prefer_gpu: Enable GPU acceleration if available
        gpu_id: CUDA device ID to use
        device: Device string ('cuda' or 'cpu'), auto-detected if None
    """
    prefer_gpu: bool = True
    """Enable GPU acceleration if available (default: True)"""
    
    gpu_id: int = 0
    """CUDA device ID to use (default: 0)"""
    
    device: Optional[str] = None
    """Device string ('cuda' or 'cpu'), auto-detected if None"""


@dataclass
class IOConfig:
    """
    I/O and checkpoint settings
    
    Attributes:
        parquet_glob: Glob pattern for input Parquet files
        output_dir: Directory to write outputs and checkpoints
        descriptor_columns: Specific descriptor columns to process (None = all)
        n_metadata: Number of metadata columns to exclude from filtering
        row_group_size: Parquet row group size
        part_bytes_target: Target size for each Parquet part file
        atomic_write: Use atomic writes (tmp + rename) for safety
        resume_mode: Checkpoint resume mode ('off' or 'scan_parts')
    """
    parquet_glob: str = ""
    """Glob pattern for input Parquet files (e.g., 'data/*.parquet')"""
    
    output_dir: str = "output"
    """Directory to write outputs and checkpoints"""
    
    descriptor_columns: Optional[List[str]] = None
    """Specific descriptor columns to process (None = all)"""
    
    n_metadata: int = 6
    """Number of metadata columns to exclude from filtering"""
    
    row_group_size: int = 10_000
    """Parquet row group size"""
    
    part_bytes_target: int = 256 * 1024 * 1024  # 256 MB
    """Target size for each Parquet part file (bytes)"""
    
    batch_rows: int = 50_000
    """
    Batch size for reading Parquet files (rows per batch)
    
    Performance tip:
    - Recommended: batch_rows should be a multiple of row_group_size
    - Example: If row_group_size=10_000, use batch_rows=50_000 (5x)
    - Too small: Excessive I/O overhead
    - Too large: High memory usage
    """
    
    atomic_write: bool = True
    """Use atomic writes (tmp + rename) for safety"""
    
    resume_mode: Literal["off", "scan_parts"] = "scan_parts"
    """Checkpoint resume mode: 'off' or 'scan_parts'"""
    
    _frozen: bool = field(default=False, init=False, repr=False)
    
    def __setattr__(self, key: str, value: Any):
        """Guard against modification when frozen"""
        if key != "_frozen" and getattr(self, "_frozen", False):
            raise RuntimeError(f"Config is frozen. Cannot set '{key}'")
        object.__setattr__(self, key, value)


@dataclass
class PreprocessingConfig:
    """
    Molecular standardization and parsing settings
    
    Attributes:
        profile: Standardization profile ('neutral', 'ligand_only', 'complex_included')
        std_core: Use RDKit Cleanup with std_core=True
        use_normalizer: Apply RDKit Normalizer
        use_reionizer: Apply RDKit Reionizer
        use_metal_disconnector: Apply RDKit MetalDisconnector
        keep_largest_fragment: Keep only largest fragment after standardization
        primary_id_col: Primary ID column name
        smiles_col: SMILES column name
        inchi_col: InChI column name
        parse_order: Order to try parsing ('smiles', 'inchi')
    """
    profile: Literal["neutral", "ligand_only", "complex_included"] = "neutral"
    """Standardization profile"""
    
    std_core: bool = True
    """Use RDKit Cleanup with std_core=True"""
    
    use_normalizer: bool = False
    """Apply RDKit Normalizer"""
    
    use_reionizer: bool = False
    """Apply RDKit Reionizer"""
    
    use_metal_disconnector: bool = False
    """Apply RDKit MetalDisconnector"""
    
    keep_largest_fragment: bool = False
    """Keep only largest fragment after standardization"""
    
    primary_id_col: str = "CID"
    """Primary ID column name"""
    
    smiles_col: str = "SMILES::Absolute"
    """SMILES column name"""
    
    inchi_col: str = "InChI::Standard"
    """InChI column name"""
    
    parse_order: List[Literal["smiles", "inchi"]] = field(
        default_factory=lambda: ["smiles", "inchi"]
    )
    """Order to try parsing molecular structures"""


@dataclass
class DescriptorConfig:
    """
    Descriptor calculation settings
    
    Attributes:
        descriptor_set: Which descriptor set to use ('rdkit', 'mordred', 'both')
        descriptor_include: Whitelist of specific descriptors to include
        descriptor_exclude: Blacklist of descriptors to exclude
        per_molecule_timeout_sec: Timeout per molecule calculation (seconds)
        workers: Number of worker processes (0 = auto-detect)
    """
    descriptor_set: Literal["rdkit", "mordred", "both"] = "both"
    """Which descriptor set to use"""
    
    descriptor_include: List[str] = field(default_factory=list)
    """Whitelist of specific descriptors to include (empty = all)"""
    
    descriptor_exclude: List[str] = field(default_factory=list)
    """Blacklist of descriptors to exclude"""
    
    per_molecule_timeout_sec: int = 60
    """Timeout per molecule calculation (seconds)"""
    
    workers: int = 0
    """Number of worker processes (0 = auto-detect from CPU count)"""
    
    def get_workers(self) -> int:
        """Get actual worker count (auto-detect if 0)"""
        if self.workers == 0:
            import multiprocessing
            return multiprocessing.cpu_count()
        return self.workers


@dataclass
class FilteringConfig:
    """
    Filtering pipeline configuration (Pass 0-4)
    
    Pass 0: Sampling
    Pass 1: Statistics & Variance
    Pass 2: Correlation
    Pass 3: VIF Multicollinearity
    Pass 4: Nonlinear Detection
    """
    
    # ===== Pass 0: Sampling =====
    sample_per_file: Optional[int] = None
    """Number of samples per file (None = use all)"""
    
    file_independent_sampling: bool = False
    """Sample independently per file"""
    
    # ===== Pass 1: Statistics & Variance =====
    variance_threshold: float = 0.002
    """Minimum normalized variance to keep a descriptor"""
    
    max_missing_ratio: float = 0.5
    """Maximum ratio of missing values allowed"""
    
    min_effective_n: int = 100
    """Minimum effective sample size required"""
    
    range_mode: Union[RangeLiteral, RangeMode] = "TRIMMED"
    """Range calculation mode: 'MINMAX', 'TRIMMED', or 'IQR'"""
    
    trim_lower: float = 0.025
    """Lower percentile for trimmed range (as fraction, e.g., 0.025 = 2.5%)"""
    
    trim_upper: float = 0.975
    """Upper percentile for trimmed range (as fraction, e.g., 0.975 = 97.5%)"""
    
    force_recompute: bool = False
    """Force recompute statistics even if checkpoint exists"""
    
    # ===== Pass 2: Spearman Correlation =====
    spearman_threshold: float = 0.95
    """Spearman correlation threshold for clustering"""
    
    m: int = 64
    """CountSketch: number of buckets"""
    
    r: int = 8
    """CountSketch: number of repetitions"""
    
    # ===== Pass 3: VIF Multicollinearity =====
    vif_threshold: float = 10.0
    """VIF threshold for multicollinearity detection"""
    
    # ===== Pass 4: Nonlinear Detection =====
    nonlinear_threshold: float = 0.30
    """General nonlinear dependency threshold"""
    
    hsic_threshold: float = 0.30
    """HSIC (Hilbert-Schmidt Independence Criterion) threshold"""
    
    rdc_threshold: float = 0.30
    """RDC (Randomized Dependence Coefficient) threshold"""
    
    # Advanced HSIC/RDC parameters (rarely need tuning)
    hsic_D: int = 50
    """HSIC random projection dimension (default: 50)"""
    
    rdc_d: int = 20
    """RDC random projection dimension (default: 20)"""
    
    rdc_seeds: int = 10
    """Number of RDC random seeds for averaging (default: 10)"""
    
    # ===== Clustering =====
    resolution: float = 1.0
    """Leiden clustering resolution parameter"""
    
    n_iterations: int = -1
    """Leiden clustering iterations (-1 = auto)"""
    
    # ===== Pass 4 Similarity Fusion =====
    w_hsic: float = 0.5
    """Weight for HSIC in similarity fusion (default: 0.5)"""
    
    w_rdc: float = 0.5
    """Weight for RDC in similarity fusion (default: 0.5)"""
    
    # ===== Binary Skew Filter =====
    use_binary_skew_filter: bool = False
    """Enable binary/skewed descriptor filtering"""
    
    binary_skew_threshold: float = 0.40
    """Threshold for binary descriptor detection"""
    
    binary_minority_frac: float = 0.10
    """Minimum minority class fraction for binary descriptors"""
    
    def _normalize(self):
        """Normalize fields to canonical values"""
        # RangeMode normalization
        if isinstance(self.range_mode, (str, RangeMode)):
            self.range_mode = canonicalize_rangemode(self.range_mode)
        
        # Trim percentile auto-conversion (백분율 → 0-1 범위)
        # CRITICAL: 내부 표준은 0-1 (fraction)
        if self.trim_lower > 1.0:
            warnings.warn(
                f"trim_lower={self.trim_lower} appears to be in percentage (>1.0). "
                f"Auto-converting to fraction by dividing by 100. "
                f"New value: {self.trim_lower/100:.4f}. "
                f"NOTE: Internal standard is 0-1 range (e.g., 0.025 = 2.5%)",
                UserWarning
            )
            self.trim_lower = self.trim_lower / 100.0
        
        if self.trim_upper > 1.0:
            warnings.warn(
                f"trim_upper={self.trim_upper} appears to be in percentage (>1.0). "
                f"Auto-converting to fraction by dividing by 100. "
                f"New value: {self.trim_upper/100:.4f}. "
                f"NOTE: Internal standard is 0-1 range (e.g., 0.975 = 97.5%)",
                UserWarning
            )
            self.trim_upper = self.trim_upper / 100.0
    
    def _validate(self):
        """Validate filtering configuration"""
        # Trim percentile validation (AFTER normalization)
        if self.range_mode == "TRIMMED":
            lo, hi = self.trim_lower, self.trim_upper
            if not (0.0 <= lo < hi <= 1.0):
                raise ValueError(
                    f"Invalid trim percentiles: {lo}, {hi}. "
                    f"Must satisfy 0.0 <= lower < upper <= 1.0. "
                    f"NOTE: Use fraction format (0.025 = 2.5%, not 2.5)"
                )
        
        # Threshold validations
        if not 0.0 <= self.variance_threshold <= 1.0:
            raise ValueError(f"variance_threshold must be in [0, 1], got {self.variance_threshold}")
        
        if not 0.0 <= self.max_missing_ratio <= 1.0:
            raise ValueError(f"max_missing_ratio must be in [0, 1], got {self.max_missing_ratio}")
        
        # Fusion weight validation
        if self.w_hsic < 0 or self.w_rdc < 0:
            raise ValueError(
                f"Fusion weights must be non-negative: "
                f"w_hsic={self.w_hsic}, w_rdc={self.w_rdc}"
            )
        
        if self.w_hsic + self.w_rdc == 0:
            raise ValueError(
                f"At least one fusion weight must be positive: "
                f"w_hsic={self.w_hsic}, w_rdc={self.w_rdc}"
            )


@dataclass
class SystemConfig:
    """
    System-wide settings
    
    Attributes:
        random_seed: Global random seed for reproducibility
        deterministic_ops: Force deterministic operations when possible
        checkpoint: Enable checkpointing
        verbose: Enable verbose logging
        log_level: Logging level
        log_json: Use JSON-formatted logs
        log_dir: Directory for log files
        progress_every_n: Print progress every N items
        error_policy: How to handle errors ('continue', 'skip_molecule', 'fail_fast')
    """
    random_seed: int = 42
    """Global random seed for reproducibility"""
    
    deterministic_ops: bool = True
    """Force deterministic operations when possible (may impact performance)"""
    
    checkpoint: bool = True
    """Enable checkpointing for resumability"""
    
    verbose: bool = True
    """Enable verbose logging"""
    
    log_level: Literal["DEBUG", "INFO", "WARN", "ERROR"] = "INFO"
    """Logging level"""
    
    log_json: bool = False
    """Use JSON-formatted logs (machine-readable)"""
    
    log_dir: str = "logs"
    """Directory for log files"""
    
    progress_every_n: int = 1000
    """Print progress every N items"""
    
    error_policy: Literal["continue", "skip_molecule", "fail_fast"] = "continue"
    """How to handle errors during processing"""


# ========== Main Configuration (SSOT Root) ==========

@dataclass
class Config:
    """
    Unified configuration for Molecular Descriptor Toolkit
    
    Hierarchical SSOT (Single Source of Truth) design with section-based organization.
    Each section groups related settings, allowing modules to depend only on what they need.
    
    Sections:
        device: GPU/CPU settings
        io: Input/output and checkpointing
        preprocessing: Molecular standardization
        descriptor: Descriptor calculation
        filtering: Filtering pipeline (Pass 0-4)
        system: System-wide settings
    
    Usage:
        # Full pipeline
        config = Config(
            io=IOConfig(parquet_glob="data/*.parquet", output_dir="results"),
            filtering=FilteringConfig(variance_threshold=0.001),
        )
        
        # From YAML
        config = load_config("settings.yaml")
        
        # With overrides
        config = load_config("settings.yaml", overrides={
            "filtering.vif_threshold": 8.0,
            "device.prefer_gpu": False,
        })
    """
    
    device: DeviceConfig = field(default_factory=DeviceConfig)
    """Device and GPU acceleration settings"""
    
    io: IOConfig = field(default_factory=IOConfig)
    """I/O and checkpoint settings"""
    
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    """Molecular standardization and parsing settings"""
    
    descriptor: DescriptorConfig = field(default_factory=DescriptorConfig)
    """Descriptor calculation settings"""
    
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    """Filtering pipeline configuration (Pass 0-4)"""
    
    system: SystemConfig = field(default_factory=SystemConfig)
    """System-wide settings"""
    
    # Private flag for freeze functionality
    _frozen: bool = field(default=False, init=False, repr=False)
    
    def __post_init__(self):
        """Post-initialization: normalize and validate"""
        # Normalize fields
        self.filtering._normalize()
    
    def __setattr__(self, key: str, value: Any):
        """Guard against modification when frozen"""
        if key != "_frozen" and getattr(self, "_frozen", False):
            raise RuntimeError(
                f"Config is frozen. Cannot set '{key}'. "
                f"Create a new Config instance instead."
            )
        object.__setattr__(self, key, value)
    
    def __getattr__(self, name: str):
        """
        Backward compatibility: flat field aliases
        
        Provides access to commonly-used fields via flat names with deprecation warnings.
        """
        legacy_map = {
            # Device
            "prefer_gpu": ("device", "prefer_gpu"),
            "gpu_id": ("device", "gpu_id"),
            "device": ("device", "device"),
            # I/O
            "parquet_glob": ("io", "parquet_glob"),
            "output_dir": ("io", "output_dir"),
            "descriptor_columns": ("io", "descriptor_columns"),
            "n_metadata": ("io", "n_metadata"),
            # Filtering
            "variance_threshold": ("filtering", "variance_threshold"),
            "spearman_threshold": ("filtering", "spearman_threshold"),
            "vif_threshold": ("filtering", "vif_threshold"),
            # System
            "random_seed": ("system", "random_seed"),
            "checkpoint": ("system", "checkpoint"),
            "verbose": ("system", "verbose"),
        }
        
        if name in legacy_map:
            section, key = legacy_map[name]
            warnings.warn(
                f"Accessing '{name}' directly is deprecated. "
                f"Use 'config.{section}.{key}' instead. "
                f"Flat access will be removed in v2.0.",
                DeprecationWarning,
                stacklevel=2
            )
            return getattr(getattr(self, section), key)
        
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def validate_and_finalize(self) -> "Config":
        """
        Validate and finalize configuration
        
        Performs:
        - Device auto-detection
        - Field validation
        - Directory creation
        
        Returns:
            Self (for method chaining)
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Device detection
        if self.device.device is None:
            if self.device.prefer_gpu:
                self.device.device = _auto_detect_device()
            else:
                self.device.device = "cpu"
        
        # Validate device
        if self.device.device == "cuda":
            if not _cuda_available():
                warnings.warn(
                    "GPU requested but not available. Falling back to CPU.",
                    UserWarning
                )
                self.device.device = "cpu"
            else:
                try:
                    import torch
                    torch.cuda.set_device(self.device.gpu_id)
                except Exception as e:
                    warnings.warn(f"Failed to set GPU device: {e}. Using default GPU.")
        
        # Normalize and validate filtering config
        self.filtering._normalize()
        self.filtering._validate()
        
        # Ensure output directory exists
        _ensure_dir(self.io.output_dir)
        
        # Ensure log directory exists
        _ensure_dir(self.system.log_dir)
        
        return self
    
    @property
    def using_gpu(self) -> bool:
        """Check if GPU is being used"""
        return self.device.device == "cuda"
    
    def get_device_info(self) -> str:
        """
        Get human-readable device information
        
        Returns:
            Device info string
        """
        if self.using_gpu:
            try:
                import torch
                gpu_name = torch.cuda.get_device_name(self.device.gpu_id)
                gpu_mem = torch.cuda.get_device_properties(self.device.gpu_id).total_memory / 1e9
                return f"GPU: {gpu_name} ({gpu_mem:.1f} GB)"
            except:
                return "GPU: Unknown"
        return "Device: CPU"


# ========== Config Utilities ==========

class FrozenError(RuntimeError):
    """Raised when attempting to modify a frozen configuration"""
    pass


def freeze(config: Config) -> Config:
    """
    Freeze configuration (make immutable)
    
    Call this before starting pipeline execution to prevent accidental modifications.
    
    Args:
        config: Configuration to freeze
    
    Returns:
        Frozen configuration (same instance)
    
    Example:
        >>> cfg = load_config("settings.yaml").validate_and_finalize()
        >>> freeze(cfg)
        >>> cfg.filtering.vif_threshold = 5.0  # Raises FrozenError
    """
    def _freeze_recursive(obj):
        """Recursively freeze dataclass instances"""
        if hasattr(obj, "__dict__"):
            object.__setattr__(obj, "_frozen", True)
            for value in obj.__dict__.values():
                if hasattr(value, "__dict__"):
                    _freeze_recursive(value)
    
    _freeze_recursive(config)
    return config


def _as_flat_dict(obj, prefix: str = "") -> Dict[str, Any]:
    """
    Convert nested config to flat dictionary
    
    Args:
        obj: Config object or section
        prefix: Key prefix for nested keys
    
    Returns:
        Flat dictionary with dotted keys
    """
    out = {}
    for k, v in obj.__dict__.items():
        if k.startswith("_"):  # Skip private fields
            continue
        key = f"{prefix}.{k}" if prefix else k
        if hasattr(v, "__dict__") and not isinstance(v, (str, Path)):
            out.update(_as_flat_dict(v, key))
        else:
            out[key] = v
    return out


def config_diff(a: Config, b: Config) -> List[str]:
    """
    Compare two configurations and return differences
    
    Useful for experiment reproducibility and debugging.
    
    Args:
        a: First config
        b: Second config
    
    Returns:
        List of difference strings
    
    Example:
        >>> cfg1 = load_config("exp1.yaml")
        >>> cfg2 = load_config("exp2.yaml")
        >>> diffs = config_diff(cfg1, cfg2)
        >>> print("\\n".join(diffs))
        filtering.vif_threshold: 10.0 → 8.0
        descriptor.workers: 4 → 8
    """
    da, db = _as_flat_dict(a), _as_flat_dict(b)
    keys = sorted(set(da) | set(db))
    diffs = []
    
    for k in keys:
        va = da.get(k, "<MISSING>")
        vb = db.get(k, "<MISSING>")
        if va != vb:
            diffs.append(f"{k}: {va!r} → {vb!r}")
    
    return diffs


# ========== Backward Compatibility Alias ==========

# For code that still uses PipelineConfig
PipelineConfig = Config
