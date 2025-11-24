"""Configuration module for Molecular Descriptor Toolkit"""

from molecular_descriptor_toolkit.config.settings import (
    Config,
    DeviceConfig,
    IOConfig,
    PreprocessingConfig,
    DescriptorConfig,
    FilteringConfig,
    SystemConfig,
    RangeMode,
    RangeLiteral,
    canonicalize_rangemode,
    freeze,
    config_diff,
    FrozenError,
    PipelineConfig,  # Backward compatibility alias
)

from molecular_descriptor_toolkit.config.loader import (
    load_config,
    save_config,
    check_unknown_keys,
)

__all__ = [
    # Main config
    "Config",
    "PipelineConfig",  # Alias for backward compatibility
    
    # Sections
    "DeviceConfig",
    "IOConfig",
    "PreprocessingConfig",
    "DescriptorConfig",
    "FilteringConfig",
    "SystemConfig",
    
    # Types
    "RangeMode",
    "RangeLiteral",
    
    # Utilities
    "canonicalize_rangemode",
    "freeze",
    "config_diff",
    "FrozenError",
    
    # Loader
    "load_config",
    "save_config",
    "check_unknown_keys",
]
