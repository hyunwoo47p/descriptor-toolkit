"""
Configuration Loader for ChemDescriptorML (CDML)

Supports YAML, JSON, environment variables, and CLI overrides.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from .settings import Config, _as_flat_dict


def load_config(
    path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    env_prefix: str = "CDML_",
) -> Config:
    """
    Load configuration with priority: defaults < file < env < overrides
    
    Priority order (later overrides earlier):
    1. Default values from dataclass definitions
    2. Configuration file (YAML or JSON)
    3. Environment variables (CDML_* prefix by default)
    4. Explicit overrides dictionary

    Args:
        path: Path to YAML or JSON config file (optional)
        overrides: Dictionary of explicit overrides (dotpath format)
        env_prefix: Prefix for environment variables (default: "CDML_")
    
    Returns:
        Loaded and validated Config instance
    
    Examples:
        # From YAML file
        >>> cfg = load_config("settings.yaml")
        
        # From YAML with overrides
        >>> cfg = load_config(
        ...     "settings.yaml",
        ...     overrides={"filtering.vif_threshold": 8.0}
        ... )
        
        # Environment variables
        >>> # CDML_DEVICE_GPU_ID=1
        >>> # CDML_FILTERING_VIF_THRESHOLD=8.0
        >>> cfg = load_config()
        
        # Overrides only (no file)
        >>> cfg = load_config(overrides={
        ...     "io.parquet_glob": "data/*.parquet",
        ...     "io.output_dir": "results",
        ... })
    """
    # Start with defaults
    cfg = Config()
    
    # Load from file if provided
    if path:
        file_data = _load_file(path)
        cfg = _update_from_dict(cfg, file_data)
    
    # Load from environment variables
    env_data = _read_env_overrides(prefix=env_prefix)
    if env_data:
        cfg = _update_from_dict(cfg, env_data)
    
    # Apply explicit overrides
    if overrides:
        cfg = _update_from_dict(cfg, overrides)
    
    # Validate and finalize
    cfg.validate_and_finalize()
    
    return cfg


def _load_file(path: str) -> Dict[str, Any]:
    """
    Load configuration file (YAML or JSON)
    
    Args:
        path: Path to config file
    
    Returns:
        Configuration dictionary
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is unsupported or invalid
    """
    file_path = Path(path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    suffix = file_path.suffix.lower()
    
    # JSON
    if suffix == ".json":
        with open(file_path, 'r') as f:
            return json.load(f)
    
    # YAML
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required to load YAML config files. "
                "Install it with: pip install pyyaml"
            )
        
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
            return data if data is not None else {}
    
    raise ValueError(
        f"Unsupported config file format: {suffix}. "
        f"Supported formats: .json, .yaml, .yml"
    )


def _read_env_overrides(prefix: str = "CDML_") -> Dict[str, Any]:
    """
    Read configuration overrides from environment variables

    Environment variable format:
        {PREFIX}{SECTION}_{KEY}=value

    Type conversion (automatic via JSON parsing):
        - Booleans: "true"/"false" (case-sensitive) → bool
        - Numbers: "42" → int, "3.14" → float
        - Strings: Used as-is (if not valid JSON)

    Examples:
        CDML_DEVICE_GPU_ID=1 → device.gpu_id = 1
        CDML_DEVICE_PREFER_GPU=true → device.prefer_gpu = True
        CDML_FILTERING_VIF_THRESHOLD=8.0 → filtering.vif_threshold = 8.0
        CDML_IO_OUTPUT_DIR=results → io.output_dir = "results"
    
    Args:
        prefix: Environment variable prefix
    
    Returns:
        Dictionary with dotpath keys
    """
    overrides = {}
    
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        
        # Remove prefix and convert to dotpath
        # CDML_DEVICE_GPU_ID → device.gpu_id
        dotpath = key[len(prefix):].lower().replace("_", ".", 1).replace("_", "")
        
        # Try to parse as JSON (handles numbers, booleans, etc.)
        try:
            parsed_value = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            # Keep as string if not valid JSON
            parsed_value = value
        
        overrides[dotpath] = parsed_value
    
    return overrides


def _update_from_dict(cfg: Config, data: Dict[str, Any]) -> Config:
    """
    Update config from dictionary with dotpath keys
    
    Supports both nested dictionaries and dotpath keys:
    
    Nested dict format:
        {
            "device": {"gpu_id": 1},
            "filtering": {"vif_threshold": 8.0}
        }
    
    Dotpath format:
        {
            "device.gpu_id": 1,
            "filtering.vif_threshold": 8.0
        }
    
    Args:
        cfg: Config instance to update
        data: Dictionary with config values
    
    Returns:
        Updated config instance
    
    Raises:
        AttributeError: If dotpath references non-existent section or field
        ValueError: If value type is incompatible
    """
    # Flatten nested dicts to dotpath format
    flat_data = _flatten_dict(data)
    
    # Apply each override
    for dotpath, value in flat_data.items():
        _set_dotpath(cfg, dotpath, value)
    
    return cfg


def _flatten_dict(d: Dict[str, Any], parent_key: str = "") -> Dict[str, Any]:
    """
    Flatten nested dictionary to dotpath format
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
    
    Returns:
        Flattened dictionary
    
    Example:
        >>> _flatten_dict({"a": {"b": 1, "c": 2}})
        {"a.b": 1, "a.c": 2}
    """
    items = []
    
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        
        if isinstance(v, dict):
            # Recurse into nested dict
            items.extend(_flatten_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    
    return dict(items)


def _set_dotpath(obj: Any, dotpath: str, value: Any):
    """
    Set value using dotpath notation
    
    Args:
        obj: Root object
        dotpath: Dotted path (e.g., "device.gpu_id")
        value: Value to set
    
    Raises:
        AttributeError: If path is invalid
    
    Example:
        >>> _set_dotpath(cfg, "device.gpu_id", 1)
        >>> # cfg.device.gpu_id = 1
    """
    parts = dotpath.split(".")
    
    # Navigate to parent
    node = obj
    for part in parts[:-1]:
        try:
            node = getattr(node, part)
        except AttributeError:
            # 허용된 섹션 리스트 제안
            valid_sections = [attr for attr in dir(obj) if not attr.startswith('_') and hasattr(getattr(obj, attr), '__dict__')]
            raise AttributeError(
                f"Invalid config path: '{dotpath}'. "
                f"Section '{part}' not found. "
                f"Valid sections: {', '.join(valid_sections)}"
            )
    
    # Set value
    final_key = parts[-1]
    
    if not hasattr(node, final_key):
        # 허용된 필드 리스트 제안
        valid_fields = [attr for attr in dir(node) if not attr.startswith('_') and not callable(getattr(node, attr))]
        section_name = parts[-2] if len(parts) > 1 else 'root'
        raise AttributeError(
            f"Invalid config path: '{dotpath}'. "
            f"Field '{final_key}' not found in section '{section_name}'. "
            f"Valid fields: {', '.join(valid_fields[:10])}{'...' if len(valid_fields) > 10 else ''}"
        )
    
    try:
        setattr(node, final_key, value)
    except Exception as e:
        raise ValueError(
            f"Failed to set '{dotpath}' = {value!r}: {e}"
        )


def save_config(cfg: Config, path: str, format: Optional[str] = None):
    """
    Save configuration to file
    
    Args:
        cfg: Config instance to save
        path: Output file path
        format: File format ('json' or 'yaml'), auto-detected from extension if None
    
    Examples:
        >>> save_config(cfg, "settings.yaml")
        >>> save_config(cfg, "settings.json")
    """
    file_path = Path(path)
    
    # Auto-detect format from extension
    if format is None:
        suffix = file_path.suffix.lower()
        if suffix == ".json":
            format = "json"
        elif suffix in {".yaml", ".yml"}:
            format = "yaml"
        else:
            raise ValueError(
                f"Cannot auto-detect format from extension '{suffix}'. "
                f"Specify format explicitly: format='json' or format='yaml'"
            )
    
    # Get config as nested dict
    data = _config_to_dict(cfg)
    
    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save
    if format == "json":
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    elif format == "yaml":
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required to save YAML config files. "
                "Install it with: pip install pyyaml"
            )
        
        with open(file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'yaml'.")


def _config_to_dict(cfg: Config) -> Dict[str, Any]:
    """
    Convert Config to nested dictionary (for saving)
    
    Args:
        cfg: Config instance
    
    Returns:
        Nested dictionary representation
    """
    def _dataclass_to_dict(obj):
        """Recursively convert dataclass to dict"""
        if not hasattr(obj, "__dict__"):
            return obj
        
        result = {}
        for k, v in obj.__dict__.items():
            if k.startswith("_"):  # Skip private fields
                continue
            
            if hasattr(v, "__dict__"):
                result[k] = _dataclass_to_dict(v)
            elif isinstance(v, (list, tuple)):
                result[k] = [_dataclass_to_dict(x) for x in v]
            elif isinstance(v, Path):
                result[k] = str(v)
            else:
                result[k] = v
        
        return result
    
    return _dataclass_to_dict(cfg)


def check_unknown_keys(data: Dict[str, Any], cfg: Config) -> list[str]:
    """
    Check for unknown keys in config data
    
    Useful for catching typos in config files.
    
    Args:
        data: Configuration dictionary (can be nested)
        cfg: Config instance (for validation)
    
    Returns:
        List of unknown keys (dotpath format)
    
    Example:
        >>> data = {"device": {"gpu_idd": 1}}  # typo
        >>> unknown = check_unknown_keys(data, Config())
        >>> print(unknown)
        ['device.gpu_idd']
    """
    flat_data = _flatten_dict(data)
    valid_keys = set(_as_flat_dict(cfg).keys())
    unknown = [key for key in flat_data if key not in valid_keys]
    return unknown
